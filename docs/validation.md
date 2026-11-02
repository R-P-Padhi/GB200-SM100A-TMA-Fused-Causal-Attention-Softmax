# Validation record — 2026-11-02

**Environment:** Linux x86_64, pinned Rust **1.99.1**, NVIDIA cuda-oxide **v0.3.0** (`99e571f4`), CUDA Toolkit **13.0** developer components, `ptxas` for `sm_100a`, `libclang` 18.1.1. Dependencies were installed into ignored `.tools/` by `python3 scripts/bootstrap.py`; the CUDA package checksums were validated. **No B200/GB200 was present**.

| Command | Result | What it establishes |
|---|---|---|
| `cargo test --locked -p causal-reference -p softmax-host` | **PASS, 42/42** | 21 CPU/reference/online tests + 21 host/config/numerical/traffic-model tests |
| `python3 -B -m unittest discover -s scripts -p 'test_*.py' -v` | **PASS, 21/21** | PTX/ptxas parsers reject missing entries, wrong target, misplaced instructions, TMA copy-before-expect, spills and excess registers |
| `cargo fmt --all -- --check` | **PASS** | Rust formatting |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | **PASS** | All three crates, including GPU host code, compile without Clippy warnings |
| `cd gpu && cargo oxide build --arch sm_100a` | **PASS** | Rust GPU module compiled and emitted `gb200_tma_causal_softmax.ptx` |
| `python3 scripts/check_ptx.py gb200_tma_causal_softmax.ptx` | **PASS** | All 6 entries; TMA instructions, expect-before-copy text order, barrier/fence/exp; online parity waits and shuffles |
| `python3 scripts/check_resources.py ... --output evidence/sm100a.cubin --log evidence/sm100a.ptxas.log` | **PASS** | Assembled cubin, per-entry register cap, no spills, register-limited occupancy bound |
| Blackwell numerical tests, sanitizer, GPU timing | **NOT RUN** | Requires physical B200/GB200 with driver |

## Exact assembler allocation

From [`evidence/sm100a.ptxas.log`](../evidence/sm100a.ptxas.log) generated against this **merged** source:

| GPU entry | Registers/thread | Spill stores | Spill loads | Stack frame | Shared memory |
|---|---:|---:|---:|---:|---:|
| `probe` | 8 | 0 B | 0 B | 0 B | 0 B |
| `causal_softmax_baseline` | 26 | 0 B | 0 B | 0 B | 36 B |
| `causal_softmax_warp` | 20 | 0 B | 0 B | 0 B | 0 B |
| `causal_softmax_tma` | 31 | 0 B | 0 B | 32 B | 4,232 B |
| `causal_softmax_tma_pipelined` | 30 | 0 B | 0 B | 32 B | 6,288 B |
| `causal_softmax_tma_online` | 29 | 0 B | 0 B | 40 B | 6,160 B |

All register counts are at most 128. The register-only estimator in `check_resources.py` gives a 100% ceiling at the maximum 31 registers/thread; **this is not achieved occupancy**. Importantly, TMA kernels still contain local-memory instructions for their explicit **32–40 B stack frames**. They are **not counted as register spills by ptxas** and may still affect speed. We preserve this distinction instead of claiming zero local-memory traffic.

## Second-pass repairs

The first merged revision registered expected TMA bytes **after** issuing the copy in the single- and double-buffered three-pass kernels. That ordering can race completion with transaction accounting; all TMA loaders now register bytes **before** the asynchronous copy. The PTX checker gained a regression test for this source/assembly ordering. This is still a static guard, **not** a substitute for GPU race testing.

The previous benchmark assigned 8 bytes/element to the narrow-warp and scalar baseline kernels, overstating their requested input traffic. The host now computes logical traffic separately for causal-prefix reads and dense stores, while preserving 8 B/element for three-pass TMA and 6 B/element for online TMA. New tests cover square, short and tall matrices. The CI reference job now explicitly installs `rustfmt`/`clippy`, and the device job additionally lints the GPU host crate.

## Evidence and reproducibility

The [`evidence/`](../evidence/) directory contains `sm100a.ptx`, `sm100a.cubin`, `sm100a.ptxas.log` and `source-sha256.json`. These are generated device artifacts; they have **not** been launched on a GPU. CI independently recompiles from source and uploads generated PTX, cubin, ptxas and a SHA-256 source manifest. A future GitHub Actions badge must refer to an **actual completed run**, not merely the existence of `.github/workflows/ci.yml`.

After setting up a Linux CUDA toolchain, reproduce from repo root:

```bash
python3 scripts/bootstrap.py                  # optional for non-CUDA Linux hosts
source .tools/env.sh                           # only when using bootstrap
cd gpu && cargo oxide build --arch sm_100a && cd ..
python3 scripts/check_ptx.py gb200_tma_causal_softmax.ptx
python3 scripts/check_resources.py gb200_tma_causal_softmax.ptx \
    --output sm100a.cubin --log sm100a.ptxas.log
```

If cuda-oxide or CUDA is already installed, omit the bootstrap and use the existing configured environment. On Windows, the repository's Linux CI performs these steps without WSL or a local CUDA installation.

## What remains unproven

1. Actual accuracy on B200/GB200, especially odd sizes, padded tails, zero scale and large exponent gaps. The host has `--gpu-tests` and `--gpu-stress`, but **neither was executed**.
2. Synchronization and memory safety under Compute Sanitizer. PTX and resource checks cannot detect TMA barrier races.
3. Performance: CUDA-event latency, Nsight counters, realistic occupancy, DRAM transactions, and any speedup over the baseline. Current `--bench` produces **host wall-clock time** only if run on a matching GPU.
4. The CI CUDA 13.3.1/Clang 21 container still needs its own recorded run; this document attests to the local CUDA 13.0 compilation, not remote CI success.

## Earlier project comparison

The supplied older project (source revision `b631530700a8a4a5f165456f2d71fe9195b323b5`) introduced the useful online 8×128 TMA algorithm and had valid archived 29-register/zero-spill assembly evidence, but lacked the breadth of this merged host/reference, five-kernel dispatch and adversarial validators. The online algorithm and source-pinned API audit were retained; the merged revision was **recompiled independently**, so the new `evidence/` files are attributable to the code in this archive.
