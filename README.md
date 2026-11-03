# GB200 SM_100A TMA Fused Causal Attention Softmax

Rust · NVIDIA cuda-oxide · FP16 I/O · FP32 math · Blackwell `sm_100a`

**Build status:** All five kernels and the probe compile to `sm_100a` PTX and assemble with `ptxas` (Rust 1.99.1, cuda-oxide v0.3.0, CUDA 13.0). Compute kernels use **20–31 registers/thread with zero reported spills**. [Linux CI (2026-11-03)](https://github.com/R-P-Padhi/GB200-SM100A-TMA-Fused-Causal-Attention-Softmax/actions/runs/41595787096) passed **45 Rust tests, 28 Python tests**, and a fresh SM100a static build. The local PTX, cubin and assembly log are in [`evidence/`](evidence/). Device correctness and GB200 performance have not been measured.

## Scope

- Input: precomputed attention scores `S ∈ FP16[M,N]`, row-major, optionally padded to an 8-element stride. This project does **not** compute `Q × Kᵀ`, multiply by `V`, or implement FlashAttention.
- Output: `P[r,c] = exp(scale*S[r,c] - max_r) / sum_{j<=r} exp(scale*S[r,j] - max_r)` for `c<=r`, otherwise `0`, rounded to FP16. Causal indices are absolute, starting at zero; no KV-cache offset or arbitrary mask.
- Precision: FP16 storage; FP32 scaling, maximum, approximate exponential (`ex2.approx.f32`), sum, normalization. Finite input and scale in `[0, 1e30]` required. Zero scale gives a uniform causal distribution.
- Default `4096×4096` (32 MiB per dense FP16 matrix); stress `8192×8192` (128 MiB per matrix). Small odd and rectangular shapes supported through 16-byte-aligned pitch and TMA out-of-bounds fill.
- TMA work assignment: every CTA has 256 threads (eight warps). The three-pass TMA kernels handle `4×256` tiles with eight-warp reductions, in three streaming passes (max → sum → output). The **online TMA** kernel handles `8×128` tiles with one warp per row, four values/lane, and **two passes** (running max+sum → output). Both designs fuse computation into one GPU launch.
- Pipelining: two aligned FP16 shared input tiles and two mbarriers alternate G2S transfers. The online variant uses one producer arrival per barrier phase and parity waits; the three-pass variant uses 256 arrivals and token waits. **Every TMA load registers expected bytes before issuing the asynchronous copy.** CTA synchronization prevents buffer reuse before consumption; S2G stores wait for bulk-group completion. Overlap is a hypothesis, not a benchmark result.
- The runtime builds **separate tensor maps** for `4×256` and `8×128`; reusing the wrong descriptor would copy the wrong geometry. TMA source/storage and descriptor operations are inherently unsafe low-level intrinsics; host interfaces and CPU math are ordinary safe Rust. Safety comments identify manually indexed device stores.

## Kernels

| Kernel | Rows/CTA | Transfer | Purpose |
|---|---:|---|---|
| `causal_softmax_baseline` | 1 | Ordinary global loads/stores | Simple comparison/control kernel |
| `causal_softmax_warp` | 8 | Ordinary global loads/stores | Narrow matrices (`N <= 32`), one warp per row |
| `causal_softmax_tma` | 4 | 2D TMA, one shared input tile | Single-buffered TMA control |
| `causal_softmax_tma_pipelined` | 4 | 2D TMA, two input tiles | In-flight next-tile load while computing current tile |
| `causal_softmax_tma_online` | 8 | 2D TMA, two input tiles | Warp-per-row online max/sum, two source passes |

Default dispatch: warp for `N <= 32`, otherwise two-pass online TMA. `--kernel baseline|warp|tma|pipeline|online` explicitly selects a variant.

## Files

| Path | Content |
|---|---|
| `gpu/src/main.rs` | five GPU kernels, probe, tensor-map encoder |
| `gpu/src/driver.rs` | typed host dispatch, buffer ownership, GPU tests and dispatch |
| `gpu/src/driver/bench.rs` | CUDA-event benchmarks, transfers, queue depths and stream concurrency |
| `host/src/lib.rs` | GPU-independent CLI, patterns, numerical verifier and benchmark stats |
| `host/examples/cpu_bench.rs` | CPU-only reference timing harness |
| `reference/src/lib.rs` | independent CPU spec, IEEE half conversion, padding, FP64 oracle and tests |
| `reference/src/online.rs` | warp butterfly + two-pass tiled arithmetic model and edge tests |
| `scripts/check_ptx.py` | inspect each entry separately; reject comments and missing TMA/warp instructions |
| `scripts/check_resources.py` | per-kernel `ptxas` gate: <=128 registers, zero spills, >=25% register-limited occupancy; report stack separately |
| `scripts/test_validation.py` | 21 failure-oriented tests, including TMA copy-before-barrier-registration regression |
| `scripts/test_benchmark_reports.py` | seven benchmark-log parser tests |
| `scripts/summarize_bench.py` | convert benchmark logs to typed JSON |
| `scripts/source_manifest.py` | SHA-256 manifest of source and toolchain inputs for CI evidence |
| `scripts/bootstrap.py`, `scripts/cuda-packages.json` | optional local driverless CUDA 13 setup with official package SHA-256 pins |
| `.github/workflows/ci.yml` | locked CPU tests, formatting, GPU-free SM100a compile/assemble |
| `docs/architecture.md` | execution layout, synchronization, constraints |
| `docs/benchmarking.md` | benchmark workload matrix, clocks and result interpretation |
| `docs/evaluation.md` | report material, slide outline, viva answers |
| `docs/validation.md`, `evidence/` | reproducible compile/test results, archived PTX/cubin/ptxas logs |
| `docs/sources.md` | pinned cuda-oxide API/source map and architecture compatibility |
| `THIRD_PARTY_NOTICES.md` | external dependency and CUDA license ownership |

## Toolchain

- NVIDIA official [CUDA Rust](https://github.com/NVIDIA/cuda-rust), **cuda-oxide v0.3.0**, commit [`99e571f49baaefe4cd69157950c89ac9d4012728`](https://github.com/NVIDIA/cuda-rust/commit/99e571f49baaefe4cd69157950c89ac9d4012728) pinned for the CUDA crates and `cargo-oxide`. This project uses cuda-oxide's single-source `#[cuda_module]` APIs; not the unrelated older Rust-CUDA stack.
- Rust **1.99.1** with `rust-src`, `rustc-dev`, `llvm-tools`; `.cargo/config.toml` enables `RUSTC_BOOTSTRAP=1` for upstream's experimental f16 primitive/codegen. Rust 1.89 is **not** the supported toolchain for this release.
- Linux x86_64 static compilation was **tested locally** with a checksum-verified CUDA Toolkit **13.0** subset (including `ptxas`), libclang 18.1.1 and the pinned Rust/cuda-oxide toolchain; CI uses Ubuntu 24.04, CUDA 13.3.1 and Clang/LLVM 21. No Blackwell GPU is required for `cargo oxide build`, PTX inspection, or `ptxas` assembly. GPU execution requires B200/GB200 with a compatible driver.
- Optional Linux-only local bootstrap: `python3 scripts/bootstrap.py` downloads pinned CUDA 13.0 developer packages (SHA-256 checked) under ignored `.tools/`, and installs the **same pinned** `cargo-oxide`. It does not install a GPU driver; Windows users can rely exclusively on GitHub Actions. This optional route uses pinned libclang 18.1.1; CI uses system Clang 21.
- `sm_100a` is architecture-specific; binaries are not portable across other SM series. Compatibility for this **build**, not cuda-oxide generally:

  | GPU | Compute capability | This kernel |
  |---|---:|---|
  | B200 / GB200 | 10.0 (`sm_100a`) | Target; execution awaiting validation |
  | B300 / GB300 | 10.3 (`sm_103a`) | Not supported |
  | H100 / H200 | 9.0 (`sm_90`) | Not supported |
  | A100 / RTX 4090 / GTX 1650 | 8.0 / 8.9 / 7.5 | Not supported |
  | RTX 5090 | 12.0 (`sm_120a`) | Not supported |

## Commands

```bash
# CPU + test-tool unit tests (host-only):
cargo test --locked -p causal-reference -p softmax-host
cargo fmt --all -- --check
cargo clippy --locked -p causal-reference -p softmax-host -- -D warnings
python3 -m unittest discover -s scripts -p 'test_*.py' -v

# Linux CI/installed CUDA Rust toolchain, no GPU required for building:
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git \
  --rev 99e571f49baaefe4cd69157950c89ac9d4012728 cargo-oxide
cd gpu
CUDA_OXIDE_TARGET=sm_100a cargo oxide build --arch sm_100a
cd ..
cargo clippy --locked -p gb200-tma-causal-softmax --all-targets -- -D warnings
python3 scripts/check_ptx.py gb200_tma_causal_softmax.ptx
python3 scripts/check_resources.py gb200_tma_causal_softmax.ptx \
  --output sm100a.cubin --log sm100a.ptxas.log

# On a matching B200/GB200 only:
cd gpu
cargo oxide run --arch sm_100a -- --probe
cargo oxide run --arch sm_100a -- --gpu-tests
cargo oxide run --arch sm_100a -- --gpu-stress
cargo oxide run --arch sm_100a -- --rows 4096 --cols 4096 --scale 0.125
cargo oxide run --arch sm_100a -- --stress
cargo oxide run --arch sm_100a -- --kernel baseline --rows 257 --cols 513
cargo oxide run --arch sm_100a -- --kernel online --rows 137 --cols 259 --scale 0
cargo oxide run --arch sm_100a -- --bench --iterations 30 --warmup 5
cargo oxide run --arch sm_100a -- --bench --kernel pipeline --stress
cargo oxide run --arch sm_100a -- --help
```

`cargo oxide run` needs a compatible B200/GB200. The CI job uses `cargo oxide build` and does not launch GPU code.

## Static pass/fail gates

1. Rust CPU reference tests complete.
2. `sm_100a` PTX contains all **six entries** (probe + five kernels); all TMA entries individually contain 2D loads/stores, barriers, async proxy fences, FP32 exponent, bulk wait, and **expected-byte registration before TMA loads**. Online additionally requires parity waits, FP16 conversions and warp shuffles. Comments cannot satisfy these checks.
3. CUDA 13 `ptxas` assembles the PTX for `sm_100a`; **every** entry reports <=128 registers, zero spill loads/stores and register-limited occupancy >=25%. `ptxas` also reports compiler stack scratch (32–40 B for TMA variants) and PTX local-memory instructions; **these are not reported register spills**, but may still cost memory traffic.
4. If target assembly fails, register count is unknown, or any required instruction is absent, CI fails. `ptxas` success is **not** GPU runtime correctness. No GPU job runs automatically on GitHub-hosted CPU runners.

## GPU correctness and benchmarking

- `--gpu-tests`: varied row/column shapes crossing 8/32/256/512/1024 boundaries; zero, spike and pseudorandom patterns; compare every supported kernel to CPU reference.
- `--gpu-stress`: additional shapes (including 8192-wide rectangles), seven patterns, scales `0/0.125/0.5/1.0/1e30`. Case failures include kernel, shape, pattern and scale.
- Validator checks every FP16 probability, exact future-token zeros, finite/nonnegative output and row sums. TMA paths also require zeroed pitch padding; device allocations begin with a nonzero sentinel to detect missing writes.
- `--bench`: warmups, minimum/median/p95/mean synchronized **host wall-clock** timing. CPU comparison is enabled by default; `--no-verify` opts out. Estimated logical traffic is **8 bytes/element for three-pass TMA**, **6 for online TMA**; the scalar baseline (three causal-prefix reads plus dense output) and narrow-warp variant (one causal-prefix read plus dense output) use shape-aware counts. The current TMA kernels still transfer causally masked input tiles; masking suppresses the arithmetic, not the copy. Padding, caches and real transactions are excluded; this is neither a hardware DRAM counter nor CUDA-event timing.

## Boundaries / further work

- `ex2.approx.f32` is an approximate FP32 instruction; GPU comparisons allow 0.003 absolute + 0.003 relative difference from the CPU reference. Run hardware/sanitizer tests before claiming exact accuracy or safety.
- TMA load/store descriptors assume 16-byte-aligned pointers and row pitches. Host padding satisfies pitch; CUDA device allocations supply aligned base addresses. Workloads with nonfinite values are rejected by the CPU spec, and not defined for the kernel.
- The 128-register and zero-spill **static gates passed**: the five compute kernels use 20/26/29/30/31 registers as detailed in `docs/validation.md`. The register-only occupancy calculation is not a measured residency or throughput result. Shared-memory, block/SM, register allocation granularity and architecture limits can reduce realized occupancy.
- Still no cross-block row reductions, arbitrary masks, score tensor generation, tensor-core MMA, GB300 compilation or real hardware timing. CUDA-event/Nsight measurements and further tuning follow hardware correctness and sanitizer tests on a B200/GB200.

## Benchmarking

`--bench-suite` measures CUDA-event kernel latency (min/median/p95/p99/mean/stddev/CV), logical bandwidth, elements/s, transfer times, device-to-device copies, queue depth (1/4/16/64) and independent-stream throughput (1/2/4/8 streams). It also sweeps shapes and score distributions. Correctness checks run outside timed windows; `--bench` retains the synchronized host-wall comparison.

- On a B200/GB200, from `gpu/`: `cargo oxide run --arch sm_100a -- --bench-suite --iterations 40 --warmup 10`.
- Add `--stress` for 8192×8192 or `--kernel online` to select one variant.
- CPU-only CI runs `softmax-host --example cpu_bench` and uploads timing logs; these measure the CPU reference, not the GPU.

Workloads, timing definitions and remaining measurements: [docs/benchmarking.md](docs/benchmarking.md).

## Sources

- [NVIDIA CUDA Rust official repo](https://github.com/NVIDIA/cuda-rust), [cuda-oxide installation](https://github.com/NVIDIA/cuda-rust/blob/cuda-oxide/v0.3.0/cuda-oxide/cuda-oxide-book/getting-started/installation.md), [upstream TMA example](https://github.com/NVIDIA/cuda-rust/blob/cuda-oxide/v0.3.0/cuda-oxide/crates/rustc-codegen-cuda/examples/tma_copy/src/main.rs), [warp reduction example](https://github.com/NVIDIA/cuda-rust/blob/cuda-oxide/v0.3.0/cuda-oxide/crates/rustc-codegen-cuda/examples/warp_reduce/src/main.rs).
- [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/), [CUDA Tensor Map API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html).

MIT-licensed project source. NVIDIA's external crates retain their own Apache-2.0 license.

