# Evaluation brief

## Abstract

An experimental **GB200/B200 `sm_100a`** causal attention-score softmax in Rust using NVIDIA cuda-oxide v0.3.0. Five GPU variants cover global-memory baseline, narrow warp specialization, single-buffer TMA, three-pass double-buffered TMA, and **two-pass online double-buffered TMA**. All take FP16 attention scores and emit FP16 causal probabilities using stable FP32 math. This is *not* FlashAttention; it does not multiply `Q×Kᵀ` or apply `P×V`.

## Evidence matrix

| Claim | How to verify | Status as of 2026-11-02 |
|---|---|---|
| CPU reference, tiled model and binary16 conversion | `cargo test --locked -p causal-reference` | **21 tests passed** locally; FP16 bit-pattern roundtrip, FP64 oracle, online recurrence and warp butterfly |
| Host CLI, inputs, verifier and statistics | `cargo test --locked -p softmax-host` | **21 tests passed** locally; includes causal-prefix benchmark traffic |
| PTX checker and resource parser | `python3 -m unittest discover -s scripts -p 'test_*.py'` | **21 tests passed** locally; per-entry signatures, barrier ordering, incomplete logs, zero-spill enforcement and wrong targets |
| Rust syntax/format and host Clippy | `cargo fmt --all -- --check` and `cargo clippy --locked -p causal-reference -p softmax-host -- -D warnings` | **Passed** locally |
| Device Rust typechecking / PTX | `cd gpu && cargo oxide build --arch sm_100a` | **PASS** locally with pinned Rust 1.99.1, cuda-oxide v0.3.0, CUDA 13.0 |
| TMA, barriers, shuffles, exponent in PTX | `python3 scripts/check_ptx.py gb200_tma_causal_softmax.ptx` | **PASS**, all six PTX entries including three TMA variants |
| `ptxas` assembly, registers ≤128, zero spills | `python3 scripts/check_resources.py gb200_tma_causal_softmax.ptx` | **PASS**, 20–31 compute registers, zero spills; explicit 32–40 B stack scratch for TMA |
| Correct output on B200/GB200 | `--gpu-tests` then `--gpu-stress` | **Not run**; hardware unavailable |
| Speedup versus baseline | `--bench` on matching hardware | **Not measured** |

The GitHub Actions Linux job is configured to run CPU tests and, separately, CUDA 13 static compile/assembly without a GPU. **Local CUDA 13.0 compilation and assembly passed**; a committed workflow is still not evidence that its remote job passed. Attach an actual run URL when it does. The generated local PTX, cubin and ptxas log are included under `evidence/`.

The independently uploaded **older** two-kernel codebase had archived `sm_100a` PTX and a `ptxas` log showing a 29-register, zero-spill online TMA entry. Its source pin is `b6315307`; this merged workspace uses the v0.3.0 release pin `99e571f4` and has extra kernels/changes. The old report was **not** reused: a fresh assembler report now independently records 29 registers and zero spills for the **merged** online kernel.

## GPU test matrix

- Quick suite: 14 irregular/rectangular shapes (1×1 through 64×65, 1×8, 2×17, 10×257, 17×1024), three patterns, 0.125 scale. Every applicable kernel is compared against CPU output.
- Stress suite: 18 shapes, including 129×8192 and 257×257, seven deterministic patterns, scales `0/0.125/0.5/1/1e30`; all five applicable kernel variants are compared against CPU results.
- Verifier: outputs finite and nonnegative; future-token entries *exactly* zero; elementwise 0.003 absolute + 0.003 relative tolerance; row sums approximately one. TMA paths check padded columns; all output allocations start with nonzero sentinels to catch unwritten cells.
- Benchmarks: `--bench --warmup 5 --iterations 30`, CPU verification on by default, synchronized wall-clock min/median/p95/mean and estimated logical effective bandwidth. Three-pass TMA uses 8 B/element; online TMA uses 6 B/element; baseline and warp use shape-specific causal-prefix load counts. These estimates exclude padding and caching, include host timing overhead and are **not** CUDA-event timings or hardware DRAM counters.

The large `8192×8192` workload is a *manual* stress/benchmark preset, not a default correctness-CI test. First test 1×1 and irregular small shapes on actual hardware before attempting memory-intensive cases.

## Suggested presentation: 10 slides

1. Scope and boundaries: softmax of existing attention scores, not full attention.
2. FP16 data, FP32 arithmetic, scaled causal-softmax formula.
3. Why NVIDIA B200 `sm_100a`, Rust and official cuda-oxide release pin.
4. CPU reference and adversarial test patterns.
5. Baseline: one CTA per row, eight warp partials.
6. Specialization: one warp per short row, eight rows/CTA.
7. 2D TMA single-buffer tile geometry and descriptors.
8. Double-buffering and online max/sum: producer-only barrier arrival, parity waits, two versus three passes.
9. Validation gates: source tests, PTX signatures, ptxas spill/register caps, GPU suite.
10. Results **only once measured**, limitations and next experiments.

## Viva questions and direct answers

**What is a CUDA kernel?** A function run by many GPU threads in parallel. A block is a cooperating group; here it always has 256 threads.

**Why is this not FlashAttention?** It receives the score matrix. FlashAttention also computes attention products with Q/K/V while minimizing intermediate movement.

**Why subtract the maximum?** `exp(x - max(x))` leaves normalized probabilities unchanged and prevents positive exponential overflow.

**Why FP32 inside an FP16 pipeline?** Maximum, exponent and accumulation need more range/precision than FP16 supplies.

**How does the causal mask work?** Score `(r,c)` contributes only if `c <= r`; masked positions get zero probability.

**What is TMA?** Tensor Memory Accelerator, an asynchronous tensor-descriptor-driven data transfer between global and shared memory.

**What does an mbarrier do?** Lets cooperating threads know when an asynchronous transfer has completed before they consume shared data.

**What exactly overlaps?** The double-buffer candidate starts the next G2S load into the alternate input tile before processing the current tile.

**Why keep a baseline?** It provides a simpler, independent GPU comparison for correctness and eventually performance; optimizations can be wrong or slower.

**Why specialize narrow rows?** A single warp can reduce up to 32 score columns directly without an eight-warp shared-memory reduction.

**Why three versus two passes?** The control TMA kernels separately stream max, denominator and output. Online statistics update `(max,sum)` together, so the online variant requires one source traversal for statistics and one for normalization.

**How does online softmax stay stable?** Rescale the old denominator by `exp(old_max - new_max)` when a tile increases the running maximum; add the tile's exponentials relative to that new maximum.

**Why do the mbarriers differ?** One algorithm uses a collective 256-thread arrival, while online TMA has one elected producer arrival. In both, all consumers must wait until TMA has filled the current shared tile.

**Is `unsafe` eliminated by Rust?** No. CUDA intrinsics and disjoint device writes require carefully scoped `unsafe`; static checks and GPU sanitizer tests are still needed.

**What does `ptxas -v` prove?** It reports assembled code's register and spill allocations and catches assembler errors, not data-race freedom or correct output.

**What is 25% occupancy?** A theoretical occupancy target derived from resident warps/threads under resource limits; occupancy is not speedup.

**Why not report GB200 performance now?** This environment has no GB200 and cannot run `sm_100a` code; a fabricated benchmark would not be an engineering result.

**What should be done next?** Confirm the remote CI run, then run the quick GPU suite and Compute Sanitizer on Blackwell; only after device correctness passes, compare variants with CUDA events and Nsight Compute.
