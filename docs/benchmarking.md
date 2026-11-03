# Performance benchmark protocol

Keep three timing sources separate: **CUDA events** for device work, **host wall-clock** for copies/launch/sync, and **CPU wall-clock** for the reference. Speedups must use comparable clocks and workloads. GPU figures require a B200/GB200 and a compatible driver; CPU-only GitHub runners collect reference timing and static assembly data.

## Workloads and measured dimensions

| Command / category | What the executable measures | Timing source |
|---|---|---|
| --bench | Five available kernel variants, single shape, launch+sync | Host wall |
| --bench-suite: shape | 1x1, 31x31, 32x32, 64x64, 128x257, 256x256, 512x512, 1024x1024, 128x4096, 4096x128, 2048x2048, 4096x4096, specified shape; include 8192x8192 with --stress | CUDA events |
| --bench-suite: queue | 1, 4, 16, 64 back-to-back launches per timed event window, per-launch amortized | CUDA events |
| --bench-suite: transfer | Blocking H2D, D2H, H2D+kernel+D2H on preallocated buffers | Host wall |
| --bench-suite: device_copy | Device-to-device pitched FP16 memcpy baseline (one read, one write) | CUDA events |
| --bench-suite: pattern | Zeros, ramp, alternating, spikes, extreme, random; scales 0, 0.125, 1, 1e30 | CUDA events |
| --bench-suite: concurrency | 1, 2, 4, 8 nonblocking CUDA streams, independent buffers and four queued launches/stream | Host wall, synchronized across all streams |
| cpu_bench | CPU reference vs CPU online model on nine shapes and two input distributions | CPU host wall |

Every sample group reports: sample count, minimum, mean, median, p95, p99, maximum, population standard deviation and coefficient of variation. Additional metrics include:

- Output elements/s, effective rows/s (derive as elements/s divided by columns), model-based logical GB/s, and speedup versus baseline for each shape when the baseline variant is included.
- H2D and D2H pageable host-transfer latency, end-to-end latency and throughput, plus a device-to-device copy bandwidth baseline.
- Concurrent kernel launches/s, 1/2/4/8-stream aggregate throughput, speedup relative to one stream.
- Queue-depth effects, rectangular/ragged/tiny shape performance, pattern and scale sensitivity.
- Correctness against the CPU reference unless --no-verify is requested, plus the independent GPU stress suite.
- Static register counts, spill loads/stores, shared-memory footprint and register occupancy bound from ptxas (not GPU runtime numbers).

The logical-byte estimate is derived from per-kernel FP16 read/write/pass counts in softmax-host; it is **not** measured DRAM traffic or a literal physical read/write instruction count. In the present online TMA code, all column tiles are copied twice even when the causal mask excludes their values from the softmax; the 6 B/element figure models these dense tile transfers and the dense output. TMA padding, cache effects and memory transactions change actual traffic. NVIDIA Nsight Compute hardware counters are required for per-direction DRAM byte counts, achieved bandwidth, cache hit rates, SM occupancy, stalls and memory-pipeline overlap.

## Commands on B200 / GB200

With the pinned toolchain from the root README and a compatible GPU, run from `gpu/`:

```bash
cargo oxide run --arch sm_100a -- --gpu-tests
cargo oxide run --arch sm_100a -- --gpu-stress
cargo oxide run --arch sm_100a -- --bench-suite --iterations 40 --warmup 10
cargo oxide run --arch sm_100a -- --bench-suite --stress --iterations 20 --warmup 5
cargo oxide run --arch sm_100a -- --bench-suite --kernel online --rows 4096 --cols 4096 --iterations 100
cargo oxide run --arch sm_100a -- --bench --kernel auto --iterations 100
```

Each result is a machine-readable `BENCH key=value` line. Record the Git commit, GPU model, CUDA driver/toolkit, clock and thermal state, power limit and competing workloads. Warm up first; repeat the full sweep three times to check variance.

From the repository root on CPU-only Linux or GitHub Actions:

    cargo test --locked -p causal-reference -p softmax-host
    cargo run --locked --release -p softmax-host --example cpu_bench -- --iterations 5
    python3 -m unittest discover -s scripts -p 'test_*.py' -v

CI saves `cpu-bench.log` and a parsed `cpu-bench.json` using `scripts/summarize_bench.py`. These cover CPU reference work only. Shared-runner timing is variable, so CI does not enforce a performance threshold.

## Interpreting results

- CUDA-event elapsed time surrounds only device enqueue operations in one stream. For queued batches the elapsed time is divided by the number of launches, which gives steady-state amortized latency, not isolated single-launch latency.
- Host-wall kernel time includes enqueue overhead plus synchronization and must not be compared directly with event time.
- Concurrent-stream numbers use distinct allocations and stream synchronization after each wave. Kernel overlap is permitted, not guaranteed. The speedup compares aggregate kernels/s, since 8 streams do eight times the work of one stream per wave.
- Host copies use safe synchronous cuda-core APIs with pageable host vectors. These are **not** pinned-memory overlap benchmarks, nor do they isolate PCIe/NVLink from the driver overhead.
- Accuracy checks occur before or after timed loops, not inside the CUDA-event window. Profiling with an external tool can perturb time and should be kept separate.
- A lower register count does not itself prove better speed; compare event latency, achieved occupancy and actual hardware counters.
- Use `NOT MEASURED` for missing GPU results, not numeric zero.

## Results status

| Evidence | Status |
|---|---|
| Rust/host tests, compiler, PTX/ptxas static checks | GitHub Actions Validation workflow; refer to current run for status |
| CPU reference performance | Recorded by cpu-bench.log CI artifact; machine-dependent |
| GB200 CUDA-event timings / streams / transfers | **NOT MEASURED: no GB200 GPU available in this environment** |
| Nsight DRAM read/write transactions, L2 behavior and SM stalls | **NOT MEASURED: requires GB200 hardware and profiler** |
