# Toolchain and API audit

Verified against the [official NVIDIA cuda-rust `cuda-oxide/v0.3.0` tag](https://github.com/NVIDIA/cuda-rust/tree/cuda-oxide/v0.3.0), commit `99e571f49baaefe4cd69157950c89ac9d4012728` (2026-10-31). Keep the Rust crates and `cargo-oxide` backend on the **same revision**. This is not the older `Rust-GPU/rust-cuda` project.

| Code use | Source under the pinned NVIDIA revision |
|---|---|
| Rust 1.99.1, compiler components | `cuda-oxide/rust-toolchain.toml` |
| `#[kernel]`, `#[launch_bounds(256, 2)]`, `#[cuda_module]` | `cuda-oxide/crates/cuda-macros/src/lib.rs` and `rustc-codegen-cuda/examples/tma_copy/src/main.rs` |
| TMA 2D load/store, bulk group commit/wait | `cuda-oxide/crates/cuda-device/src/generated/tma.rs` |
| Tensor-map 128-byte layout/alignment | `cuda-oxide/crates/cuda-device/src/tma.rs` |
| `mbarrier_init`, `mbarrier_inval` | `cuda-oxide/crates/cuda-device/src/generated/mbarrier_basic.rs` |
| Arrive/expected bytes, token/parity wait, fence | `cuda-oxide/crates/cuda-device/src/generated/mbarrier_extended.rs` and `barrier.rs` |
| FP32 `ex2_approx_f32` | `cuda-oxide/crates/cuda-device/src/generated/float.rs` |
| XOR warp shuffles and reductions | `cuda-oxide/crates/cuda-device/src/warp.rs` |
| Aligned shared arrays, disjoint outputs | `cuda-oxide/crates/cuda-device/src/shared.rs` and `disjoint.rs` |
| Device buffers, stream, context | `cuda-core/src/device_buffer.rs`, `runtime.rs`, `context.rs` |
| Driver tensor-map encoding | CUDA driver API `cuTensorMapEncodeTiled` |

Barrier ordering also follows the official [NVIDIA CUTLASS asynchronous-barrier guidance](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html): establish the expected transaction byte count **before** launching a TMA copy, then wait for the phase to finish before reading shared memory.

Host input/output are stored as **`u16` half-float bit patterns** and converted in the CPU reference; native `f16` is confined to the GPU kernel for low-level TMA tile arithmetic. This avoids requiring a host-facing `cuda-core/f16` buffer type for the new online path. `RUSTC_BOOTSTRAP=1` is needed for the experimental compiler integration even though NVIDIA now pins stable Rust 1.99.1.

NVIDIA lists GB200/B200 as CC **10.0**, GB300/B300 as **10.3**, H100/H200 as **9.0**, and RTX 5090 as **12.0**. Architecture-specific `sm_100a` PTX/cubins are neither backward- nor generally forward-compatible; this project deliberately rejects other devices. See [NVIDIA GPU capabilities](https://developer.nvidia.com/cuda/gpus) and the [Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).

The older uploaded standalone project used commit `b631530700a8a4a5f165456f2d71fe9195b323b5` (2026-11-02). Its archived measurements belong to **that older source only**. This merged code was **independently rebuilt and assembled**, with fresh PTX and per-kernel resource evidence in [`evidence/`](../evidence/).
