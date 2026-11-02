#![feature(f16)]
#![allow(clippy::missing_safety_doc)]

use causal_reference::{causal_softmax, pad, padded_pitch, unpad};
use cuda_core::sys::{
    self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
    mbarrier_init, mbarrier_inval, mbarrier_try_wait, mbarrier_try_wait_parity,
};
use cuda_device::float::ex2_approx_f32;
use cuda_device::tma::{
    TmaDescriptor, cp_async_bulk_commit_group, cp_async_bulk_tensor_2d_g2s_cta,
    cp_async_bulk_tensor_2d_s2g, cp_async_bulk_wait_group,
};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread, warp};
use cuda_host::cuda_module;
use std::{error::Error, ffi::c_void, mem::MaybeUninit};

const THREADS: u32 = 256;
const TILE_COLS: usize = 256; // Maximum descriptor box dimension.
const TILE_ROWS: usize = 4; // Four independent values per thread.
const TILE_ELEMS: usize = TILE_COLS * TILE_ROWS;
const TILE_BYTES: u32 = (TILE_ELEMS * 2) as u32;
const LOG2_E: f32 = core::f32::consts::LOG2_E;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn online_exp(x: f32) -> f32 {
        ex2_approx_f32(x * LOG2_E)
    }

    // Butterfly reduction broadcasts the final result to every lane.
    #[inline(always)]
    fn online_max(mut x: f32) -> f32 {
        for offset in [16, 8, 4, 2, 1] {
            x = x.max(warp::shuffle_xor_f32(x, offset));
        }
        x
    }

    #[inline(always)]
    fn online_sum(mut x: f32) -> f32 {
        for offset in [16, 8, 4, 2, 1] {
            x += warp::shuffle_xor_f32(x, offset);
        }
        x
    }

    #[kernel]
    pub fn probe(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let value = idx.get() as u32 ^ 0x5a5a_5a5a;
        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }

    /// Reference device implementation: one CTA per row; scalar global loads/stores,
    /// two-stage reductions, FP32 accumulation, no tensor map or shared input tile.
    #[kernel]
    #[launch_bounds(256)]
    pub fn causal_softmax_baseline(
        input: &[u16],
        mut output: DisjointSlice<u16>,
        cols: i32,
        pitch: i32,
        scale: f32,
    ) {
        static mut PARTIAL: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut REDUCED: SharedArray<f32, 1> = SharedArray::UNINIT;
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let cols = cols as usize;
        let pitch = pitch as usize;
        let mut local_max = f32::NEG_INFINITY;
        let mut col = tid;
        while col < cols {
            if col <= row {
                let score = f16::from_bits(input[row * pitch + col]) as f32 * scale;
                local_max = local_max.max(score);
            }
            col += THREADS as usize;
        }
        let warp_max = warp::reduce_max_f32(local_max);
        if tid.is_multiple_of(32) {
            unsafe {
                PARTIAL[tid / 32] = warp_max;
            }
        }
        thread::sync_threads();
        let seed = if tid < 8 {
            unsafe { PARTIAL[tid] }
        } else {
            f32::NEG_INFINITY
        };
        let max = warp::reduce_max_f32(seed);
        if tid == 0 {
            unsafe {
                REDUCED[0] = max;
            }
        }
        thread::sync_threads();
        let row_max = unsafe { REDUCED[0] };

        let mut local_sum = 0.0f32;
        col = tid;
        while col < cols {
            if col <= row {
                let score = f16::from_bits(input[row * pitch + col]) as f32 * scale;
                local_sum += ex2_approx_f32((score - row_max) * LOG2_E);
            }
            col += THREADS as usize;
        }
        let warp_sum = warp::reduce_sum_f32(local_sum);
        if tid.is_multiple_of(32) {
            unsafe {
                PARTIAL[tid / 32] = warp_sum;
            }
        }
        thread::sync_threads();
        let seed = if tid < 8 {
            unsafe { PARTIAL[tid] }
        } else {
            0.0
        };
        let sum = warp::reduce_sum_f32(seed);
        if tid == 0 {
            unsafe {
                REDUCED[0] = sum;
            }
        }
        thread::sync_threads();
        let denom = unsafe { REDUCED[0] };

        col = tid;
        while col < cols {
            let p = if col <= row {
                let score = f16::from_bits(input[row * pitch + col]) as f32 * scale;
                ex2_approx_f32((score - row_max) * LOG2_E) / denom
            } else {
                0.0
            };
            // SAFETY: each (row, col) is visited by one CTA/lane; the host
            // validates pitch and output allocation before launch.
            unsafe {
                *output.get_unchecked_mut(row * pitch + col) = (p as f16).to_bits();
            }
            col += THREADS as usize;
        }
    }

    /// One warp owns one complete row; eight independent rows per 256-lane CTA.
    /// Only used for N <= 32, so no CTA barrier or shared-memory reduction.
    #[kernel]
    #[launch_bounds(256)]
    pub fn causal_softmax_warp(
        input: &[u16],
        mut output: DisjointSlice<u16>,
        rows: i32,
        cols: i32,
        pitch: i32,
        scale: f32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let row = thread::blockIdx_x() as usize * 8 + tid / 32;
        let valid = row < rows as usize && lane < cols as usize;
        let keep = valid && lane <= row;
        let score = if keep {
            f16::from_bits(input[row * pitch as usize + lane]) as f32 * scale
        } else {
            f32::NEG_INFINITY
        };
        let max = warp::reduce_max_f32(score);
        let value = if keep {
            ex2_approx_f32((score - max) * LOG2_E)
        } else {
            0.0
        };
        let denom = warp::reduce_sum_f32(value);
        if valid {
            // SAFETY: exactly one lane writes each valid element of its warp's row.
            unsafe {
                *output.get_unchecked_mut(row * pitch as usize + lane) =
                    ((value / denom) as f16).to_bits();
            }
        }
    }

    /// One 256-thread CTA handles four rows; each tile is 256x4 FP16.
    /// Three streaming passes: max, exp-sum, then normalize/TMA store.
    #[kernel]
    #[launch_bounds(256)]
    pub fn causal_softmax_tma(
        #[grid_constant] src_map: &TmaDescriptor,
        #[grid_constant] dst_map: &TmaDescriptor,
        rows: i32,
        cols: i32,
        scale: f32,
    ) {
        static mut INPUT: SharedArray<u16, TILE_ELEMS, 128> = SharedArray::UNINIT;
        static mut OUTPUT: SharedArray<u16, TILE_ELEMS, 128> = SharedArray::UNINIT;
        static mut LOAD_BAR: Barrier = Barrier::UNINIT;
        static mut PARTIAL: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut REDUCED: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row0 = (thread::blockIdx_x() as usize) * TILE_ROWS;
        let tiles = (cols as usize).div_ceil(TILE_COLS);
        let mut maxima = [f32::NEG_INFINITY; TILE_ROWS];
        let mut totals = [0.0f32; TILE_ROWS];

        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut LOAD_BAR, THREADS);
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        for phase in 0..3 {
            for tile in 0..tiles {
                let token = unsafe {
                    if tid == 0 {
                        // Register bytes before TMA can complete against this phase.
                        let token = mbarrier_arrive_expect_tx(&raw const LOAD_BAR, 1, TILE_BYTES);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            &raw mut INPUT as *mut u8,
                            src_map,
                            (tile * TILE_COLS) as i32,
                            row0 as i32,
                            &raw mut LOAD_BAR,
                        );
                        token
                    } else {
                        mbarrier_arrive(&raw const LOAD_BAR)
                    }
                };
                unsafe { while !mbarrier_try_wait(&raw const LOAD_BAR, token) {} }
                thread::sync_threads();

                let col = tile * TILE_COLS + tid;
                for r in 0..TILE_ROWS {
                    let row = row0 + r;
                    let keep = row < rows as usize && col < cols as usize && col <= row;
                    let scaled = if keep {
                        (f16::from_bits(unsafe { INPUT[r * TILE_COLS + tid] }) as f32) * scale
                    } else {
                        f32::NEG_INFINITY
                    };

                    if phase == 2 {
                        let p = if keep {
                            ex2_approx_f32((scaled - maxima[r]) * LOG2_E) / totals[r]
                        } else {
                            0.0
                        };
                        unsafe {
                            OUTPUT[r * TILE_COLS + tid] = (p as f16).to_bits();
                        }
                        continue;
                    }

                    let local = if phase == 0 {
                        scaled
                    } else if keep {
                        ex2_approx_f32((scaled - maxima[r]) * LOG2_E)
                    } else {
                        0.0
                    };
                    let warp_value = if phase == 0 {
                        warp::reduce_max_f32(local)
                    } else {
                        warp::reduce_sum_f32(local)
                    };
                    if tid.is_multiple_of(32) {
                        unsafe {
                            PARTIAL[tid / 32] = warp_value;
                        }
                    }
                    thread::sync_threads();

                    // Every lane participates; only warp zero's result is read.
                    let part = if tid < 8 {
                        unsafe { PARTIAL[tid] }
                    } else if phase == 0 {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                    let all = if phase == 0 {
                        warp::reduce_max_f32(part)
                    } else {
                        warp::reduce_sum_f32(part)
                    };
                    if tid == 0 {
                        unsafe {
                            REDUCED[0] = all;
                        }
                    }
                    thread::sync_threads();
                    let value = unsafe { REDUCED[0] };
                    if phase == 0 {
                        maxima[r] = maxima[r].max(value);
                    } else {
                        totals[r] += value;
                    }
                    thread::sync_threads();
                }

                if phase == 2 {
                    thread::sync_threads();
                    if tid == 0 {
                        unsafe {
                            fence_proxy_async_shared_cta();
                            cp_async_bulk_tensor_2d_s2g(
                                &raw const OUTPUT as *const u8,
                                dst_map,
                                (tile * TILE_COLS) as i32,
                                row0 as i32,
                            );
                        }
                        cp_async_bulk_commit_group();
                        cp_async_bulk_wait_group(0);
                    }
                }
                thread::sync_threads(); // Buffer not reused until async copy finishes.
            }
        }
    }

    /// Double-buffered 2D TMA pipeline. Tile n+1 transfers into the other
    /// shared tile while all warps reduce tile n. Barriers are reused only after
    /// the previous load on that buffer has completed and the consumers sync.
    #[kernel]
    #[launch_bounds(256)]
    pub fn causal_softmax_tma_pipelined(
        #[grid_constant] src_map: &TmaDescriptor,
        #[grid_constant] dst_map: &TmaDescriptor,
        rows: i32,
        cols: i32,
        scale: f32,
    ) {
        static mut IN0: SharedArray<u16, TILE_ELEMS, 128> = SharedArray::UNINIT;
        static mut IN1: SharedArray<u16, TILE_ELEMS, 128> = SharedArray::UNINIT;
        static mut OUT: SharedArray<u16, TILE_ELEMS, 128> = SharedArray::UNINIT;
        static mut BAR0: Barrier = Barrier::UNINIT;
        static mut BAR1: Barrier = Barrier::UNINIT;
        static mut PARTIAL: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut REDUCED: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row0 = thread::blockIdx_x() as usize * TILE_ROWS;
        let tiles = (cols as usize).div_ceil(TILE_COLS);
        let mut maxima = [f32::NEG_INFINITY; TILE_ROWS];
        let mut totals = [0.0f32; TILE_ROWS];

        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR0, THREADS);
                mbarrier_init(&raw mut BAR1, THREADS);
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        for phase in 0..3 {
            // Prime slot zero. Every thread arrives, but no one waits yet.
            let mut current_token = unsafe {
                if tid == 0 {
                    let token = mbarrier_arrive_expect_tx(&raw const BAR0, 1, TILE_BYTES);
                    cp_async_bulk_tensor_2d_g2s_cta(
                        &raw mut IN0 as *mut u8,
                        src_map,
                        0,
                        row0 as i32,
                        &raw mut BAR0,
                    );
                    token
                } else {
                    mbarrier_arrive(&raw const BAR0)
                }
            };

            for tile in 0..tiles {
                let slot0 = tile % 2 == 0;
                if slot0 {
                    unsafe { while !mbarrier_try_wait(&raw const BAR0, current_token) {} }
                } else {
                    unsafe { while !mbarrier_try_wait(&raw const BAR1, current_token) {} }
                }
                thread::sync_threads();

                // Launch the next load before computing the current tile.
                // Opposite slots guarantee the in-flight TMA cannot overwrite
                // data that this CTA is still consuming.
                let next_token = if tile + 1 < tiles {
                    let x = ((tile + 1) * TILE_COLS) as i32;
                    if slot0 {
                        unsafe {
                            if tid == 0 {
                                let token =
                                    mbarrier_arrive_expect_tx(&raw const BAR1, 1, TILE_BYTES);
                                cp_async_bulk_tensor_2d_g2s_cta(
                                    &raw mut IN1 as *mut u8,
                                    src_map,
                                    x,
                                    row0 as i32,
                                    &raw mut BAR1,
                                );
                                token
                            } else {
                                mbarrier_arrive(&raw const BAR1)
                            }
                        }
                    } else {
                        unsafe {
                            if tid == 0 {
                                let token =
                                    mbarrier_arrive_expect_tx(&raw const BAR0, 1, TILE_BYTES);
                                cp_async_bulk_tensor_2d_g2s_cta(
                                    &raw mut IN0 as *mut u8,
                                    src_map,
                                    x,
                                    row0 as i32,
                                    &raw mut BAR0,
                                );
                                token
                            } else {
                                mbarrier_arrive(&raw const BAR0)
                            }
                        }
                    }
                } else {
                    current_token
                };

                let col = tile * TILE_COLS + tid;
                for r in 0..TILE_ROWS {
                    let row = row0 + r;
                    let keep = row < rows as usize && col < cols as usize && col <= row;
                    let bits = if slot0 {
                        unsafe { IN0[r * TILE_COLS + tid] }
                    } else {
                        unsafe { IN1[r * TILE_COLS + tid] }
                    };
                    let scaled = if keep {
                        (f16::from_bits(bits) as f32) * scale
                    } else {
                        f32::NEG_INFINITY
                    };

                    if phase == 2 {
                        let p = if keep {
                            ex2_approx_f32((scaled - maxima[r]) * LOG2_E) / totals[r]
                        } else {
                            0.0
                        };
                        unsafe {
                            OUT[r * TILE_COLS + tid] = (p as f16).to_bits();
                        }
                        continue;
                    }
                    let value = if phase == 0 {
                        scaled
                    } else if keep {
                        ex2_approx_f32((scaled - maxima[r]) * LOG2_E)
                    } else {
                        0.0
                    };
                    let warp_value = if phase == 0 {
                        warp::reduce_max_f32(value)
                    } else {
                        warp::reduce_sum_f32(value)
                    };
                    if tid.is_multiple_of(32) {
                        unsafe {
                            PARTIAL[tid / 32] = warp_value;
                        }
                    }
                    thread::sync_threads();
                    let seed = if tid < 8 {
                        unsafe { PARTIAL[tid] }
                    } else if phase == 0 {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                    let block_value = if phase == 0 {
                        warp::reduce_max_f32(seed)
                    } else {
                        warp::reduce_sum_f32(seed)
                    };
                    if tid == 0 {
                        unsafe {
                            REDUCED[0] = block_value;
                        }
                    }
                    thread::sync_threads();
                    let reduced = unsafe { REDUCED[0] };
                    if phase == 0 {
                        maxima[r] = maxima[r].max(reduced);
                    } else {
                        totals[r] += reduced;
                    }
                    thread::sync_threads();
                }

                if phase == 2 {
                    thread::sync_threads();
                    if tid == 0 {
                        unsafe {
                            fence_proxy_async_shared_cta();
                            cp_async_bulk_tensor_2d_s2g(
                                &raw const OUT as *const u8,
                                dst_map,
                                (tile * TILE_COLS) as i32,
                                row0 as i32,
                            );
                        }
                        cp_async_bulk_commit_group();
                        cp_async_bulk_wait_group(0);
                    }
                }
                thread::sync_threads();
                current_token = next_token;
            }
            // Both input slots are quiescent before the next full-row pass.
            thread::sync_threads();
        }
    }

    /// Eight rows/CTA, one warp/row, 128 columns/tile. Online (max, sum)
    /// statistics need one input traversal; normalization makes the second.
    /// The producer alone arrives at each mbarrier; every warp waits.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn causal_softmax_tma_online(
        #[grid_constant] src_map: &TmaDescriptor,
        #[grid_constant] dst_map: &TmaDescriptor,
        rows: i32,
        cols: i32,
        scale: f32,
    ) {
        static mut IN0: SharedArray<f16, 1024, 128> = SharedArray::UNINIT;
        static mut IN1: SharedArray<f16, 1024, 128> = SharedArray::UNINIT;
        static mut OUT: SharedArray<f16, 1024, 128> = SharedArray::UNINIT;
        static mut BAR0: Barrier = Barrier::UNINIT;
        static mut BAR1: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp_id = tid / 32;
        let row_base = thread::blockIdx_x() * 8;
        let row = row_base + warp_id;
        let local = (warp_id * 128 + lane) as usize;
        let tiles = (cols as u32).div_ceil(128);
        let mut maximum = f32::NEG_INFINITY;
        let mut denominator = 0.0f32;

        for pass in 0..2 {
            if tid == 0 {
                unsafe {
                    mbarrier_init(&raw mut BAR0, 1);
                    mbarrier_init(&raw mut BAR1, 1);
                    fence_proxy_async_shared_cta();
                }
            }
            thread::sync_threads();
            if tid == 0 {
                unsafe {
                    mbarrier_arrive_expect_tx(&raw const BAR0, 1, 2048);
                    cp_async_bulk_tensor_2d_g2s_cta(
                        &raw mut IN0 as *mut u8,
                        src_map,
                        0,
                        row_base as i32,
                        &raw mut BAR0,
                    );
                }
            }

            for tile in 0..tiles {
                let slot = tile & 1;
                let parity = (tile / 2) & 1;
                let bar = if slot == 0 {
                    &raw const BAR0
                } else {
                    &raw const BAR1
                };
                unsafe { while !mbarrier_try_wait_parity(bar, parity) {} }

                if tid == 0 && tile + 1 < tiles {
                    unsafe {
                        let next_bar = if slot == 0 {
                            &raw mut BAR1
                        } else {
                            &raw mut BAR0
                        };
                        let next_dst = if slot == 0 {
                            &raw mut IN1 as *mut u8
                        } else {
                            &raw mut IN0 as *mut u8
                        };
                        mbarrier_arrive_expect_tx(next_bar, 1, 2048);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            next_dst,
                            src_map,
                            ((tile + 1) * 128) as i32,
                            row_base as i32,
                            next_bar,
                        );
                    }
                }

                let src = if slot == 0 {
                    &raw const IN0 as *const f16
                } else {
                    &raw const IN1 as *const f16
                };
                let c0 = tile * 128 + lane;
                let c1 = c0 + 32;
                let c2 = c0 + 64;
                let c3 = c0 + 96;
                let keep0 = row < rows as u32 && c0 < cols as u32 && c0 <= row;
                let keep1 = row < rows as u32 && c1 < cols as u32 && c1 <= row;
                let keep2 = row < rows as u32 && c2 < cols as u32 && c2 <= row;
                let keep3 = row < rows as u32 && c3 < cols as u32 && c3 <= row;
                let x0 = if keep0 {
                    (unsafe { *src.add(local) }) as f32 * scale
                } else {
                    f32::NEG_INFINITY
                };
                let x1 = if keep1 {
                    (unsafe { *src.add(local + 32) }) as f32 * scale
                } else {
                    f32::NEG_INFINITY
                };
                let x2 = if keep2 {
                    (unsafe { *src.add(local + 64) }) as f32 * scale
                } else {
                    f32::NEG_INFINITY
                };
                let x3 = if keep3 {
                    (unsafe { *src.add(local + 96) }) as f32 * scale
                } else {
                    f32::NEG_INFINITY
                };

                if pass == 0 {
                    // This condition is warp-uniform and avoids -inf - -inf.
                    if row < rows as u32 && tile * 128 <= row && tile * 128 < cols as u32 {
                        let next = maximum.max(online_max(x0.max(x1).max(x2.max(x3))));
                        let e0 = online_exp(x0 - next);
                        let e1 = online_exp(x1 - next);
                        let e2 = online_exp(x2 - next);
                        let e3 = online_exp(x3 - next);
                        denominator = denominator * online_exp(maximum - next)
                            + online_sum((e0 + e1) + (e2 + e3));
                        maximum = next;
                    }
                } else {
                    let p0 = if keep0 {
                        online_exp(x0 - maximum) / denominator
                    } else {
                        0.0
                    };
                    let p1 = if keep1 {
                        online_exp(x1 - maximum) / denominator
                    } else {
                        0.0
                    };
                    let p2 = if keep2 {
                        online_exp(x2 - maximum) / denominator
                    } else {
                        0.0
                    };
                    let p3 = if keep3 {
                        online_exp(x3 - maximum) / denominator
                    } else {
                        0.0
                    };
                    unsafe {
                        let dst = &raw mut OUT as *mut f16;
                        *dst.add(local) = p0 as f16;
                        *dst.add(local + 32) = p1 as f16;
                        *dst.add(local + 64) = p2 as f16;
                        *dst.add(local + 96) = p3 as f16;
                        fence_proxy_async_shared_cta();
                    }
                    thread::sync_threads();
                    if tid == 0 {
                        unsafe {
                            cp_async_bulk_tensor_2d_s2g(
                                &raw const OUT as *const u8,
                                dst_map,
                                (tile * 128) as i32,
                                row_base as i32,
                            );
                        }
                        cp_async_bulk_commit_group();
                        cp_async_bulk_wait_group(0);
                    }
                }
                thread::sync_threads();
            }
            if tid == 0 {
                unsafe {
                    mbarrier_inval(&raw mut BAR0);
                    mbarrier_inval(&raw mut BAR1);
                }
            }
            thread::sync_threads();
        }
    }
}

fn tensor_map(
    ptr: *mut c_void,
    pitch: usize,
    rows: usize,
    box_cols: u32,
    box_rows: u32,
) -> Result<TmaDescriptor, Box<dyn Error>> {
    if ptr.is_null() || !(ptr as usize).is_multiple_of(16) {
        return Err("TMA base address must be nonnull and 16-byte aligned".into());
    }
    if rows == 0 || pitch == 0 || !pitch.is_multiple_of(8) {
        return Err("TMA matrix must have positive rows and 16-byte-aligned row strides".into());
    }
    let mut desc = MaybeUninit::<CUtensorMap>::uninit();
    let dims = [pitch as u64, rows as u64];
    let strides = [(pitch * 2) as u64];
    let box_dims = [box_cols, box_rows];
    let element_strides = [1u32, 1u32];
    let code = unsafe {
        cuTensorMapEncodeTiled(
            desc.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            ptr,
            dims.as_ptr(),
            strides.as_ptr(),
            box_dims.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if code != cuda_sys::cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled: {code:?}").into());
    }
    const {
        assert!(size_of::<TmaDescriptor>() == size_of::<CUtensorMap>());
    }
    const {
        assert!(align_of::<CUtensorMap>() >= align_of::<TmaDescriptor>());
    }
    Ok(unsafe { desc.as_ptr().cast::<TmaDescriptor>().read() })
}

mod driver;

fn main() -> Result<(), Box<dyn Error>> {
    driver::main_entry()
}
