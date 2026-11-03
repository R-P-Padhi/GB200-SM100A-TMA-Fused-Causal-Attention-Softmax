//! Host-only allocation, dispatch, correctness suites and timing.
use super::*;
use cuda_core::CudaStream;

mod bench;
use softmax_host::{Accuracy, Action, BenchStats, Config, Kernel, Pattern, make_input, verify};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

struct Session {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}

struct Buffers {
    input: DeviceBuffer<u16>,
    output: DeviceBuffer<u16>,
    source_map: TmaDescriptor,
    target_map: TmaDescriptor,
    online_source_map: TmaDescriptor,
    online_target_map: TmaDescriptor,
    pitch: usize,
}

impl Session {
    fn new() -> Result<Self, Box<dyn Error>> {
        let context = CudaContext::new(0)?;
        let (major, minor) = context.compute_capability()?;
        if (major, minor) != (10, 0) {
            return Err(format!("GB200/sm_100a requested, found sm_{major}{minor}").into());
        }
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        Ok(Self {
            _context: context,
            stream,
            module,
        })
    }

    fn probe(&self) -> Result<(), Box<dyn Error>> {
        let mut out = DeviceBuffer::<u32>::zeroed(&self.stream, THREADS as usize)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { self.module.probe(&self.stream, cfg, &mut out) }?;
        for (i, value) in out.to_host_vec(&self.stream)?.iter().enumerate() {
            if *value != (i as u32 ^ 0x5a5a_5a5a) {
                return Err(format!("probe mismatch at {i}").into());
            }
        }
        println!("PASS probe: all 256 lanes");
        Ok(())
    }

    fn prepare(&self, raw: &[u16], rows: usize, cols: usize) -> Result<Buffers, Box<dyn Error>> {
        self.prepare_on(&self.stream, raw, rows, cols)
    }

    fn prepare_on(
        &self,
        stream: &CudaStream,
        raw: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<Buffers, Box<dyn Error>> {
        if raw.len() != rows * cols {
            return Err("input length does not match shape".into());
        }
        let pitch = padded_pitch(cols);
        let pitched = pad(raw, rows, cols);
        let input = DeviceBuffer::from_host(stream, &pitched)?;
        // A nonzero sentinel reveals missing stores in the logical output.
        let output = DeviceBuffer::from_host(stream, &vec![0xfb00u16; pitched.len()])?;
        let input_ptr = input.cu_deviceptr() as *mut c_void;
        let output_ptr = output.cu_deviceptr() as *mut c_void;
        let source_map = tensor_map(input_ptr, pitch, rows, 256, 4)?;
        let target_map = tensor_map(output_ptr, pitch, rows, 256, 4)?;
        let online_source_map = tensor_map(input_ptr, pitch, rows, 128, 8)?;
        let online_target_map = tensor_map(output_ptr, pitch, rows, 128, 8)?;
        Ok(Buffers {
            input,
            output,
            source_map,
            target_map,
            online_source_map,
            online_target_map,
            pitch,
        })
    }

    fn launch(
        &self,
        kind: Kernel,
        b: &mut Buffers,
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<(), Box<dyn Error>> {
        self.launch_on(&self.stream, kind, b, rows, cols, scale)
    }

    fn launch_on(
        &self,
        stream: &CudaStream,
        kind: Kernel,
        b: &mut Buffers,
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<(), Box<dyn Error>> {
        let cfg = LaunchConfig {
            grid_dim: (
                match kind {
                    Kernel::Warp => rows.div_ceil(8),
                    Kernel::Baseline => rows,
                    Kernel::Tma | Kernel::Pipeline => rows.div_ceil(TILE_ROWS),
                    Kernel::Online => rows.div_ceil(8),
                    Kernel::Auto => unreachable!("resolve auto before launch"),
                } as u32,
                1,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        match kind {
            Kernel::Baseline => {
                // SAFETY: each lane/CTA owns disjoint row-column coordinates;
                // allocation and pitch were checked when creating Buffers.
                unsafe {
                    self.module.causal_softmax_baseline(
                        stream,
                        cfg,
                        &b.input,
                        &mut b.output,
                        cols as i32,
                        b.pitch as i32,
                        scale,
                    )
                }?;
            }
            Kernel::Warp => {
                if cols > 32 {
                    return Err("warp specialization requires cols<=32".into());
                }
                unsafe {
                    self.module.causal_softmax_warp(
                        stream,
                        cfg,
                        &b.input,
                        &mut b.output,
                        rows as i32,
                        cols as i32,
                        b.pitch as i32,
                        scale,
                    )
                }?;
            }
            Kernel::Tma => {
                // SAFETY: descriptors point to live buffers and describe disjoint CTA
                // output rows; sync before buffers are moved/freed.
                unsafe {
                    self.module.causal_softmax_tma(
                        stream,
                        cfg,
                        b.source_map,
                        b.target_map,
                        rows as i32,
                        cols as i32,
                        scale,
                    )
                }?;
            }
            Kernel::Pipeline => {
                unsafe {
                    self.module.causal_softmax_tma_pipelined(
                        stream,
                        cfg,
                        b.source_map,
                        b.target_map,
                        rows as i32,
                        cols as i32,
                        scale,
                    )
                }?;
            }
            Kernel::Online => {
                // SAFETY: two TMA descriptors point to live nonaliasing buffers;
                // the barrier protocol is handled uniformly by every CTA.
                unsafe {
                    self.module.causal_softmax_tma_online(
                        stream,
                        cfg,
                        b.online_source_map,
                        b.online_target_map,
                        rows as i32,
                        cols as i32,
                        scale,
                    )
                }?;
            }
            Kernel::Auto => unreachable!("resolve auto before launch"),
        }
        Ok(())
    }

    fn result(
        &self,
        kind: Kernel,
        b: &Buffers,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u16>, Box<dyn Error>> {
        self.stream.synchronize()?;
        let physical = b.output.to_host_vec(&self.stream)?;
        if matches!(kind, Kernel::Tma | Kernel::Pipeline | Kernel::Online) {
            for row in 0..rows {
                for col in cols..b.pitch {
                    if physical[row * b.pitch + col] != 0 {
                        return Err(format!("nonzero padded output at ({row}, {col})").into());
                    }
                }
            }
        }
        Ok(unpad(&physical, rows, cols))
    }
}

fn run_case(
    s: &Session,
    kind: Kernel,
    rows: usize,
    cols: usize,
    scale: f32,
    input: &[u16],
) -> Result<Accuracy, Box<dyn Error>> {
    let mut buffers = s.prepare(input, rows, cols)?;
    s.launch(kind, &mut buffers, rows, cols, scale)?;
    let actual = s.result(kind, &buffers, rows, cols)?;
    let expected = causal_softmax(input, rows, cols, scale)?;
    Ok(verify(&actual, &expected, rows, cols)?)
}

const QUICK: &[(usize, usize)] = &[
    (1, 1),
    (1, 8),
    (2, 17),
    (3, 31),
    (4, 32),
    (5, 33),
    (7, 64),
    (8, 127),
    (9, 256),
    (10, 257),
    (13, 513),
    (17, 1024),
    (33, 9),
    (64, 65),
];
const MORE: &[(usize, usize)] = &[
    (1, 129),
    (2, 255),
    (3, 256),
    (4, 257),
    (5, 511),
    (6, 512),
    (7, 513),
    (8, 1025),
    (15, 2048),
    (16, 16),
    (31, 31),
    (32, 32),
    (33, 33),
    (64, 128),
    (65, 256),
    (127, 512),
    (129, 8192),
    (257, 257),
];

fn gpu_tests(s: &Session, stress: bool) -> Result<(), Box<dyn Error>> {
    let shapes = if stress { MORE } else { QUICK };
    let patterns: &[Pattern] = if stress {
        &Pattern::ALL
    } else {
        &[Pattern::Zeros, Pattern::Spikes, Pattern::Random]
    };
    let mut passed = 0usize;
    for &(rows, cols) in shapes {
        for &pattern in patterns {
            let input = make_input(rows, cols, pattern, 0x13579bdf);
            for kind in Kernel::ALL {
                if !kind.supports(cols) {
                    continue;
                }
                for scale in if stress {
                    &[0.0, 0.125, 0.5, 1.0, 1.0e30][..]
                } else {
                    &[0.125][..]
                } {
                    let score = run_case(s, kind, rows, cols, *scale, &input).map_err(|e| {
                        format!(
                            "FAIL {} {}x{} {} scale={scale}: {e}",
                            kind.name(),
                            rows,
                            cols,
                            pattern.name()
                        )
                    })?;
                    println!(
                        "PASS {:8} {:4}x{:<5} {:11} scale={scale:<5} max_err={:.5}",
                        kind.name(),
                        rows,
                        cols,
                        pattern.name(),
                        score.max_abs
                    );
                    passed += 1;
                }
            }
        }
    }
    println!("PASS GPU correctness: {passed} cases (executed on GB200)");
    Ok(())
}

fn benchmark(s: &Session, c: &Config) -> Result<(), Box<dyn Error>> {
    let input = make_input(c.rows, c.cols, Pattern::Random, 0x13579bdf);
    let candidates: Vec<Kernel> = if c.kernel == Kernel::Auto {
        Kernel::ALL
            .into_iter()
            .filter(|k| k.supports(c.cols))
            .collect()
    } else {
        vec![c.kernel]
    };
    for kind in candidates {
        let mut buffers = s.prepare(&input, c.rows, c.cols)?;
        if c.verify {
            s.launch(kind, &mut buffers, c.rows, c.cols, c.scale)?;
            let actual = s.result(kind, &buffers, c.rows, c.cols)?;
            let expected = causal_softmax(&input, c.rows, c.cols, c.scale)?;
            let accuracy = verify(&actual, &expected, c.rows, c.cols)?;
            println!(
                "validated {} max_abs={:.6} max_row_sum_err={:.6}",
                kind.name(),
                accuracy.max_abs,
                accuracy.max_row_sum_error
            );
        }
        for _ in 0..c.warmup {
            s.launch(kind, &mut buffers, c.rows, c.cols, c.scale)?;
            s.stream.synchronize()?;
        }
        let mut samples = Vec::<Duration>::with_capacity(c.iterations);
        for _ in 0..c.iterations {
            s.stream.synchronize()?;
            let start = Instant::now();
            s.launch(kind, &mut buffers, c.rows, c.cols, c.scale)?;
            s.stream.synchronize()?;
            samples.push(start.elapsed());
        }
        let stats = BenchStats::from_samples(&samples)?;
        println!(
            "{}",
            stats.report(
                kind.name(),
                c.rows,
                c.cols,
                kind.logical_bytes(c.rows, c.cols)
            )
        );
    }
    println!(
        "Timing: host wall-clock around enqueue + stream synchronization; not CUDA-event kernel time."
    );
    Ok(())
}

pub(super) fn main_entry() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", Config::help());
        return Ok(());
    }
    let c = Config::parse(args.iter())?;
    let session = Session::new()?;
    match c.action {
        Action::Probe => session.probe(),
        Action::GpuTests => gpu_tests(&session, false),
        Action::GpuStress => gpu_tests(&session, true),
        Action::Bench => benchmark(&session, &c),
        Action::BenchSuite => bench::suite(&session, &c),
        Action::Run => {
            let input = make_input(c.rows, c.cols, Pattern::Random, 0x13579bdf);
            let kind = c.resolved_kernel();
            if c.verify {
                let accuracy = run_case(&session, kind, c.rows, c.cols, c.scale, &input)?;
                println!(
                    "PASS {} {}x{} scale={} max_abs={:.6} max_row_sum_err={:.6}",
                    kind.name(),
                    c.rows,
                    c.cols,
                    c.scale,
                    accuracy.max_abs,
                    accuracy.max_row_sum_error
                );
            } else {
                let mut buffers = session.prepare(&input, c.rows, c.cols)?;
                session.launch(kind, &mut buffers, c.rows, c.cols, c.scale)?;
                session.stream.synchronize()?;
                println!(
                    "launched {} {}x{} (comparison disabled)",
                    kind.name(),
                    c.rows,
                    c.cols
                );
            }
            Ok(())
        }
    }
}
