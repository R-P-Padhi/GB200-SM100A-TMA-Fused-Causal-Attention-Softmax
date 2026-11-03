//! Hardware-only GB200 benchmark matrix; never executed in CPU-only CI.
use super::*;
use std::time::Instant;

const EVENT_FLAGS: cuda_sys::CUevent_flags = cuda_sys::CUevent_flags_enum_CU_EVENT_DEFAULT;

#[allow(clippy::too_many_arguments)]
fn kernel_event_ms(
    session: &Session,
    stream: &CudaStream,
    kind: Kernel,
    buffers: &mut Buffers,
    rows: usize,
    cols: usize,
    scale: f32,
    queued: usize,
) -> Result<f64, Box<dyn Error>> {
    let start = stream.record_event(Some(EVENT_FLAGS))?;
    for _ in 0..queued {
        session.launch_on(stream, kind, buffers, rows, cols, scale)?;
    }
    let stop = stream.record_event(Some(EVENT_FLAGS))?;
    let ms = f64::from(start.elapsed_ms(&stop)?) / queued as f64;
    if !ms.is_finite() || ms < 0.0 {
        return Err("invalid CUDA event timing".into());
    }
    Ok(ms)
}

#[allow(clippy::too_many_arguments)]
fn sample_events(
    session: &Session,
    kind: Kernel,
    buffers: &mut Buffers,
    rows: usize,
    cols: usize,
    scale: f32,
    warmup: usize,
    samples: usize,
    queued: usize,
) -> Result<BenchStats, Box<dyn Error>> {
    for _ in 0..warmup {
        session.launch(kind, buffers, rows, cols, scale)?;
    }
    session.stream.synchronize()?;
    let mut ms = Vec::with_capacity(samples);
    for _ in 0..samples {
        ms.push(kernel_event_ms(
            session,
            &session.stream,
            kind,
            buffers,
            rows,
            cols,
            scale,
            queued,
        )?);
    }
    Ok(BenchStats::from_millis(&ms)?)
}

fn candidates(selected: Kernel, cols: usize) -> Vec<Kernel> {
    if selected == Kernel::Auto {
        Kernel::ALL
            .into_iter()
            .filter(|kind| kind.supports(cols))
            .collect()
    } else if selected.supports(cols) {
        vec![selected]
    } else {
        Vec::new()
    }
}

fn shape_sweep(session: &Session, config: &Config) -> Result<(), Box<dyn Error>> {
    let mut cases = vec![
        (1, 1),
        (31, 31),
        (32, 32),
        (64, 64),
        (128, 257),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (128, 4096),
        (4096, 128),
        (2048, 2048),
        (4096, 4096),
    ];
    cases.push((config.rows, config.cols));
    if config.rows >= 8192 || config.cols >= 8192 {
        cases.push((8192, 8192));
    }
    cases.sort_unstable();
    cases.dedup();
    for (rows, cols) in cases {
        let input = make_input(rows, cols, Pattern::Random, 0x7381_7cde);
        // Independent host reference is computed once per shape.
        let expected = if config.verify {
            Some(causal_softmax(&input, rows, cols, config.scale)?)
        } else {
            None
        };
        for kind in candidates(config.kernel, cols) {
            let mut buffers = session.prepare(&input, rows, cols)?;
            if let Some(expected) = &expected {
                session.launch(kind, &mut buffers, rows, cols, config.scale)?;
                let actual = session.result(kind, &buffers, rows, cols)?;
                verify(&actual, expected, rows, cols)?;
            }
            let stats = sample_events(
                session,
                kind,
                &mut buffers,
                rows,
                cols,
                config.scale,
                config.warmup,
                config.iterations,
                1,
            )?;
            println!(
                "BENCH category=shape clock=cuda_event {}",
                stats.report(kind.name(), rows, cols, kind.logical_bytes(rows, cols))
            );
        }
    }
    Ok(())
}

fn launch_queue(session: &Session, config: &Config) -> Result<(), Box<dyn Error>> {
    let rows = config.rows.min(1024);
    let cols = config.cols.min(1024);
    for kind in candidates(config.kernel, cols) {
        let input = make_input(rows, cols, Pattern::Random, 0x513e_79a2);
        let mut buffers = session.prepare(&input, rows, cols)?;
        for queued in [1, 4, 16, 64] {
            let stats = sample_events(
                session,
                kind,
                &mut buffers,
                rows,
                cols,
                config.scale,
                config.warmup,
                config.iterations,
                queued,
            )?;
            println!(
                "BENCH category=queue clock=cuda_event queued={queued} {}",
                stats.report(kind.name(), rows, cols, kind.logical_bytes(rows, cols))
            );
        }
    }
    Ok(())
}

fn transfers(session: &Session, config: &Config) -> Result<(), Box<dyn Error>> {
    let rows = config.rows;
    let cols = config.cols;
    let kind = config.resolved_kernel();
    let input = make_input(rows, cols, Pattern::Random, 0x11ba_4ff2);
    let pitched = pad(&input, rows, cols);
    let mut buffers = session.prepare(&input, rows, cols)?;
    session.launch(kind, &mut buffers, rows, cols, config.scale)?;
    session.stream.synchronize()?;
    let mut output = vec![0u16; pitched.len()];
    let mut h2d = Vec::with_capacity(config.iterations);
    let mut d2h = Vec::with_capacity(config.iterations);
    let mut full = Vec::with_capacity(config.iterations);
    for _ in 0..config.warmup {
        buffers.input.copy_from_host(&session.stream, &pitched)?;
        session.launch(kind, &mut buffers, rows, cols, config.scale)?;
        buffers.output.copy_to_host(&session.stream, &mut output)?;
    }
    for _ in 0..config.iterations {
        let t = Instant::now();
        buffers.input.copy_from_host(&session.stream, &pitched)?;
        h2d.push(t.elapsed());

        let t = Instant::now();
        buffers.output.copy_to_host(&session.stream, &mut output)?;
        d2h.push(t.elapsed());

        let t = Instant::now();
        buffers.input.copy_from_host(&session.stream, &pitched)?;
        session.launch(kind, &mut buffers, rows, cols, config.scale)?;
        buffers.output.copy_to_host(&session.stream, &mut output)?;
        full.push(t.elapsed());
    }
    if config.verify {
        let actual = unpad(&output, rows, cols);
        let expected = causal_softmax(&input, rows, cols, config.scale)?;
        verify(&actual, &expected, rows, cols)?;
    }
    let bytes = pitched.len() as f64 * 2.0;
    for (stage, timings, traffic) in [
        ("h2d", &h2d, bytes),
        ("d2h", &d2h, bytes),
        (
            "h2d_kernel_d2h",
            &full,
            2.0 * bytes + kind.logical_bytes(rows, cols),
        ),
    ] {
        let stats = BenchStats::from_samples(timings)?;
        println!(
            "BENCH category=transfer stage={stage} clock=host_wall {}",
            stats.report(kind.name(), rows, cols, traffic)
        );
    }
    Ok(())
}

fn pattern_sweep(session: &Session, config: &Config) -> Result<(), Box<dyn Error>> {
    let (rows, cols) = (config.rows.min(256), config.cols.min(256));
    for pattern in [
        Pattern::Zeros,
        Pattern::Ramp,
        Pattern::Alternating,
        Pattern::Spikes,
        Pattern::Extreme,
        Pattern::Random,
    ] {
        let input = make_input(rows, cols, pattern, 0xd3ad_87c1);
        for scale in [0.0, 0.125, 1.0, 1.0e30] {
            for kind in candidates(config.kernel, cols) {
                let mut buffers = session.prepare(&input, rows, cols)?;
                if config.verify {
                    session.launch(kind, &mut buffers, rows, cols, scale)?;
                    let actual = session.result(kind, &buffers, rows, cols)?;
                    let expected = causal_softmax(&input, rows, cols, scale)?;
                    verify(&actual, &expected, rows, cols)?;
                }
                let stats = sample_events(
                    session,
                    kind,
                    &mut buffers,
                    rows,
                    cols,
                    scale,
                    config.warmup.min(3),
                    config.iterations.min(12),
                    1,
                )?;
                println!(
                    "BENCH category=pattern pattern={} scale={scale} clock=cuda_event {}",
                    pattern.name(),
                    stats.report(kind.name(), rows, cols, kind.logical_bytes(rows, cols))
                );
            }
        }
    }
    Ok(())
}

fn multi_stream(session: &Session, config: &Config) -> Result<(), Box<dyn Error>> {
    let rows = config.rows.min(512);
    let kind = config.resolved_kernel();
    let cols = if kind == Kernel::Warp {
        config.cols.min(32)
    } else {
        config.cols.min(512)
    };
    let input = make_input(rows, cols, Pattern::Random, 0x58ee_7719);
    let expected = if config.verify {
        Some(causal_softmax(&input, rows, cols, config.scale)?)
    } else {
        None
    };
    let queued = 4usize;
    let mut baseline_throughput = None::<f64>;
    for count in [1usize, 2, 4, 8] {
        // New nonblocking streams and distinct allocations: no output sharing.
        let streams = (0..count)
            .map(|_| session._context.new_stream())
            .collect::<Result<Vec<_>, _>>()?;
        let mut buffers = streams
            .iter()
            .map(|stream| session.prepare_on(stream, &input, rows, cols))
            .collect::<Result<Vec<_>, _>>()?;
        for _ in 0..config.warmup {
            for (stream, buffer) in streams.iter().zip(buffers.iter_mut()) {
                session.launch_on(stream, kind, buffer, rows, cols, config.scale)?;
            }
        }
        for stream in &streams {
            stream.synchronize()?;
        }
        let mut wall = Vec::with_capacity(config.iterations);
        for _ in 0..config.iterations {
            let t = Instant::now();
            for (stream, buffer) in streams.iter().zip(buffers.iter_mut()) {
                for _ in 0..queued {
                    session.launch_on(stream, kind, buffer, rows, cols, config.scale)?;
                }
            }
            for stream in &streams {
                stream.synchronize()?;
            }
            wall.push(t.elapsed());
        }
        if let Some(expected) = &expected {
            for (stream, buffer) in streams.iter().zip(buffers.iter()) {
                let output = buffer.output.to_host_vec(stream)?;
                verify(&unpad(&output, rows, cols), expected, rows, cols)?;
            }
        }
        let stats = BenchStats::from_samples(&wall)?;
        let calls = (count * queued) as f64;
        let calls_per_s = calls / (stats.median_ms / 1e3);
        let baseline = *baseline_throughput.get_or_insert(calls_per_s);
        let speedup = if baseline > 0.0 {
            calls_per_s / baseline
        } else {
            0.0
        };
        let gbps = calls_per_s * kind.logical_bytes(rows, cols) / 1e9;
        println!(
            "BENCH category=concurrency clock=host_wall streams={count} calls_per_stream={queued} \
             kernel={} rows={rows} cols={cols} n={} batch_p50_ms={:.5} batch_p95_ms={:.5} \
             batch_p99_ms={:.5} kernels_per_s={calls_per_s:.2} throughput_speedup={speedup:.3} \
             approx_logical_GBps={gbps:.2}",
            kind.name(),
            stats.samples,
            stats.median_ms,
            stats.p95_ms,
            stats.p99_ms
        );
    }
    Ok(())
}

pub(super) fn suite(session: &Session, config: &Config) -> Result<(), Box<dyn Error>> {
    println!(
        "BENCH hardware=GB200 clock=CUDA_event_and_host_wall verify={} warmup={} repetitions={} \
         warning=no_Nsight_counters_no_actual_DRAM_bytes",
        config.verify, config.warmup, config.iterations
    );
    shape_sweep(session, config)?;
    launch_queue(session, config)?;
    transfers(session, config)?;
    pattern_sweep(session, config)?;
    multi_stream(session, config)?;
    println!(
        "BENCH complete: CUDA events are device timing; host-wall includes synchronization. \
             Logical GB/s is an estimated traffic model, not an Nsight DRAM counter."
    );
    Ok(())
}
