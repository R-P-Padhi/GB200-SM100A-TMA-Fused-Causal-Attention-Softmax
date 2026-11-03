//! CPU-only timing harness. These figures are never presented as GPU performance.
use causal_reference::{causal_softmax, online::tiled_online_softmax};
use softmax_host::{BenchStats, Pattern, make_input, verify};
use std::{
    error::Error,
    hint::black_box,
    time::Instant,
};

fn sample<F>(iterations: usize, mut f: F) -> Result<BenchStats, Box<dyn Error>>
where
    F: FnMut() -> Result<Vec<u16>, causal_reference::ShapeError>,
{
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let output = f()?;
        black_box(output);
        times.push(started.elapsed());
    }
    Ok(BenchStats::from_samples(&times)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut iterations = 7usize;
    let mut stress = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iterations" => {
                iterations = args.next().ok_or("missing --iterations value")?.parse()?;
            }
            "--stress" => stress = true,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if !(1..=1_000).contains(&iterations) {
        return Err("iterations must be 1..=1000".into());
    }
    let mut cases = vec![
        (1, 8192),
        (8192, 1),
        (31, 31),
        (64, 64),
        (128, 128),
        (128, 512),
        (512, 128),
        (256, 256),
        (512, 512),
    ];
    if stress {
        cases.extend([(1024, 1024), (2048, 2048)]);
    }
    println!("CPU_REFERENCE_ONLY timing=host_Instant repeats={iterations} no_gpu=true");
    for (rows, cols) in cases {
        for pattern in [Pattern::Random, Pattern::Extreme] {
            let input = make_input(rows, cols, pattern, 0x17a6_29bf);
            let expected = causal_softmax(&input, rows, cols, 0.125)?;
            let online = tiled_online_softmax(&input, rows, cols, 0.125)?;
            verify(&online, &expected, rows, cols)?;
            let algorithms: [(
                &str,
                fn(&[u16], usize, usize, f32) -> Result<Vec<u16>, causal_reference::ShapeError>,
            ); 2] = [
                ("cpu_reference", causal_softmax),
                ("cpu_online_model", tiled_online_softmax),
            ];
            for (name, compute) in algorithms {
                // One untimed warmup, then independent allocations just like the reference.
                black_box(compute(&input, rows, cols, 0.125)?);
                let stats = sample(iterations, || compute(black_box(&input), rows, cols, 0.125))?;
                let logical_bytes = (rows as f64) * (cols as f64) * 4.0;
                println!(
                    "model={name} pattern={} {}",
                    pattern.name(),
                    stats.report(name, rows, cols, logical_bytes)
                );
            }
        }
    }
    Ok(())
}
