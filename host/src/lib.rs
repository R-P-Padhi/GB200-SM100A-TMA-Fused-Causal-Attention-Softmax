//! GPU-independent CLI, vectors, result checks and timing statistics.
use causal_reference::{MAX_SCALE, f32_to_half, half_to_f32};
use std::{fmt, time::Duration};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Kernel {
    Auto,
    Baseline,
    Warp,
    Tma,
    Pipeline,
    Online,
}
impl Kernel {
    pub const ALL: [Self; 5] = [
        Self::Baseline,
        Self::Warp,
        Self::Tma,
        Self::Pipeline,
        Self::Online,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Baseline => "baseline",
            Self::Warp => "warp",
            Self::Tma => "tma",
            Self::Pipeline => "pipeline",
            Self::Online => "online",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "baseline" => Ok(Self::Baseline),
            "warp" => Ok(Self::Warp),
            "tma" => Ok(Self::Tma),
            "pipeline" => Ok(Self::Pipeline),
            "online" => Ok(Self::Online),
            _ => Err(format!("unknown kernel: {s}")),
        }
    }
    pub fn supports(self, cols: usize) -> bool {
        self != Self::Warp || cols <= 32
    }

    /// Approximate requested logical FP16 traffic; excludes TMA padding,
    /// cache effects, transaction overhead and internal shared-memory traffic.
    pub fn logical_bytes(self, rows: usize, cols: usize) -> f64 {
        let (r, c) = (rows as f64, cols as f64);
        let elements = r * c;
        let causal = if rows <= cols {
            r * (r + 1.0) / 2.0
        } else {
            c * (c + 1.0) / 2.0 + (r - c) * c
        };
        match self {
            Self::Baseline => 6.0 * causal + 2.0 * elements,
            Self::Warp => 2.0 * causal + 2.0 * elements,
            Self::Tma | Self::Pipeline => 8.0 * elements,
            Self::Online => 6.0 * elements,
            Self::Auto => unreachable!("resolve auto before estimating traffic"),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Action {
    Run,
    Probe,
    GpuTests,
    GpuStress,
    Bench,
    BenchSuite,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub action: Action,
    pub kernel: Kernel,
    pub rows: usize,
    pub cols: usize,
    pub scale: f32,
    pub iterations: usize,
    pub warmup: usize,
    pub verify: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            action: Action::Run,
            kernel: Kernel::Auto,
            rows: 4096,
            cols: 4096,
            scale: 0.125,
            iterations: 20,
            warmup: 5,
            verify: true,
        }
    }
}
impl Config {
    pub fn parse<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut c = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let key = arg.as_ref();
            match key {
                "--probe" => c.action = Action::Probe,
                "--gpu-tests" => c.action = Action::GpuTests,
                "--gpu-stress" => c.action = Action::GpuStress,
                "--bench" => c.action = Action::Bench,
                "--bench-suite" => c.action = Action::BenchSuite,
                "--no-verify" => c.verify = false,
                "--stress" => (c.rows, c.cols) = (8192, 8192),
                "--rows" | "--cols" | "--scale" | "--kernel" | "--iterations" | "--warmup" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for {key}"))?;
                    let value = value.as_ref();
                    match key {
                        "--rows" => c.rows = value.parse().map_err(|_| "invalid rows")?,
                        "--cols" => c.cols = value.parse().map_err(|_| "invalid cols")?,
                        "--scale" => c.scale = value.parse().map_err(|_| "invalid scale")?,
                        "--kernel" => c.kernel = Kernel::parse(value)?,
                        "--iterations" => {
                            c.iterations = value.parse().map_err(|_| "invalid iterations")?
                        }
                        _ => c.warmup = value.parse().map_err(|_| "invalid warmup")?,
                    }
                }
                "--help" | "-h" => return Err(Self::help().to_owned()),
                unknown => return Err(format!("unknown flag: {unknown}\n{}", Self::help())),
            }
        }
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.rows == 0 || self.cols == 0 {
            return Err("rows and cols must be positive".into());
        }
        if self.rows > i32::MAX as usize || self.cols > i32::MAX as usize {
            return Err("dimensions exceed i32 TMA coordinates".into());
        }
        if ((self.cols + 7) & !7) > i32::MAX as usize {
            return Err("padded pitch exceeds device i32 indexing".into());
        }
        if self.rows.checked_mul(self.cols).is_none()
            || self.rows.checked_mul((self.cols + 7) & !7).is_none()
        {
            return Err("matrix size overflow".into());
        }
        if !self.scale.is_finite() || !(0.0..=MAX_SCALE).contains(&self.scale) {
            return Err("scale must be finite and in [0, 1e30]".into());
        }
        if self.iterations == 0 || self.iterations > 100_000 || self.warmup > 10_000 {
            return Err("invalid benchmark repetition count".into());
        }
        if !self.kernel.supports(self.cols) {
            return Err("warp specialization requires cols <= 32".into());
        }
        Ok(())
    }
    pub fn resolved_kernel(&self) -> Kernel {
        if self.kernel != Kernel::Auto {
            self.kernel
        } else if self.cols <= 32 {
            Kernel::Warp
        } else {
            Kernel::Online
        }
    }
    pub const fn help() -> &'static str {
        "--probe | --gpu-tests | --gpu-stress | --bench | --bench-suite [--rows M] [--cols N] [--scale F] \
         [--kernel auto|baseline|warp|tma|pipeline|online] [--iterations N] [--warmup N] \
         [--stress] [--no-verify]"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    Zeros,
    Ramp,
    Alternating,
    Spikes,
    Extreme,
    Tiny,
    Random,
}
impl Pattern {
    pub const ALL: [Self; 7] = [
        Self::Zeros,
        Self::Ramp,
        Self::Alternating,
        Self::Spikes,
        Self::Extreme,
        Self::Tiny,
        Self::Random,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Zeros => "zeros",
            Self::Ramp => "ramp",
            Self::Alternating => "alternating",
            Self::Spikes => "spikes",
            Self::Extreme => "extreme",
            Self::Tiny => "tiny",
            Self::Random => "random",
        }
    }
}

pub fn make_input(rows: usize, cols: usize, pattern: Pattern, seed: u32) -> Vec<u16> {
    let mut state = seed;
    (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = match pattern {
                Pattern::Zeros => 0.0,
                Pattern::Ramp => (c as f32 - r as f32) * 0.125,
                Pattern::Alternating => {
                    if c % 2 == 0 {
                        -11.0
                    } else {
                        11.0
                    }
                }
                Pattern::Spikes => {
                    if c == r.min(cols - 1) {
                        28.0
                    } else {
                        -23.0
                    }
                }
                Pattern::Extreme => {
                    if c % 5 == 0 {
                        65000.0
                    } else {
                        -65000.0
                    }
                }
                Pattern::Tiny => (c % 11) as f32 * 0.00005 - 0.00025,
                Pattern::Random => ((state >> 16) as f32 / 65536.0 - 0.5) * 32.0,
            };
            f32_to_half(value)
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accuracy {
    pub max_abs: f32,
    pub max_relative: f32,
    pub max_row_sum_error: f32,
}
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyError {
    Length,
    NonFinite(usize),
    Negative(usize),
    Mask(usize),
    Element { index: usize, got: f32, want: f32 },
    RowSum { row: usize, sum: f32 },
}
impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for VerifyError {}

pub fn verify(
    got: &[u16],
    expected: &[u16],
    rows: usize,
    cols: usize,
) -> Result<Accuracy, VerifyError> {
    if got.len() != expected.len() || rows.checked_mul(cols) != Some(got.len()) {
        return Err(VerifyError::Length);
    }
    let mut metrics = Accuracy {
        max_abs: 0.0,
        max_relative: 0.0,
        max_row_sum_error: 0.0,
    };
    for row in 0..rows {
        let mut sum = 0.0f32;
        for col in 0..cols {
            let i = row * cols + col;
            let (g, w) = (half_to_f32(got[i]), half_to_f32(expected[i]));
            if !g.is_finite() {
                return Err(VerifyError::NonFinite(i));
            }
            if g < 0.0 {
                return Err(VerifyError::Negative(i));
            }
            if col > row && got[i] != 0 {
                return Err(VerifyError::Mask(i));
            }
            let err = (g - w).abs();
            metrics.max_abs = metrics.max_abs.max(err);
            if w > 0.0001 {
                metrics.max_relative = metrics.max_relative.max(err / w);
            }
            if err > 0.003 + 0.003 * w.abs() {
                return Err(VerifyError::Element {
                    index: i,
                    got: g,
                    want: w,
                });
            }
            sum += g;
        }
        metrics.max_row_sum_error = metrics.max_row_sum_error.max((sum - 1.0).abs());
        if (sum - 1.0).abs() > 0.02 {
            return Err(VerifyError::RowSum { row, sum });
        }
    }
    Ok(metrics)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchStats {
    pub samples: usize,
    pub min_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub stddev_ms: f64,
    pub cv_pct: f64,
}
impl BenchStats {
    /// Nearest-rank tails; standard deviation is population SD, not standard error.
    pub fn from_millis(values: &[f64]) -> Result<Self, String> {
        if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("benchmark needs finite nonnegative timing samples".into());
        }
        let mut ms = values.to_vec();
        ms.sort_by(f64::total_cmp);
        let n = ms.len();
        let mean = ms.iter().sum::<f64>() / n as f64;
        let variance = ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let median = if n % 2 == 0 {
            (ms[n / 2 - 1] + ms[n / 2]) / 2.0
        } else {
            ms[n / 2]
        };
        let stddev = variance.sqrt();
        Ok(Self {
            samples: n,
            min_ms: ms[0],
            median_ms: median,
            p95_ms: ms[(n * 95).div_ceil(100) - 1],
            p99_ms: ms[(n * 99).div_ceil(100) - 1],
            max_ms: ms[n - 1],
            mean_ms: mean,
            stddev_ms: stddev,
            cv_pct: if mean > 0.0 { 100.0 * stddev / mean } else { 0.0 },
        })
    }
    pub fn from_samples(samples: &[Duration]) -> Result<Self, String> {
        Self::from_millis(
            &samples
                .iter()
                .map(|x| x.as_secs_f64() * 1000.0)
                .collect::<Vec<_>>(),
        )
    }
    pub fn report(&self, name: &str, rows: usize, cols: usize, logical_bytes: f64) -> String {
        let gbps = if self.median_ms > 0.0 {
            logical_bytes / (self.median_ms / 1000.0) / 1e9
        } else {
            0.0
        };
        let melems = if self.median_ms > 0.0 {
            rows as f64 * cols as f64 / (self.median_ms / 1000.0) / 1e6
        } else {
            0.0
        };
        format!(
            "kernel={name} rows={rows} cols={cols} n={} min_ms={:.5} median_ms={:.5} p95_ms={:.5} p99_ms={:.5} max_ms={:.5} mean_ms={:.5} stddev_ms={:.5} cv_pct={:.2} approx_logical_GBps={gbps:.2} Melem_per_s={melems:.2}",
            self.samples, self.min_ms, self.median_ms, self.p95_ms, self.p99_ms,
            self.max_ms, self.mean_ms, self.stddev_ms, self.cv_pct
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults() {
        let c = Config::default();
        assert_eq!(c.resolved_kernel(), Kernel::Online);
    }
    #[test]
    fn parse_cli() {
        let c = Config::parse([
            "--bench",
            "--rows",
            "12",
            "--cols",
            "17",
            "--scale",
            "0.5",
            "--iterations",
            "7",
        ])
        .unwrap();
        assert_eq!(
            (c.action, c.rows, c.cols, c.iterations),
            (Action::Bench, 12, 17, 7)
        );
        assert_eq!(c.scale, 0.5);
    }
    #[test]
    fn rejects_zero() {
        assert!(Config::parse(["--rows", "0"]).is_err());
    }
    #[test]
    fn rejects_bad_scale() {
        assert!(Config::parse(["--scale", "NaN"]).is_err());
        assert!(Config::parse(["--scale", "-0.5"]).is_err());
        assert!(Config::parse(["--scale", "1e31"]).is_err());
        assert_eq!(Config::parse(["--scale", "0"]).unwrap().scale, 0.0);
    }
    #[test]
    fn rejects_missing_value() {
        assert!(Config::parse(["--cols"]).is_err());
    }
    #[test]
    fn rejects_unknown() {
        assert!(Config::parse(["--x"]).is_err());
    }
    #[test]
    fn rejects_bad_kernel() {
        assert!(Config::parse(["--kernel", "wrong"]).is_err());
    }
    #[test]
    fn warp_width_limit() {
        assert!(Config::parse(["--cols", "33", "--kernel", "warp"]).is_err());
    }
    #[test]
    fn warp_auto() {
        assert_eq!(
            Config::parse(["--cols", "31"]).unwrap().resolved_kernel(),
            Kernel::Warp
        );
    }
    #[test]
    fn logical_bytes_respect_causal_loads_and_kernel_passes() {
        for (rows, cols, eligible) in [(4, 4, 10.0), (2, 4, 3.0), (4, 2, 7.0)] {
            let full = (rows * cols) as f64;
            assert_eq!(
                Kernel::Baseline.logical_bytes(rows, cols),
                6.0 * eligible + 2.0 * full
            );
            assert_eq!(
                Kernel::Warp.logical_bytes(rows, cols),
                2.0 * eligible + 2.0 * full
            );
            assert_eq!(Kernel::Tma.logical_bytes(rows, cols), 8.0 * full);
            assert_eq!(Kernel::Pipeline.logical_bytes(rows, cols), 8.0 * full);
            assert_eq!(Kernel::Online.logical_bytes(rows, cols), 6.0 * full);
        }
    }
    #[test]
    fn all_patterns_finite() {
        for p in Pattern::ALL {
            let v = make_input(9, 37, p, 17);
            assert_eq!(v.len(), 333);
            assert!(v.iter().all(|&b| half_to_f32(b).is_finite()));
        }
    }
    #[test]
    fn seeded_random_repeatable() {
        let a = make_input(3, 9, Pattern::Random, 2);
        assert_eq!(a, make_input(3, 9, Pattern::Random, 2));
        assert_ne!(a, make_input(3, 9, Pattern::Random, 3));
    }
    #[test]
    fn verify_accepts_cpu() {
        let v = vec![f32_to_half(0.0); 16];
        let y = causal_reference::causal_softmax(&v, 4, 4, 1.0).unwrap();
        assert!(verify(&y, &y, 4, 4).is_ok());
    }
    #[test]
    fn verify_rejects_mask() {
        let y = vec![f32_to_half(1.0); 9];
        assert!(matches!(verify(&y, &y, 3, 3), Err(VerifyError::Mask(_))));
    }
    #[test]
    fn verify_rejects_nonfinite() {
        let y = vec![0x7c00];
        assert!(matches!(
            verify(&y, &y, 1, 1),
            Err(VerifyError::NonFinite(0))
        ));
    }
    #[test]
    fn verify_rejects_negative_probabilities() {
        let negative = [f32_to_half(-0.0005)];
        let expected = [f32_to_half(0.0)];
        assert_eq!(
            verify(&negative, &expected, 1, 1),
            Err(VerifyError::Negative(0))
        );
    }
    #[test]
    fn bench_stats() {
        let d = [
            Duration::from_millis(5),
            Duration::from_millis(2),
            Duration::from_millis(9),
        ];
        let b = BenchStats::from_samples(&d).unwrap();
        assert_eq!(b.min_ms, 2.0);
        assert_eq!(b.median_ms, 5.0);
        assert_eq!(b.p95_ms, 9.0);
    }
    #[test]
    fn bench_empty() {
        assert!(BenchStats::from_samples(&[]).is_err());
    }
    #[test]
    fn bench_even_median() {
        let d = [Duration::from_millis(2), Duration::from_millis(8)];
        assert_eq!(BenchStats::from_samples(&d).unwrap().median_ms, 5.0);
    }
    #[test]
    fn rejects_pitch_wrap() {
        assert!(Config::parse(["--cols", "2147483647"]).is_err());
    }
    #[test]
    fn verify_rejects_length_overflow() {
        assert_eq!(verify(&[], &[], usize::MAX, 2), Err(VerifyError::Length));
    }
    #[test]
    fn bench_percentiles_and_spread() {
        let b = BenchStats::from_millis(&[1.0, 3.0, 5.0, 7.0, 9.0]).unwrap();
        assert_eq!((b.samples, b.median_ms, b.p95_ms, b.p99_ms), (5, 5.0, 9.0, 9.0));
        assert_eq!(b.mean_ms, 5.0);
        assert!((b.stddev_ms - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!(b.cv_pct > 0.0);
    }
    #[test]
    fn bench_rejects_invalid_timings() {
        assert!(BenchStats::from_millis(&[]).is_err());
        assert!(BenchStats::from_millis(&[f64::NAN]).is_err());
        assert!(BenchStats::from_millis(&[-1.0]).is_err());
        assert_eq!(BenchStats::from_millis(&[0.0]).unwrap().cv_pct, 0.0);
    }
    #[test]
    fn suite_cli_and_bounds() {
        let c = Config::parse(["--bench-suite", "--iterations", "8", "--warmup", "2"]).unwrap();
        assert_eq!(c.action, Action::BenchSuite);
        assert_eq!(c.iterations, 8);
        assert!(Config::parse(["--bench-suite", "--iterations", "0"]).is_err());
    }

}
