//! CPU specification. Input and output store IEEE binary16 as native `u16` bits.

pub mod online;

pub const MAX_SCALE: f32 = 1.0e30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    Empty,
    WrongLength,
    BadScale,
    NonFinite,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Empty => "matrix dimensions must be positive",
            Self::WrongLength => "score buffer length does not match matrix dimensions",
            Self::BadScale => "attention scale must be finite and in [0, 1e30]",
            Self::NonFinite => "score matrix contains NaN or infinity",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ShapeError {}

/// Exact half -> single conversion, including subnormals, infinities and NaNs.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 31) as u32;
    let frac = (h & 1023) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            let shift = frac.leading_zeros() - 21;
            let mant = frac << shift;
            sign | ((113 - shift) << 23) | ((mant & 1023) << 13)
        }
        31 => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// Round-to-nearest-even single -> half. Includes subnormals and overflow.
pub fn f32_to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let exp = ((bits >> 23) & 255) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    if exp == 0 {
        return sign;
    } // f32 subnormals are below half range
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let v = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mask = (1u32 << shift) - 1;
        let halfway = 1u32 << (shift - 1);
        let mut r = v >> shift;
        if (v & mask) > halfway || ((v & mask) == halfway && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut m = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
    }
    sign | (((e as u32) << 10).wrapping_add(m) as u16)
}

pub fn causal_softmax(
    scores: &[u16],
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<Vec<u16>, ShapeError> {
    validate(scores, rows, cols, scale)?;
    let mut output = vec![0u16; scores.len()];
    for row in 0..rows {
        let valid = (row + 1).min(cols);
        let offset = row * cols;
        let max = (0..valid)
            .map(|c| half_to_f32(scores[offset + c]) * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = (0..valid)
            .map(|c| (half_to_f32(scores[offset + c]) * scale - max).exp())
            .sum();
        for col in 0..valid {
            let p = (half_to_f32(scores[offset + col]) * scale - max).exp() / denom;
            output[offset + col] = f32_to_half(p);
        }
    }
    Ok(output)
}

pub(crate) fn validate(
    scores: &[u16],
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<(), ShapeError> {
    if rows == 0 || cols == 0 {
        return Err(ShapeError::Empty);
    }
    if rows.checked_mul(cols) != Some(scores.len()) {
        return Err(ShapeError::WrongLength);
    }
    if !scale.is_finite() || !(0.0..=MAX_SCALE).contains(&scale) {
        return Err(ShapeError::BadScale);
    }
    if scores.iter().any(|&v| !half_to_f32(v).is_finite()) {
        return Err(ShapeError::NonFinite);
    }
    Ok(())
}

/// Pack a logical MxN matrix into a TMA-compatible pitch: 16-byte row stride.
pub fn padded_pitch(cols: usize) -> usize {
    (cols + 7) & !7
}

pub fn pad(scores: &[u16], rows: usize, cols: usize) -> Vec<u16> {
    let pitch = padded_pitch(cols);
    let mut out = vec![0; rows * pitch];
    for row in 0..rows {
        out[row * pitch..row * pitch + cols].copy_from_slice(&scores[row * cols..(row + 1) * cols]);
    }
    out
}

pub fn unpad(padded: &[u16], rows: usize, cols: usize) -> Vec<u16> {
    let pitch = padded_pitch(cols);
    let mut out = vec![0; rows * cols];
    for row in 0..rows {
        out[row * cols..(row + 1) * cols].copy_from_slice(&padded[row * pitch..row * pitch + cols]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn h(x: f32) -> u16 {
        f32_to_half(x)
    }
    fn f(x: u16) -> f32 {
        half_to_f32(x)
    }

    #[test]
    fn half_specials() {
        assert_eq!(h(0.0), 0);
        assert_eq!(h(-0.0), 0x8000);
        assert_eq!(h(1.0), 0x3c00);
        assert_eq!(h(-2.0), 0xc000);
        assert_eq!(h(65504.0), 0x7bff);
        assert_eq!(h(70000.0), 0x7c00);
        assert_eq!(f(0x0001), 2f32.powi(-24));
        assert_eq!(f(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(f(0x0400), 2f32.powi(-14));
        assert!(f(0x7e00).is_nan());
    }
    #[test]
    fn half_ties_even() {
        assert_eq!(h(1.0 + 2f32.powi(-11)), h(1.0));
        assert_eq!(h(1.0 + 3.0 * 2f32.powi(-11)), h(1.0 + 2f32.powi(-10) * 2.0));
        assert_eq!(h(2f32.powi(-25)), 0);
        assert_eq!(h(3.0 * 2f32.powi(-25)), 2);
    }
    #[test]
    fn one_by_one() {
        assert_eq!(causal_softmax(&[h(-9.0)], 1, 1, 0.5).unwrap(), vec![h(1.0)]);
    }
    #[test]
    fn all_zero_triangular() {
        let y = causal_softmax(&[h(0.0); 16], 4, 4, 1.0).unwrap();
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    f(y[r * 4 + c]),
                    if c <= r {
                        f(h(1.0 / (r + 1) as f32))
                    } else {
                        0.0
                    }
                );
            }
        }
    }
    #[test]
    fn large_logits_stable() {
        let y =
            causal_softmax(&[h(65000.0), h(64992.0), h(65000.0), h(64992.0)], 2, 2, 1.0).unwrap();
        assert_eq!(f(y[0]), 1.0);
        assert_eq!(f(y[1]), 0.0);
        assert!((f(y[2]) + f(y[3]) - 1.0).abs() < 0.002);
    }
    #[test]
    fn rectangular_and_mask() {
        let scores: Vec<u16> = (0..35).map(|i| h((i as f32 % 7.0) - 2.0)).collect();
        let y = causal_softmax(&scores, 5, 7, 0.25).unwrap();
        for r in 0..5 {
            let sum: f32 = (0..7).map(|c| f(y[r * 7 + c])).sum();
            assert!((sum - 1.0).abs() < 0.002, "row {r}: {sum}");
            for c in r + 1..7 {
                assert_eq!(y[r * 7 + c], 0);
            }
        }
    }
    #[test]
    fn skinny_and_wide() {
        for (r, c) in [(1, 17), (7, 1), (3, 257), (5, 15), (9, 256)] {
            let input = vec![h(0.125); r * c];
            let y = causal_softmax(&input, r, c, 0.2).unwrap();
            for row in 0..r {
                assert!((0..c).map(|col| f(y[row * c + col])).sum::<f32>() > 0.99);
            }
        }
    }
    #[test]
    fn padding_roundtrip() {
        for (r, c) in [(1, 1), (5, 17), (4, 256), (3, 257)] {
            let x: Vec<u16> = (0..r * c).map(|i| i as u16).collect();
            assert_eq!(unpad(&pad(&x, r, c), r, c), x);
            assert_eq!(pad(&x, r, c).len() % 8, 0);
        }
    }
    #[test]
    fn errors() {
        assert_eq!(causal_softmax(&[], 0, 1, 1.0), Err(ShapeError::Empty));
        assert_eq!(
            causal_softmax(&[0], 1, 2, 1.0),
            Err(ShapeError::WrongLength)
        );
        assert_eq!(causal_softmax(&[0], 1, 1, -0.1), Err(ShapeError::BadScale));
        assert_eq!(
            causal_softmax(&[0], 1, 1, 1.0e31),
            Err(ShapeError::BadScale)
        );
        assert_eq!(
            causal_softmax(&[0x7e00], 1, 1, 1.0),
            Err(ShapeError::NonFinite)
        );
    }
    #[test]
    fn monotonic() {
        let x = [h(0.0), h(4.0), h(-4.0), h(1.0), h(2.0), h(3.0)];
        let y = causal_softmax(&x, 2, 3, 1.0).unwrap();
        assert!(f(y[4]) > f(y[3]));
        assert_eq!(y[2], 0);
    }

    #[test]
    fn every_finite_half_roundtrips() {
        for bits in 0u32..=u16::MAX as u32 {
            let bits = bits as u16;
            if bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0 {
                continue;
            }
            assert_eq!(f32_to_half(half_to_f32(bits)), bits, "0x{bits:04x}");
        }
    }

    #[test]
    fn independent_f64_oracle() {
        for (rows, cols) in [(1, 1), (2, 9), (7, 3), (16, 17), (33, 257)] {
            let x: Vec<u16> = (0..rows * cols)
                .map(|i| h(((i * 37 % 101) as f32 - 50.0) / 9.0))
                .collect();
            let result = causal_softmax(&x, rows, cols, 0.25).unwrap();
            for row in 0..rows {
                let n = (row + 1).min(cols);
                let scores: Vec<f64> = (0..n).map(|c| f(x[row * cols + c]) as f64 * 0.25).collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denom: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for col in 0..cols {
                    let expected = if col < n {
                        ((scores[col] - max).exp() / denom) as f32
                    } else {
                        0.0
                    };
                    assert!((f(result[row * cols + col]) - expected).abs() < 0.001);
                }
            }
        }
    }

    #[test]
    fn subnormal_inputs_remain_finite() {
        let v = [
            0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x0000, 0x8000, 0x3c00,
        ];
        let p = causal_softmax(&v, 3, 3, 0.125).unwrap();
        assert!(p.iter().all(|&x| f(x).is_finite()));
        for r in 0..3 {
            assert!((0..3).map(|c| f(p[r * 3 + c])).sum::<f32>() > 0.999);
        }
    }

    #[test]
    fn severe_logit_contrast() {
        let x = vec![h(-65000.0), h(65000.0), h(-65000.0), h(65000.0)];
        let out = causal_softmax(&x, 2, 2, 1.0).unwrap();
        assert_eq!(&out[..2], &[h(1.0), 0]);
        assert_eq!(&out[2..], &[0, h(1.0)]);
    }

    #[test]
    fn all_rows_normalize_under_varied_scales() {
        let x: Vec<u16> = (0..35).map(|i| h((i as f32 % 13.0 - 6.0) * 0.75)).collect();
        for scale in [0.001, 0.125, 1.0, 10.0] {
            let y = causal_softmax(&x, 7, 5, scale).unwrap();
            for row in 0..7 {
                let total: f32 = (0..5).map(|c| f(y[row * 5 + c])).sum();
                assert!((total - 1.0).abs() < 0.002, "{row}, {scale}: {total}");
            }
        }
    }

    #[test]
    fn reference_deterministic() {
        let x: Vec<u16> = (0..1024).map(|i| h((i % 17) as f32 / 8.0)).collect();
        assert_eq!(
            causal_softmax(&x, 32, 32, 0.3),
            causal_softmax(&x, 32, 32, 0.3)
        );
    }

    #[test]
    fn invalid_shape_cannot_overflow() {
        assert_eq!(
            causal_softmax(&[], usize::MAX, 2, 1.0),
            Err(ShapeError::WrongLength)
        );
    }

    #[test]
    fn infinity_is_rejected() {
        assert_eq!(
            causal_softmax(&[0x7c00], 1, 1, 1.0),
            Err(ShapeError::NonFinite)
        );
        assert_eq!(
            causal_softmax(&[0xfc00], 1, 1, 1.0),
            Err(ShapeError::NonFinite)
        );
    }
}
