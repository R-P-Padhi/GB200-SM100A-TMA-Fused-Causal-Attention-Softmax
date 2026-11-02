//! CPU execution model for the 8-warp, two-pass online TMA variant.
//! This models arithmetic and warp reductions, not CUDA barriers or transfers.
use super::{ShapeError, f32_to_half, half_to_f32, validate};

fn butterfly(mut lanes: [f32; 32], maximum: bool) -> [f32; 32] {
    for offset in [16, 8, 4, 2, 1] {
        let before = lanes;
        for lane in 0..32 {
            lanes[lane] = if maximum {
                before[lane].max(before[lane ^ offset])
            } else {
                before[lane] + before[lane ^ offset]
            };
        }
    }
    lanes
}

pub fn tiled_online_softmax(
    scores: &[u16],
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<Vec<u16>, ShapeError> {
    validate(scores, rows, cols, scale)?;
    let mut output = vec![0u16; scores.len()];
    for row in 0..rows {
        let valid = (row + 1).min(cols);
        let mut maximum = f32::NEG_INFINITY;
        let mut denominator = 0.0f32;
        for col0 in (0..valid).step_by(128) {
            let mut lanes = [[f32::NEG_INFINITY; 4]; 32];
            let mut lane_max = [f32::NEG_INFINITY; 32];
            for (lane, values) in lanes.iter_mut().enumerate() {
                for (item, score) in values.iter_mut().enumerate() {
                    let col = col0 + lane + item * 32;
                    if col < valid {
                        *score = half_to_f32(scores[row * cols + col]) * scale;
                    }
                    lane_max[lane] = lane_max[lane].max(*score);
                }
            }
            let next = maximum.max(butterfly(lane_max, true)[0]);
            let lane_exp = std::array::from_fn(|lane| {
                let v = lanes[lane];
                let e0 = (v[0] - next).exp();
                let e1 = (v[1] - next).exp();
                let e2 = (v[2] - next).exp();
                let e3 = (v[3] - next).exp();
                (e0 + e1) + (e2 + e3)
            });
            denominator = denominator * (maximum - next).exp() + butterfly(lane_exp, false)[0];
            maximum = next;
        }
        for col in 0..valid {
            let score = half_to_f32(scores[row * cols + col]) * scale;
            output[row * cols + col] = f32_to_half((score - maximum).exp() / denominator);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{causal_softmax, f32_to_half as h};

    #[test]
    fn complete_butterfly_reaches_every_lane() {
        let x = std::array::from_fn(|i| (i + 1) as f32);
        assert_eq!(butterfly(x, false), [528.0; 32]);
        assert_eq!(butterfly(x, true), [32.0; 32]);
        let mut sparse = [f32::NEG_INFINITY; 32];
        sparse[31] = -7.0;
        assert_eq!(butterfly(sparse, true), [-7.0; 32]);
    }

    #[test]
    fn ragged_tiles_match_independent_reference() {
        for (rows, cols) in [(1, 1), (7, 17), (9, 129), (65, 257), (137, 259), (257, 513)] {
            let scores: Vec<_> = (0..rows * cols)
                .map(|i| h(((i * 7919 % 251) as f32 - 125.0) * 0.125))
                .collect();
            for scale in [0.0, 0.001, 0.125, 1.0, 10.0] {
                let actual = tiled_online_softmax(&scores, rows, cols, scale).unwrap();
                let expected = causal_softmax(&scores, rows, cols, scale).unwrap();
                for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                    let g = half_to_f32(got);
                    let w = half_to_f32(want);
                    assert!(
                        (g - w).abs() <= 0.003 + 0.003 * w,
                        "{rows}x{cols} scale={scale} index={i}: {g} vs {w}"
                    );
                    if (i % cols) > (i / cols) {
                        assert_eq!(got, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn extremes_are_finite_and_zero_scale_is_uniform() {
        let rows = 139;
        let cols = 257;
        let scores: Vec<_> = (0..rows * cols)
            .map(|i| h(if i % 3 == 0 { 65504.0 } else { -65504.0 }))
            .collect();
        for scale in [0.0, 1.0, 1.0e30] {
            let result = tiled_online_softmax(&scores, rows, cols, scale).unwrap();
            assert!(result.iter().all(|&v| half_to_f32(v).is_finite()));
            for r in 0..rows {
                let sum: f32 = (0..cols).map(|c| half_to_f32(result[r * cols + c])).sum();
                assert!((sum - 1.0).abs() < 0.003);
                if scale == 0.0 {
                    for c in 0..=r {
                        assert_eq!(result[r * cols + c], h(1.0 / (r + 1) as f32));
                    }
                }
            }
        }
    }
}
