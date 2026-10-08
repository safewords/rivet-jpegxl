//! XYB, the perceptual colour space lossy JPEG XL codes in: linear sRGB
//! mixed into cone responses, biased, cube-rooted, and turned into an
//! opponent pair (X), a luminance (Y) and a blue channel (B).

use super::header::OpsinInverse;

/// 3x3 matrices, row major.
type M3 = [[f64; 3]; 3];

fn inverse(m: &M3) -> M3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |a: usize, b: usize, d: usize, e: usize| m[a][b] * m[d][e];
    [
        [
            (c(1, 1, 2, 2) - c(1, 2, 2, 1)) / det,
            (c(0, 2, 2, 1) - c(0, 1, 2, 2)) / det,
            (c(0, 1, 1, 2) - c(0, 2, 1, 1)) / det,
        ],
        [
            (c(1, 2, 2, 0) - c(1, 0, 2, 2)) / det,
            (c(0, 0, 2, 2) - c(0, 2, 2, 0)) / det,
            (c(0, 2, 1, 0) - c(0, 0, 1, 2)) / det,
        ],
        [
            (c(1, 0, 2, 1) - c(1, 1, 2, 0)) / det,
            (c(0, 1, 2, 0) - c(0, 0, 2, 1)) / det,
            (c(0, 0, 1, 1) - c(0, 1, 1, 0)) / det,
        ],
    ]
}

/// The forward XYB transform the decoder inverts with `opsin` and
/// `intensity_target`.
#[derive(Clone, Debug)]
pub struct Xyb {
    to_lms: M3,
    bias: [f64; 3],
    bias_cbrt: [f64; 3],
    /// Linear 1.0 is this many times the decoder's LMS unit.
    intensity_scale: f64,
}

impl Xyb {
    pub fn new(opsin: &OpsinInverse, intensity_target: f32) -> Self {
        let m = opsin.inverse_matrix;
        let inv: M3 = [
            [f64::from(m[0]), f64::from(m[1]), f64::from(m[2])],
            [f64::from(m[3]), f64::from(m[4]), f64::from(m[5])],
            [f64::from(m[6]), f64::from(m[7]), f64::from(m[8])],
        ];
        let bias = opsin.opsin_biases.map(f64::from);
        Xyb {
            to_lms: inverse(&inv),
            bias,
            bias_cbrt: bias.map(f64::cbrt),
            intensity_scale: 255.0 / f64::from(intensity_target),
        }
    }

    /// Linear sRGB (1.0 = the intensity target) to (X, Y, B).
    #[inline]
    pub fn from_linear(&self, rgb: [f32; 3]) -> [f32; 3] {
        let v = rgb.map(f64::from);
        let mut lms = [0f64; 3];
        for (k, l) in lms.iter_mut().enumerate() {
            let mixed =
                self.to_lms[k][0] * v[0] + self.to_lms[k][1] * v[1] + self.to_lms[k][2] * v[2];
            // The decoder: linear = intensity_scale * (lms_cubed + bias).
            let cubed = mixed / self.intensity_scale - self.bias[k];
            *l = cubed.cbrt() + self.bias_cbrt[k];
        }
        [
            ((lms[0] - lms[1]) / 2.0) as f32,
            ((lms[0] + lms[1]) / 2.0) as f32,
            lms[2] as f32,
        ]
    }

    /// The decoder's inverse, for tests and error estimates.
    pub fn to_linear(&self, xyb: [f32; 3]) -> [f32; 3] {
        let [x, y, b] = xyb.map(f64::from);
        let l = y + x - self.bias_cbrt[0];
        let m = y - x - self.bias_cbrt[1];
        let s = b - self.bias_cbrt[2];
        let lms = [l, m, s]
            .iter()
            .zip(&self.bias)
            .map(|(v, bias)| (v * v * v + bias) * self.intensity_scale)
            .collect::<Vec<_>>();
        let inv = inverse(&self.to_lms);
        [0, 1, 2].map(|k| (inv[k][0] * lms[0] + inv[k][1] * lms[1] + inv[k][2] * lms[2]) as f32)
    }
}

/// The sRGB transfer curve, decoded.
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let x = Xyb::new(&OpsinInverse::default(), 255.0);
        for rgb in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.7, 0.1],
            [1.0, 0.0, 0.5],
        ] {
            let back = x.to_linear(x.from_linear(rgb));
            for k in 0..3 {
                assert!((back[k] - rgb[k]).abs() < 1e-5, "{rgb:?} -> {back:?}");
            }
        }
        // Gray has no X.
        assert!(x.from_linear([0.5, 0.5, 0.5])[0].abs() < 1e-4);
    }
}
