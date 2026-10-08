//! The VarDCT transforms: what each of the 27 block transforms does to a
//! block's coefficients and LF, and its inverse, the encoder's forward
//! transform.
//!
//! Coefficients are stored "wide": a block `H` tall and `W` wide keeps its
//! `min(H, W) x max(H, W)` coefficients, rows the vertical frequencies of a
//! wide block and the horizontal ones of a square or tall block. The DCTs are scaled so the DC is the mean;
//! larger ones take their lowest frequencies from the 8x8-block means
//! (the LF image) through the "reinterpreting" DCT.

use std::f64::consts::{PI, SQRT_2};
use std::sync::OnceLock;

/// A block transform, by its code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TransformType {
    Dct8 = 0,
    Identity = 1,
    Dct2 = 2,
    Dct4 = 3,
    Dct16 = 4,
    Dct32 = 5,
    /// 16 tall, 8 wide.
    Dct16x8 = 6,
    /// 8 tall, 16 wide.
    Dct8x16 = 7,
    Dct32x8 = 8,
    Dct8x32 = 9,
    Dct32x16 = 10,
    Dct16x32 = 11,
    /// Two 4-tall halves.
    Dct4x8 = 12,
    /// Two 4-wide halves.
    Dct8x4 = 13,
    Afv0 = 14,
    Afv1 = 15,
    Afv2 = 16,
    Afv3 = 17,
    Dct64 = 18,
    Dct64x32 = 19,
    Dct32x64 = 20,
    Dct128 = 21,
    Dct128x64 = 22,
    Dct64x128 = 23,
    Dct256 = 24,
    Dct256x128 = 25,
    Dct128x256 = 26,
}

impl TransformType {
    pub const ALL: [TransformType; 27] = [
        TransformType::Dct8,
        TransformType::Identity,
        TransformType::Dct2,
        TransformType::Dct4,
        TransformType::Dct16,
        TransformType::Dct32,
        TransformType::Dct16x8,
        TransformType::Dct8x16,
        TransformType::Dct32x8,
        TransformType::Dct8x32,
        TransformType::Dct32x16,
        TransformType::Dct16x32,
        TransformType::Dct4x8,
        TransformType::Dct8x4,
        TransformType::Afv0,
        TransformType::Afv1,
        TransformType::Afv2,
        TransformType::Afv3,
        TransformType::Dct64,
        TransformType::Dct64x32,
        TransformType::Dct32x64,
        TransformType::Dct128,
        TransformType::Dct128x64,
        TransformType::Dct64x128,
        TransformType::Dct256,
        TransformType::Dct256x128,
        TransformType::Dct128x256,
    ];

    /// Blocks covered (across, down).
    pub fn covered(self) -> (usize, usize) {
        const X: [usize; 27] = [
            1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32,
        ];
        const Y: [usize; 27] = [
            1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16,
        ];
        (X[self as usize], Y[self as usize])
    }

    /// The coefficient order (and context) shape.
    pub fn shape(self) -> usize {
        const S: [usize; 27] = [
            0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
        ];
        S[self as usize]
    }

    /// The dequantisation table it uses.
    pub fn quant_table(self) -> usize {
        use TransformType::*;
        match self {
            Dct8 => 0,
            Identity => 1,
            Dct2 => 2,
            Dct4 => 3,
            Dct16 => 4,
            Dct32 => 5,
            Dct16x8 | Dct8x16 => 6,
            Dct32x8 | Dct8x32 => 7,
            Dct32x16 | Dct16x32 => 8,
            Dct4x8 | Dct8x4 => 9,
            Afv0 | Afv1 | Afv2 | Afv3 => 10,
            Dct64 => 11,
            Dct64x32 | Dct32x64 => 12,
            Dct128 => 13,
            Dct128x64 | Dct64x128 => 14,
            Dct256 => 15,
            Dct256x128 | Dct128x256 => 16,
        }
    }

    fn is_plain_dct(self) -> bool {
        use TransformType::*;
        !matches!(
            self,
            Identity | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3
        )
    }
}

/// The DCT basis: `x[n] = sum_k c[k] w_k cos(pi (2n + 1) k / 2N)`, `w_0 = 1`,
/// `w_k = sqrt 2`.
fn basis(n: usize) -> &'static [f64] {
    static TABLES: [OnceLock<Vec<f64>>; 9] = [const { OnceLock::new() }; 9];
    let log = n.trailing_zeros() as usize;
    TABLES[log].get_or_init(|| {
        let mut t = vec![0.0; n * n];
        for k in 0..n {
            let w = if k == 0 { 1.0 } else { SQRT_2 };
            for x in 0..n {
                t[k * n + x] = w * (PI * (2 * x + 1) as f64 * k as f64 / (2 * n) as f64).cos();
            }
        }
        t
    })
}

/// 1-D: coefficients to samples.
fn idct1(c: &[f64], out: &mut [f64]) {
    let n = c.len();
    let b = basis(n);
    for x in 0..n {
        out[x] = (0..n).map(|k| c[k] * b[k * n + x]).sum();
    }
}

/// 1-D: samples to coefficients (the inverse of `idct1`).
fn dct1(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    let b = basis(n);
    for k in 0..n {
        let s: f64 = (0..n).map(|i| x[i] * b[k * n + i]).sum();
        out[k] = s / n as f64;
    }
}

/// Apply `f` along the rows then the columns of an `h x w` matrix.
fn separable(data: &mut [f64], h: usize, w: usize, f: fn(&[f64], &mut [f64])) {
    let mut tmp = vec![0.0; w.max(h)];
    for r in 0..h {
        f(&data[r * w..(r + 1) * w], &mut tmp[..w]);
        data[r * w..(r + 1) * w].copy_from_slice(&tmp[..w]);
    }
    let mut col = vec![0.0; h];
    for c in 0..w {
        for r in 0..h {
            col[r] = data[r * w + c];
        }
        f(&col, &mut tmp[..h]);
        for r in 0..h {
            data[r * w + c] = tmp[r];
        }
    }
}

/// Pixels (`h` tall, `w` wide) from wide-stored coefficients.
fn idct2d(coeffs: &[f64], h: usize, w: usize) -> Vec<f64> {
    if w > h {
        let mut d = coeffs.to_vec();
        separable(&mut d, h, w, idct1);
        d
    } else {
        // Stored transposed: rows are the horizontal frequencies.
        let mut d = coeffs.to_vec(); // w rows, h columns
        separable(&mut d, w, h, idct1);
        transpose(&d, w, h)
    }
}

/// Wide-stored coefficients from pixels (`h` tall, `w` wide).
fn dct2d(pixels: &[f64], h: usize, w: usize) -> Vec<f64> {
    if w > h {
        let mut d = pixels.to_vec();
        separable(&mut d, h, w, dct1);
        d
    } else {
        let mut d = transpose(pixels, h, w);
        separable(&mut d, w, h, dct1);
        d
    }
}

fn transpose(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// The reinterpreting DCT's scale for frequency `k` of `n` LF samples: the
/// 8n-point coefficient whose 8-sample means have unit k-th coefficient.
fn resample_scale(k: usize, n: usize) -> f64 {
    static TABLES: [OnceLock<Vec<f64>>; 6] = [const { OnceLock::new() }; 6];
    let log = n.trailing_zeros() as usize;
    TABLES[log].get_or_init(|| {
        (0..n)
            .map(|k| {
                let len = 8 * n;
                let w = if k == 0 { 1.0 } else { SQRT_2 };
                let means: Vec<f64> = (0..n)
                    .map(|j| {
                        (0..8)
                            .map(|i| {
                                w * (PI * (2 * (8 * j + i) + 1) as f64 * k as f64
                                    / (2 * len) as f64)
                                    .cos()
                            })
                            .sum::<f64>()
                            / 8.0
                    })
                    .collect();
                let mut c = vec![0.0; n];
                dct1(&means, &mut c);
                1.0 / c[k]
            })
            .collect()
    })[k]
}

/// The LLF a plain DCT gets from its LF samples (`cy` rows of `cx`), as a
/// wide-stored `min x max` corner.
#[cfg_attr(not(test), allow(dead_code))]
fn llf_from_lf(lf: &[f64], cx: usize, cy: usize) -> Vec<f64> {
    // DCT the LF samples, then scale each frequency.
    let mut d = lf.to_vec();
    separable(&mut d, cy, cx, dct1);
    for ky in 0..cy {
        for kx in 0..cx {
            d[ky * cx + kx] *= resample_scale(ky, cy) * resample_scale(kx, cx);
        }
    }
    if cx > cy { d } else { transpose(&d, cy, cx) }
}

/// The LF samples (`cy` rows of `cx`) giving a plain DCT's LLF.
fn lf_from_llf(llf: &[f64], cx: usize, cy: usize) -> Vec<f64> {
    let mut d = if cx > cy {
        llf.to_vec()
    } else {
        transpose(llf, cx, cy)
    };
    for ky in 0..cy {
        for kx in 0..cx {
            d[ky * cx + kx] /= resample_scale(ky, cy) * resample_scale(kx, cx);
        }
    }
    separable(&mut d, cy, cx, idct1);
    d
}

/// The inverse transform: pixels (row major, `8 cy` by `8 cx`) from
/// coefficients (wide-stored) and LF (`cy` rows of `cx`).
/// (The decoder's; the encoder checks its forward transform with it.)
#[cfg_attr(not(test), allow(dead_code))]
pub fn inverse(t: TransformType, coeffs: &[f32], lf: &[f32]) -> Vec<f32> {
    let (cx, cy) = t.covered();
    let (w, h) = (cx * 8, cy * 8);
    let mut c: Vec<f64> = coeffs[..w * h].iter().map(|&v| f64::from(v)).collect();
    let out = if t.is_plain_dct() {
        let lf: Vec<f64> = lf[..cx * cy].iter().map(|&v| f64::from(v)).collect();
        let llf = llf_from_lf(&lf, cx, cy);
        let (rows, cols) = (cy.min(cx), cx.max(cy));
        let stride = w.max(h);
        for r in 0..rows {
            for k in 0..cols {
                c[r * stride + k] = llf[r * cols + k];
            }
        }
        idct2d(&c, h, w)
    } else {
        c[0] = f64::from(lf[0]);
        small_inverse(t, &c)
    };
    out.into_iter().map(|v| v as f32).collect()
}

/// The forward transform: wide-stored coefficients (the LLF positions
/// holding the LLF) and the LF samples, from pixels (`8 cy` by `8 cx`).
pub fn forward(t: TransformType, pixels: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (cx, cy) = t.covered();
    let (w, h) = (cx * 8, cy * 8);
    let p: Vec<f64> = pixels[..w * h].iter().map(|&v| f64::from(v)).collect();
    if t.is_plain_dct() {
        let c = dct2d(&p, h, w);
        let (rows, cols) = (cy.min(cx), cx.max(cy));
        let stride = w.max(h);
        let mut llf = vec![0.0; rows * cols];
        for r in 0..rows {
            for k in 0..cols {
                llf[r * cols + k] = c[r * stride + k];
            }
        }
        let lf = lf_from_llf(&llf, cx, cy);
        (
            c.into_iter().map(|v| v as f32).collect(),
            lf.into_iter().map(|v| v as f32).collect(),
        )
    } else {
        let m = small_forward_matrix(t);
        let mut c = vec![0f32; 64];
        for (k, ck) in c.iter_mut().enumerate() {
            *ck = (0..64).map(|i| m[k * 64 + i] * p[i]).sum::<f64>() as f32;
        }
        let lf = vec![c[0]];
        (c, lf)
    }
}

/// The small transforms, as the decoder applies them (coefficient 0 holds
/// the LF).
fn small_inverse(t: TransformType, c: &[f64]) -> Vec<f64> {
    use TransformType::*;
    let mut out = vec![0.0; 64];
    match t {
        Identity => {
            let dcs = quad(c[0], c[1], c[8], c[9]);
            for y in 0..2 {
                for x in 0..2 {
                    let block_dc = dcs[y * 2 + x];
                    let mut residual_sum = 0.0;
                    for iy in 0..4 {
                        for ix in 0..4 {
                            if ix == 0 && iy == 0 {
                                continue;
                            }
                            residual_sum += c[(y + iy * 2) * 8 + x + ix * 2];
                        }
                    }
                    let centre = block_dc - residual_sum / 16.0;
                    out[(4 * y + 1) * 8 + 4 * x + 1] = centre;
                    for iy in 0..4 {
                        for ix in 0..4 {
                            if ix == 1 && iy == 1 {
                                continue;
                            }
                            out[(y * 4 + iy) * 8 + x * 4 + ix] =
                                c[(y + iy * 2) * 8 + x + ix * 2] + centre;
                        }
                    }
                    out[y * 4 * 8 + x * 4] = c[(y + 2) * 8 + x + 2] + centre;
                }
            }
        }
        Dct2 => {
            let mut a = c.to_vec();
            let mut b = vec![0.0; 64];
            haar_top(2, &a, &mut b);
            haar_top(4, &b, &mut a);
            haar_top(8, &a, &mut out);
        }
        Dct4 => {
            let dcs = quad(c[0], c[1], c[8], c[9]);
            for y in 0..2 {
                for x in 0..2 {
                    let mut block = vec![0.0; 16];
                    block[0] = dcs[y * 2 + x];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            if ix == 0 && iy == 0 {
                                continue;
                            }
                            block[iy * 4 + ix] = c[(y + iy * 2) * 8 + x + ix * 2];
                        }
                    }
                    let px = idct2d(&block, 4, 4);
                    for iy in 0..4 {
                        for ix in 0..4 {
                            out[(y * 4 + iy) * 8 + x * 4 + ix] = px[iy * 4 + ix];
                        }
                    }
                }
            }
        }
        Dct8x4 => {
            let dcs = [c[0] + c[8], c[0] - c[8]];
            for x in 0..2 {
                let mut block = vec![0.0; 32];
                for iy in 0..4 {
                    for ix in 0..8 {
                        block[iy * 8 + ix] = if ix == 0 && iy == 0 {
                            dcs[x]
                        } else {
                            c[(x + iy * 2) * 8 + ix]
                        };
                    }
                }
                // 8 tall, 4 wide: coefficients stored 4 x 8.
                let px = idct2d(&block, 8, 4);
                for iy in 0..8 {
                    for ix in 0..4 {
                        out[iy * 8 + x * 4 + ix] = px[iy * 4 + ix];
                    }
                }
            }
        }
        Dct4x8 => {
            let dcs = [c[0] + c[8], c[0] - c[8]];
            for y in 0..2 {
                let mut block = vec![0.0; 32];
                for iy in 0..4 {
                    for ix in 0..8 {
                        block[iy * 8 + ix] = if ix == 0 && iy == 0 {
                            dcs[y]
                        } else {
                            c[(y + iy * 2) * 8 + ix]
                        };
                    }
                }
                let px = idct2d(&block, 4, 8);
                for iy in 0..4 {
                    for ix in 0..8 {
                        out[(y * 4 + iy) * 8 + ix] = px[iy * 8 + ix];
                    }
                }
            }
        }
        Afv0 | Afv1 | Afv2 | Afv3 => {
            let kind = t as usize - Afv0 as usize;
            let (afv_x, afv_y) = (kind & 1, kind / 2);
            let dcs = [(c[0] + c[8] + c[1]) * 4.0, c[0] + c[8] - c[1], c[0] - c[8]];
            // The corner: a 4x4 non-separable basis.
            let mut coeff = [0.0; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    coeff[iy * 4 + ix] = if ix == 0 && iy == 0 {
                        dcs[0]
                    } else {
                        c[iy * 2 * 8 + ix * 2]
                    };
                }
            }
            let basis = afv_basis();
            let mut block = [0.0; 16];
            for i in 0..16 {
                block[i] = (0..16).map(|j| coeff[j] * basis[j * 16 + i]).sum();
            }
            for iy in 0..4 {
                let by = if afv_y == 1 { 3 - iy } else { iy };
                for ix in 0..4 {
                    let bx = if afv_x == 1 { 3 - ix } else { ix };
                    out[(iy + afv_y * 4) * 8 + afv_x * 4 + ix] = block[by * 4 + bx];
                }
            }
            // The 4x4 beside it.
            let mut b4 = vec![0.0; 16];
            for iy in 0..4 {
                for ix in 0..4 {
                    b4[iy * 4 + ix] = if ix == 0 && iy == 0 {
                        dcs[1]
                    } else {
                        c[iy * 2 * 8 + ix * 2 + 1]
                    };
                }
            }
            let px = idct2d(&b4, 4, 4);
            for iy in 0..4 {
                for ix in 0..4 {
                    out[(iy + afv_y * 4) * 8 + (1 - afv_x) * 4 + ix] = px[iy * 4 + ix];
                }
            }
            // The 4x8 half.
            let mut b8 = vec![0.0; 32];
            for iy in 0..4 {
                for ix in 0..8 {
                    b8[iy * 8 + ix] = if ix == 0 && iy == 0 {
                        dcs[2]
                    } else {
                        c[(1 + iy * 2) * 8 + ix]
                    };
                }
            }
            let px = idct2d(&b8, 4, 8);
            for iy in 0..4 {
                for ix in 0..8 {
                    out[(iy + (1 - afv_y) * 4) * 8 + ix] = px[iy * 8 + ix];
                }
            }
        }
        _ => unreachable!(),
    }
    out
}

fn quad(b00: f64, b01: f64, b10: f64, b11: f64) -> [f64; 4] {
    [
        b00 + b01 + b10 + b11,
        b00 + b01 - b10 - b11,
        b00 - b01 + b10 - b11,
        b00 - b01 - b10 + b11,
    ]
}

/// One level of the 2x2 Haar the DCT2 transform repeats.
fn haar_top(s: usize, input: &[f64], output: &mut [f64]) {
    output.copy_from_slice(input);
    let n = s / 2;
    for y in 0..n {
        for x in 0..n {
            let c00 = input[y * 8 + x];
            let c01 = input[y * 8 + n + x];
            let c10 = input[(y + n) * 8 + x];
            let c11 = input[(y + n) * 8 + n + x];
            output[y * 2 * 8 + x * 2] = c00 + c01 + c10 + c11;
            output[y * 2 * 8 + x * 2 + 1] = c00 + c01 - c10 - c11;
            output[(y * 2 + 1) * 8 + x * 2] = c00 - c01 + c10 - c11;
            output[(y * 2 + 1) * 8 + x * 2 + 1] = c00 - c01 - c10 + c11;
        }
    }
}

/// The AFV corner's 4x4 basis: 16 orthonormal vectors (row `j` is basis
/// function `j` over the 16 corner samples).
fn afv_basis() -> &'static [f64; 256] {
    static B: OnceLock<[f64; 256]> = OnceLock::new();
    B.get_or_init(|| {
        let mut b = [0.0; 256];
        for (i, v) in AFV_BASIS.iter().enumerate() {
            b[i] = *v;
        }
        b
    })
}

/// The forward matrix of a small transform: the inverse of its (linear)
/// inverse, by Gauss-Jordan elimination.
fn small_forward_matrix(t: TransformType) -> &'static [f64] {
    static M: [OnceLock<Vec<f64>>; 27] = [const { OnceLock::new() }; 27];
    M[t as usize].get_or_init(|| {
        // Columns: the inverse of each unit coefficient.
        let mut a = vec![0.0; 64 * 64];
        for k in 0..64 {
            let mut c = vec![0.0; 64];
            c[k] = 1.0;
            let px = small_inverse(t, &c);
            for i in 0..64 {
                a[i * 64 + k] = px[i];
            }
        }
        invert(&a, 64)
    })
}

fn invert(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| m[x * n + col].abs().total_cmp(&m[y * n + col].abs()))
            .unwrap();
        for k in 0..n {
            m.swap(col * n + k, pivot * n + k);
            inv.swap(col * n + k, pivot * n + k);
        }
        let p = m[col * n + col];
        assert!(p.abs() > 1e-12, "a singular transform");
        for k in 0..n {
            m[col * n + k] /= p;
            inv[col * n + k] /= p;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * n + col];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[col * n + k];
                        inv[r * n + k] -= f * inv[col * n + k];
                    }
                }
            }
        }
    }
    inv
}

/// The AFV corner basis, as the format defines it.
#[rustfmt::skip]
#[allow(clippy::approx_constant)]
const AFV_BASIS: [f64; 256] = include!("afv_basis.in");

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_transforms::transform_map::HfTransformType;

    fn oracle(t: TransformType, coeffs: &[f32], lf: &[f32]) -> Vec<f32> {
        let (cx, cy) = t.covered();
        let mut buf = coeffs.to_vec();
        buf.resize((cx * cy * 64).max(64), 0.0);
        let mut lf = lf.to_vec();
        lf.resize(cx * cy, 0.0);
        jxl_transforms::transform::transform_to_pixels(
            HfTransformType::from_usize(t as usize).unwrap(),
            &mut lf,
            &mut buf,
        );
        buf.truncate(cx * cy * 64);
        buf
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn inverse_matches_the_decoder() {
        for t in TransformType::ALL {
            let (cx, cy) = t.covered();
            let n = cx * cy * 64;
            let coeffs = noise(n, t as u64 + 1);
            let lf = noise(cx * cy, t as u64 + 100);
            let want = oracle(t, &coeffs, &lf);
            let got = inverse(t, &coeffs, &lf);
            let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
            for i in 0..n {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-4 * scale,
                    "{t:?} sample {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn forward_inverts() {
        for t in TransformType::ALL {
            let (cx, cy) = t.covered();
            let n = cx * cy * 64;
            let pixels = noise(n, t as u64 + 7);
            let (coeffs, lf) = forward(t, &pixels);
            let back = oracle(t, &coeffs, &lf);
            for i in 0..n {
                assert!(
                    (back[i] - pixels[i]).abs() < 1e-3,
                    "{t:?} sample {i}: {} vs {}",
                    back[i],
                    pixels[i]
                );
            }
        }
    }
}
