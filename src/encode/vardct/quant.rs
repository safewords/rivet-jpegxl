//! Dequantisation matrices: how each of the 17 tables is described (the
//! library default, or one of seven parametric or raw encodings), the
//! weights the decoder computes from that, and how a description is sent.

use std::f32::consts::SQRT_2;

use super::library::Library;
use crate::encode::bits::BitWriter;
use crate::encode::header::write_f16;
use crate::{Error, Result};

/// The tables' sizes in blocks (rows, columns of the wide layout / 8).
pub const REQUIRED_SIZE_X: [usize; 17] = [1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 8, 4, 16, 8, 32, 16];
pub const REQUIRED_SIZE_Y: [usize; 17] = [1, 1, 1, 1, 2, 4, 2, 4, 4, 1, 1, 8, 8, 16, 16, 32, 32];

/// Distance bands: per channel, the first weight then the ratios to the
/// next (as `mult`).
#[derive(Clone, Debug, PartialEq)]
pub struct Bands {
    pub params: [Vec<f32>; 3],
}

impl Bands {
    pub fn from_array<const N: usize>(values: &[[f32; N]; 3]) -> Self {
        Bands {
            params: [values[0].to_vec(), values[1].to_vec(), values[2].to_vec()],
        }
    }

    fn num_bands(&self) -> usize {
        self.params[0].len()
    }

    fn write(&self, w: &mut BitWriter) -> Result<()> {
        let n = self.num_bands();
        if n == 0 || n > 17 || self.params.iter().any(|p| p.len() != n) {
            return Err(Error::InvalidInput(
                "1 to 17 distance bands, the same per channel".into(),
            ));
        }
        w.write(4, n as u32 - 1);
        for p in &self.params {
            write_f16(w, p[0] / 64.0)?;
            for &v in &p[1..] {
                write_f16(w, v)?;
            }
        }
        Ok(())
    }
}

/// How a table is described.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantEncoding {
    Library,
    Identity {
        xyb_weights: [[f32; 3]; 3],
    },
    Dct2 {
        xyb_weights: [[f32; 6]; 3],
    },
    Dct4 {
        params: Bands,
        xyb_mul: [[f32; 2]; 3],
    },
    Dct4x8 {
        params: Bands,
        xyb_mul: [f32; 3],
    },
    Afv {
        params4x8: Bands,
        params4x4: Bands,
        weights: [[f32; 9]; 3],
    },
    Dct {
        params: Bands,
    },
    /// Explicit integer weights (`1 / (den * q)`), modular coded.
    Raw {
        qtable: Vec<i32>,
        qtable_den: f32,
    },
}

const ALMOST_ZERO: f32 = 1e-8;

fn mult(v: f32) -> f32 {
    if v > 0.0 { 1.0 + v } else { 1.0 / (1.0 - v) }
}

fn interpolate_vec(scaled_pos: f32, array: &[f32]) -> f32 {
    let idxf = scaled_pos.floor();
    let frac = scaled_pos - idxf;
    let idx = idxf as usize;
    let a = array[idx];
    let b = array[idx + 1];
    (b / a).powf(frac) * a
}

fn interpolate(pos: f32, max: f32, array: &[f32]) -> f32 {
    let scaled = pos * (array.len() - 1) as f32 / max;
    let idx = scaled as usize;
    let a = array[idx];
    let b = array[idx + 1];
    a * (b / a).powf(scaled - idx as f32)
}

/// The bands of one channel, and one more past the end (0, as the
/// decoder's fixed-size array holds) for interpolation at the last.
fn bands_of(p: &[f32]) -> Result<Vec<f32>> {
    let mut b = vec![0.0; 18];
    b[0] = p[0];
    for i in 1..p.len() {
        b[i] = b[i - 1] * mult(p[i]);
    }
    if b[..p.len()].iter().any(|&v| v < ALMOST_ZERO) {
        return Err(Error::InvalidInput("a distance band weight of zero".into()));
    }
    Ok(b)
}

fn quant_weights(rows: usize, cols: usize, bands: &Bands, out: &mut [f32]) -> Result<()> {
    let n = bands.num_bands();
    for c in 0..3 {
        let b = bands_of(&bands.params[c])?;
        let scale = (n - 1) as f32 / (SQRT_2 + 1e-6);
        let rcpcol = scale / (cols - 1) as f32;
        let rcprow = scale / (rows - 1) as f32;
        for y in 0..rows {
            let dy = y as f32 * rcprow;
            let dy2 = dy * dy;
            for x in 0..cols {
                let dx = x as f32 * rcpcol;
                let d = (dx * dx + dy2).sqrt();
                out[c * rows * cols + y * cols + x] =
                    if n == 1 { b[0] } else { interpolate_vec(d, &b) };
            }
        }
    }
    Ok(())
}

/// The table's dequantisation weights (the decoder's: one over the
/// quantisation weights), three channels of `64 * x * y`.
pub fn compute(encoding: &QuantEncoding, idx: usize) -> Result<Vec<f32>> {
    let rows = 8 * REQUIRED_SIZE_X[idx];
    let cols = 8 * REQUIRED_SIZE_Y[idx];
    let num = rows * cols;
    let mut w = vec![0f32; 3 * num];
    match encoding {
        QuantEncoding::Library => return compute(&Library::get_library_encoding(idx), idx),
        QuantEncoding::Identity { xyb_weights } => {
            for c in 0..3 {
                for i in 0..64 {
                    w[64 * c + i] = xyb_weights[c][0];
                }
                w[64 * c + 1] = xyb_weights[c][1];
                w[64 * c + 8] = xyb_weights[c][1];
                w[64 * c + 9] = xyb_weights[c][2];
            }
        }
        QuantEncoding::Dct2 { xyb_weights } => {
            for (c, xw) in xyb_weights.iter().enumerate() {
                let s = c * 64;
                w[s] = 0xBAD as f32;
                w[s + 1] = xw[0];
                w[s + 8] = xw[0];
                w[s + 9] = xw[1];
                for y in 0..2 {
                    for x in 0..2 {
                        w[s + y * 8 + x + 2] = xw[2];
                        w[s + (y + 2) * 8 + x] = xw[2];
                        w[s + (y + 2) * 8 + x + 2] = xw[3];
                    }
                }
                for y in 0..4 {
                    for x in 0..4 {
                        w[s + y * 8 + x + 4] = xw[4];
                        w[s + (y + 4) * 8 + x] = xw[4];
                        w[s + (y + 4) * 8 + x + 4] = xw[5];
                    }
                }
            }
        }
        QuantEncoding::Dct4 { params, xyb_mul } => {
            let mut w4 = [0f32; 48];
            quant_weights(4, 4, params, &mut w4)?;
            for c in 0..3 {
                for y in 0..8 {
                    for x in 0..8 {
                        w[c * num + y * 8 + x] = w4[c * 16 + (y / 2) * 4 + x / 2];
                    }
                }
                w[c * num + 1] /= xyb_mul[c][0];
                w[c * num + 8] /= xyb_mul[c][0];
                w[c * num + 9] /= xyb_mul[c][1];
            }
        }
        QuantEncoding::Dct4x8 { params, xyb_mul } => {
            let mut w48 = [0f32; 96];
            quant_weights(4, 8, params, &mut w48)?;
            for c in 0..3 {
                for y in 0..8 {
                    for x in 0..8 {
                        w[c * num + y * 8 + x] = w48[c * 32 + (y / 2) * 8 + x];
                    }
                }
                w[c * num + 8] /= xyb_mul[c];
            }
        }
        QuantEncoding::Dct { params } => quant_weights(rows, cols, params, &mut w)?,
        QuantEncoding::Raw { qtable, qtable_den } => {
            if qtable.len() != 3 * num {
                return Err(Error::InvalidInput(format!(
                    "a raw quant table of {} entries, not {}",
                    qtable.len(),
                    3 * num
                )));
            }
            for i in 0..3 * num {
                w[i] = 1.0 / (qtable_den * qtable[i] as f32);
            }
        }
        QuantEncoding::Afv {
            params4x8,
            params4x4,
            weights,
        } => {
            const FREQS: [f32; 16] = [
                0xBAD as f32,
                0xBAD as f32,
                0.8517778890324296,
                5.37778436506804,
                0xBAD as f32,
                0xBAD as f32,
                4.734747904497923,
                5.449245381693219,
                1.6598270267479331,
                4.0,
                7.275749096817861,
                10.423227632456525,
                2.662932286148962,
                7.630657783650829,
                8.962388608184032,
                12.97166202570235,
            ];
            const LO: f32 = 0.8517778890324296;
            const HI: f32 = 12.97166202570235f32 - LO + 1e-6f32;
            let mut w48 = [0f32; 96];
            quant_weights(4, 8, params4x8, &mut w48)?;
            let mut w44 = [0f32; 48];
            quant_weights(4, 4, params4x4, &mut w44)?;
            for c in 0..3 {
                let mut bands = [0f32; 4];
                bands[0] = weights[c][5];
                for i in 1..4 {
                    bands[i] = bands[i - 1] * mult(weights[c][i + 5]);
                }
                if bands.iter().any(|&b| b < ALMOST_ZERO) {
                    return Err(Error::InvalidInput("an AFV band weight of zero".into()));
                }
                let s = c * 64;
                w[s] = 1.0;
                let set = |w: &mut [f32], x: usize, y: usize, v: f32| w[s + y * 8 + x] = v;
                set(&mut w, 0, 1, weights[c][0]);
                set(&mut w, 1, 0, weights[c][1]);
                set(&mut w, 0, 2, weights[c][2]);
                set(&mut w, 2, 0, weights[c][3]);
                set(&mut w, 2, 2, weights[c][4]);
                for y in 0..4 {
                    for x in 0..4 {
                        if x < 2 && y < 2 {
                            continue;
                        }
                        let v = interpolate(FREQS[y * 4 + x] - LO, HI, &bands);
                        set(&mut w, 2 * x, 2 * y, v);
                    }
                }
                for y in 0..4 {
                    for x in 0..8 {
                        if x == 0 && y == 0 {
                            continue;
                        }
                        w[c * num + (2 * y + 1) * 8 + x] = w48[c * 32 + y * 8 + x];
                    }
                }
                for y in 0..4 {
                    for x in 0..4 {
                        if x == 0 && y == 0 {
                            continue;
                        }
                        w[c * num + 2 * y * 8 + 2 * x + 1] = w44[c * 16 + y * 4 + x];
                    }
                }
            }
        }
    }
    for v in &mut w {
        if !(ALMOST_ZERO..=1.0 / ALMOST_ZERO).contains(v) {
            return Err(Error::InvalidInput(format!("a quantisation weight of {v}")));
        }
        *v = 1.0 / *v;
    }
    Ok(w)
}

/// An encoding's header (a `Raw` table's samples follow as a modular
/// stream, which the caller writes).
pub fn write_encoding(w: &mut BitWriter, e: &QuantEncoding, idx: usize) -> Result<()> {
    let single = REQUIRED_SIZE_X[idx] * REQUIRED_SIZE_Y[idx] == 1;
    let need_single = |name: &str| {
        if single {
            Ok(())
        } else {
            Err(Error::InvalidInput(format!(
                "{name} weights for a table of more than one block"
            )))
        }
    };
    match e {
        QuantEncoding::Library => w.write(3, 0),
        QuantEncoding::Identity { xyb_weights } => {
            need_single("identity")?;
            w.write(3, 1);
            for row in xyb_weights {
                for &v in row {
                    write_f16(w, v / 64.0)?;
                }
            }
        }
        QuantEncoding::Dct2 { xyb_weights } => {
            need_single("DCT2")?;
            w.write(3, 2);
            for row in xyb_weights {
                for &v in row {
                    write_f16(w, v / 64.0)?;
                }
            }
        }
        QuantEncoding::Dct4 { params, xyb_mul } => {
            need_single("DCT4")?;
            w.write(3, 3);
            for row in xyb_mul {
                for &v in row {
                    write_f16(w, v)?;
                }
            }
            params.write(w)?;
        }
        QuantEncoding::Dct4x8 { params, xyb_mul } => {
            need_single("DCT4x8")?;
            w.write(3, 4);
            for &v in xyb_mul {
                write_f16(w, v)?;
            }
            params.write(w)?;
        }
        QuantEncoding::Afv {
            params4x8,
            params4x4,
            weights,
        } => {
            need_single("AFV")?;
            w.write(3, 5);
            for row in weights {
                for (i, &v) in row.iter().enumerate() {
                    write_f16(w, if i < 6 { v / 64.0 } else { v })?;
                }
            }
            params4x8.write(w)?;
            params4x4.write(w)?;
        }
        QuantEncoding::Dct { params } => {
            w.write(3, 6);
            params.write(w)?;
        }
        QuantEncoding::Raw { qtable_den, .. } => {
            w.write(3, 7);
            write_f16(w, *qtable_den)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_tables_compute() {
        for idx in 0..17 {
            let t = compute(&QuantEncoding::Library, idx).unwrap();
            assert_eq!(
                t.len(),
                3 * 64 * REQUIRED_SIZE_X[idx] * REQUIRED_SIZE_Y[idx]
            );
            assert!(t.iter().all(|v| v.is_finite() && *v > 0.0));
        }
    }
}
