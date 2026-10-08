//! How HF coefficients are coded: their order per block shape, the block
//! context map, and the contexts of the non-zero count and of each
//! coefficient.

use super::transform::TransformType;
use crate::encode::bits::BitWriter;
use crate::encode::entropy::{EntropyOptions, ceil_log2};
use crate::{Error, Result};

pub const NUM_ORDERS: usize = 13;
pub const NON_ZERO_BUCKETS: usize = 37;
pub const ZERO_DENSITY_CONTEXT_COUNT: usize = 458;

/// The transform each order (shape) is defined by.
pub const ORDER_TRANSFORM: [TransformType; NUM_ORDERS] = [
    TransformType::Dct8,
    TransformType::Identity,
    TransformType::Dct16,
    TransformType::Dct32,
    TransformType::Dct8x16,
    TransformType::Dct8x32,
    TransformType::Dct16x32,
    TransformType::Dct64,
    TransformType::Dct32x64,
    TransformType::Dct128,
    TransformType::Dct64x128,
    TransformType::Dct256,
    TransformType::Dct128x256,
];

const COEFF_FREQ_CONTEXT: [usize; 64] = [
    0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];
const COEFF_NUM_NONZERO_CONTEXT: [usize; 64] = [
    0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206,
];

/// The context of the coefficient at order position `k` with
/// `nonzeros_left` still to come, `prev` whether the one before was not
/// zero.
#[inline]
pub fn zero_density_context(
    nonzeros_left: usize,
    k: usize,
    log_num_blocks: usize,
    prev: usize,
) -> usize {
    let nz = nonzeros_left.div_ceil(1 << log_num_blocks);
    let kn = k >> log_num_blocks;
    (COEFF_NUM_NONZERO_CONTEXT[nz & 63] + COEFF_FREQ_CONTEXT[kn & 63]) * 2 + prev
}

/// The natural coefficient order of an order's shape: the LLF corner
/// first, then zig-zag (positions in the wide layout).
pub fn natural_order(shape: usize) -> Vec<u32> {
    let (cx, cy) = ORDER_TRANSFORM[shape].covered();
    let xsize = cx * 8;
    let xs = cx / cy;
    let xsm = xs - 1;
    let xss = ceil_log2(xs as u32) as usize;
    let mut out = vec![0u32; cx * cy * 64];
    let mut cur = cx * cy;
    for i in 0..xsize {
        for j in 0..=i {
            let (mut x, mut y) = (j, i - j);
            if i % 2 != 0 {
                std::mem::swap(&mut x, &mut y);
            }
            if y & xsm != 0 {
                continue;
            }
            y >>= xss;
            let val = if x < cx && y < cy {
                y * cx + x
            } else {
                cur += 1;
                cur - 1
            };
            out[val] = (y * xsize + x) as u32;
        }
    }
    for ir in 1..xsize {
        let ip = xsize - ir;
        let i = ip - 1;
        for j in 0..=i {
            let (mut x, mut y) = (xsize - 1 - (i - j), xsize - 1 - j);
            if i % 2 != 0 {
                std::mem::swap(&mut x, &mut y);
            }
            if y & xsm != 0 {
                continue;
            }
            y >>= xss;
            out[cur] = (y * xsize + x) as u32;
            cur += 1;
        }
    }
    out
}

/// The block context map: LF and quant thresholds, and the map from
/// (channel, shape, quant bucket, LF bucket) to a block context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockContextMap {
    /// Per channel (X, Y, B), the LF thresholds.
    pub lf_thresholds: [Vec<i32>; 3],
    pub qf_thresholds: Vec<u32>,
    /// `3 * 13 * (qf + 1) * lf` entries.
    pub context_map: Vec<u8>,
}

impl Default for BlockContextMap {
    fn default() -> Self {
        BlockContextMap {
            lf_thresholds: [vec![], vec![], vec![]],
            qf_thresholds: vec![],
            context_map: vec![
                0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, //
                7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
                7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
            ],
        }
    }
}

impl BlockContextMap {
    pub fn num_lf_contexts(&self) -> usize {
        self.lf_thresholds.iter().map(|t| t.len() + 1).product()
    }

    pub fn num_contexts(&self) -> usize {
        usize::from(*self.context_map.iter().max().unwrap()) + 1
    }

    pub fn num_ac_contexts(&self) -> usize {
        self.num_contexts() * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT)
    }

    /// A block's context.
    pub fn block_context(&self, lf_idx: usize, qf: u32, shape: usize, c: usize) -> usize {
        let qf_idx = self.qf_thresholds.iter().filter(|&&t| qf > t).count();
        let mut idx = if c < 2 { c ^ 1 } else { 2 };
        idx = idx * NUM_ORDERS + shape;
        idx = idx * (self.qf_thresholds.len() + 1) + qf_idx;
        idx = idx * self.num_lf_contexts() + lf_idx;
        usize::from(self.context_map[idx])
    }

    /// The LF bucket of a block from its quantised LF (X, Y, B).
    pub fn lf_index(&self, q: [i32; 3]) -> usize {
        let bucket = |c: usize| self.lf_thresholds[c].iter().filter(|&&t| q[c] > t).count();
        let mut b = bucket(0);
        b *= self.lf_thresholds[2].len() + 1;
        b += bucket(2);
        b *= self.lf_thresholds[1].len() + 1;
        b += bucket(1);
        b
    }

    pub fn nonzero_context(&self, nonzeros: usize, block_context: usize) -> usize {
        let ctx = if nonzeros < 8 {
            nonzeros
        } else if nonzeros < 64 {
            4 + nonzeros / 2
        } else {
            36
        };
        ctx * self.num_contexts() + block_context
    }

    pub fn zero_density_offset(&self, block_context: usize) -> usize {
        self.num_contexts() * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_context
    }

    pub(crate) fn write(&self, w: &mut BitWriter, entropy: &EntropyOptions) -> Result<()> {
        if *self == BlockContextMap::default() {
            w.bit(true);
            return Ok(());
        }
        w.bit(false);
        for t in &self.lf_thresholds {
            if t.len() > 15 {
                return Err(Error::InvalidInput("at most 15 LF thresholds".into()));
            }
            w.write(4, t.len() as u32);
            for &v in t {
                let u = crate::encode::entropy::pack_signed(v);
                match u {
                    0..16 => {
                        w.write(2, 0);
                        w.write(4, u);
                    }
                    16..272 => {
                        w.write(2, 1);
                        w.write(8, u - 16);
                    }
                    272..65808 => {
                        w.write(2, 2);
                        w.write(16, u - 272);
                    }
                    _ => {
                        w.write(2, 3);
                        w.write(32, u - 65808);
                    }
                }
            }
        }
        if self.qf_thresholds.len() > 15 {
            return Err(Error::InvalidInput("at most 15 quant thresholds".into()));
        }
        w.write(4, self.qf_thresholds.len() as u32);
        for &q in &self.qf_thresholds {
            let v = q
                .checked_sub(1)
                .ok_or_else(|| Error::InvalidInput("a quant threshold of 0".into()))?;
            match v {
                0..4 => {
                    w.write(2, 0);
                    w.write(2, v);
                }
                4..12 => {
                    w.write(2, 1);
                    w.write(3, v - 4);
                }
                12..44 => {
                    w.write(2, 2);
                    w.write(5, v - 12);
                }
                44..300 => {
                    w.write(2, 3);
                    w.write(8, v - 44);
                }
                _ => return Err(Error::InvalidInput(format!("a quant threshold of {q}"))),
            }
        }
        let size = 3 * NUM_ORDERS * self.num_lf_contexts() * (self.qf_thresholds.len() + 1);
        if self.num_lf_contexts() * (self.qf_thresholds.len() + 1) > 64
            || self.context_map.len() != size
            || self.num_contexts() > 16
        {
            return Err(Error::InvalidInput(
                "a block context map of the wrong size".into(),
            ));
        }
        // Every context up to the largest must be used.
        for k in 0..self.num_contexts() {
            if !self.context_map.contains(&(k as u8)) {
                return Err(Error::InvalidInput(format!("block context {k} unused")));
            }
        }
        crate::encode::entropy::write_context_map(w, &self.context_map, entropy);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_orders_match_the_format() {
        const ORDER_1X1: [u32; 64] = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37,
            44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        ];
        assert_eq!(natural_order(0), ORDER_1X1);
        for shape in 0..NUM_ORDERS {
            let mut o = natural_order(shape);
            o.sort_unstable();
            assert!(
                o.iter().enumerate().all(|(i, &v)| v as usize == i),
                "shape {shape}"
            );
        }
    }
}
