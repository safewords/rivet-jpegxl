//! The hybrid-uint split of a value into a token, coded with the histogram,
//! and raw bits, written as they are.

use crate::encode::bits::BitWriter;

/// A hybrid-uint configuration: values under `1 << split_exponent` are
/// their own token; above, the token carries the value's bit length and
/// `msb_in_token` bits under the leading one and its `lsb_in_token` lowest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HybridUint {
    pub split_exponent: u32,
    pub msb_in_token: u32,
    pub lsb_in_token: u32,
}

/// A value as its token, and the raw bits that follow it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Split {
    pub token: u32,
    pub nbits: u32,
    pub bits: u32,
}

impl HybridUint {
    pub(crate) const fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        HybridUint {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }

    /// The configurations the encoder chooses among for a histogram.
    pub(crate) const CANDIDATES: [HybridUint; 10] = [
        HybridUint::new(4, 2, 0),
        HybridUint::new(4, 1, 0),
        HybridUint::new(4, 0, 0),
        HybridUint::new(0, 0, 0),
        HybridUint::new(2, 0, 0),
        HybridUint::new(4, 1, 1),
        HybridUint::new(5, 2, 0),
        HybridUint::new(6, 2, 0),
        HybridUint::new(4, 0, 1),
        HybridUint::new(3, 1, 0),
    ];

    #[inline]
    pub(crate) fn split(&self, value: u32) -> Split {
        let split_token = 1u32 << self.split_exponent;
        if value < split_token {
            return Split {
                token: value,
                nbits: 0,
                bits: 0,
            };
        }
        let n = 31 - value.leading_zeros();
        let m = (value >> (n - self.msb_in_token)) & ((1 << self.msb_in_token) - 1);
        let low = value & ((1 << self.lsb_in_token) - 1);
        let in_token = self.msb_in_token + self.lsb_in_token;
        let nbits = n - in_token;
        Split {
            token: split_token
                + ((n - self.split_exponent) << in_token)
                + (m << self.lsb_in_token)
                + low,
            nbits,
            bits: (value >> self.lsb_in_token) & ((1u64 << nbits) - 1) as u32,
        }
    }

    /// The largest token any 32-bit value makes.
    #[cfg(test)]
    pub(crate) fn max_token(&self) -> u32 {
        self.split(u32::MAX).token
    }

    /// The configuration as the histogram header codes it.
    pub(crate) fn write(&self, w: &mut BitWriter, log_alpha_size: u32) {
        w.write(ceil_log2(log_alpha_size + 1), self.split_exponent);
        if self.split_exponent == log_alpha_size {
            return;
        }
        w.write(ceil_log2(self.split_exponent + 1), self.msb_in_token);
        w.write(
            ceil_log2(self.split_exponent - self.msb_in_token + 1),
            self.lsb_in_token,
        );
    }
}

pub(crate) fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

/// A signed value as the unsigned one it is coded as: 0, -1, 1, -2, 2, …
#[inline]
pub(crate) fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder's read.
    fn join(c: &HybridUint, s: Split) -> u32 {
        let split_token = 1u32 << c.split_exponent;
        if s.token < split_token {
            return s.token;
        }
        let in_token = c.lsb_in_token + c.msb_in_token;
        let nbits = c.split_exponent - in_token + ((s.token - split_token) >> in_token);
        assert_eq!(nbits, s.nbits);
        let low = s.token & ((1 << c.lsb_in_token) - 1);
        let nolow = s.token >> c.lsb_in_token;
        let hi = (nolow & ((1 << c.msb_in_token) - 1)) | (1 << c.msb_in_token);
        (((((hi as u64) << nbits) | s.bits as u64) << c.lsb_in_token) | low as u64) as u32
    }

    #[test]
    fn every_configuration_round_trips() {
        for c in HybridUint::CANDIDATES {
            for v in (0..3000).chain([65535, 65536, 1 << 20, u32::MAX >> 1, u32::MAX]) {
                assert_eq!(join(&c, c.split(v)), v, "{c:?} {v}");
            }
            assert!(c.max_token() < 224, "{c:?}");
        }
    }

    #[test]
    fn signed_packing_interleaves() {
        let packed: Vec<u32> = [0, -1, 1, -2, 2, i32::MIN, i32::MAX]
            .iter()
            .map(|&v| pack_signed(v))
            .collect();
        assert_eq!(packed, [0, 1, 2, 3, 4, u32::MAX, u32::MAX - 1]);
    }
}
