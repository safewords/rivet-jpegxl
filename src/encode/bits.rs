//! The bit writer: JPEG XL packs bits least significant first, and codes
//! most header fields with the `U32` selector distributions.

/// One of a `U32` field's four distributions.
#[derive(Clone, Copy)]
pub(super) enum Dist {
    /// The value itself, no bits.
    Val(u32),
    /// `n` bits, plus `offset`.
    Bits(u32, u32),
}

/// The distributions an enum field is coded with.
const ENUM: [Dist; 4] = [
    Dist::Val(0),
    Dist::Val(1),
    Dist::Bits(4, 2),
    Dist::Bits(6, 18),
];

#[derive(Default)]
pub(super) struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    pending: u32,
}

impl BitWriter {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// The low `n` bits of `value` (`n` up to 32).
    pub(super) fn write(&mut self, n: u32, value: u32) {
        debug_assert!(n <= 32);
        debug_assert!(n == 32 || value >> n == 0, "{value} does not fit {n} bits");
        if n == 0 {
            return;
        }
        self.acc |= u64::from(value) << self.pending;
        self.pending += n;
        while self.pending >= 8 {
            self.bytes.push(self.acc as u8);
            self.acc >>= 8;
            self.pending -= 8;
        }
    }

    pub(super) fn bit(&mut self, on: bool) {
        self.write(1, u32::from(on));
    }

    /// A `U32` field: the first distribution that can hold `value`.
    pub(super) fn u32(&mut self, value: u32, dists: [Dist; 4]) {
        for (selector, dist) in dists.into_iter().enumerate() {
            let fits = match dist {
                Dist::Val(v) => value == v,
                Dist::Bits(n, offset) => {
                    value >= offset && (n == 32 || u64::from(value - offset) < 1u64 << n)
                }
            };
            if fits {
                self.write(2, selector as u32);
                if let Dist::Bits(n, offset) = dist {
                    self.write(n, value - offset);
                }
                return;
            }
        }
        unreachable!("{value} fits none of the field's distributions");
    }

    /// An enum field, by its value.
    pub(super) fn enumeration(&mut self, value: u32) {
        self.u32(value, ENUM);
    }

    /// A `U64` field holding 0 (`extensions`, frame `flags`).
    pub(super) fn u64_zero(&mut self) {
        self.write(2, 0);
    }

    /// An empty name.
    pub(super) fn empty_string(&mut self) {
        self.write(2, 0);
    }

    /// The `varint16` of the entropy-code headers.
    pub(super) fn varint16(&mut self, value: u16) {
        if value == 0 {
            self.bit(false);
            return;
        }
        self.bit(true);
        let nbits = 15 - value.leading_zeros();
        self.write(4, nbits);
        self.write(nbits, u32::from(value) - (1 << nbits));
    }

    /// Zeros up to the next byte boundary.
    pub(super) fn pad_to_byte(&mut self) {
        if self.pending > 0 {
            self.write(8 - self.pending, 0);
        }
    }

    /// Bits written so far.
    pub(super) fn bits_written(&self) -> usize {
        self.bytes.len() * 8 + self.pending as usize
    }

    /// Another writer's bits, after these.
    pub(super) fn append_bits(&mut self, other: &BitWriter) {
        if self.pending == 0 {
            self.bytes.extend_from_slice(&other.bytes);
        } else {
            for &b in &other.bytes {
                self.write(8, u32::from(b));
            }
        }
        self.write(other.pending, other.acc as u32);
    }

    pub(super) fn append_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.pending, 0);
        self.bytes.extend_from_slice(bytes);
    }

    /// The bytes, the last one padded with zeros.
    pub(super) fn finish(mut self) -> Vec<u8> {
        self.pad_to_byte();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_go_least_significant_first() {
        let mut w = BitWriter::new();
        w.write(2, 0b10);
        w.write(3, 0b101);
        w.write(8, 0xff);
        assert_eq!(w.finish(), vec![0b1111_0110, 0b0001_1111]);
    }

    #[test]
    fn u32_picks_the_first_fitting_distribution() {
        let mut w = BitWriter::new();
        w.enumeration(13); // selector 2, then 13 - 2 in 4 bits
        assert_eq!(w.finish(), vec![0b0010_1110]);
    }
}
