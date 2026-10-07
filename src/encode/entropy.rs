//! Entropy coding with prefix codes: the Brotli-format Huffman tables JPEG XL
//! allows in place of ANS, one histogram per context, values split into a
//! token and raw bits by the hybrid-uint configuration 4/2/0.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::bits::BitWriter;

/// Prefix codes are at most this long.
const MAX_CODE_LENGTH: u8 = 15;
/// The code-length code's codes are at most this long.
const MAX_CODE_LENGTH_CODE_LENGTH: u8 = 5;
/// The order the code-length code's lengths are sent in.
const CODE_LENGTH_CODE_ORDER: [usize; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// The static code the code-length code's lengths are sent with: for a
/// length 0..=5, its bit count and codeword (least significant bit first).
const CODE_LENGTH_CODE_LENGTH_CODE: [(u32, u32); 6] =
    [(2, 0), (4, 7), (3, 3), (2, 2), (2, 1), (4, 15)];

/// Tokens below this are the value itself.
const SPLIT_TOKEN: u32 = 16;
/// Values up to 2^32 - 1 make tokens below this.
const MAX_TOKENS: usize = 128;

/// A value as its token, and the raw bits that follow it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Token {
    pub symbol: u32,
    pub nbits: u32,
    pub bits: u32,
}

/// The hybrid-uint split with `split_exponent` 4, `msb_in_token` 2,
/// `lsb_in_token` 0: values under 16 are their own token; above, the token
/// carries the bit length and the two bits under the leading one.
pub(super) fn tokenize(value: u32) -> Token {
    if value < SPLIT_TOKEN {
        return Token {
            symbol: value,
            nbits: 0,
            bits: 0,
        };
    }
    let n = 31 - value.leading_zeros(); // 4 or more
    let nbits = n - 2;
    let top = value >> nbits; // 0b1xx
    Token {
        symbol: SPLIT_TOKEN + ((nbits - 2) << 2) + (top & 3),
        nbits,
        bits: value & ((1 << nbits) - 1),
    }
}

/// A signed value as the unsigned one it is coded as: 0, -1, 1, -2, 2, …
pub(super) fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

/// Token counts, per context.
pub(super) struct Histograms {
    counts: Vec<[u64; MAX_TOKENS]>,
}

impl Histograms {
    pub(super) fn new(contexts: usize) -> Self {
        Histograms {
            counts: vec![[0; MAX_TOKENS]; contexts],
        }
    }

    pub(super) fn add(&mut self, context: usize, value: u32) {
        self.counts[context][tokenize(value).symbol as usize] += 1;
    }
}

/// A prefix code: per symbol, its length and its codeword, bit-reversed so
/// it can be written least significant bit first.
struct PrefixCode {
    lengths: Vec<u8>,
    codewords: Vec<u32>,
    /// The symbol of a code with only one (or none: then 0), which takes
    /// no bits.
    single: Option<usize>,
}

impl PrefixCode {
    fn new(counts: &[u64], max_length: u8) -> Self {
        let lengths = code_lengths(counts, max_length);
        let codewords = canonical_codewords(&lengths);
        let single = match counts.iter().filter(|&&c| c > 0).count() {
            0 => Some(0),
            1 => counts.iter().position(|&c| c > 0),
            _ => None,
        };
        PrefixCode {
            lengths,
            codewords,
            single,
        }
    }

    fn write(&self, w: &mut BitWriter, symbol: usize) {
        if self.single.is_none() {
            w.write(u32::from(self.lengths[symbol]), self.codewords[symbol]);
        }
    }

    /// The alphabet the code is sent for: up to the last symbol used.
    fn alphabet_size(&self) -> usize {
        match self.single {
            Some(s) => s + 1,
            None => self.lengths.iter().rposition(|&l| l > 0).unwrap() + 1,
        }
    }

    /// The code's table, in the Brotli format: nothing for a one-symbol
    /// alphabet, the "simple" form for up to four symbols, else the code
    /// lengths, themselves prefix coded.
    fn write_table(&self, w: &mut BitWriter) {
        let alphabet = self.alphabet_size();
        if alphabet == 1 {
            return;
        }
        let symbol_bits = usize::BITS - (alphabet - 1).leading_zeros();
        if let Some(symbol) = self.single {
            w.write(2, 1);
            w.write(2, 0);
            w.write(symbol_bits, symbol as u32);
            return;
        }
        let mut used: Vec<usize> = (0..alphabet).filter(|&s| self.lengths[s] > 0).collect();
        if used.len() <= 4 {
            // The decoder infers the lengths from the count and the order:
            // 2 symbols 1,1; 3 symbols 1,2,2 (the first sent is the short
            // one); 4 symbols 2,2,2,2 or, flagged, 1,2,3,3 in the order sent.
            used.sort_by_key(|&s| (self.lengths[s], s));
            w.write(2, 1);
            w.write(2, used.len() as u32 - 1);
            for &s in &used {
                w.write(symbol_bits, s as u32);
            }
            if used.len() == 4 {
                w.bit(self.lengths[used[0]] == 1);
            }
            return;
        }

        // The code-length code, over lengths 0..=15 (no repeat codes).
        let mut length_counts = [0u64; 18];
        for &l in &self.lengths[..alphabet] {
            length_counts[usize::from(l)] += 1;
        }
        let mut length_code = PrefixCode::new(&length_counts, MAX_CODE_LENGTH_CODE_LENGTH);
        if let Some(only) = length_code.single {
            // One length for every symbol: its code has no bits, but is sent
            // with a length that is not zero.
            length_code.lengths[only] = 1;
        }

        w.write(2, 0); // no lengths skipped
        let mut space = 32i32;
        for &i in &CODE_LENGTH_CODE_ORDER {
            if space <= 0 {
                break;
            }
            let length = length_code.lengths[i];
            let (n, codeword) = CODE_LENGTH_CODE_LENGTH_CODE[usize::from(length)];
            w.write(n, codeword);
            if length != 0 {
                space -= 32 >> length;
            }
        }
        for &l in &self.lengths[..alphabet] {
            length_code.write(w, usize::from(l));
        }
    }
}

/// Length-limited Huffman code lengths. A single used symbol gets length 0
/// (a code with no bits); unused symbols get 0 too.
fn code_lengths(counts: &[u64], max_length: u8) -> Vec<u8> {
    let used = counts.iter().filter(|&&c| c > 0).count();
    let mut lengths = vec![0u8; counts.len()];
    if used <= 1 {
        return lengths;
    }
    let mut counts = counts.to_vec();
    loop {
        // Huffman's algorithm over the used symbols: nodes 0..n are the
        // leaves, the rest are merges, each recording its parent.
        let leaves: Vec<usize> = (0..counts.len()).filter(|&s| counts[s] > 0).collect();
        let mut parent = vec![usize::MAX; 2 * leaves.len() - 1];
        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = leaves
            .iter()
            .enumerate()
            .map(|(node, &s)| Reverse((counts[s], node)))
            .collect();
        let mut next = leaves.len();
        while heap.len() > 1 {
            let Reverse((a, na)) = heap.pop().unwrap();
            let Reverse((b, nb)) = heap.pop().unwrap();
            parent[na] = next;
            parent[nb] = next;
            heap.push(Reverse((a + b, next)));
            next += 1;
        }
        let mut depth = vec![0u8; parent.len()];
        for node in (0..parent.len() - 1).rev() {
            depth[node] = depth[parent[node]] + 1;
        }
        if depth[..leaves.len()].iter().all(|&d| d <= max_length) {
            for (node, &s) in leaves.iter().enumerate() {
                lengths[s] = depth[node];
            }
            return lengths;
        }
        // Too deep: flatten the distribution and try again.
        for c in counts.iter_mut().filter(|c| **c > 0) {
            *c = c.div_ceil(2);
        }
    }
}

/// Canonical codewords for `lengths` (shorter first, then by symbol),
/// bit-reversed to be written least significant bit first.
fn canonical_codewords(lengths: &[u8]) -> Vec<u32> {
    let mut codewords = vec![0u32; lengths.len()];
    let mut count = [0u32; MAX_CODE_LENGTH as usize + 1];
    for &l in lengths {
        count[usize::from(l)] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; MAX_CODE_LENGTH as usize + 2];
    let mut code = 0u32;
    for len in 1..=MAX_CODE_LENGTH as usize {
        code = (code + count[len - 1]) << 1;
        next[len] = code;
    }
    for (s, &l) in lengths.iter().enumerate() {
        if l > 0 {
            let c = next[usize::from(l)];
            next[usize::from(l)] += 1;
            codewords[s] = c.reverse_bits() >> (32 - u32::from(l));
        }
    }
    codewords
}

/// The codes for a set of contexts, one histogram each, and how they are
/// sent.
pub(super) struct EntropyCode {
    codes: Vec<PrefixCode>,
}

impl EntropyCode {
    pub(super) fn new(histograms: &Histograms) -> Self {
        let codes = histograms
            .counts
            .iter()
            .map(|counts| PrefixCode::new(counts, MAX_CODE_LENGTH))
            .collect();
        EntropyCode { codes }
    }

    /// The header the decoder reads before the tokens: no LZ77, the context
    /// map (each context its own histogram), prefix codes, the hybrid-uint
    /// configurations, the code tables.
    pub(super) fn write_header(&self, w: &mut BitWriter) {
        w.bit(false); // LZ77
        let contexts = self.codes.len();
        if contexts > 1 {
            write_identity_context_map(w, contexts);
        }
        w.bit(true); // prefix codes
        for _ in 0..contexts {
            w.write(4, 4); // split_exponent
            w.write(3, 2); // msb_in_token
            w.write(2, 0); // lsb_in_token
        }
        for code in &self.codes {
            w.varint16((code.alphabet_size() - 1) as u16);
        }
        for code in &self.codes {
            code.write_table(w);
        }
    }

    /// `value` in `context`.
    pub(super) fn write(&self, w: &mut BitWriter, context: usize, value: u32) {
        let token = tokenize(value);
        self.codes[context].write(w, token.symbol as usize);
        w.write(token.nbits, token.bits);
    }
}

/// The context map giving each context its own histogram: in the simple
/// form for up to 8, else itself entropy coded.
fn write_identity_context_map(w: &mut BitWriter, contexts: usize) {
    let bits = usize::BITS - (contexts - 1).leading_zeros();
    if bits <= 3 {
        w.bit(true); // simple
        w.write(2, bits);
        for c in 0..contexts {
            w.write(bits, c as u32);
        }
        return;
    }
    assert!(
        contexts <= 256,
        "{contexts} contexts, a context map holds 256"
    );
    w.bit(false); // not simple
    w.bit(false); // no move-to-front
    let mut histograms = Histograms::new(1);
    for c in 0..contexts {
        histograms.add(0, c as u32);
    }
    let code = EntropyCode::new(&histograms);
    code.write_header(w);
    for c in 0..contexts {
        code.write(w, 0, c as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder's hybrid-uint 4/2/0 read.
    fn detokenize(t: Token) -> u32 {
        if t.symbol < 16 {
            return t.symbol;
        }
        let nbits = 2 + ((t.symbol - 16) >> 2);
        assert_eq!(nbits, t.nbits);
        let hi = (t.symbol & 3) | 4;
        (hi << nbits) | t.bits
    }

    #[test]
    fn tokens_round_trip() {
        for v in (0..5000).chain([65535, 65536, 1 << 20, u32::MAX >> 1, u32::MAX]) {
            let t = tokenize(v);
            assert!((t.symbol as usize) < MAX_TOKENS, "{v}");
            assert_eq!(detokenize(t), v, "{v}");
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

    #[test]
    fn lengths_are_limited_and_complete() {
        // Fibonacci counts make the deepest unlimited Huffman tree.
        let mut counts = vec![0u64; 40];
        let (mut a, mut b) = (1u64, 1u64);
        for c in counts.iter_mut() {
            *c = a;
            (a, b) = (b, a + b);
        }
        let lengths = code_lengths(&counts, 15);
        assert!(lengths.iter().all(|&l| (1..=15).contains(&l)));
        let kraft: f64 = lengths.iter().map(|&l| 0.5f64.powi(i32::from(l))).sum();
        assert!((kraft - 1.0).abs() < 1e-12);
    }
}
