//! Prefix codes: the Brotli-format Huffman tables JPEG XL allows in place of
//! ANS.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::encode::bits::BitWriter;

/// Prefix codes are at most this long.
pub(super) const MAX_CODE_LENGTH: u8 = 15;
/// The code-length code's codes are at most this long.
const MAX_CODE_LENGTH_CODE_LENGTH: u8 = 5;
/// The order the code-length code's lengths are sent in.
const CODE_LENGTH_CODE_ORDER: [usize; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// The static code the code-length code's lengths are sent with: for a
/// length 0..=5, its bit count and codeword (least significant bit first).
const CODE_LENGTH_CODE_LENGTH_CODE: [(u32, u32); 6] =
    [(2, 0), (4, 7), (3, 3), (2, 2), (2, 1), (4, 15)];
/// The code-length symbol repeating the previous non-zero length.
const REPEAT_PREVIOUS: usize = 16;
/// The code-length symbol repeating zero.
const REPEAT_ZERO: usize = 17;

/// A prefix code: per symbol, its length and its codeword, bit-reversed so
/// it can be written least significant bit first.
pub(super) struct PrefixCode {
    lengths: Vec<u8>,
    codewords: Vec<u32>,
    /// The symbol of a code with only one (or none: then 0), which takes
    /// no bits.
    single: Option<usize>,
}

impl PrefixCode {
    pub(super) fn new(counts: &[u64], max_length: u8) -> Self {
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

    pub(super) fn write(&self, w: &mut BitWriter, symbol: usize) {
        if self.single.is_none() {
            w.write(u32::from(self.lengths[symbol]), self.codewords[symbol]);
        }
    }

    /// Bits `symbol` takes.
    pub(super) fn length(&self, symbol: usize) -> u32 {
        if self.single.is_some() {
            0
        } else {
            u32::from(self.lengths[symbol])
        }
    }

    /// The alphabet the code is sent for: up to the last symbol used.
    pub(super) fn alphabet_size(&self) -> usize {
        match self.single {
            Some(s) => s + 1,
            None => self.lengths.iter().rposition(|&l| l > 0).unwrap() + 1,
        }
    }

    /// The code's table, in the Brotli format: nothing for a one-symbol
    /// alphabet, the "simple" form for up to four symbols, else the code
    /// lengths, themselves prefix coded.
    pub(super) fn write_table(&self, w: &mut BitWriter) {
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
        write_complex_table(w, &self.lengths[..alphabet]);
    }
}

/// The code-length symbols for `lengths`: each length, or a run of zeros
/// (17) or of the previous length (16) with its extra bits.
fn length_symbols(lengths: &[u8]) -> Vec<(usize, u32, u32)> {
    // Repeat codes extend a run by 3..=6 (16) or 3..=10 (17) at first, and
    // each further repeat code multiplies: libjxl/Brotli's run splitting,
    // here kept to single repeat codes per run chunk for simplicity.
    let mut out = Vec::new();
    let mut prev = 8u8; // the decoder's initial "previous" length
    let mut i = 0;
    while i < lengths.len() {
        let l = lengths[i];
        let run = lengths[i..].iter().take_while(|&&x| x == l).count();
        if l == 0 && run >= 3 {
            // One 17 per up-to-10 zeros, never two in a row (that would
            // multiply).
            let mut left = run;
            while left >= 3 {
                let n = left.min(10);
                out.push((REPEAT_ZERO, 3, (n - 3) as u32));
                left -= n;
                if left >= 3 {
                    // Break the multiplying chain with a literal zero.
                    out.push((0, 0, 0));
                    left -= 1;
                }
            }
            for _ in 0..left {
                out.push((0, 0, 0));
            }
            i += run;
            continue;
        }
        if l != 0 && l == prev && run >= 3 {
            let mut left = run;
            while left >= 3 {
                let n = left.min(6);
                out.push((REPEAT_PREVIOUS, 2, (n - 3) as u32));
                left -= n;
                if left >= 3 {
                    out.push((usize::from(l), 0, 0));
                    left -= 1;
                }
            }
            for _ in 0..left {
                out.push((usize::from(l), 0, 0));
            }
            i += run;
            continue;
        }
        out.push((usize::from(l), 0, 0));
        if l != 0 {
            prev = l;
        }
        i += 1;
    }
    out
}

fn write_complex_table(w: &mut BitWriter, lengths: &[u8]) {
    let symbols = length_symbols(lengths);
    let mut length_counts = [0u64; 18];
    for &(s, _, _) in &symbols {
        length_counts[s] += 1;
    }
    let mut length_code = PrefixCode::new(&length_counts, MAX_CODE_LENGTH_CODE_LENGTH);
    if let Some(only) = length_code.single {
        // One code-length symbol for everything: its code has no bits, but
        // is sent with a length that is not zero.
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
    for &(s, nbits, extra) in &symbols {
        length_code.write(w, s);
        w.write(nbits, extra);
    }
}

/// Length-limited Huffman code lengths. A single used symbol gets length 0
/// (a code with no bits); unused symbols get 0 too.
pub(super) fn code_lengths(counts: &[u64], max_length: u8) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The decoder's expansion of code-length symbols.
    fn expand(symbols: &[(usize, u32, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut prev = 8u8;
        let mut repeat = 0usize;
        let mut repeat_len = 0u8;
        for &(s, _, extra) in symbols {
            if s < 16 {
                repeat = 0;
                out.push(s as u8);
                if s != 0 {
                    prev = s as u8;
                }
                continue;
            }
            let bits = if s == 16 { 2 } else { 3 };
            let new_len = if s == 16 { prev } else { 0 };
            if repeat_len != new_len {
                repeat = 0;
                repeat_len = new_len;
            }
            let old = repeat;
            if repeat > 0 {
                repeat -= 2;
                repeat <<= bits;
            }
            repeat += extra as usize + 3;
            for _ in old..repeat {
                out.push(repeat_len);
            }
        }
        out
    }

    #[test]
    fn runs_expand_back() {
        let mut lengths = vec![0u8; 40];
        lengths.extend([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0, 0, 0, 5]);
        lengths.extend(vec![7; 25]);
        lengths.extend(vec![0; 23]);
        lengths.push(1);
        assert_eq!(expand(&length_symbols(&lengths)), lengths);
    }
}
