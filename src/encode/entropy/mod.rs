//! Entropy coding: the codestream's one scheme for every stream of symbols
//! (the MA tree, modular residuals, VarDCT coefficients, the ICC profile,
//! the context maps themselves). Values in contexts; contexts mapped to
//! clustered histograms; each value split into a token and raw bits; tokens
//! coded with ANS or prefix codes; optionally LZ77 copies.

mod ans;
mod cluster;
mod hybrid;
mod lz77;
mod prefix;

use crate::encode::bits::{BitWriter, Dist};
use ans::{AnsTable, FINAL_STATE};
pub(crate) use hybrid::{HybridUint, pack_signed};
pub use lz77::Lz77Mode;
use prefix::PrefixCode;

/// A value in a context, or an LZ77 copy's length (its distance follows as
/// the next token, in the distance context).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Token {
    pub context: u32,
    pub value: u32,
    pub lz77_length: bool,
}

impl Token {
    #[inline]
    pub(crate) fn new(context: u32, value: u32) -> Self {
        Token {
            context,
            value,
            lz77_length: false,
        }
    }

    #[inline]
    pub(crate) fn signed(context: u32, value: i32) -> Self {
        Token::new(context, pack_signed(value))
    }
}

/// How the entropy coder is driven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyOptions {
    /// ANS (smaller) rather than prefix codes (each symbol a whole number
    /// of bits).
    pub ans: bool,
    /// LZ77 copies, kept only where they make the stream smaller.
    pub lz77: Lz77Mode,
    /// Let contexts share histograms.
    pub clustering: bool,
    /// The most histograms a set of contexts gets (1..=256).
    pub max_histograms: usize,
    /// Choose each histogram's hybrid-uint split rather than 4/2/0.
    pub optimize_uint: bool,
}

impl Default for EntropyOptions {
    fn default() -> Self {
        EntropyOptions {
            ans: true,
            lz77: Lz77Mode::Full,
            clustering: true,
            max_histograms: 256,
            optimize_uint: true,
        }
    }
}

/// A stream's tokens, and the row length for its LZ77 2-D distances (0:
/// none; a modular stream's widest channel).
pub(crate) struct Stream {
    pub tokens: Vec<Token>,
    pub width: u32,
}

impl Stream {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Stream { tokens, width: 0 }
    }
}

/// The LZ77 header fields.
#[derive(Clone, Copy, Debug)]
struct Lz77 {
    min_symbol: u32,
    min_length: u32,
    length_uint: HybridUint,
}

const MIN_SYMBOL: u32 = 224;
const MIN_LENGTH: u32 = 3;
const LENGTH_UINT: HybridUint = HybridUint::new(0, 0, 0);
const DEFAULT_UINT: HybridUint = HybridUint::new(4, 2, 0);

enum Codes {
    Prefix(Vec<PrefixCode>),
    Ans(Vec<AnsTable>, Vec<BitWriter>),
}

/// The entropy code for a set of streams sharing contexts: what the header
/// sends, and how each token is written.
pub(crate) struct EntropyCode {
    context_map: Vec<u8>,
    lz77: Option<Lz77>,
    uint: Vec<HybridUint>,
    log_alpha_size: u32,
    codes: Codes,
    /// The context map, as written (it may itself be entropy coded).
    map_header: BitWriter,
}

impl EntropyCode {
    /// The code for `streams`, over `num_contexts` contexts, and the
    /// streams' tokens as they are to be written (with any LZ77 copies).
    pub(crate) fn build(
        num_contexts: usize,
        streams: Vec<Stream>,
        options: &EntropyOptions,
        allow_lz77: bool,
    ) -> (Self, Vec<Vec<Token>>) {
        let plain: Vec<Vec<Token>> = streams.iter().map(|s| s.tokens.clone()).collect();
        let plain_code = Self::for_tokens(num_contexts, &plain, options, None);
        if !allow_lz77 || options.lz77 == Lz77Mode::Off {
            return (plain_code, plain);
        }
        let lz77 = Lz77 {
            min_symbol: MIN_SYMBOL,
            min_length: MIN_LENGTH,
            length_uint: LENGTH_UINT,
        };
        let copied: Vec<Vec<Token>> = streams
            .iter()
            .map(|s| {
                lz77::apply(
                    &s.tokens,
                    options.lz77,
                    s.width,
                    lz77.min_length,
                    num_contexts as u32,
                )
            })
            .collect();
        if copied.iter().zip(&plain).all(|(c, p)| c.len() == p.len()) {
            return (plain_code, plain);
        }
        let copied_code = Self::for_tokens(num_contexts + 1, &copied, options, Some(lz77));
        if copied_code.cost(&copied) < plain_code.cost(&plain) {
            (copied_code, copied)
        } else {
            (plain_code, plain)
        }
    }

    fn for_tokens(
        num_contexts: usize,
        streams: &[Vec<Token>],
        options: &EntropyOptions,
        lz77: Option<Lz77>,
    ) -> Self {
        let num_contexts = num_contexts.max(1);
        let symbol_of = |t: &Token, uint: &HybridUint| -> u32 {
            match lz77 {
                Some(l) if t.lz77_length => l.min_symbol + l.length_uint.split(t.value).token,
                _ => uint.split(t.value).token,
            }
        };

        // Contexts to clusters, by their default-split histograms.
        let max_symbol = 288usize;
        let mut histograms = vec![vec![0u64; max_symbol]; num_contexts];
        for t in streams.iter().flatten() {
            histograms[t.context as usize][symbol_of(t, &DEFAULT_UINT) as usize] += 1;
        }
        let context_map = if options.clustering || num_contexts > cluster::MAX_CLUSTERS {
            let cap = if options.clustering {
                options.max_histograms
            } else {
                cluster::MAX_CLUSTERS
            };
            cluster::cluster(&histograms, cap)
        } else {
            (0..num_contexts).map(|c| c as u8).collect()
        };
        let num_clusters = usize::from(*context_map.iter().max().unwrap()) + 1;

        // Each cluster's split.
        let mut uint = vec![DEFAULT_UINT; num_clusters];
        if options.optimize_uint {
            // Per cluster, the values coded with its split (not copies).
            const SMALL: usize = 1 << 12;
            let mut small = vec![vec![0u64; SMALL]; num_clusters];
            let mut large: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
            for t in streams.iter().flatten() {
                if t.lz77_length {
                    continue;
                }
                let k = usize::from(context_map[t.context as usize]);
                if (t.value as usize) < SMALL {
                    small[k][t.value as usize] += 1;
                } else {
                    large[k].push(t.value);
                }
            }
            for k in 0..num_clusters {
                let mut best = (f64::INFINITY, DEFAULT_UINT);
                for candidate in HybridUint::CANDIDATES {
                    let mut counts = vec![0u64; max_symbol];
                    let mut raw = 0u64;
                    for (v, &n) in small[k].iter().enumerate() {
                        if n > 0 {
                            let s = candidate.split(v as u32);
                            counts[s.token as usize] += n;
                            raw += n * u64::from(s.nbits);
                        }
                    }
                    for &v in &large[k] {
                        let s = candidate.split(v);
                        counts[s.token as usize] += 1;
                        raw += u64::from(s.nbits);
                    }
                    let bits = cluster::cost(&counts) + raw as f64;
                    if bits < best.0 {
                        best = (bits, candidate);
                    }
                }
                uint[k] = best.1;
            }
        }

        // The final token counts per cluster.
        let count = |uint: &[HybridUint]| {
            let mut counts = vec![vec![0u64; max_symbol]; num_clusters];
            for t in streams.iter().flatten() {
                let k = usize::from(context_map[t.context as usize]);
                counts[k][symbol_of(t, &uint[k]) as usize] += 1;
            }
            let alphabet = counts
                .iter()
                .filter_map(|c| c.iter().rposition(|&n| n > 0))
                .max()
                .map_or(1, |p| p + 1);
            (counts, alphabet)
        };
        let (mut counts, mut alphabet) = count(&uint);
        let log_alpha_of = |alphabet: usize| {
            if options.ans {
                hybrid::ceil_log2(alphabet as u32).clamp(5, 8)
            } else {
                15
            }
        };
        // A split must be codable against the header's alphabet size: one
        // equal to it carries no in-token bits, one above it is not read.
        loop {
            let log_alpha_size = log_alpha_of(alphabet);
            let mut changed = false;
            for u in &mut uint {
                let field_bits = hybrid::ceil_log2(log_alpha_size + 1);
                let unreadable = u.split_exponent >= 1 << field_bits;
                let implied = u.split_exponent == log_alpha_size
                    && (u.msb_in_token != 0 || u.lsb_in_token != 0);
                if unreadable || implied {
                    *u = DEFAULT_UINT;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            (counts, alphabet) = count(&uint);
        }
        assert!(
            !options.ans || alphabet <= 256,
            "{alphabet} symbols for ANS"
        );

        let (log_alpha_size, codes) = if options.ans {
            let log_alpha_size = log_alpha_of(alphabet);
            let size = 1usize << log_alpha_size;
            let (tables, headers) = counts
                .iter()
                .map(|c| AnsTable::new(&c[..size.min(c.len())], log_alpha_size))
                .unzip();
            (log_alpha_size, Codes::Ans(tables, headers))
        } else {
            let codes = counts
                .iter()
                .map(|c| PrefixCode::new(&c[..alphabet], prefix::MAX_CODE_LENGTH))
                .collect();
            (15, Codes::Prefix(codes))
        };
        let map_header = if num_contexts > 1 {
            context_map_header(&context_map, options)
        } else {
            BitWriter::new()
        };
        EntropyCode {
            context_map,
            lz77,
            uint,
            log_alpha_size,
            codes,
            map_header,
        }
    }

    fn split(&self, t: &Token) -> (usize, hybrid::Split) {
        let k = usize::from(self.context_map[t.context as usize]);
        match self.lz77 {
            Some(l) if t.lz77_length => {
                let mut s = l.length_uint.split(t.value);
                s.token += l.min_symbol;
                (k, s)
            }
            _ => (k, self.uint[k].split(t.value)),
        }
    }

    /// An estimate of the bits the header and `streams` take.
    pub(crate) fn cost(&self, streams: &[Vec<Token>]) -> f64 {
        let mut header = BitWriter::new();
        self.write_header(&mut header);
        let mut bits = header.bits_written() as f64;
        for t in streams.iter().flatten() {
            let (k, s) = self.split(t);
            bits += f64::from(s.nbits);
            bits += match &self.codes {
                Codes::Prefix(codes) => f64::from(codes[k].length(s.token as usize)),
                Codes::Ans(tables, _) => {
                    (4096.0 / f64::from(tables[k].probability(s.token as usize))).log2()
                }
            };
        }
        bits
    }

    /// The header: LZ77 parameters, the context map, the code type, each
    /// histogram's split, the histograms.
    pub(crate) fn write_header(&self, w: &mut BitWriter) {
        match self.lz77 {
            None => w.bit(false),
            Some(l) => {
                w.bit(true);
                w.u32(
                    l.min_symbol,
                    [
                        Dist::Val(224),
                        Dist::Val(512),
                        Dist::Val(4096),
                        Dist::Bits(15, 8),
                    ],
                );
                w.u32(
                    l.min_length,
                    [
                        Dist::Val(3),
                        Dist::Val(4),
                        Dist::Bits(2, 5),
                        Dist::Bits(8, 9),
                    ],
                );
                l.length_uint.write(w, 8);
            }
        }
        w.append_bits(&self.map_header);
        match &self.codes {
            Codes::Prefix(_) => w.bit(true),
            Codes::Ans(..) => {
                w.bit(false);
                w.write(2, self.log_alpha_size - 5);
            }
        }
        for u in &self.uint {
            u.write(w, self.log_alpha_size);
        }
        match &self.codes {
            Codes::Prefix(codes) => {
                for code in codes {
                    w.varint16((code.alphabet_size() - 1) as u16);
                }
                for code in codes {
                    code.write_table(w);
                }
            }
            Codes::Ans(_, headers) => {
                for h in headers {
                    w.append_bits(h);
                }
            }
        }
    }

    /// A stream's tokens (one symbol reader's worth: ANS state included).
    pub(crate) fn write_tokens(&self, w: &mut BitWriter, tokens: &[Token]) {
        match &self.codes {
            Codes::Prefix(codes) => {
                for t in tokens {
                    let (k, s) = self.split(t);
                    codes[k].write(w, s.token as usize);
                    w.write(s.nbits, s.bits);
                }
            }
            Codes::Ans(tables, _) => {
                let splits: Vec<(usize, hybrid::Split)> =
                    tokens.iter().map(|t| self.split(t)).collect();
                let mut state = FINAL_STATE;
                let mut refills = vec![None; splits.len()];
                for (i, (k, s)) in splits.iter().enumerate().rev() {
                    refills[i] = tables[*k].encode(&mut state, s.token as usize);
                }
                w.write(32, state);
                for ((_, s), refill) in splits.iter().zip(refills) {
                    if let Some(bits) = refill {
                        w.write(16, u32::from(bits));
                    }
                    w.write(s.nbits, s.bits);
                }
            }
        }
    }

    /// Header and one stream, for codes with a single stream.
    pub(crate) fn write_all(&self, w: &mut BitWriter, tokens: &[Token]) {
        self.write_header(w);
        self.write_tokens(w, tokens);
    }
}

/// The context map, in the cheapest of: the simple form (entries of at
/// most 3 bits), or entropy coded, with or without move-to-front.
fn context_map_header(map: &[u8], options: &EntropyOptions) -> BitWriter {
    let clusters = u32::from(*map.iter().max().unwrap()) + 1;
    let mut candidates = Vec::new();
    let bits = hybrid::ceil_log2(clusters);
    if bits <= 3 {
        let mut w = BitWriter::new();
        w.bit(true);
        w.write(2, bits);
        for &k in map {
            w.write(bits, u32::from(k));
        }
        candidates.push(w);
    }
    if bits > 0 {
        for mtf in [false, true] {
            let values: Vec<u8> = if mtf {
                move_to_front(map)
            } else {
                map.to_vec()
            };
            let tokens: Vec<Token> = values
                .iter()
                .map(|&v| Token::new(0, u32::from(v)))
                .collect();
            let mut w = BitWriter::new();
            w.bit(false);
            w.bit(mtf);
            let (code, streams) =
                EntropyCode::build(1, vec![Stream::new(tokens)], options, map.len() > 2);
            code.write_all(&mut w, &streams[0]);
            candidates.push(w);
        }
    }
    candidates
        .into_iter()
        .min_by_key(|w| w.bits_written())
        .unwrap()
}

fn move_to_front(values: &[u8]) -> Vec<u8> {
    let mut order: Vec<u8> = (0..=255).collect();
    values
        .iter()
        .map(|&v| {
            let index = order.iter().position(|&o| o == v).unwrap();
            if index != 0 {
                order.remove(index);
                order.insert(0, v);
            }
            index as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_to_front_inverts() {
        let values = [3u8, 3, 0, 7, 3, 255, 0, 0, 7];
        let coded = move_to_front(&values);
        // The decoder's inverse.
        let mut order: Vec<u8> = (0..=255).collect();
        let decoded: Vec<u8> = coded
            .iter()
            .map(|&i| {
                let v = order[usize::from(i)];
                if i != 0 {
                    order.remove(usize::from(i));
                    order.insert(0, v);
                }
                v
            })
            .collect();
        assert_eq!(decoded, values);
    }
}
