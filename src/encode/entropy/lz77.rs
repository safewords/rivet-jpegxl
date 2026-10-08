//! LZ77 over a stream's values: a copy of earlier values is coded as a
//! length (in the context the copy starts in) and a distance (in its own
//! context), the distance either plain or, for a picture, one of 120 short
//! 2-D offsets in units of the row.

use super::Token;

/// The decoder's window.
const WINDOW: usize = 1 << 20;
/// The decoder's 2-D distances: (dx, dy) for a distance of `dy * width + dx`.
#[rustfmt::skip]
const SPECIAL_DISTANCES: [(i8, u8); 120] = [
    ( 0, 1), ( 1, 0), ( 1, 1), (-1, 1), ( 0, 2), ( 2, 0), ( 1, 2), (-1, 2), ( 2, 1), (-2, 1),
    ( 2, 2), (-2, 2), ( 0, 3), ( 3, 0), ( 1, 3), (-1, 3), ( 3, 1), (-3, 1), ( 2, 3), (-2, 3),
    ( 3, 2), (-3, 2), ( 0, 4), ( 4, 0), ( 1, 4), (-1, 4), ( 4, 1), (-4, 1), ( 3, 3), (-3, 3),
    ( 2, 4), (-2, 4), ( 4, 2), (-4, 2), ( 0, 5), ( 3, 4), (-3, 4), ( 4, 3), (-4, 3), ( 5, 0),
    ( 1, 5), (-1, 5), ( 5, 1), (-5, 1), ( 2, 5), (-2, 5), ( 5, 2), (-5, 2), ( 4, 4), (-4, 4),
    ( 3, 5), (-3, 5), ( 5, 3), (-5, 3), ( 0, 6), ( 6, 0), ( 1, 6), (-1, 6), ( 6, 1), (-6, 1),
    ( 2, 6), (-2, 6), ( 6, 2), (-6, 2), ( 4, 5), (-4, 5), ( 5, 4), (-5, 4), ( 3, 6), (-3, 6),
    ( 6, 3), (-6, 3), ( 0, 7), ( 7, 0), ( 1, 7), (-1, 7), ( 5, 5), (-5, 5), ( 7, 1), (-7, 1),
    ( 4, 6), (-4, 6), ( 6, 4), (-6, 4), ( 2, 7), (-2, 7), ( 7, 2), (-7, 2), ( 3, 7), (-3, 7),
    ( 7, 3), (-7, 3), ( 5, 6), (-5, 6), ( 6, 5), (-6, 5), ( 8, 0), ( 4, 7), (-4, 7), ( 7, 4),
    (-7, 4), ( 8, 1), ( 8, 2), ( 6, 6), (-6, 6), ( 8, 3), ( 5, 7), (-5, 7), ( 7, 5), (-7, 5),
    ( 8, 4), ( 6, 7), (-6, 7), ( 7, 6), (-7, 6), ( 8, 5), ( 7, 7), (-7, 7), ( 8, 6), ( 8, 7),
];

/// How a stream's values are searched for copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Lz77Mode {
    /// None.
    #[default]
    Off,
    /// Runs only: copies of the value just before.
    Rle,
    /// Any copy in the window, found through a hash chain.
    Full,
}

/// The distance symbol for `distance` (at least 1), with `width` the row
/// length for 2-D distances (0: none).
pub(super) fn distance_symbol(distance: usize, width: u32) -> u32 {
    if width == 0 {
        return distance as u32 - 1;
    }
    let w = width as i64;
    for (i, &(dx, dy)) in SPECIAL_DISTANCES.iter().enumerate() {
        if i64::from(dy) * w + i64::from(dx) == distance as i64 {
            return i as u32;
        }
    }
    distance as u32 - 1 + 120
}

/// The stream with copies found: values replaced by `(length token,
/// distance token)` pairs where that is likely cheaper. `min_length` is the
/// shortest copy; `distance_context` the copies' distance context.
pub(super) fn apply(
    tokens: &[Token],
    mode: Lz77Mode,
    width: u32,
    min_length: u32,
    distance_context: u32,
) -> Vec<Token> {
    let n = tokens.len();
    let mut out = Vec::with_capacity(n);
    let min_len = min_length as usize;
    let value = |i: usize| tokens[i].value;

    const HASH_BITS: u32 = 16;
    const CHAIN: usize = 32;
    let hash = |i: usize| -> usize {
        let mut h = 0u64;
        for k in 0..min_len.min(4) {
            h = h
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(u64::from(value(i + k)) + 1);
        }
        (h >> (64 - HASH_BITS)) as usize
    };
    let mut head = if mode == Lz77Mode::Full {
        vec![usize::MAX; 1 << HASH_BITS]
    } else {
        Vec::new()
    };
    let mut prev = if mode == Lz77Mode::Full {
        vec![usize::MAX; n]
    } else {
        Vec::new()
    };
    let insert = |i: usize, head: &mut Vec<usize>, prev: &mut Vec<usize>| {
        if mode == Lz77Mode::Full && i + min_len <= n {
            let h = hash(i);
            prev[i] = head[h];
            head[h] = i;
        }
    };

    let mut i = 0;
    while i < n {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i > 0 && i + min_len <= n {
            // A run of the previous value.
            let run = (i..n).take_while(|&j| value(j) == value(i - 1)).count();
            if run >= min_len {
                best_len = run;
                best_dist = 1;
            }
            if mode == Lz77Mode::Full {
                let mut candidate = head[hash(i)];
                let mut steps = 0;
                while candidate != usize::MAX && steps < CHAIN && i - candidate <= WINDOW {
                    let len = (0..n - i)
                        .take_while(|&k| value(candidate + k) == value(i + k))
                        .count();
                    if len > best_len + 1 || (len > best_len && i - candidate < best_dist) {
                        best_len = len;
                        best_dist = i - candidate;
                    }
                    candidate = prev[candidate];
                    steps += 1;
                }
            }
        }
        if best_len >= min_len {
            out.push(Token {
                context: tokens[i].context,
                value: (best_len - min_len) as u32,
                lz77_length: true,
            });
            out.push(Token {
                context: distance_context,
                value: distance_symbol(best_dist, width),
                lz77_length: false,
            });
            for k in 0..best_len {
                insert(i + k, &mut head, &mut prev);
            }
            i += best_len;
        } else {
            out.push(tokens[i]);
            insert(i, &mut head, &mut prev);
            i += 1;
        }
    }
    out
}

/// The values a stream with copies decodes to (for tests).
#[cfg(test)]
pub(super) fn expand(tokens: &[Token], width: u32, min_length: u32) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        if t.lz77_length {
            let len = (t.value + min_length) as usize;
            let sym = tokens[i + 1].value;
            let dist_sub_1 = if width == 0 {
                sym
            } else if sym >= 120 {
                sym - 120
            } else {
                let (dx, dy) = SPECIAL_DISTANCES[sym as usize];
                (width as i64 * i64::from(dy) + i64::from(dx) - 1).max(0) as u32
            };
            let distance = ((dist_sub_1 as usize).min(WINDOW - 1) + 1).min(out.len());
            let start = out.len() - distance;
            for k in 0..len {
                out.push(out[start + k]);
            }
            i += 2;
        } else {
            out.push(t.value);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_expand_back() {
        let mut values: Vec<u32> = (0..200).map(|i| (i * 7 % 13) as u32).collect();
        values.extend(std::iter::repeat_n(5, 50));
        values.extend_from_slice(&values.clone()[..120]);
        values.extend((0..64).map(|i| (i % 8) as u32));
        let tokens: Vec<Token> = values
            .iter()
            .map(|&v| Token {
                context: 0,
                value: v,
                lz77_length: false,
            })
            .collect();
        for (mode, width) in [(Lz77Mode::Rle, 0), (Lz77Mode::Full, 0), (Lz77Mode::Full, 8)] {
            let coded = apply(&tokens, mode, width, 3, 1);
            assert!(coded.len() < tokens.len());
            assert_eq!(expand(&coded, width, 3), values, "{mode:?} {width}");
        }
    }
}
