//! rANS: a histogram's distribution, normalised to 4096, how it is sent,
//! and the alias table the decoder builds from it — which the encoder needs
//! to map each symbol's slots to the decoder's state positions.

use crate::encode::bits::BitWriter;

const LOG_SUM_PROBS: u32 = 12;
const SUM_PROBS: u32 = 1 << LOG_SUM_PROBS;
/// The code marking a run of repeated probabilities.
const RLE_MARKER: u32 = LOG_SUM_PROBS + 1;
/// The state the decoder must end in.
pub(super) const FINAL_STATE: u32 = 0x130000;

/// The prefix code the decoder reads the probabilities' bit lengths with:
/// for a code 0..=13, its bit count and codeword (least significant first).
const LOG_COUNT_CODE: [(u32, u32); 14] = [
    (5, 17),
    (4, 11),
    (4, 15),
    (4, 3),
    (4, 9),
    (4, 7),
    (3, 4),
    (3, 2),
    (3, 5),
    (3, 6),
    (3, 0),
    (6, 33),
    (7, 1),
    (7, 65),
];

/// A histogram as ANS codes it.
pub(super) struct AnsTable {
    /// Each symbol's probability, out of 4096.
    dist: Vec<u32>,
    /// For each symbol, its slots' positions in the decoder's state space.
    slots: Vec<Vec<u16>>,
}

impl AnsTable {
    /// The table for `counts` (symbols past `alphabet` unused), with
    /// `log_alpha_size` the header's table size.
    pub(super) fn new(counts: &[u64], log_alpha_size: u32) -> (Self, BitWriter) {
        let table_size = 1usize << log_alpha_size;
        assert!(counts.len() <= table_size);
        let mut header = BitWriter::new();
        let dist = write_distribution(&mut header, counts, table_size);
        let slots = alias_slots(&dist, log_alpha_size);
        (AnsTable { dist, slots }, header)
    }

    pub(super) fn probability(&self, symbol: usize) -> u32 {
        self.dist.get(symbol).copied().unwrap_or(0)
    }

    /// One step of the encoder, going backwards: from the state after
    /// `symbol` to the one before it, and the 16 bits the decoder will read
    /// back in between, if any.
    #[inline]
    pub(super) fn encode(&self, state: &mut u32, symbol: usize) -> Option<u16> {
        let freq = self.dist[symbol];
        debug_assert!(freq > 0, "symbol {symbol} has no probability");
        let mut out = None;
        if (*state >> (32 - LOG_SUM_PROBS)) >= freq {
            out = Some(*state as u16);
            *state >>= 16;
        }
        let slot = self.slots[symbol][(*state % freq) as usize];
        *state = ((*state / freq) << LOG_SUM_PROBS) + u32::from(slot);
        out
    }
}

/// The decoder's `read_u8`: 0, or a bit length and the bits under it.
fn write_u8(w: &mut BitWriter, value: u32) {
    debug_assert!(value < 256);
    if value == 0 {
        w.bit(false);
        return;
    }
    w.bit(true);
    let n = 31 - value.leading_zeros();
    w.write(3, n);
    w.write(n, value - (1 << n));
}

/// Normalise `counts` to a distribution summing to 4096, every used symbol
/// at least 1, and write it in the cheapest of the header's forms. Returns
/// the distribution as the decoder will see it.
fn write_distribution(w: &mut BitWriter, counts: &[u64], table_size: usize) -> Vec<u32> {
    let used: Vec<usize> = (0..counts.len()).filter(|&s| counts[s] > 0).collect();
    let mut dist = vec![0u32; table_size];
    match used.len() {
        0 | 1 => {
            let s = used.first().copied().unwrap_or(0);
            w.bit(true);
            w.bit(false);
            write_u8(w, s as u32);
            dist[s] = SUM_PROBS;
            return dist;
        }
        2 => {
            let total = counts[used[0]] + counts[used[1]];
            let p0 = ((counts[used[0]] * u64::from(SUM_PROBS) + total / 2) / total)
                .clamp(1, u64::from(SUM_PROBS) - 1) as u32;
            w.bit(true);
            w.bit(true);
            write_u8(w, used[0] as u32);
            write_u8(w, used[1] as u32);
            w.write(LOG_SUM_PROBS, p0);
            dist[used[0]] = p0;
            dist[used[1]] = SUM_PROBS - p0;
            return dist;
        }
        _ => {}
    }

    // Evenly distributed, when every symbol up to the last is used equally.
    let alphabet = used.last().unwrap() + 1;
    if alphabet <= 256
        && used.len() == alphabet
        && counts[..alphabet].iter().all(|&c| c == counts[0])
    {
        w.bit(false);
        w.bit(true);
        write_u8(w, alphabet as u32 - 1);
        let base = SUM_PROBS / alphabet as u32;
        let remainder = (SUM_PROBS % alphabet as u32) as usize;
        for (s, d) in dist[..alphabet].iter_mut().enumerate() {
            *d = base + u32::from(s < remainder);
        }
        return dist;
    }

    // The general form: each probability's bit length, then its bits under
    // the leading one to a precision `shift` sets; one symbol — the first
    // with the longest — is left out and gets what remains. Try every
    // precision and keep the smallest header and data.
    let mut best: Option<(f64, BitWriter, Vec<u32>)> = None;
    for shift in 0..=LOG_SUM_PROBS + 1 {
        let Some(normalized) = normalize(counts, alphabet, shift) else {
            continue;
        };
        let mut header = BitWriter::new();
        header.bit(false);
        header.bit(false);
        write_complex(&mut header, &normalized, alphabet, shift);
        let data_bits: f64 = (0..alphabet)
            .filter(|&s| counts[s] > 0)
            .map(|s| counts[s] as f64 * (f64::from(SUM_PROBS) / f64::from(normalized[s])).log2())
            .sum();
        let cost = data_bits + header.bits_written() as f64;
        if best.as_ref().is_none_or(|(c, _, _)| cost < *c) {
            best = Some((cost, header, normalized));
        }
    }
    let (_, header, normalized) = best.expect("no precision fits the histogram");
    w.append_bits(&header);
    dist[..alphabet].copy_from_slice(&normalized[..alphabet]);
    dist
}

/// The code for a probability: 0 for none, else its bit length plus one.
fn log_code(p: u32) -> u32 {
    if p == 0 { 0 } else { 32 - p.leading_zeros() }
}

/// How many bits under the leading one a probability of bit length
/// `zeros + 1` keeps at precision `shift`.
fn kept_bits(zeros: u32, shift: u32) -> u32 {
    (shift as i32 - ((LOG_SUM_PROBS as i32 - zeros as i32) >> 1)).clamp(0, zeros as i32) as u32
}

/// `p` rounded down to what precision `shift` represents (at least 1).
fn representable(p: u32, shift: u32) -> u32 {
    if p <= 1 {
        return p;
    }
    let zeros = 31 - p.leading_zeros();
    let drop = zeros - kept_bits(zeros, shift);
    (p >> drop) << drop
}

/// The distribution for `counts` at precision `shift`, or none if the left
/// out symbol would be left nothing.
fn normalize(counts: &[u64], alphabet: usize, shift: u32) -> Option<Vec<u32>> {
    let total: u64 = counts[..alphabet].iter().sum();
    let mut dist = vec![0u32; alphabet];
    for s in 0..alphabet {
        if counts[s] > 0 {
            let p = (counts[s] as f64 * f64::from(SUM_PROBS) / total as f64).round() as u32;
            dist[s] = representable(p.clamp(1, SUM_PROBS - 1), shift);
        }
    }
    // The left out symbol: the first with the longest code.
    let omit = omitted(&dist);
    loop {
        let others: u32 = (0..alphabet).filter(|&s| s != omit).map(|s| dist[s]).sum();
        if others < SUM_PROBS {
            // Its code must still be the first longest after taking the rest.
            dist[omit] = SUM_PROBS - others;
            if omitted(&dist) == omit {
                return Some(dist);
            }
        }
        // Too much given away: take the largest other down a step.
        let (s, _) = (0..alphabet)
            .filter(|&s| s != omit && dist[s] > 1)
            .map(|s| (s, dist[s]))
            .max_by_key(|&(_, p)| p)?;
        let lowered = representable(dist[s] - 1, shift).max(1);
        dist[s] = lowered;
    }
}

fn omitted(dist: &[u32]) -> usize {
    let mut best = 0;
    for s in 1..dist.len() {
        if log_code(dist[s]) > log_code(dist[best]) {
            best = s;
        }
    }
    best
}

fn write_complex(w: &mut BitWriter, dist: &[u32], alphabet: usize, shift: u32) {
    // The precision: a unary length (at most 3), then that many bits.
    let len = 32 - (shift + 1).leading_zeros() - 1; // floor(log2(shift + 1))
    let len = len.min(3);
    for _ in 0..len {
        w.bit(true);
    }
    if len < 3 {
        w.bit(false);
    }
    w.write(len, shift + 1 - (1 << len));
    write_u8(w, alphabet as u32 - 3);

    let omit = omitted(dist);
    let codes: Vec<u32> = dist[..alphabet].iter().map(|&p| log_code(p)).collect();
    // The codes, with runs of four or more equal probabilities after the
    // first sent as one repeat.
    let mut runs = vec![0usize; alphabet]; // at i: how many after i repeat it
    let mut i = 0;
    while i < alphabet {
        let mut j = i + 1;
        while j < alphabet && dist[j] == dist[i] && j != omit {
            j += 1;
        }
        // A repeat may not follow the left out symbol, which reads as 0.
        if j - i - 1 >= 4 && i != omit {
            runs[i] = j - i - 1;
        }
        i = j;
    }
    let mut s = 0;
    while s < alphabet {
        let (n, c) = LOG_COUNT_CODE[codes[s] as usize];
        w.write(n, c);
        if runs[s] > 0 {
            let (n, c) = LOG_COUNT_CODE[RLE_MARKER as usize];
            w.write(n, c);
            write_u8(w, runs[s] as u32 - 4);
            s += runs[s] + 1;
        } else {
            s += 1;
        }
    }
    // The bits under each leading one, for the symbols not repeated or left
    // out, in order.
    let mut s = 0;
    while s < alphabet {
        let p = dist[s];
        if s != omit && p > 1 {
            let zeros = codes[s] - 1;
            let kept = kept_bits(zeros, shift);
            w.write(kept, (p - (1 << zeros)) >> (zeros - kept));
        }
        s += runs[s] + 1;
    }
}

/// For each symbol, the decoder's state positions (0..4096) of its slots,
/// in slot order — from the alias table the decoder builds.
fn alias_slots(dist: &[u32], log_alpha_size: u32) -> Vec<Vec<u16>> {
    let table_size = dist.len();
    let log_bucket_size = LOG_SUM_PROBS - log_alpha_size;
    let bucket_size = 1u32 << log_bucket_size;
    let mut slots: Vec<Vec<u16>> = dist.iter().map(|&d| vec![0; d as usize]).collect();

    if let Some(single) = dist.iter().position(|&d| d == SUM_PROBS) {
        for (pos, slot) in slots[single].iter_mut().enumerate() {
            *slot = pos as u16;
        }
        return slots;
    }

    // The decoder's construction, step for step.
    struct Working {
        alias_symbol: u32,
        alias_offset: u32,
        alias_cutoff: u32,
    }
    let alphabet = dist.iter().rposition(|&d| d > 0).map_or(0, |p| p + 1);
    let mut buckets: Vec<Working> = dist
        .iter()
        .enumerate()
        .map(|(i, &d)| Working {
            alias_symbol: if i < alphabet { i as u32 } else { 0 },
            alias_offset: 0,
            alias_cutoff: d,
        })
        .collect();
    let mut underfull = Vec::new();
    let mut overfull = Vec::new();
    for (i, &d) in dist.iter().enumerate() {
        match d.cmp(&bucket_size) {
            std::cmp::Ordering::Less => underfull.push(i),
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => overfull.push(i),
        }
    }
    while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
        let by = bucket_size - buckets[u].alias_cutoff;
        buckets[o].alias_cutoff -= by;
        buckets[u].alias_symbol = o as u32;
        buckets[u].alias_offset = buckets[o].alias_cutoff;
        match buckets[o].alias_cutoff.cmp(&bucket_size) {
            std::cmp::Ordering::Less => underfull.push(o),
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => overfull.push(o),
        }
    }
    // Walk every state position as the decoder reads it.
    for i in 0..table_size {
        let b = &buckets[i];
        let full = b.alias_cutoff == bucket_size;
        for pos in 0..bucket_size {
            let idx = (i as u32) << log_bucket_size | pos;
            let (symbol, offset) = if full || pos < b.alias_cutoff {
                (i, pos)
            } else {
                (
                    b.alias_symbol as usize,
                    b.alias_offset + pos - b.alias_cutoff,
                )
            };
            slots[symbol][offset as usize] = idx as u16;
        }
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_count_code_is_the_decoders() {
        // The decoder's table, indexed by 7 peeked bits.
        #[rustfmt::skip]
        const TABLE: [(u8, u8); 128] = [
            (10, 3), (12, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (13, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        ];
        for (code, &(n, c)) in LOG_COUNT_CODE.iter().enumerate() {
            for high in 0..(1u32 << (7 - n)) {
                let idx = (c | (high << n)) as usize;
                assert_eq!(TABLE[idx], (code as u8, n as u8), "code {code}");
            }
        }
    }

    #[test]
    fn slots_cover_the_state_space_once() {
        let counts = [100u64, 3, 0, 50, 1, 1, 7, 900, 0, 2];
        let (table, _) = AnsTable::new(&counts, 5);
        let mut seen = vec![false; 4096];
        for (s, slots) in table.slots.iter().enumerate() {
            assert_eq!(slots.len() as u32, table.dist[s]);
            for &p in slots {
                assert!(!seen[p as usize]);
                seen[p as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn normalised_distributions_sum_to_4096() {
        for counts in [
            vec![1u64, 1, 1, 1000000],
            vec![5, 5, 5, 5, 5, 6],
            vec![3, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 1],
            (1..200).collect::<Vec<u64>>(),
        ] {
            for shift in 0..=13 {
                if let Some(d) = normalize(&counts, counts.len(), shift) {
                    assert_eq!(d.iter().sum::<u32>(), 4096, "{counts:?} {shift}");
                    for (s, &c) in counts.iter().enumerate() {
                        assert_eq!(c > 0, d[s] > 0);
                    }
                }
            }
        }
    }
}
