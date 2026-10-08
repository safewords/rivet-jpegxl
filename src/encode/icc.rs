//! An ICC profile as the codestream carries it: the header predicted, the
//! tag table and common tag data turned into commands, the rest copied;
//! that stream of bytes then entropy coded in 41 contexts from the two bytes
//! before.

use super::bits::BitWriter;
use super::entropy::{EntropyCode, EntropyOptions, Stream, Token};
use crate::{Error, Result};

const HEADER_SIZE: usize = 128;
const CONTEXTS: usize = 41;
const COMMON_TAGS: [&[u8; 4]; 19] = [
    b"rTRC", b"rXYZ", b"cprt", b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"rTRC",
    b"gTRC", b"bTRC", b"kTRC", b"chad", b"desc", b"chrm", b"dmnd", b"dmdd", b"lumi",
];
const COMMON_DATA: [&[u8; 4]; 8] = [
    b"XYZ ", b"desc", b"text", b"mluc", b"para", b"curv", b"sf32", b"gbd ",
];

/// The decoder's prediction of header byte `idx` (`header`: the coded
/// bytes so far).
fn predict_header(idx: usize, size: u32, header: &[u8]) -> u8 {
    match idx {
        0..=3 => size.to_be_bytes()[idx],
        8 => 4,
        12..=23 => b"mntrRGB XYZ "[idx - 12],
        36..=39 => b"acsp"[idx - 36],
        41 | 42 if header[40] == b'A' => b'P',
        43 if header[40] == b'A' => b'L',
        41 if header[40] == b'M' => b'S',
        42 if header[40] == b'M' => b'F',
        43 if header[40] == b'M' => b'T',
        42 if header[40] == b'S' && header[41] == b'G' => b'I',
        43 if header[40] == b'S' && header[41] == b'G' => b' ',
        42 if header[40] == b'S' && header[41] == b'U' => b'N',
        43 if header[40] == b'S' && header[41] == b'U' => b'W',
        70 => 246,
        71 => 214,
        73 => 1,
        78 => 211,
        79 => 45,
        80..=83 => header[4 + idx - 80],
        _ => 0,
    }
}

fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn be32(p: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]])
}

/// The tag table's entries, if the profile has a sane one.
fn tag_table(p: &[u8]) -> Option<Vec<([u8; 4], u32, u32)>> {
    if p.len() < HEADER_SIZE + 4 {
        return None;
    }
    let n = be32(p, HEADER_SIZE) as usize;
    if n == 0 || HEADER_SIZE + 4 + 12 * n > p.len() {
        return None;
    }
    let mut tags = Vec::with_capacity(n);
    for i in 0..n {
        let at = HEADER_SIZE + 4 + 12 * i;
        let sig = [p[at], p[at + 1], p[at + 2], p[at + 3]];
        let (start, size) = (be32(p, at + 4), be32(p, at + 8));
        if u64::from(start) + u64::from(size) > p.len() as u64 {
            return None;
        }
        tags.push((sig, start, size));
    }
    Some(tags)
}

/// The profile as the decoder's command and data streams.
pub(crate) fn compress(profile: &[u8]) -> Vec<u8> {
    let size = profile.len();
    let mut data = Vec::new();
    let mut commands = Vec::new();

    let header_len = size.min(HEADER_SIZE);
    let mut coded_header = Vec::with_capacity(header_len);
    for idx in 0..header_len {
        let p = predict_header(idx, size as u32, &coded_header);
        coded_header.push(profile[idx].wrapping_sub(p));
    }
    data.extend_from_slice(&coded_header);

    if size > HEADER_SIZE {
        let mut pos = HEADER_SIZE;
        match tag_table(profile) {
            Some(tags) => {
                varint(&mut commands, tags.len() as u64 + 1);
                pos += 4 + 12 * tags.len();
                // The decoder's first default start leaves out the tag count.
                let mut prev_start = (HEADER_SIZE + 12 * tags.len()) as u32;
                let mut prev_size = 0u32;
                let mut i = 0;
                while i < tags.len() {
                    let (sig, start, tsize) = tags[i];
                    // Three tags in one: r/g/b TRC sharing data, or r/g/b XYZ
                    // in a row.
                    let triple = |a: &[u8; 4], b: &[u8; 4], c: &[u8; 4], step: u32| {
                        i + 2 < tags.len()
                            && &sig == a
                            && &tags[i + 1].0 == b
                            && &tags[i + 2].0 == c
                            && tags[i + 1].2 == tsize
                            && tags[i + 2].2 == tsize
                            && tags[i + 1].1 == start.wrapping_add(step)
                            && tags[i + 2].1 == start.wrapping_add(2 * step)
                    };
                    let (code, count) = if triple(b"rTRC", b"gTRC", b"bTRC", 0) {
                        (2u8, 3)
                    } else if triple(b"rXYZ", b"gXYZ", b"bXYZ", tsize) {
                        (3, 3)
                    } else if let Some(k) = COMMON_TAGS.iter().skip(2).position(|t| **t == sig) {
                        // A single tag (the triple codes 2 and 3 excepted).
                        (k as u8 + 4, 1)
                    } else {
                        (1, 1)
                    };
                    let mut command = code;
                    let default_start = prev_start.wrapping_add(prev_size);
                    if start != default_start {
                        command |= 64;
                    }
                    let default_size = match &sig {
                        b"rXYZ" | b"gXYZ" | b"bXYZ" | b"kXYZ" | b"wtpt" | b"bkpt" | b"lumi" => 20,
                        _ => prev_size,
                    };
                    if tsize != default_size {
                        command |= 128;
                    }
                    commands.push(command);
                    if code == 1 {
                        data.extend_from_slice(&sig);
                    }
                    if command & 64 != 0 {
                        varint(&mut commands, u64::from(start));
                    }
                    if command & 128 != 0 {
                        varint(&mut commands, u64::from(tsize));
                    }
                    prev_start = start;
                    prev_size = tsize;
                    i += count;
                }
                // The end of the tag list.
                commands.push(0);
                // The tag data.
                let mut starts: Vec<(usize, usize)> = tags
                    .iter()
                    .map(|&(_, s, l)| (s as usize, l as usize))
                    .collect();
                starts.sort_unstable();
                starts.dedup();
                let mut copy_from = pos;
                let flush = |commands: &mut Vec<u8>, data: &mut Vec<u8>, from: usize, to: usize| {
                    if to > from {
                        commands.push(1);
                        varint(commands, (to - from) as u64);
                        data.extend_from_slice(&profile[from..to]);
                    }
                };
                for (s, l) in starts {
                    if s < pos || s + 8 > size {
                        continue;
                    }
                    let sig = &profile[s..s + 4];
                    let zeros = profile[s + 4..s + 8] == [0, 0, 0, 0];
                    if sig == b"XYZ " && zeros && l >= 20 && s + 20 <= size {
                        flush(&mut commands, &mut data, copy_from, s);
                        commands.push(10);
                        data.extend_from_slice(&profile[s + 8..s + 20]);
                        pos = s + 20;
                        copy_from = pos;
                    } else if let Some(k) = COMMON_DATA.iter().position(|t| t[..] == *sig)
                        && zeros
                    {
                        flush(&mut commands, &mut data, copy_from, s);
                        commands.push(16 + k as u8);
                        pos = s + 8;
                        copy_from = pos;
                    }
                }
                flush(&mut commands, &mut data, copy_from, size);
            }
            None => {
                varint(&mut commands, 0);
                commands.push(1);
                varint(&mut commands, (size - HEADER_SIZE) as u64);
                data.extend_from_slice(&profile[HEADER_SIZE..]);
            }
        }
    }

    let mut out = Vec::new();
    varint(&mut out, size as u64);
    varint(&mut out, commands.len() as u64);
    out.extend_from_slice(&commands);
    out.extend_from_slice(&data);
    out
}

/// The context of a coded byte from the two before it.
fn context(i: usize, b1: u8, b2: u8) -> u32 {
    if i <= HEADER_SIZE {
        return 0;
    }
    let p1 = match b1 {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=1 => 2 + u32::from(b1),
        2..=15 => 4,
        241..=254 => 5,
        255 => 6,
        _ => 7,
    };
    let p2 = match b2 {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=15 => 2,
        241..=255 => 3,
        _ => 4,
    };
    1 + p1 + 8 * p2
}

/// The profile as the codestream carries it, after the colour encoding.
pub(crate) fn write(w: &mut BitWriter, profile: &[u8]) -> Result<()> {
    if profile.is_empty() || profile.len() > 1 << 28 {
        return Err(Error::InvalidInput(format!(
            "an ICC profile of {} bytes",
            profile.len()
        )));
    }
    let stream = compress(profile);
    if stream.len() > 1 << 24 {
        return Err(Error::InvalidInput(
            "the ICC profile does not compress below 16 MiB".into(),
        ));
    }
    write_u64(w, stream.len() as u64);
    let tokens: Vec<Token> = (0..stream.len())
        .map(|i| {
            let b1 = if i >= 1 { stream[i - 1] } else { 0 };
            let b2 = if i >= 2 { stream[i - 2] } else { 0 };
            Token::new(context(i, b1, b2), u32::from(stream[i]))
        })
        .collect();
    let (code, streams) = EntropyCode::build(
        CONTEXTS,
        vec![Stream::new(tokens)],
        &EntropyOptions::default(),
        true,
    );
    code.write_all(w, &streams[0]);
    Ok(())
}

/// A `U64` field.
pub(crate) fn write_u64(w: &mut BitWriter, v: u64) {
    match v {
        0 => w.write(2, 0),
        1..=16 => {
            w.write(2, 1);
            w.write(4, (v - 1) as u32);
        }
        17..=272 => {
            w.write(2, 2);
            w.write(8, (v - 17) as u32);
        }
        _ => {
            w.write(2, 3);
            w.write(12, (v & 0xfff) as u32);
            let mut rest = v >> 12;
            let mut shift = 12;
            while rest > 0 {
                w.bit(true);
                if shift == 60 {
                    w.write(4, (rest & 0xf) as u32);
                    return;
                }
                w.write(8, (rest & 0xff) as u32);
                rest >>= 8;
                shift += 8;
            }
            w.bit(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_round_trips() {
        let mut profile: Vec<u8> = (0..100u32).map(|i| (i * 37 % 251) as u8).collect();
        profile[..4].copy_from_slice(&100u32.to_be_bytes());
        profile[36..40].copy_from_slice(b"acsp");
        profile[40..44].copy_from_slice(b"APPL");
        let s = compress(&profile);
        assert_eq!(&s[..2], &[100, 0]);
        // The decoder's reconstruction.
        let coded = &s[2..];
        let decoded: Vec<u8> = (0..100)
            .map(|i| predict_header(i, 100, coded).wrapping_add(coded[i]))
            .collect();
        assert_eq!(decoded, profile);
        // Its predictions hit: the size, "acsp", "PPL".
        assert!(coded[..4].iter().all(|&b| b == 0));
        assert!(coded[36..40].iter().all(|&b| b == 0));
        assert!(coded[41..44].iter().all(|&b| b == 0));
    }
}
