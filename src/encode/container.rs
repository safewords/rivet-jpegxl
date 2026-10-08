//! The ISOBMFF-style container: the codestream in a `jxlc` box or split in
//! `jxlp` boxes, with metadata boxes (Exif, XMP, JUMBF, any other) around
//! it, optionally Brotli-wrapped (`brob`), and the level box.

use crate::{Error, Result};

/// A metadata box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataBox {
    /// The box type, e.g. `*b"Exif"`, `*b"xml "`, `*b"jumb"`.
    pub kind: [u8; 4],
    /// The content (for Exif: the 4-byte TIFF header offset, then the Exif
    /// data).
    pub data: Vec<u8>,
    /// Wrapped in a `brob` box (Brotli; stored blocks, readable by any
    /// Brotli decoder).
    pub brotli: bool,
    /// After the codestream rather than before it.
    pub after_codestream: bool,
}

impl MetadataBox {
    /// An Exif box from TIFF-headed Exif data.
    pub fn exif(tiff: Vec<u8>) -> Self {
        let mut data = vec![0, 0, 0, 0];
        data.extend(tiff);
        MetadataBox {
            kind: *b"Exif",
            data,
            brotli: false,
            after_codestream: false,
        }
    }

    pub fn xmp(xml: Vec<u8>) -> Self {
        MetadataBox {
            kind: *b"xml ",
            data: xml,
            brotli: false,
            after_codestream: false,
        }
    }
}

/// How the container is laid out.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Container {
    /// A `jxll` box with this conformance level (5 or 10).
    pub level: Option<u8>,
    pub boxes: Vec<MetadataBox>,
    /// Split the codestream into `jxlp` boxes at these byte offsets (none:
    /// one `jxlc` box).
    pub split_at: Option<Vec<usize>>,
    /// Write the `jxlp` boxes out of order (needs `ftyp` version 1).
    pub jxlp_out_of_order: bool,
    /// Write box sizes in the 64-bit form.
    pub large_sizes: bool,
    /// Leave the last box's size open (0: to the end of the file).
    pub open_last_box: bool,
}

const SIGNATURE: [u8; 12] = [
    0, 0, 0, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
];

fn push_box(
    out: &mut Vec<u8>,
    kind: &[u8; 4],
    prefix: &[u8],
    content: &[u8],
    large: bool,
    open: bool,
) {
    let len = prefix.len() + content.len();
    if open {
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(kind);
    } else if large || len + 8 > u32::MAX as usize {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(&((len + 16) as u64).to_be_bytes());
    } else {
        out.extend_from_slice(&((len + 8) as u32).to_be_bytes());
        out.extend_from_slice(kind);
    }
    out.extend_from_slice(prefix);
    out.extend_from_slice(content);
}

/// A Brotli stream holding `data` in stored meta-blocks.
pub fn brotli_stored(data: &[u8]) -> Vec<u8> {
    let mut w = crate::encode::bits::BitWriter::new();
    w.bit(false); // WBITS 16
    for chunk in data.chunks(1 << 24) {
        let mlen = chunk.len() as u32 - 1;
        let bits = 32 - mlen.leading_zeros();
        let nibbles = bits.div_ceil(4).max(4);
        w.bit(false); // ISLAST
        w.write(2, nibbles - 4);
        w.write(nibbles * 4, mlen);
        w.bit(true); // ISUNCOMPRESSED
        w.pad_to_byte();
        w.append_bytes(chunk);
    }
    w.bit(true); // ISLAST
    w.bit(true); // ISLASTEMPTY
    w.finish()
}

/// `codestream` in a container.
pub fn wrap(codestream: &[u8], c: &Container) -> Result<Vec<u8>> {
    let mut out = SIGNATURE.to_vec();
    let version = u32::from(c.jxlp_out_of_order);
    let mut ftyp = b"jxl ".to_vec();
    ftyp.extend_from_slice(&version.to_be_bytes());
    ftyp.extend_from_slice(b"jxl ");
    push_box(&mut out, b"ftyp", &[], &ftyp, c.large_sizes, false);
    if let Some(level) = c.level {
        push_box(&mut out, b"jxll", &[], &[level], c.large_sizes, false);
    }

    // The boxes in order, the last possibly open.
    enum Item<'a> {
        Meta(&'a MetadataBox),
        Code,
    }
    let mut items: Vec<Item> = c
        .boxes
        .iter()
        .filter(|b| !b.after_codestream)
        .map(Item::Meta)
        .collect();
    items.push(Item::Code);
    items.extend(
        c.boxes
            .iter()
            .filter(|b| b.after_codestream)
            .map(Item::Meta),
    );
    let last = items.len() - 1;
    for (i, item) in items.iter().enumerate() {
        let open = c.open_last_box && i == last;
        match item {
            Item::Meta(b) => {
                if b.kind[..3] == *b"jxl" || &b.kind == b"brob" || &b.kind == b"ftyp" {
                    return Err(Error::InvalidInput(format!(
                        "a metadata box of type {:?}",
                        String::from_utf8_lossy(&b.kind)
                    )));
                }
                if b.brotli {
                    push_box(
                        &mut out,
                        b"brob",
                        &b.kind,
                        &brotli_stored(&b.data),
                        c.large_sizes,
                        open,
                    );
                } else {
                    push_box(&mut out, &b.kind, &[], &b.data, c.large_sizes, open);
                }
            }
            Item::Code => match &c.split_at {
                None => push_box(&mut out, b"jxlc", &[], codestream, c.large_sizes, open),
                Some(at) => {
                    let mut cuts = vec![0];
                    cuts.extend(
                        at.iter()
                            .copied()
                            .filter(|&p| p > 0 && p < codestream.len()),
                    );
                    cuts.push(codestream.len());
                    cuts.sort_unstable();
                    cuts.dedup();
                    let parts: Vec<&[u8]> =
                        cuts.windows(2).map(|w| &codestream[w[0]..w[1]]).collect();
                    let n = parts.len();
                    let order: Vec<usize> = if c.jxlp_out_of_order {
                        // The first in place, the rest reversed.
                        std::iter::once(0).chain((1..n).rev()).collect()
                    } else {
                        (0..n).collect()
                    };
                    for (k, &p) in order.iter().enumerate() {
                        let mut index = p as u32;
                        if p + 1 == n {
                            index |= 0x8000_0000;
                        }
                        let open = open && k + 1 == n;
                        push_box(
                            &mut out,
                            b"jxlp",
                            &index.to_be_bytes(),
                            parts[p],
                            c.large_sizes,
                            open,
                        );
                    }
                }
            },
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_brotli_headers() {
        let s = brotli_stored(b"abc");
        // WBITS 0, ISLAST 0, MNIBBLES 4 (00), MLEN-1 = 2 in 16 bits,
        // ISUNCOMPRESSED 1, pad; data; ISLAST, ISLASTEMPTY.
        assert_eq!(s[0], 0b0010_0000);
        assert_eq!(s[1], 0);
        assert_eq!(s[2], 0b0001_0000);
        assert_eq!(&s[3..6], b"abc");
        assert_eq!(s[6], 0b11);
    }
}
