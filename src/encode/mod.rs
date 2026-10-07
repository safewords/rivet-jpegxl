//! Lossless JPEG XL encoding: a bare codestream with one modular frame.
//!
//! Every sample is coded exactly — predicted from its neighbours, the
//! residual prefix coded — so decoding gives back the same samples. The
//! colour space is sRGB (or gray with the sRGB transfer curve); the alpha,
//! when there is one, is straight.

mod bits;
mod entropy;
mod headers;
mod modular;

use crate::{Channels, Error, Result};
use bits::BitWriter;
use headers::ImageHeader;
use modular::{ModularImage, Plane};

/// Samples to encode, interleaved as [`Channels`] says, rows top to bottom,
/// no padding.
#[derive(Clone, Copy, Debug)]
pub enum Samples<'a> {
    /// 8 bits a sample.
    U8(&'a [u8]),
    /// 16 bits a sample, the full range.
    U16(&'a [u16]),
}

impl Samples<'_> {
    fn len(&self) -> usize {
        match self {
            Samples::U8(s) => s.len(),
            Samples::U16(s) => s.len(),
        }
    }

    fn bits_per_sample(&self) -> u32 {
        match self {
            Samples::U8(_) => 8,
            Samples::U16(_) => 16,
        }
    }

    fn get(&self, i: usize) -> i32 {
        match self {
            Samples::U8(s) => i32::from(s[i]),
            Samples::U16(s) => i32::from(s[i]),
        }
    }
}

/// The largest side the codestream's size header holds.
const MAX_DIMENSION: u32 = 1 << 30;

/// Encode a picture losslessly as a JPEG XL codestream.
///
/// ```
/// let pixels = vec![0u8; 16 * 16 * 3];
/// let jxl = jpegxl::encode_lossless(16, 16, jpegxl::Channels::Rgb, jpegxl::Samples::U8(&pixels))?;
/// let image = jpegxl::decode(&jxl)?;
/// assert_eq!(image.pixels, jpegxl::Pixels::U8(pixels));
/// # Ok::<(), jpegxl::Error>(())
/// ```
pub fn encode_lossless(
    width: u32,
    height: u32,
    channels: Channels,
    samples: Samples<'_>,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::InvalidInput(format!(
            "{width}x{height}: each side must be 1 to {MAX_DIMENSION}"
        )));
    }
    let count = channels.count();
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(count))
        .ok_or_else(|| Error::InvalidInput(format!("{width}x{height} is too large")))?;
    if samples.len() != expected {
        return Err(Error::InvalidInput(format!(
            "{} samples for {width}x{height} {channels:?}, which takes {expected}",
            samples.len()
        )));
    }

    let (w, h) = (width as usize, height as usize);
    let planes: Vec<Plane> = (0..count)
        .map(|c| Plane {
            width: w,
            height: h,
            samples: (0..w * h).map(|i| samples.get(i * count + c)).collect(),
        })
        .collect();
    let image = ModularImage::new(planes);

    let mut out = BitWriter::new();
    ImageHeader {
        width,
        height,
        bits_per_sample: samples.bits_per_sample(),
        gray: channels.is_gray(),
        alpha: channels.has_alpha(),
    }
    .write(&mut out);
    headers::write_frame_header(&mut out, u32::from(channels.has_alpha()));

    // The sections: LfGlobal; then, for more than one group, an LfGroup per
    // 2048-pixel square and HfGlobal (all empty for this frame) and a
    // section per group.
    let mut global = BitWriter::new();
    image.write_global(&mut global);
    let mut sections = vec![global.finish()];
    if !image.single_group() {
        let lf_dim = (headers::GROUP_DIM * 8) as usize;
        let lf_groups = w.div_ceil(lf_dim) * h.div_ceil(lf_dim);
        sections.extend(std::iter::repeat_n(Vec::new(), lf_groups + 1));
        for g in 0..image.groups() {
            let mut group = BitWriter::new();
            image.write_group(&mut group, g);
            sections.push(group.finish());
        }
    }
    let sizes: Vec<usize> = sections.iter().map(Vec::len).collect();
    headers::write_toc(&mut out, &sizes);
    for section in &sections {
        out.append_bytes(section);
    }
    Ok(out.finish())
}
