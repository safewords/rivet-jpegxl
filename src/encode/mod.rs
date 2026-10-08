//! Lossless JPEG XL encoding: a bare codestream with one modular frame.
//!
//! Every sample is coded exactly — predicted from its neighbours, the
//! residual prefix coded — so decoding gives back the same samples. The
//! colour space is sRGB (or gray with the sRGB transfer curve); the alpha,
//! when there is one, is straight.

// Codec loops index several parallel arrays by position; iterators would
// obscure the arithmetic the decoder's own loops are written in.
#![allow(clippy::needless_range_loop)]

mod bits;
mod entropy;
mod headers;
mod modular;

use crate::{Channels, Error, Result};
use bits::BitWriter;
pub use entropy::{EntropyOptions, Lz77Mode};
use headers::ImageHeader;
use modular::{Channel, ImageStreams, Layout, ModularCoded};
pub use modular::{
    ModularOptions, Palette, Predictor, Rct, SqueezeStep, Transform, TreeMode, WeightedParams,
};

/// Options for lossless encoding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LosslessOptions {
    pub modular: ModularOptions,
    /// The global transforms, applied in order.
    pub transforms: Vec<Transform>,
}

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
    encode_lossless_with(
        width,
        height,
        channels,
        samples,
        &LosslessOptions::default(),
    )
}

/// [`encode_lossless`], with the entropy coder's options.
pub fn encode_lossless_with(
    width: u32,
    height: u32,
    channels: Channels,
    samples: Samples<'_>,
    options: &LosslessOptions,
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
    let mut coded: Vec<Channel> = (0..count)
        .map(|c| {
            Channel::new(
                w,
                h,
                Some((0, 0)),
                (0..w * h).map(|i| samples.get(i * count + c)).collect(),
            )
        })
        .collect();
    let bits = samples.bits_per_sample();
    let mut sixteen_bit = bits <= 12 && modular::fits_i16(&coded);
    for t in &options.transforms {
        t.apply(&mut coded, bits, options.modular.weighted)
            .map_err(Error::InvalidInput)?;
        sixteen_bit &= modular::fits_i16(&coded);
    }

    let gd = headers::GROUP_DIM as usize;
    let layout = Layout {
        group_dim: gd,
        groups_x: w.div_ceil(gd),
        groups_y: h.div_ceil(gd),
        lf_groups_x: w.div_ceil(gd * 8),
        lf_groups_y: h.div_ceil(gd * 8),
        pass_shifts: vec![(0, 2)],
        num_lf_groups: w.div_ceil(gd * 8) * h.div_ceil(gd * 8),
    };
    let split = ImageStreams::split(coded, options.transforms.clone(), &layout);
    let mut streams = vec![split.global];
    streams.extend(split.lf);
    for pass in split.hf {
        streams.extend(pass);
    }
    let size_limit = (1024 + w * h * count / 16).min(1 << 22);
    let modular = ModularCoded::new(&streams, &options.modular, size_limit);

    let mut out = BitWriter::new();
    ImageHeader {
        width,
        height,
        bits_per_sample: bits,
        gray: channels.is_gray(),
        alpha: channels.has_alpha(),
        modular_16bit: sixteen_bit,
    }
    .write(&mut out);
    headers::write_frame_header(&mut out, u32::from(channels.has_alpha()));

    // The sections: LfGlobal (the default LF quantisation, the global tree,
    // the global image), each LF group, HfGlobal (empty), each group.
    let mut global = BitWriter::new();
    global.bit(true); // LfQuant: all_default
    modular.write_global_tree(&mut global);
    modular.write_stream(&mut global, 0);
    let num_lf = layout.num_lf_groups;
    let num_groups = layout.num_groups();
    let mut sections = vec![global];
    for g in 0..num_lf {
        let mut s = BitWriter::new();
        modular.write_stream(&mut s, 1 + g);
        sections.push(s);
    }
    sections.push(BitWriter::new()); // HfGlobal
    for g in 0..num_groups {
        let mut s = BitWriter::new();
        modular.write_stream(&mut s, 1 + num_lf + g);
        sections.push(s);
    }
    let sections: Vec<Vec<u8>> = if num_groups == 1 {
        // One section holds them all, read on from one to the next.
        let mut all = BitWriter::new();
        for s in &sections {
            all.append_bits(s);
        }
        vec![all.finish()]
    } else {
        sections.into_iter().map(BitWriter::finish).collect()
    };
    let sizes: Vec<usize> = sections.iter().map(Vec::len).collect();
    headers::write_toc(&mut out, &sizes);
    for section in &sections {
        out.append_bytes(section);
    }
    Ok(out.finish())
}
