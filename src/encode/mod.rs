//! JPEG XL encoding, this crate's own.
//!
//! [`Encoder`] writes a codestream with every feature its header and frames
//! can carry: any sample format, extra channels of every kind, named colour
//! encodings or ICC profiles, orientation, previews, animation, and frames
//! with crops, blending, references, upsampling and passes.
//! [`encode_lossless`] is the short way to a lossless still.

// Codec loops index several parallel arrays by position; iterators would
// obscure the arithmetic the decoder's own loops are written in.
#![allow(clippy::needless_range_loop)]
// The format's constants are given to the precision the spec gives them.
#![allow(clippy::excessive_precision)]

mod bits;
mod container;
mod encoder;
mod entropy;
mod features;
mod frame;
mod header;
mod icc;
mod modular;
mod vardct;
mod xyb;

use crate::{Channels, Error, Result};
pub use container::{Container, MetadataBox, brotli_stored, wrap};
pub use encoder::{Encoder, Frame, FrameContent, FrameOptions, ModularFrame, float_to_format_bits};
pub use entropy::{EntropyOptions, Lz77Mode};
pub use features::{
    Features, Noise, Patch, PatchBlendMode, PatchBlending, PatchPlacement, QuantizedSpline, Splines,
};
pub use frame::{BlendMode, Blending, Crop, FrameType, Passes, Restoration};
pub use header::{
    Animation, Chromaticity, ColorEncoding, ColorSpace, ColorSpec, ExtraChannel, ExtraChannelKind,
    ImageInfo, OpsinInverse, Primaries, RenderingIntent, SampleFormat, ToneMapping,
    TransferFunction, UpsamplingWeights, WhitePoint,
};
pub use modular::{
    ModularOptions, Palette, Predictor, Rct, SqueezeStep, Transform, TreeMode, WeightedParams,
};
pub use vardct::coeffs::BlockContextMap;
pub use vardct::encode::{ColorCorrelation, Strategy, VarDctFrame, VarDctOptions};
pub use vardct::quant::{Bands, QuantEncoding};
pub use vardct::transform::TransformType;
pub use xyb::{Xyb, srgb_to_linear};

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
    let mut info = ImageInfo::new(width, height);
    info.format = SampleFormat::Int(samples.bits_per_sample());
    if channels.is_gray() {
        info.color = ColorSpec::Encoding(ColorEncoding::GRAY);
    }
    if channels.has_alpha() {
        info.extra_channels
            .push(ExtraChannel::alpha(SampleFormat::Int(
                samples.bits_per_sample(),
            )));
    }
    let plane = |c: usize| -> Vec<i32> { (0..w * h).map(|i| samples.get(i * count + c)).collect() };
    let nc = if channels.is_gray() { 1 } else { 3 };
    let mut encoder = Encoder::new(info)?;
    encoder.add_frame(Frame {
        options: FrameOptions::default(),
        content: FrameContent::Modular(ModularFrame {
            color: (0..nc).map(plane).collect(),
            extra: (nc..count).map(plane).collect(),
            options: options.modular.clone(),
            transforms: options.transforms.clone(),
            ..Default::default()
        }),
    });
    encoder.finish()
}

/// Encode a picture lossily (VarDCT, XYB) at `distance` (1.0: about
/// visually lossless; lower is better). The samples are sRGB (gray: the
/// sRGB curve); an alpha channel is kept exactly.
///
/// ```
/// let pixels = vec![128u8; 32 * 32 * 3];
/// let jxl = jpegxl::encode_lossy(32, 32, jpegxl::Channels::Rgb, jpegxl::Samples::U8(&pixels), 1.0)?;
/// assert_eq!(jpegxl::probe(&jxl)?.width, 32);
/// # Ok::<(), jpegxl::Error>(())
/// ```
pub fn encode_lossy(
    width: u32,
    height: u32,
    channels: Channels,
    samples: Samples<'_>,
    distance: f32,
) -> Result<Vec<u8>> {
    encode_lossy_with(
        width,
        height,
        channels,
        samples,
        &VarDctOptions {
            distance,
            ..Default::default()
        },
    )
}

/// [`encode_lossy`] with every VarDCT option.
pub fn encode_lossy_with(
    width: u32,
    height: u32,
    channels: Channels,
    samples: Samples<'_>,
    options: &VarDctOptions,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::InvalidInput(format!(
            "{width}x{height}: each side must be 1 to {MAX_DIMENSION}"
        )));
    }
    let count = channels.count();
    let (w, h) = (width as usize, height as usize);
    if samples.len() != w * h * count {
        return Err(Error::InvalidInput(format!(
            "{} samples for {width}x{height} {channels:?}, which takes {}",
            samples.len(),
            w * h * count
        )));
    }
    let bits = samples.bits_per_sample();
    let max = ((1u32 << bits) - 1) as f32;
    let mut info = ImageInfo::new(width, height);
    info.xyb = true;
    info.format = SampleFormat::Int(bits);
    if channels.is_gray() {
        info.color = ColorSpec::Encoding(ColorEncoding::GRAY);
    }
    if channels.has_alpha() {
        info.extra_channels
            .push(ExtraChannel::alpha(SampleFormat::Int(bits)));
    }
    let xyb = Xyb::new(&OpsinInverse::default(), 255.0);
    // The sRGB curve as a table over the sample values.
    let curve: Vec<f32> = (0..=max as u32)
        .map(|v| srgb_to_linear(v as f32 / max))
        .collect();
    let gray = channels.is_gray();
    let mut color = vec![vec![0f32; w * h]; 3];
    for i in 0..w * h {
        let at = |c: usize| curve[samples.get(i * count + c) as usize];
        let lin = if gray {
            let g = at(0);
            [g, g, g]
        } else {
            [at(0), at(1), at(2)]
        };
        let v = xyb.from_linear(lin);
        for c in 0..3 {
            color[c][i] = v[c];
        }
    }
    let extra = if channels.has_alpha() {
        vec![
            (0..w * h)
                .map(|i| samples.get(i * count + count - 1))
                .collect(),
        ]
    } else {
        Vec::new()
    };
    let mut encoder = Encoder::new(info)?;
    encoder.add_frame(Frame {
        options: FrameOptions {
            restoration: Restoration::DEFAULT,
            ..Default::default()
        },
        content: FrameContent::VarDct(VarDctFrame {
            color,
            extra,
            options: options.clone(),
        }),
    });
    encoder.finish()
}
