//! # rivet-jpegxl
//!
//! JPEG XL decoding with a small, typed API, over
//! [jxl-rs](https://github.com/libjxl/jxl-rs) — the JPEG XL project's own
//! pure-Rust decoder. No C, no system libraries, no build script.
//!
//! ```no_run
//! # fn main() -> jpegxl::Result<()> {
//! let data = std::fs::read("in.jxl").unwrap();
//! let info = jpegxl::probe(&data)?;
//! println!("{}x{}, {} bits, alpha {}", info.width, info.height, info.bits_per_sample, info.has_alpha);
//!
//! let image = jpegxl::decode(&data)?;
//! match &image.pixels {
//!     jpegxl::Pixels::U8(p) => println!("{} bytes of 8-bit {:?}", p.len(), image.channels),
//!     jpegxl::Pixels::U16(p) => println!("{} 16-bit samples", p.len()),
//!     jpegxl::Pixels::F32(p) => println!("{} float samples", p.len()),
//! }
//! # Ok(()) }
//! ```
//!
//! **Pixels** come out interleaved, rows top to bottom with no padding,
//! straight alpha, at the precision the file was coded at
//! ([`SampleType::Auto`]): 8-bit for up to 8 bits a sample, 16-bit for
//! more, float for a float image (an HDR one, usually) — or at the one asked
//! for. Gray stays gray.
//!
//! **Colour** is the file's own: the pixels are in the colour space
//! [`Image::icc_profile`] describes (the embedded profile, or one made from
//! the codestream's colour encoding — sRGB, Display P3, BT.2100 PQ / HLG, …),
//! so a caller with a colour manager converts from it, and one without can
//! read [`Info::color`] for the common cases.
//!
//! **Orientation** is applied by default (the picture comes out upright, as a
//! viewer shows it); [`DecodeOptions::apply_orientation`] turns that off and
//! [`Info::orientation`] says what it was.

mod error;
mod runner;

use std::borrow::Cow;

use jxl::api::{
    Endianness, JxlAuxBoxType, JxlBitDepth, JxlColorType, JxlDataFormat, JxlDecoder,
    JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};
use jxl::headers::extra_channels::ExtraChannel;

pub use error::{Error, Result};

/// The underlying decoder, for what this crate does not wrap.
pub use jxl;

/// The bare codestream's signature.
const CODESTREAM_SIGNATURE: [u8; 2] = [0xff, 0x0a];
/// The container's signature box.
const CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
];

/// Whether `data` starts like a JPEG XL file: a bare codestream (`FF 0A`) or
/// the ISOBMFF-style container (`JXL ` signature box).
pub fn is_jxl(data: &[u8]) -> bool {
    data.starts_with(&CODESTREAM_SIGNATURE) || data.starts_with(&CONTAINER_SIGNATURE)
}

/// Whether the file is a bare codestream or the container, which can also
/// carry metadata boxes (Exif, XMP) and a JPEG reconstruction box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrapping {
    Codestream,
    Container,
}

/// The channels a picture has, in the order they are interleaved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channels {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

impl Channels {
    /// Samples per pixel.
    pub fn count(self) -> usize {
        match self {
            Channels::Gray => 1,
            Channels::GrayAlpha => 2,
            Channels::Rgb => 3,
            Channels::Rgba => 4,
        }
    }

    pub fn has_alpha(self) -> bool {
        matches!(self, Channels::GrayAlpha | Channels::Rgba)
    }

    pub fn is_gray(self) -> bool {
        matches!(self, Channels::Gray | Channels::GrayAlpha)
    }

    fn of(gray: bool, alpha: bool) -> Self {
        match (gray, alpha) {
            (true, false) => Channels::Gray,
            (true, true) => Channels::GrayAlpha,
            (false, false) => Channels::Rgb,
            (false, true) => Channels::Rgba,
        }
    }

    fn jxl(self) -> JxlColorType {
        match self {
            Channels::Gray => JxlColorType::Grayscale,
            Channels::GrayAlpha => JxlColorType::GrayscaleAlpha,
            Channels::Rgb => JxlColorType::Rgb,
            Channels::Rgba => JxlColorType::Rgba,
        }
    }
}

/// The sample type pixels are decoded to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SampleType {
    /// The file's own precision: `U8` for up to 8 bits a sample, `U16` for
    /// more, `F32` for a float image.
    #[default]
    Auto,
    /// 8 bits a sample, 0..=255.
    U8,
    /// 16 bits a sample, 0..=65535 (the full range, whatever the coded depth).
    U16,
    /// Float, nominal range 0.0..=1.0 (an HDR picture may go beyond 1.0).
    F32,
}

/// Decoded samples, interleaved as [`Channels`] says.
#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl Pixels {
    /// Samples (not pixels).
    pub fn len(&self) -> usize {
        match self {
            Pixels::U8(p) => p.len(),
            Pixels::U16(p) => p.len(),
            Pixels::F32(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A decoded picture.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    /// Width in pixels, upright when the orientation was applied.
    pub width: u32,
    pub height: u32,
    pub channels: Channels,
    /// `width * height * channels.count()` samples, rows top to bottom.
    pub pixels: Pixels,
    /// Bits a sample as coded (the precision the samples carry).
    pub bits_per_sample: u32,
    /// The colour space of the pixels, as an ICC profile.
    pub icc_profile: Option<Vec<u8>>,
}

/// A frame of an animation.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// The frame, composited on the canvas.
    pub image: Image,
    /// How long it is shown, milliseconds (0 for a still).
    pub duration_ms: f64,
    /// When it is first shown, milliseconds from the start.
    pub timestamp_ms: f64,
}

/// The colour a codestream declares, for a caller without a colour manager.
/// Files with an embedded ICC profile say [`Color::Icc`]; read the profile.
#[derive(Clone, Debug, PartialEq)]
pub enum Color {
    /// sRGB primaries (or gray), D65, the sRGB transfer.
    Srgb,
    /// sRGB primaries (or gray), D65, linear.
    LinearSrgb,
    /// sRGB primaries (or gray), D65, a pure power transfer: the encoding
    /// exponent, `0.45455` for gamma 2.2.
    Gamma(f32),
    /// Display P3: P3 primaries, D65, the sRGB transfer.
    DisplayP3,
    /// BT.2100 (BT.2020 primaries) with the PQ transfer (HDR10).
    Bt2100Pq,
    /// BT.2100 (BT.2020 primaries) with the HLG transfer.
    Bt2100Hlg,
    /// Something else the codestream describes, in jxl-rs's notation
    /// (`colour space_white point_primaries_intent_transfer`).
    Other(String),
    /// An embedded ICC profile.
    Icc,
}

/// Animation timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Animation {
    /// Ticks a second, `numerator / denominator`.
    pub ticks_per_second: (u32, u32),
    /// Times to play, 0 for forever.
    pub loop_count: u32,
}

/// A file's description, read without decoding its pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// Width as shown: the orientation's transposition (5..=8) applied.
    pub width: u32,
    /// Height as shown.
    pub height: u32,
    /// Bits a sample as coded.
    pub bits_per_sample: u32,
    /// Whether samples are float (an HDR picture, usually).
    pub float: bool,
    pub has_alpha: bool,
    pub gray: bool,
    /// Whether there is a black (K) channel: a CMYK picture, decoded to RGB.
    pub cmyk: bool,
    /// The Exif orientation, 1..=8 (1 is upright). Applied on decode unless
    /// [`DecodeOptions::apply_orientation`] is off.
    pub orientation: u8,
    /// Present for an animation.
    pub animation: Option<Animation>,
    /// The colour the codestream declares.
    pub color: Color,
    /// Whether the picture is coded in its original colour space (lossless,
    /// or lossy without XYB) rather than XYB.
    pub uses_original_profile: bool,
    /// The intended peak luminance, nits (255 for SDR; 10000 for PQ, 1000 for
    /// HLG unless the file says otherwise).
    pub intensity_target: f32,
    /// Container or bare codestream.
    pub wrapping: Wrapping,
    /// The `Exif` box, when one comes before the codestream (the TIFF header
    /// on, its four-byte offset dropped). `None` for a bare codestream.
    pub exif: Option<Vec<u8>>,
    /// The ICC profile of the pixels as [`decode`] makes them.
    pub icc_profile: Option<Vec<u8>>,
}

impl Info {
    /// The size as stored, before the orientation: width and height swapped
    /// back for a transposing orientation (5..=8). What
    /// [`DecodeOptions::apply_orientation`] `false` decodes to.
    pub fn stored_dims(&self) -> (u32, u32) {
        if self.orientation >= 5 {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        }
    }
}

/// Bounds on what a file may make the decoder do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The most pixels a picture (or any frame) may have (default 2^28,
    /// 16384 x 16384).
    pub max_pixels: u64,
    /// The most frames an animation may have when decoding them all
    /// (default 100 000).
    pub max_frames: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_pixels: 1 << 28,
            max_frames: 100_000,
        }
    }
}

/// How to decode.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeOptions {
    /// Turn the picture upright by its orientation (default `true`).
    pub apply_orientation: bool,
    /// The sample type to decode to.
    pub sample_type: SampleType,
    /// Decode gray pictures as RGB (default `false`: gray stays gray).
    pub gray_to_rgb: bool,
    /// Worker threads; 0 is one per core, 1 the caller's alone.
    pub threads: usize,
    pub limits: Limits,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            apply_orientation: true,
            sample_type: SampleType::Auto,
            gray_to_rgb: false,
            threads: 0,
            limits: Limits::default(),
        }
    }
}

/// Decodes a JPEG XL file: a still, or an animation's first frame.
pub fn decode(data: &[u8]) -> Result<Image> {
    Decoder::new(data)?.decode()
}

/// [`decode`] with options.
pub fn decode_with(data: &[u8], options: &DecodeOptions) -> Result<Image> {
    Decoder::with_options(data, options.clone())?.decode()
}

/// Reads a file's description without decoding its pixels.
pub fn probe(data: &[u8]) -> Result<Info> {
    Ok(Decoder::new(data)?.info().clone())
}

/// A JPEG XL decoder over a file in memory. The headers are read (and
/// checked) by [`Decoder::new`]; pixels are decoded on demand.
pub struct Decoder<'a> {
    data: &'a [u8],
    options: DecodeOptions,
    info: Info,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        Self::with_options(data, DecodeOptions::default())
    }

    pub fn with_options(data: &'a [u8], options: DecodeOptions) -> Result<Self> {
        if !is_jxl(data) {
            return Err(Error::NotJpegXl);
        }
        let (decoder, _) = header(data, &options)?;
        let info = info_of(&decoder, data)?;
        let pixels = u64::from(info.width) * u64::from(info.height);
        if pixels > options.limits.max_pixels {
            return Err(Error::LimitExceeded(format!(
                "{}x{} is {pixels} pixels, over the limit of {}",
                info.width, info.height, options.limits.max_pixels
            )));
        }
        Ok(Self {
            data,
            options,
            info,
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    /// The still, or the animation's first frame.
    pub fn decode(&self) -> Result<Image> {
        let mut frames = self.frames();
        match frames.next() {
            Some(f) => Ok(f?.image),
            None => Err(Error::Truncated),
        }
    }

    /// Every frame, in order (one for a still). Each is decoded as it is
    /// asked for.
    pub fn frames(&self) -> Frames<'a> {
        Frames {
            data: self.data,
            options: self.options.clone(),
            state: None,
            input: self.data,
            done: false,
            count: 0,
            timestamp_ms: 0.0,
            first: true,
        }
    }
}

/// The frames of a file, decoded one at a time ([`Decoder::frames`]).
pub struct Frames<'a> {
    data: &'a [u8],
    options: DecodeOptions,
    state: Option<(JxlDecoder<states::WithImageInfo>, Output)>,
    input: &'a [u8],
    done: bool,
    count: u32,
    timestamp_ms: f64,
    first: bool,
}

impl Iterator for Frames<'_> {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Result<Frame>> {
        if self.done {
            return None;
        }
        let r = self.step();
        if r.is_err() {
            self.done = true;
        }
        r.transpose()
    }
}

impl Frames<'_> {
    fn step(&mut self) -> Result<Option<Frame>> {
        let (decoder, output) = match self.state.take() {
            Some(s) => s,
            None if self.first => {
                self.first = false;
                let mut input = self.data;
                let (mut decoder, _) = header_from(&mut input, &self.options)?;
                let output = Output::choose(&decoder, &self.options)?;
                decoder.set_pixel_format(output.format.clone())?;
                self.input = input;
                (decoder, output)
            }
            None => return Ok(None),
        };
        if self.count > 0 && !decoder.has_more_frames() {
            self.done = true;
            return Ok(None);
        }
        if self.count >= self.options.limits.max_frames {
            return Err(Error::LimitExceeded(format!(
                "more than {} frames",
                self.options.limits.max_frames
            )));
        }
        let mut runner = runner::ThreadRunner::new(self.options.threads);
        // The frame's header.
        let mut decoder = decoder;
        let frame = loop {
            let r = if runner.is_parallel() {
                decoder.process(&mut self.input, Some(&mut runner))?
            } else {
                decoder.process(&mut self.input, None)?
            };
            match r {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if self.input.is_empty() {
                        return Err(Error::Truncated);
                    }
                    decoder = fallback;
                }
            }
        };
        let header = frame.frame_header();
        let (w, h) = header.size;
        if (w as u64) * (h as u64) > self.options.limits.max_pixels {
            return Err(Error::LimitExceeded(format!("a {w}x{h} frame")));
        }
        let samples = w * h * output.channels.count();
        let mut pixels = match output.sample {
            Sample::U8 => Pixels::U8(vec![0u8; samples]),
            Sample::U16 => Pixels::U16(vec![0u16; samples]),
            Sample::F32 => Pixels::F32(vec![0f32; samples]),
        };
        {
            let bytes: &mut [u8] = match &mut pixels {
                Pixels::U8(p) => p.as_mut_slice(),
                // SAFETY: plain integer / float samples reinterpreted as
                // their bytes, exclusively borrowed for the decode; u8 has no
                // alignment requirement and every bit pattern is valid for
                // the element types written back.
                Pixels::U16(p) => unsafe {
                    std::slice::from_raw_parts_mut(p.as_mut_ptr().cast::<u8>(), p.len() * 2)
                },
                Pixels::F32(p) => unsafe {
                    std::slice::from_raw_parts_mut(p.as_mut_ptr().cast::<u8>(), p.len() * 4)
                },
            };
            let row = w * output.channels.count() * output.sample.bytes();
            let mut buffer = JxlOutputBuffer::new(bytes, h, row);
            let mut frame = frame;
            let next = loop {
                let r = if runner.is_parallel() {
                    frame.process(
                        &mut self.input,
                        std::slice::from_mut(&mut buffer),
                        Some(&mut runner),
                    )?
                } else {
                    frame.process(&mut self.input, std::slice::from_mut(&mut buffer), None)?
                };
                match r {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if self.input.is_empty() {
                            return Err(Error::Truncated);
                        }
                        frame = fallback;
                    }
                }
            };
            self.state = Some((next, output.clone()));
        }
        let duration_ms = header.duration.unwrap_or(0.0);
        let image = Image {
            width: w as u32,
            height: h as u32,
            channels: output.channels,
            pixels,
            bits_per_sample: output.bits_per_sample,
            icc_profile: output.icc.clone(),
        };
        let at = self.timestamp_ms;
        self.timestamp_ms += duration_ms;
        self.count += 1;
        Ok(Some(Frame {
            image,
            duration_ms,
            timestamp_ms: at,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sample {
    U8,
    U16,
    F32,
}

impl Sample {
    fn bytes(self) -> usize {
        match self {
            Sample::U8 => 1,
            Sample::U16 => 2,
            Sample::F32 => 4,
        }
    }
}

/// What the decoder is asked to write.
#[derive(Clone)]
struct Output {
    channels: Channels,
    sample: Sample,
    bits_per_sample: u32,
    format: JxlPixelFormat,
    icc: Option<Vec<u8>>,
}

impl Output {
    fn choose(
        decoder: &JxlDecoder<states::WithImageInfo>,
        options: &DecodeOptions,
    ) -> Result<Self> {
        let info = decoder.basic_info();
        let gray = decoder.current_pixel_format().color_type.is_grayscale() && !options.gray_to_rgb;
        let alpha = info
            .extra_channels
            .iter()
            .any(|c| c.ec_type == ExtraChannel::Alpha);
        let channels = Channels::of(gray, alpha);
        let bits = info.bit_depth.bits_per_sample();
        let sample = match options.sample_type {
            SampleType::U8 => Sample::U8,
            SampleType::U16 => Sample::U16,
            SampleType::F32 => Sample::F32,
            SampleType::Auto => match info.bit_depth {
                JxlBitDepth::Float { .. } => Sample::F32,
                JxlBitDepth::Int { bits_per_sample } if bits_per_sample <= 8 => Sample::U8,
                JxlBitDepth::Int { .. } => Sample::U16,
            },
        };
        let data_format = match sample {
            Sample::U8 => JxlDataFormat::U8 { bit_depth: 8 },
            Sample::U16 => JxlDataFormat::U16 {
                endianness: Endianness::native(),
                bit_depth: 16,
            },
            Sample::F32 => JxlDataFormat::F32 {
                endianness: Endianness::native(),
            },
        };
        let format = JxlPixelFormat {
            color_type: channels.jxl(),
            color_data_format: Some(data_format),
            extra_channel_format: vec![None; info.extra_channels.len()],
        };
        let icc = decoder
            .output_color_profile()
            .try_as_icc()
            .map(Cow::into_owned);
        Ok(Self {
            channels,
            sample,
            bits_per_sample: bits,
            format,
            icc,
        })
    }
}

fn jxl_options(options: &DecodeOptions) -> JxlDecoderOptions {
    let mut o = JxlDecoderOptions::default();
    o.adjust_orientation = options.apply_orientation;
    o.request_aux_boxes = vec![JxlAuxBoxType::EXIF];
    // Pixels times channels, with room for alpha and a few extra channels.
    // jxl-rs counts samples (three colour channels and every extra one, a
    // row at least 16 wide); this crate checks the pixels themselves after
    // the header, so this only has to be no tighter than that.
    o.sample_limit = usize::try_from(options.limits.max_pixels.saturating_mul(16)).ok();
    o
}

/// The decoder past the image header, and how much of `data` it read.
fn header(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(JxlDecoder<states::WithImageInfo>, usize)> {
    let mut input = data;
    let decoder = header_from(&mut input, options)?.0;
    Ok((decoder, data.len() - input.len()))
}

fn header_from(
    input: &mut &[u8],
    options: &DecodeOptions,
) -> Result<(JxlDecoder<states::WithImageInfo>, ())> {
    let mut decoder = JxlDecoder::<states::Initialized>::new(jxl_options(options));
    loop {
        match decoder.process(input, None)? {
            ProcessingResult::Complete { result } => return Ok((result, ())),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(Error::Truncated);
                }
                decoder = fallback;
            }
        }
    }
}

fn info_of(decoder: &JxlDecoder<states::WithImageInfo>, data: &[u8]) -> Result<Info> {
    let basic = decoder.basic_info();
    let (width, height) = basic.size;
    let has_alpha = basic
        .extra_channels
        .iter()
        .any(|c| c.ec_type == ExtraChannel::Alpha);
    let cmyk = basic
        .extra_channels
        .iter()
        .any(|c| c.ec_type == ExtraChannel::Black);
    let embedded = decoder.embedded_color_profile();
    let color = color_of(embedded);
    let exif = decoder
        .aux_boxes(JxlAuxBoxType::EXIF)
        .first()
        .and_then(exif_payload)
        .map(|d| strip_exif_offset(&d));
    let icc_profile = decoder
        .output_color_profile()
        .try_as_icc()
        .map(Cow::into_owned);
    Ok(Info {
        width: u32::try_from(width).map_err(|_| Error::LimitExceeded(format!("width {width}")))?,
        height: u32::try_from(height)
            .map_err(|_| Error::LimitExceeded(format!("height {height}")))?,
        bits_per_sample: basic.bit_depth.bits_per_sample(),
        float: matches!(basic.bit_depth, JxlBitDepth::Float { .. }),
        has_alpha,
        gray: decoder.current_pixel_format().color_type.is_grayscale(),
        cmyk,
        orientation: basic.orientation as u8,
        animation: basic.animation.as_ref().map(|a| Animation {
            ticks_per_second: (a.tps_numerator, a.tps_denominator),
            loop_count: a.num_loops,
        }),
        color,
        uses_original_profile: basic.uses_original_profile,
        intensity_target: basic.tone_mapping.intensity_target,
        wrapping: if data.starts_with(&CONTAINER_SIGNATURE) {
            Wrapping::Container
        } else {
            Wrapping::Codestream
        },
        exif,
        icc_profile,
    })
}

/// An Exif box's contents: as stored, or Brotli-decompressed (`brob`) with
/// the `brotli` feature. `None` for a compressed box without it.
fn exif_payload(b: &jxl::api::JxlAuxBox) -> Option<Vec<u8>> {
    #[cfg(feature = "brotli")]
    {
        b.data(&[]).ok().map(Cow::into_owned)
    }
    #[cfg(not(feature = "brotli"))]
    {
        (!b.is_compressed()).then(|| b.raw_data().to_vec())
    }
}

/// An `Exif` box's payload is a four-byte offset to the TIFF header, then
/// the Exif data; what callers want is the TIFF header on.
fn strip_exif_offset(payload: &[u8]) -> Vec<u8> {
    match payload.get(..4) {
        Some(o) => {
            let offset = u32::from_be_bytes([o[0], o[1], o[2], o[3]]) as usize;
            payload.get(4 + offset..).unwrap_or(&[]).to_vec()
        }
        None => Vec::new(),
    }
}

/// The colour a profile names, read from jxl-rs's description of it.
fn color_of(profile: &jxl::api::JxlColorProfile) -> Color {
    match profile {
        jxl::api::JxlColorProfile::Icc(_) => Color::Icc,
        jxl::api::JxlColorProfile::Simple(encoding) => {
            color_of_description(&encoding.get_color_encoding_description())
        }
    }
}

/// jxl-rs's notation: `RGB_D65_SRG_Rel_SRG` -- colour space, white point,
/// primaries (absent for gray), rendering intent, transfer (`SRG`, `Lin`,
/// `PeQ`, `HLG`, `709`, `DCI`, or `g` and an exponent).
fn color_of_description(d: &str) -> Color {
    let parts: Vec<&str> = d.split('_').collect();
    let (white, primaries, transfer) = match parts.as_slice() {
        ["RGB", white, primaries, _intent, transfer] => (*white, *primaries, *transfer),
        ["Gra", white, _intent, transfer] => (*white, "SRG", *transfer),
        _ => return Color::Other(d.to_string()),
    };
    match (white, primaries, transfer) {
        ("D65", "SRG", "SRG") => Color::Srgb,
        ("D65", "SRG", "Lin") => Color::LinearSrgb,
        ("D65", "SRG", g) if g.starts_with('g') => match g[1..].parse::<f32>() {
            Ok(exponent) => Color::Gamma(exponent),
            Err(_) => Color::Other(d.to_string()),
        },
        ("D65", "DCI", "SRG") => Color::DisplayP3,
        ("D65", "202", "PeQ") => Color::Bt2100Pq,
        ("D65", "202", "HLG") => Color::Bt2100Hlg,
        _ => Color::Other(d.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_descriptions_are_read_field_by_field() {
        assert_eq!(color_of_description("RGB_D65_SRG_Rel_SRG"), Color::Srgb);
        assert_eq!(color_of_description("Gra_D65_Rel_SRG"), Color::Srgb);
        assert_eq!(
            color_of_description("RGB_D65_SRG_Per_Lin"),
            Color::LinearSrgb
        );
        assert_eq!(
            color_of_description("RGB_D65_SRG_Rel_g0.4545500"),
            Color::Gamma(0.45455)
        );
        assert_eq!(
            color_of_description("RGB_D65_DCI_Rel_SRG"),
            Color::DisplayP3
        );
        assert_eq!(color_of_description("RGB_D65_202_Rel_PeQ"), Color::Bt2100Pq);
        assert_eq!(
            color_of_description("RGB_D65_202_Per_HLG"),
            Color::Bt2100Hlg
        );
        assert!(matches!(
            color_of_description("RGB_DCI_DCI_Rel_DCI"),
            Color::Other(_)
        ));
        assert!(matches!(
            color_of_description("XYB_D65_Per"),
            Color::Other(_)
        ));
    }

    #[test]
    fn the_exif_box_offset_is_dropped() {
        assert_eq!(strip_exif_offset(&[0, 0, 0, 2, 9, 9, b'I', b'I']), b"II");
        assert!(strip_exif_offset(&[0, 0, 0, 9]).is_empty());
        assert!(strip_exif_offset(&[0]).is_empty());
    }
}
