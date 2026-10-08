//! The image header: everything the codestream says about the picture
//! before its first frame.

use super::bits::{BitWriter, Dist};
use super::entropy::pack_signed;
use crate::{Error, Result};

/// How samples are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    /// Unsigned integers of this many bits (1..=31).
    Int(u32),
    /// Floats of `bits` bits, `exponent_bits` of them exponent (IEEE-like:
    /// 32/8 is f32, 16/5 is f16).
    Float { bits: u32, exponent_bits: u32 },
}

impl Default for SampleFormat {
    fn default() -> Self {
        SampleFormat::Int(8)
    }
}

impl SampleFormat {
    pub(crate) fn check(&self) -> Result<()> {
        match *self {
            SampleFormat::Int(b) if (1..=31).contains(&b) => Ok(()),
            SampleFormat::Float {
                bits,
                exponent_bits,
            } if (2..=8).contains(&exponent_bits)
                && (2..=23).contains(&(bits as i32 - exponent_bits as i32 - 1)) =>
            {
                Ok(())
            }
            other => Err(Error::InvalidInput(format!("sample format {other:?}"))),
        }
    }

    /// Bits a sample.
    pub fn bits(&self) -> u32 {
        match *self {
            SampleFormat::Int(b) => b,
            SampleFormat::Float { bits, .. } => bits,
        }
    }

    fn write(&self, w: &mut BitWriter) {
        match *self {
            SampleFormat::Int(b) => {
                w.bit(false);
                w.u32(
                    b,
                    [Dist::Val(8), Dist::Val(10), Dist::Val(12), Dist::Bits(6, 1)],
                );
            }
            SampleFormat::Float {
                bits,
                exponent_bits,
            } => {
                w.bit(true);
                w.u32(
                    bits,
                    [
                        Dist::Val(32),
                        Dist::Val(16),
                        Dist::Val(24),
                        Dist::Bits(6, 1),
                    ],
                );
                w.write(4, exponent_bits - 1);
            }
        }
    }
}

/// A chromaticity (x, y), each in millionths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chromaticity {
    pub x: i32,
    pub y: i32,
}

impl Chromaticity {
    pub fn from_f64(x: f64, y: f64) -> Self {
        Chromaticity {
            x: (x * 1e6).round() as i32,
            y: (y * 1e6).round() as i32,
        }
    }

    fn write(&self, w: &mut BitWriter) {
        let d = [
            Dist::Bits(19, 0),
            Dist::Bits(19, 524288),
            Dist::Bits(20, 1048576),
            Dist::Bits(21, 2097152),
        ];
        w.u32(pack_signed(self.x), d);
        w.u32(pack_signed(self.y), d);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    Rgb,
    Gray,
    /// The XYB space itself (only meaningful for an XYB-encoded picture
    /// asked to stay XYB).
    Xyb,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhitePoint {
    D65,
    /// Equal energy.
    E,
    Dci,
    Custom(Chromaticity),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primaries {
    Srgb,
    Bt2100,
    P3,
    Custom {
        red: Chromaticity,
        green: Chromaticity,
        blue: Chromaticity,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferFunction {
    Bt709,
    Unknown,
    Linear,
    Srgb,
    Pq,
    Dci,
    Hlg,
    /// A pure power: the encoding exponent in ten-millionths (0.45455 →
    /// 4545500).
    Gamma(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderingIntent {
    Perceptual,
    Relative,
    Saturation,
    Absolute,
}

/// A colour encoding the codestream describes by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorEncoding {
    pub color_space: ColorSpace,
    pub white_point: WhitePoint,
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub intent: RenderingIntent,
}

impl ColorEncoding {
    pub const SRGB: ColorEncoding = ColorEncoding {
        color_space: ColorSpace::Rgb,
        white_point: WhitePoint::D65,
        primaries: Primaries::Srgb,
        transfer: TransferFunction::Srgb,
        intent: RenderingIntent::Relative,
    };
    pub const LINEAR_SRGB: ColorEncoding = ColorEncoding {
        transfer: TransferFunction::Linear,
        ..ColorEncoding::SRGB
    };
    pub const GRAY: ColorEncoding = ColorEncoding {
        color_space: ColorSpace::Gray,
        ..ColorEncoding::SRGB
    };
    pub const DISPLAY_P3: ColorEncoding = ColorEncoding {
        primaries: Primaries::P3,
        ..ColorEncoding::SRGB
    };
    pub const BT2100_PQ: ColorEncoding = ColorEncoding {
        primaries: Primaries::Bt2100,
        transfer: TransferFunction::Pq,
        ..ColorEncoding::SRGB
    };
    pub const BT2100_HLG: ColorEncoding = ColorEncoding {
        primaries: Primaries::Bt2100,
        transfer: TransferFunction::Hlg,
        ..ColorEncoding::SRGB
    };
}

/// The picture's colour space: named, or an ICC profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColorSpec {
    Encoding(ColorEncoding),
    /// An ICC profile; `gray` says whether it is a gray profile.
    Icc {
        profile: Vec<u8>,
        gray: bool,
    },
}

impl Default for ColorSpec {
    fn default() -> Self {
        ColorSpec::Encoding(ColorEncoding::SRGB)
    }
}

impl ColorSpec {
    pub(crate) fn is_gray(&self) -> bool {
        match self {
            ColorSpec::Encoding(e) => e.color_space == ColorSpace::Gray,
            ColorSpec::Icc { gray, .. } => *gray,
        }
    }

    fn write(&self, w: &mut BitWriter) {
        let (want_icc, enc) = match self {
            ColorSpec::Encoding(e) => (false, *e),
            ColorSpec::Icc { gray, .. } => (
                true,
                ColorEncoding {
                    color_space: if *gray {
                        ColorSpace::Gray
                    } else {
                        ColorSpace::Rgb
                    },
                    ..ColorEncoding::SRGB
                },
            ),
        };
        if !want_icc && enc == ColorEncoding::SRGB {
            w.bit(true); // all_default
            return;
        }
        w.bit(false);
        w.bit(want_icc);
        let cs = enc.color_space;
        w.enumeration(match cs {
            ColorSpace::Rgb => 0,
            ColorSpace::Gray => 1,
            ColorSpace::Xyb => 2,
            ColorSpace::Unknown => 3,
        });
        if want_icc {
            return;
        }
        if cs != ColorSpace::Xyb {
            w.enumeration(match enc.white_point {
                WhitePoint::D65 => 1,
                WhitePoint::Custom(_) => 2,
                WhitePoint::E => 10,
                WhitePoint::Dci => 11,
            });
            if let WhitePoint::Custom(c) = enc.white_point {
                c.write(w);
            }
        }
        if cs != ColorSpace::Xyb && cs != ColorSpace::Gray {
            w.enumeration(match enc.primaries {
                Primaries::Srgb => 1,
                Primaries::Custom { .. } => 2,
                Primaries::Bt2100 => 9,
                Primaries::P3 => 11,
            });
            if let Primaries::Custom { red, green, blue } = enc.primaries {
                red.write(w);
                green.write(w);
                blue.write(w);
            }
        }
        if cs != ColorSpace::Xyb {
            match enc.transfer {
                TransferFunction::Gamma(g) => {
                    w.bit(true);
                    w.write(24, g);
                }
                tf => {
                    w.bit(false);
                    w.enumeration(match tf {
                        TransferFunction::Bt709 => 1,
                        TransferFunction::Unknown => 2,
                        TransferFunction::Linear => 8,
                        TransferFunction::Srgb => 13,
                        TransferFunction::Pq => 16,
                        TransferFunction::Dci => 17,
                        TransferFunction::Hlg => 18,
                        TransferFunction::Gamma(_) => unreachable!(),
                    });
                }
            }
        }
        w.enumeration(match enc.intent {
            RenderingIntent::Perceptual => 0,
            RenderingIntent::Relative => 1,
            RenderingIntent::Saturation => 2,
            RenderingIntent::Absolute => 3,
        });
    }
}

/// What an extra channel holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExtraChannelKind {
    Alpha {
        /// Premultiplied.
        associated: bool,
    },
    Depth,
    /// A spot colour: its linear RGB and its strength.
    SpotColor([f32; 4]),
    SelectionMask,
    /// The K of CMYK.
    Black,
    /// A colour filter array channel, by its index.
    Cfa(u32),
    Thermal,
    /// One of the reserved kinds, 0..=7.
    Reserved(u32),
    Unknown,
    Optional,
}

/// An extra channel's description.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraChannel {
    pub kind: ExtraChannelKind,
    pub format: SampleFormat,
    /// The channel is subsampled by `1 << dim_shift` each way.
    pub dim_shift: u32,
    pub name: String,
}

impl ExtraChannel {
    pub fn alpha(format: SampleFormat) -> Self {
        ExtraChannel {
            kind: ExtraChannelKind::Alpha { associated: false },
            format,
            dim_shift: 0,
            name: String::new(),
        }
    }

    fn write(&self, w: &mut BitWriter) -> Result<()> {
        let default = ExtraChannel::alpha(SampleFormat::Int(8));
        if *self == default {
            w.bit(true);
            return Ok(());
        }
        w.bit(false);
        w.enumeration(match self.kind {
            ExtraChannelKind::Alpha { .. } => 0,
            ExtraChannelKind::Depth => 1,
            ExtraChannelKind::SpotColor(_) => 2,
            ExtraChannelKind::SelectionMask => 3,
            ExtraChannelKind::Black => 4,
            ExtraChannelKind::Cfa(_) => 5,
            ExtraChannelKind::Thermal => 6,
            ExtraChannelKind::Reserved(r) if r < 8 => 7 + r,
            ExtraChannelKind::Reserved(r) => {
                return Err(Error::InvalidInput(format!("reserved channel kind {r}")));
            }
            ExtraChannelKind::Unknown => 15,
            ExtraChannelKind::Optional => 16,
        });
        self.format.write(w);
        w.u32(
            self.dim_shift,
            [Dist::Val(0), Dist::Val(3), Dist::Val(4), Dist::Bits(3, 1)],
        );
        write_string(w, &self.name)?;
        match self.kind {
            ExtraChannelKind::Alpha { associated } => w.bit(associated),
            ExtraChannelKind::SpotColor(c) => {
                for v in c {
                    write_f16(w, v)?;
                }
            }
            ExtraChannelKind::Cfa(i) => w.u32(
                i,
                [
                    Dist::Val(1),
                    Dist::Bits(2, 0),
                    Dist::Bits(4, 3),
                    Dist::Bits(8, 19),
                ],
            ),
            _ => {}
        }
        Ok(())
    }
}

/// Animation timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Animation {
    /// Ticks per second: `tps_numerator / tps_denominator`.
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    /// 0: forever.
    pub num_loops: u32,
    pub have_timecodes: bool,
}

/// HDR tone mapping hints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneMapping {
    /// The peak, in nits.
    pub intensity_target: f32,
    pub min_nits: f32,
    pub relative_to_max_display: bool,
    pub linear_below: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        ToneMapping {
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
        }
    }
}

/// The XYB inverse the decoder uses (custom transform data).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpsinInverse {
    pub inverse_matrix: [f32; 9],
    pub opsin_biases: [f32; 3],
    pub quant_biases: [f32; 4],
}

impl Default for OpsinInverse {
    fn default() -> Self {
        OpsinInverse {
            inverse_matrix: [
                11.031566901960783,
                -9.866943921568629,
                -0.16462299647058826,
                -3.254147380392157,
                4.418770392156863,
                -0.16462299647058826,
                -3.6588512862745097,
                2.7129230470588235,
                1.9459282392156863,
            ],
            opsin_biases: [-0.0037930732552754493; 3],
            quant_biases: [
                1.0 - 0.05465007330715401,
                1.0 - 0.07005449891748593,
                1.0 - 0.049935103337343655,
                0.145,
            ],
        }
    }
}

/// Custom upsampling kernels (for 2x, 4x, 8x).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct UpsamplingWeights {
    pub weights2: Option<[f32; 15]>,
    pub weights4: Option<Vec<f32>>,
    pub weights8: Option<Vec<f32>>,
}

/// Everything the image header says.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub format: SampleFormat,
    pub color: ColorSpec,
    /// Colour samples are XYB (lossy); otherwise they are in `color`.
    pub xyb: bool,
    /// EXIF orientation, 1..=8.
    pub orientation: u32,
    pub intrinsic_size: Option<(u32, u32)>,
    pub preview: Option<(u32, u32)>,
    pub animation: Option<Animation>,
    pub tone_mapping: ToneMapping,
    pub extra_channels: Vec<ExtraChannel>,
    pub opsin_inverse: Option<OpsinInverse>,
    pub upsampling_weights: UpsamplingWeights,
}

impl ImageInfo {
    pub fn new(width: u32, height: u32) -> Self {
        ImageInfo {
            width,
            height,
            format: SampleFormat::Int(8),
            color: ColorSpec::default(),
            xyb: false,
            orientation: 1,
            intrinsic_size: None,
            preview: None,
            animation: None,
            tone_mapping: ToneMapping::default(),
            extra_channels: Vec::new(),
            opsin_inverse: None,
            upsampling_weights: UpsamplingWeights::default(),
        }
    }

    pub(crate) fn color_channels(&self) -> usize {
        if self.color.is_gray() { 1 } else { 3 }
    }

    /// Validate.
    pub(crate) fn check(&self) -> Result<()> {
        let bad = |m: String| Err(Error::InvalidInput(m));
        if self.width == 0 || self.height == 0 || self.width > 1 << 30 || self.height > 1 << 30 {
            return bad(format!("{}x{}", self.width, self.height));
        }
        if u64::from(self.width) * u64::from(self.height) > 1 << 40 {
            return bad(format!(
                "{}x{} is over 2^40 pixels",
                self.width, self.height
            ));
        }
        if !(1..=8).contains(&self.orientation) {
            return bad(format!("orientation {}", self.orientation));
        }
        self.format.check()?;
        for ec in &self.extra_channels {
            ec.format.check()?;
            if ec.dim_shift > 8 {
                return bad(format!("dim_shift {}", ec.dim_shift));
            }
        }
        if let Some((w, h)) = self.preview
            && (w == 0 || h == 0 || w > 4096 || h > 4096)
        {
            return bad(format!("a {w}x{h} preview"));
        }
        if let Some(a) = self.animation
            && (a.tps_numerator == 0 || a.tps_denominator == 0)
        {
            return bad("an animation of zero ticks a second".into());
        }
        let t = self.tone_mapping;
        if t.intensity_target <= 0.0
            || t.min_nits < 0.0
            || t.min_nits > t.intensity_target
            || t.linear_below < 0.0
            || (t.relative_to_max_display && t.linear_below > 1.0)
        {
            return bad(format!("tone mapping {t:?}"));
        }
        Ok(())
    }

    /// The signature, the size, the metadata, the transform data, any ICC
    /// profile; padded to a byte. `modular_16bit` says whether every
    /// modular sample of every frame fits 16 bits.
    pub(crate) fn write(&self, w: &mut BitWriter, modular_16bit: bool) -> Result<()> {
        self.check()?;
        w.write(8, 0xff);
        w.write(8, 0x0a);
        write_size(w, self.width, self.height);

        let extra_fields = self.orientation != 1
            || self.intrinsic_size.is_some()
            || self.preview.is_some()
            || self.animation.is_some()
            || self.tone_mapping != ToneMapping::default();
        let all_default = !extra_fields
            && self.format == SampleFormat::Int(8)
            && modular_16bit
            && self.extra_channels.is_empty()
            && self.xyb
            && self.color == ColorSpec::default();
        w.bit(all_default);
        if !all_default {
            w.bit(extra_fields);
            if extra_fields {
                w.write(3, self.orientation - 1);
                w.bit(self.intrinsic_size.is_some());
                if let Some((iw, ih)) = self.intrinsic_size {
                    write_size(w, iw, ih);
                }
                w.bit(self.preview.is_some());
                if let Some((pw, ph)) = self.preview {
                    write_preview_size(w, pw, ph);
                }
                w.bit(self.animation.is_some());
                if let Some(a) = self.animation {
                    w.u32(
                        a.tps_numerator,
                        [
                            Dist::Val(100),
                            Dist::Val(1000),
                            Dist::Bits(10, 1),
                            Dist::Bits(30, 1),
                        ],
                    );
                    w.u32(
                        a.tps_denominator,
                        [
                            Dist::Val(1),
                            Dist::Val(1001),
                            Dist::Bits(8, 1),
                            Dist::Bits(10, 1),
                        ],
                    );
                    w.u32(
                        a.num_loops,
                        [
                            Dist::Val(0),
                            Dist::Bits(3, 0),
                            Dist::Bits(16, 0),
                            Dist::Bits(32, 0),
                        ],
                    );
                    w.bit(a.have_timecodes);
                }
            }
            self.format.write(w);
            w.bit(modular_16bit);
            w.u32(
                self.extra_channels.len() as u32,
                [
                    Dist::Val(0),
                    Dist::Val(1),
                    Dist::Bits(4, 2),
                    Dist::Bits(12, 1),
                ],
            );
            for ec in &self.extra_channels {
                ec.write(w)?;
            }
            w.bit(self.xyb);
            self.color.write(w);
            if extra_fields {
                let t = self.tone_mapping;
                if t == ToneMapping::default() {
                    w.bit(true);
                } else {
                    w.bit(false);
                    write_f16(w, t.intensity_target)?;
                    write_f16(w, t.min_nits)?;
                    w.bit(t.relative_to_max_display);
                    write_f16(w, t.linear_below)?;
                }
            }
            w.u64_zero(); // extensions
        }

        // CustomTransformData.
        let uw = &self.upsampling_weights;
        let custom_opsin = self.xyb
            && self
                .opsin_inverse
                .is_some_and(|o| o != OpsinInverse::default());
        let mask = u32::from(uw.weights2.is_some())
            | u32::from(uw.weights4.is_some()) << 1
            | u32::from(uw.weights8.is_some()) << 2;
        if !custom_opsin && mask == 0 {
            w.bit(true);
        } else {
            w.bit(false);
            if self.xyb {
                match self.opsin_inverse.filter(|_| custom_opsin) {
                    None => w.bit(true),
                    Some(o) => {
                        w.bit(false);
                        for v in o
                            .inverse_matrix
                            .iter()
                            .chain(&o.opsin_biases)
                            .chain(&o.quant_biases)
                        {
                            write_f16(w, *v)?;
                        }
                    }
                }
            }
            w.write(3, mask);
            if let Some(k) = &uw.weights2 {
                for &v in k {
                    write_f16(w, v)?;
                }
            }
            for (k, n) in [(&uw.weights4, 55), (&uw.weights8, 210)] {
                if let Some(k) = k {
                    if k.len() != n {
                        return Err(Error::InvalidInput(format!(
                            "{} upsampling weights, not {n}",
                            k.len()
                        )));
                    }
                    for &v in k {
                        write_f16(w, v)?;
                    }
                }
            }
        }

        if let ColorSpec::Icc { profile, .. } = &self.color {
            super::icc::write(w, profile)?;
        }
        w.pad_to_byte();
        Ok(())
    }
}

fn dimension(w: &mut BitWriter, v: u32) {
    w.u32(
        v - 1,
        [
            Dist::Bits(9, 0),
            Dist::Bits(13, 0),
            Dist::Bits(18, 0),
            Dist::Bits(30, 0),
        ],
    );
}

/// The decoder's width for a height and an aspect ratio code.
fn ratio_width(h: u32, ratio: u32) -> u64 {
    let h = u64::from(h);
    match ratio {
        1 => h,
        2 => h * 12 / 10,
        3 => h * 4 / 3,
        4 => h * 3 / 2,
        5 => h * 16 / 9,
        6 => h * 5 / 4,
        7 => h * 2,
        _ => unreachable!(),
    }
}

fn write_size(w: &mut BitWriter, width: u32, height: u32) {
    let ratio = (1..=7)
        .find(|&r| ratio_width(height, r) == u64::from(width))
        .unwrap_or(0);
    let small = height % 8 == 0 && width % 8 == 0 && height <= 256 && width <= 256;
    w.bit(small);
    if small {
        w.write(5, height / 8 - 1);
    } else {
        dimension(w, height);
    }
    w.write(3, ratio);
    if ratio == 0 {
        if small {
            w.write(5, width / 8 - 1);
        } else {
            dimension(w, width);
        }
    }
}

fn write_preview_size(w: &mut BitWriter, width: u32, height: u32) {
    let ratio = (1..=7)
        .find(|&r| ratio_width(height, r) == u64::from(width))
        .unwrap_or(0);
    let div8 = height % 8 == 0 && width % 8 == 0;
    let div8_dist = [
        Dist::Val(16),
        Dist::Val(32),
        Dist::Bits(5, 1),
        Dist::Bits(9, 33),
    ];
    let plain = [
        Dist::Bits(6, 0),
        Dist::Bits(8, 64),
        Dist::Bits(10, 320),
        Dist::Bits(12, 1344),
    ];
    w.bit(div8);
    if div8 {
        w.u32(height / 8, div8_dist);
    } else {
        w.u32(height - 1, plain);
    }
    w.write(3, ratio);
    if ratio == 0 {
        if div8 {
            w.u32(width / 8, div8_dist);
        } else {
            w.u32(width - 1, plain);
        }
    }
}

pub(crate) fn write_string(w: &mut BitWriter, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    if bytes.len() > 1071 {
        return Err(Error::InvalidInput(format!(
            "a name of {} bytes",
            bytes.len()
        )));
    }
    w.u32(
        bytes.len() as u32,
        [
            Dist::Val(0),
            Dist::Bits(4, 0),
            Dist::Bits(5, 16),
            Dist::Bits(10, 48),
        ],
    );
    for &b in bytes {
        w.write(8, u32::from(b));
    }
    Ok(())
}

/// `v` as an IEEE half, rounded to nearest.
pub(crate) fn f16_bits(v: f32) -> Option<u16> {
    if !v.is_finite() {
        return None;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 && mant == 0 {
        return Some(sign);
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return None;
    }
    if e <= 0 {
        // Subnormal half.
        if e < -10 {
            return Some(sign);
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && h & 1 == 1) {
            h += 1;
        }
        return Some(sign | h as u16);
    }
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    if h >= 0x7c00 {
        return None;
    }
    Some(sign | h as u16)
}

pub(crate) fn write_f16(w: &mut BitWriter, v: f32) -> Result<()> {
    let h =
        f16_bits(v).ok_or_else(|| Error::InvalidInput(format!("{v} does not fit a half float")))?;
    w.write(16, u32::from(h));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves() {
        assert_eq!(f16_bits(1.0), Some(0x3c00));
        assert_eq!(f16_bits(-2.0), Some(0xc000));
        assert_eq!(f16_bits(255.0), Some(0x5bf8));
        assert_eq!(f16_bits(65504.0), Some(0x7bff));
        assert_eq!(f16_bits(70000.0), None);
        assert_eq!(f16_bits(5.960464e-8), Some(1));
        assert_eq!(f16_bits(0.0), Some(0));
    }
}
