//! The codestream encoder: an image header, an optional preview, and any
//! number of frames, each with every option the frame header has.

use super::bits::BitWriter;
use super::features::Features;
use super::frame::{
    Blending, Crop, FrameHeader, FrameType, Passes, Restoration, SKIP_ADAPTIVE_LF_SMOOTHING,
    USE_LF_FRAME, write_toc,
};
use super::header::ImageInfo;
use super::modular::{
    self, Channel, ImageStreams, Layout, ModularCoded, ModularOptions, ModularStream, Transform,
};
use crate::{Error, Result};

/// A frame's options: everything its header can say.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameOptions {
    pub frame_type: FrameType,
    /// The frame covers this part of the canvas (none: all of it).
    pub crop: Option<Crop>,
    pub blending: Blending,
    /// One per extra channel (empty: each like `blending`, replace).
    pub ec_blending: Vec<Blending>,
    /// In ticks, for an animation.
    pub duration: u32,
    pub timecode: u32,
    /// Whether this is the last frame (none: the last one added is).
    pub is_last: Option<bool>,
    /// The reference slot (0..=3) this frame is saved to.
    pub save_as_reference: u32,
    /// Save before the colour transform (none: the decoder's default).
    pub save_before_ct: Option<bool>,
    pub name: String,
    /// The frame is coded at `1 / upsampling` and upsampled (1, 2, 4, 8).
    pub upsampling: u32,
    /// Per extra channel (empty: all 1).
    pub ec_upsampling: Vec<u32>,
    pub restoration: Restoration,
    /// The samples are YCbCr (channels Cb, Y, Cr; Y less 128/255 of the
    /// range, chroma centred on 0), each channel's `jpeg_upsampling` as JPEG
    /// sampling factors: 0 the lowest, 1 twice that both ways, 2 twice
    /// across, 3 twice down — 4:2:0 is `[0, 1, 0]`, 4:2:2 `[0, 2, 0]`.
    pub ycbcr: Option<[u32; 3]>,
    pub passes: Passes,
    /// Groups of `128 << group_size_shift` pixels (modular frames).
    pub group_size_shift: u32,
    /// Take the LF from the preceding LF frame (VarDCT).
    pub use_lf_frame: bool,
    /// The sections' order in the file (none: the natural order).
    pub section_order: Option<Vec<usize>>,
    /// Patches, splines and noise.
    pub features: Features,
    /// Leave out adaptive LF smoothing (VarDCT).
    pub skip_adaptive_lf_smoothing: bool,
}

impl Default for FrameOptions {
    fn default() -> Self {
        FrameOptions {
            frame_type: FrameType::Regular,
            crop: None,
            blending: Blending::default(),
            ec_blending: Vec::new(),
            duration: 0,
            timecode: 0,
            is_last: None,
            save_as_reference: 0,
            save_before_ct: None,
            name: String::new(),
            upsampling: 1,
            ec_upsampling: Vec::new(),
            restoration: Restoration::NONE,
            ycbcr: None,
            passes: Passes::default(),
            group_size_shift: 1,
            use_lf_frame: false,
            section_order: None,
            features: Features::default(),
            skip_adaptive_lf_smoothing: false,
        }
    }
}

/// A modular frame's samples: each channel row by row at its coded size
/// (see [`Encoder::channel_sizes`]), integers in the channel's sample
/// format (a float format's bit patterns).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ModularFrame {
    pub color: Vec<Vec<i32>>,
    pub extra: Vec<Vec<i32>>,
    pub options: ModularOptions,
    /// The global transforms, in order.
    pub transforms: Vec<Transform>,
    /// Transforms applied in each group's own stream, after the global
    /// ones.
    pub group_transforms: Vec<Transform>,
    /// Lossy: squeeze residuals rounded to steps of this (see
    /// `quantize_residuals`).
    pub residual_quantization: Option<f32>,
    /// The LF quantisation factors (for an XYB image: the X, Y, B scales
    /// its integer samples are multiplied by).
    pub lf_quant: Option<[f32; 3]>,
}

impl ModularFrame {
    pub fn new(color: Vec<Vec<i32>>, extra: Vec<Vec<i32>>) -> Self {
        ModularFrame {
            color,
            extra,
            ..Default::default()
        }
    }
}

/// What a frame holds.
#[derive(Clone, Debug, PartialEq)]
pub enum FrameContent {
    Modular(ModularFrame),
}

/// A frame to encode.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub options: FrameOptions,
    pub content: FrameContent,
}

/// The encoder: an image header, then frames.
#[derive(Clone, Debug)]
pub struct Encoder {
    info: ImageInfo,
    preview: Option<Frame>,
    frames: Vec<Frame>,
}

/// A frame ready to write: its header, and its sections in natural order.
struct CodedFrame {
    header: FrameHeader,
    /// The info the header is written against (the preview's size, for the
    /// preview).
    info: ImageInfo,
    sections: Vec<BitWriter>,
    order: Option<Vec<usize>>,
}

impl Encoder {
    pub fn new(info: ImageInfo) -> Result<Self> {
        info.check()?;
        Ok(Encoder {
            info,
            preview: None,
            frames: Vec::new(),
        })
    }

    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// The preview frame (the header must declare a preview).
    pub fn set_preview(&mut self, frame: Frame) -> Result<()> {
        if self.info.preview.is_none() {
            return Err(Error::InvalidInput("no preview declared".into()));
        }
        self.preview = Some(frame);
        Ok(())
    }

    pub fn add_frame(&mut self, frame: Frame) {
        self.frames.push(frame);
    }

    /// The coded (width, height) of each channel of a frame with `options`:
    /// the colour channels, then the extra channels.
    pub fn channel_sizes(&self, options: &FrameOptions) -> Result<Vec<(usize, usize)>> {
        let header = self.frame_header(options, &self.info, true)?;
        Ok(channel_shapes(&header, &self.info)
            .into_iter()
            .map(|(w, h, _)| (w, h))
            .collect())
    }

    fn frame_header(
        &self,
        o: &FrameOptions,
        info: &ImageInfo,
        is_last: bool,
    ) -> Result<FrameHeader> {
        let ne = info.extra_channels.len();
        let ec_blending = if o.ec_blending.is_empty() {
            vec![Blending::default(); ne]
        } else {
            o.ec_blending.clone()
        };
        let ec_upsampling = if o.ec_upsampling.is_empty() {
            vec![1; ne]
        } else {
            o.ec_upsampling.clone()
        };
        let normal = matches!(
            o.frame_type,
            FrameType::Regular | FrameType::SkipProgressive
        );
        let header = FrameHeader {
            frame_type: o.frame_type,
            modular: true,
            flags: (if o.use_lf_frame { USE_LF_FRAME } else { 0 })
                | (if o.skip_adaptive_lf_smoothing {
                    SKIP_ADAPTIVE_LF_SMOOTHING
                } else {
                    0
                })
                | o.features.flags(),
            ycbcr: o.ycbcr,
            upsampling: o.upsampling,
            ec_upsampling,
            group_size_shift: o.group_size_shift,
            x_qm_scale: 3,
            b_qm_scale: 2,
            passes: o.passes.clone(),
            crop: o.crop,
            blending: o.blending,
            ec_blending,
            duration: o.duration,
            timecode: o.timecode,
            is_last: normal && o.is_last.unwrap_or(is_last),
            save_as_reference: o.save_as_reference,
            save_before_ct: o.save_before_ct,
            name: o.name.clone(),
            restoration: o.restoration.clone(),
        };
        if matches!(o.frame_type, FrameType::Lf { .. }) && o.crop.is_some() {
            return Err(Error::InvalidInput("an LF frame cannot be cropped".into()));
        }
        header.check(info)?;
        Ok(header)
    }

    /// The codestream.
    pub fn finish(&self) -> Result<Vec<u8>> {
        if self.frames.is_empty() {
            return Err(Error::InvalidInput("no frames".into()));
        }
        if self.info.preview.is_some() && self.preview.is_none() {
            return Err(Error::InvalidInput(
                "a preview declared but not given".into(),
            ));
        }
        let mut sixteen_bit = true;
        let mut coded = Vec::new();
        if let (Some(p), Some((pw, ph))) = (&self.preview, self.info.preview) {
            let mut info = self.info.clone();
            info.width = pw;
            info.height = ph;
            let header = self.frame_header(&p.options, &info, true)?;
            coded.push(self.code_frame(p, header, info, &mut sixteen_bit)?);
        }
        let n = self.frames.len();
        for (i, f) in self.frames.iter().enumerate() {
            let header = self.frame_header(&f.options, &self.info, i + 1 == n)?;
            coded.push(self.code_frame(f, header, self.info.clone(), &mut sixteen_bit)?);
        }
        if !coded.last().unwrap().header.is_last {
            return Err(Error::InvalidInput(
                "the last frame must be marked last".into(),
            ));
        }

        let mut out = BitWriter::new();
        self.info.write(&mut out, sixteen_bit)?;
        for f in coded {
            f.header.write(&mut out, &f.info)?;
            let sections: Vec<Vec<u8>> = if f.sections.len() == 1 {
                vec![f.sections.into_iter().next().unwrap().finish()]
            } else {
                f.sections.into_iter().map(BitWriter::finish).collect()
            };
            let sizes: Vec<usize> = sections.iter().map(Vec::len).collect();
            write_toc(&mut out, &sizes, f.order.as_deref())?;
            match &f.order {
                None => {
                    for s in &sections {
                        out.append_bytes(s);
                    }
                }
                Some(order) => {
                    for &s in order {
                        out.append_bytes(&sections[s]);
                    }
                }
            }
        }
        Ok(out.finish())
    }

    fn code_frame(
        &self,
        frame: &Frame,
        header: FrameHeader,
        info: ImageInfo,
        sixteen_bit: &mut bool,
    ) -> Result<CodedFrame> {
        match &frame.content {
            FrameContent::Modular(m) => {
                let sections =
                    code_modular(m, &header, &info, &frame.options.features, sixteen_bit)?;
                let single = sections.len() == 1;
                let order = frame.options.section_order.clone();
                if single && order.is_some() {
                    return Err(Error::InvalidInput(
                        "a one-section frame has no section order".into(),
                    ));
                }
                Ok(CodedFrame {
                    header,
                    info,
                    sections,
                    order,
                })
            }
        }
    }
}

/// The colour channels a frame codes: one for gray, unless the samples
/// are XYB or YCbCr.
pub(crate) fn color_channels(header: &FrameHeader, info: &ImageInfo) -> usize {
    if info.color.is_gray() && !info.xyb && header.ycbcr.is_none() {
        1
    } else {
        3
    }
}

/// Each channel's coded (width, height, shift): colour, then extra.
pub(crate) fn channel_shapes(
    header: &FrameHeader,
    info: &ImageInfo,
) -> Vec<(usize, usize, (u32, u32))> {
    let (w, h) = header.size(info);
    let mut out = Vec::new();
    for c in 0..color_channels(header, info) {
        let (sx, sy) = header.chroma_shift(c);
        out.push((w.div_ceil(1 << sx), h.div_ceil(1 << sy), (sx, sy)));
    }
    let (uw, uh) = header.size_upsampled(info);
    let color_shift = header.upsampling.ilog2();
    for i in 0..info.extra_channels.len() {
        let eff = header.ec_upsampling_effective(info, i) as usize;
        let shift = (eff.ilog2()).saturating_sub(color_shift);
        out.push((uw.div_ceil(eff), uh.div_ceil(eff), (shift, shift)));
    }
    out
}

/// A modular frame's sections, in natural order (one, when the frame has
/// one group and one pass).
fn code_modular(
    m: &ModularFrame,
    header: &FrameHeader,
    info: &ImageInfo,
    features: &Features,
    sixteen_bit: &mut bool,
) -> Result<Vec<BitWriter>> {
    let shapes = channel_shapes(header, info);
    let nc = color_channels(header, info);
    if m.color.len() != nc || m.extra.len() != info.extra_channels.len() {
        return Err(Error::InvalidInput(format!(
            "{} colour and {} extra channels for an image of {nc} and {}",
            m.color.len(),
            m.extra.len(),
            info.extra_channels.len()
        )));
    }
    let mut channels = Vec::new();
    for (i, data) in m.color.iter().chain(&m.extra).enumerate() {
        let (w, h, shift) = shapes[i];
        if data.len() != w * h {
            return Err(Error::InvalidInput(format!(
                "channel {i} has {} samples, its {w}x{h} takes {}",
                data.len(),
                w * h
            )));
        }
        channels.push(Channel::new(w, h, Some(shift), data.clone()));
    }
    let bits = info.format.bits();
    *sixteen_bit &= modular::fits_i16(&channels);
    for t in &m.transforms {
        t.apply(&mut channels, bits, m.options.weighted)
            .map_err(Error::InvalidInput)?;
        *sixteen_bit &= modular::fits_i16(&channels);
    }
    if let Some(q) = m.residual_quantization {
        modular::quantize_residuals(&mut channels, q);
    }

    let (gx, gy) = header.groups(info);
    let (lx, ly) = header.lf_groups(info);
    let layout = Layout {
        group_dim: header.group_dim(),
        groups_x: gx,
        groups_y: gy,
        lf_groups_x: lx,
        lf_groups_y: ly,
        pass_shifts: (0..header.passes.num_passes as usize)
            .map(|p| header.passes.shift_bracket(p))
            .collect(),
        num_lf_groups: lx * ly,
    };
    let mut split = ImageStreams::split(channels, m.transforms.clone(), &layout);
    if !m.group_transforms.is_empty() {
        for stream in split.hf.iter_mut().flatten() {
            if stream
                .channels
                .iter()
                .all(|c| c.width == 0 || c.height == 0)
            {
                continue;
            }
            for t in &m.group_transforms {
                t.apply(&mut stream.channels, bits, m.options.weighted)
                    .map_err(|e| Error::InvalidInput(format!("a group transform: {e}")))?;
            }
            *sixteen_bit &= modular::fits_i16(&stream.channels);
            stream.transforms = m.group_transforms.clone();
        }
    }
    let has_channels = !shapes.is_empty();
    let mut streams: Vec<ModularStream> = vec![split.global];
    let num_lf = split.lf.len();
    streams.extend(split.lf);
    for pass in split.hf {
        streams.extend(pass);
    }
    let (fw, fh) = match header.crop {
        Some(c) => (c.width as usize, c.height as usize),
        None => (info.width as usize, info.height as usize),
    };
    let size_limit = (1024 + fw * fh * (nc + info.extra_channels.len()) / 16).min(1 << 22);
    let coded = ModularCoded::new(&streams, &m.options, size_limit);

    // LfGlobal: the LF quantisation, the global tree, the global image.
    let mut global = BitWriter::new();
    features.write(&mut global, info.extra_channels.len())?;
    match m.lf_quant {
        None => global.bit(true), // LfQuant: all_default
        Some(q) => {
            global.bit(false);
            for v in q {
                crate::encode::header::write_f16(&mut global, v * 128.0)?;
            }
        }
    }
    coded.write_global_tree(&mut global);
    if has_channels {
        coded.write_stream(&mut global, 0);
    }
    let mut sections = vec![global];
    for g in 0..num_lf {
        let mut s = BitWriter::new();
        coded.write_stream(&mut s, 1 + g);
        sections.push(s);
    }
    sections.push(BitWriter::new()); // HfGlobal
    let num_groups = layout.num_groups();
    for p in 0..header.passes.num_passes as usize {
        for g in 0..num_groups {
            let mut s = BitWriter::new();
            coded.write_stream(&mut s, 1 + num_lf + p * num_groups + g);
            sections.push(s);
        }
    }
    if num_groups == 1 && header.passes.num_passes == 1 {
        let mut all = BitWriter::new();
        for s in &sections {
            all.append_bits(s);
        }
        return Ok(vec![all]);
    }
    Ok(sections)
}

/// An f32 as the bit pattern of a float sample format.
pub fn float_to_format_bits(v: f32, bits: u32, exponent_bits: u32) -> Option<i32> {
    if bits == 32 && exponent_bits == 8 {
        return Some(v.to_bits() as i32);
    }
    let mant_bits = bits - exponent_bits - 1;
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    let b = v.to_bits();
    let sign = b >> 31;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    let out_sign = sign << (bits - 1);
    if exp == 0 && mant == 0 {
        return Some(out_sign as i32);
    }
    if exp == 0xff {
        return None;
    }
    let e = exp - 127 + bias;
    let drop = 23 - mant_bits;
    if e >= (1 << exponent_bits) - 1 {
        return None;
    }
    if e <= 0 {
        // Subnormal in the target format: exact only.
        let full = mant | 0x80_0000;
        let shift = drop as i32 + 1 - e;
        if shift >= 32 || full & ((1u32 << shift) - 1) != 0 {
            return None;
        }
        return Some((out_sign | (full >> shift)) as i32);
    }
    if mant & ((1 << drop) - 1) != 0 {
        return None;
    }
    Some((out_sign | ((e as u32) << mant_bits) | (mant >> drop)) as i32)
}

/// Helpers for frame options.
impl FrameOptions {
    /// A frame for an animation: shown for `duration` ticks.
    pub fn animation_frame(duration: u32) -> Self {
        FrameOptions {
            duration,
            ..Default::default()
        }
    }
}
