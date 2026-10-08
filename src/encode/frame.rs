//! A frame's header, its table of contents, and its geometry.

use super::bits::{BitWriter, Dist};
use super::entropy::{EntropyCode, EntropyOptions, Stream, Token, pack_signed};
use super::header::{ImageInfo, write_f16, write_string};
use super::icc::write_u64;
use crate::{Error, Result};

/// What a frame is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Shown (when last, or with a duration).
    #[default]
    Regular,
    /// The LF (1/8 resolution, `level` times over) of later frames.
    Lf { level: u32 },
    /// Only saved, for later frames to blend from or reference.
    ReferenceOnly,
    /// Regular, but not a progressive step.
    SkipProgressive,
}

impl FrameType {
    fn code(self) -> u32 {
        match self {
            FrameType::Regular => 0,
            FrameType::Lf { .. } => 1,
            FrameType::ReferenceOnly => 2,
            FrameType::SkipProgressive => 3,
        }
    }

    fn is_normal(self) -> bool {
        matches!(self, FrameType::Regular | FrameType::SkipProgressive)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Replace,
    Add,
    /// Alpha blending over the source.
    Blend,
    AlphaWeightedAdd,
    Multiply,
}

impl BlendMode {
    fn code(self) -> u32 {
        match self {
            BlendMode::Replace => 0,
            BlendMode::Add => 1,
            BlendMode::Blend => 2,
            BlendMode::AlphaWeightedAdd => 3,
            BlendMode::Multiply => 4,
        }
    }

    fn uses_alpha(self) -> bool {
        matches!(self, BlendMode::Blend | BlendMode::AlphaWeightedAdd)
    }
}

/// How a frame (or one of its extra channels) is blended onto the canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Blending {
    pub mode: BlendMode,
    /// The extra channel that is the alpha.
    pub alpha_channel: u32,
    pub clamp: bool,
    /// The saved reference frame (0..=3) blended onto.
    pub source: u32,
}

impl Blending {
    fn write(&self, w: &mut BitWriter, extra_channels: usize, full_frame: bool) {
        w.u32(
            self.mode.code(),
            [Dist::Val(0), Dist::Val(1), Dist::Val(2), Dist::Bits(2, 3)],
        );
        let alpha = extra_channels > 0 && self.mode.uses_alpha();
        if alpha {
            w.u32(
                self.alpha_channel,
                [Dist::Val(0), Dist::Val(1), Dist::Val(2), Dist::Bits(3, 3)],
            );
        }
        if alpha || self.mode == BlendMode::Multiply {
            w.bit(self.clamp);
        }
        if !(full_frame && self.mode == BlendMode::Replace) {
            w.u32(
                self.source,
                [Dist::Val(0), Dist::Val(1), Dist::Val(2), Dist::Val(3)],
            );
        }
    }
}

/// Progressive passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Passes {
    pub num_passes: u32,
    /// Per pass but the last: the coefficient bits it leaves out.
    pub shift: Vec<u32>,
    /// Downsampling factors (2, 4, 8) reached ...
    pub downsample: Vec<u32>,
    /// ... by the end of these passes.
    pub last_pass: Vec<u32>,
}

impl Default for Passes {
    fn default() -> Self {
        Passes {
            num_passes: 1,
            shift: Vec::new(),
            downsample: Vec::new(),
            last_pass: Vec::new(),
        }
    }
}

impl Passes {
    fn write(&self, w: &mut BitWriter) {
        w.u32(
            self.num_passes,
            [Dist::Val(1), Dist::Val(2), Dist::Val(3), Dist::Bits(3, 4)],
        );
        if self.num_passes == 1 {
            return;
        }
        w.u32(
            self.downsample.len() as u32,
            [Dist::Val(0), Dist::Val(1), Dist::Val(2), Dist::Bits(1, 3)],
        );
        for &s in &self.shift {
            w.write(2, s);
        }
        for &d in &self.downsample {
            w.u32(d, [Dist::Val(1), Dist::Val(2), Dist::Val(4), Dist::Val(8)]);
        }
        for &l in &self.last_pass {
            w.u32(
                l,
                [Dist::Val(0), Dist::Val(1), Dist::Val(2), Dist::Bits(3, 0)],
            );
        }
    }

    fn check(&self) -> Result<()> {
        let n = self.num_passes;
        let ok = (1..=11).contains(&n)
            && self.shift.len() == (n as usize).saturating_sub(1)
            && self.shift.iter().all(|&s| s < 4)
            && self.downsample.len() == self.last_pass.len()
            && (self.downsample.len() as u32) < n.max(1)
            && self.downsample.len() <= 4
            && self.downsample.iter().all(|d| [1, 2, 4, 8].contains(d))
            && self.downsample.windows(2).all(|p| p[1] < p[0])
            && self.last_pass.windows(2).all(|p| p[1] > p[0])
            && self.last_pass.iter().all(|&l| l < n && l < 8);
        if ok || n == 1 {
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("passes {self:?}")))
        }
    }

    /// The decoder's range of channel shifts a pass's sections carry.
    pub(crate) fn shift_bracket(&self, pass: usize) -> (u32, u32) {
        let mut max_shift = 3u32;
        let mut min_shift = 3u32;
        for i in 0..=pass {
            max_shift = min_shift;
            let mut found = false;
            for j in 0..self.downsample.len() {
                if i == self.last_pass[j] as usize {
                    min_shift = self.downsample[j].ilog2();
                    found = true;
                }
            }
            if i + 1 == self.num_passes as usize {
                min_shift = 0;
                found = true;
            }
            if !found {
                min_shift = max_shift;
            }
        }
        if min_shift < max_shift {
            (min_shift, max_shift - 1)
        } else {
            (1, 0)
        }
    }
}

/// The restoration filters.
#[derive(Clone, Debug, PartialEq)]
pub struct Restoration {
    /// Gaborish smoothing, with custom weights (x1, x2, y1, y2, b1, b2) or
    /// the default ones.
    pub gaborish: Option<Option<[f32; 6]>>,
    /// Edge-preserving filter iterations, 0..=3.
    pub epf_iters: u32,
    pub epf_sharp_lut: Option<[f32; 8]>,
    /// Channel scales, then the pass-1 and pass-2 zero flushes.
    pub epf_weights: Option<([f32; 3], f32, f32)>,
    /// Quant multiplier (VarDCT), pass-0 and pass-2 sigma scales, border SAD
    /// multiplier.
    pub epf_sigma: Option<(f32, f32, f32, f32)>,
    /// The sigma a modular frame filters with.
    pub epf_sigma_for_modular: f32,
}

impl Restoration {
    /// No filtering.
    pub const NONE: Restoration = Restoration {
        gaborish: None,
        epf_iters: 0,
        epf_sharp_lut: None,
        epf_weights: None,
        epf_sigma: None,
        epf_sigma_for_modular: 1.0,
    };

    /// The decoder's defaults (Gaborish on, two EPF iterations).
    pub const DEFAULT: Restoration = Restoration {
        gaborish: Some(None),
        epf_iters: 2,
        epf_sharp_lut: None,
        epf_weights: None,
        epf_sigma: None,
        epf_sigma_for_modular: 1.0,
    };

    fn write(&self, w: &mut BitWriter, modular: bool) -> Result<()> {
        if *self == Restoration::DEFAULT {
            w.bit(true);
            return Ok(());
        }
        w.bit(false);
        w.bit(self.gaborish.is_some());
        if let Some(custom) = self.gaborish {
            w.bit(custom.is_some());
            if let Some(ws) = custom {
                for v in ws {
                    write_f16(w, v)?;
                }
            }
        }
        if self.epf_iters > 3 {
            return Err(Error::InvalidInput(format!(
                "{} EPF iterations",
                self.epf_iters
            )));
        }
        w.write(2, self.epf_iters);
        if self.epf_iters > 0 {
            if !modular {
                w.bit(self.epf_sharp_lut.is_some());
                if let Some(lut) = self.epf_sharp_lut {
                    for v in lut {
                        write_f16(w, v)?;
                    }
                }
            }
            w.bit(self.epf_weights.is_some());
            if let Some((scale, z1, z2)) = self.epf_weights {
                for v in scale {
                    write_f16(w, v)?;
                }
                write_f16(w, z1)?;
                write_f16(w, z2)?;
            }
            w.bit(self.epf_sigma.is_some());
            if let Some((quant_mul, s0, s2, border)) = self.epf_sigma {
                if !modular {
                    write_f16(w, quant_mul)?;
                }
                write_f16(w, s0)?;
                write_f16(w, s2)?;
                write_f16(w, border)?;
            }
            if modular {
                write_f16(w, self.epf_sigma_for_modular)?;
            }
        }
        w.u64_zero();
        Ok(())
    }
}

/// A placed rectangle of the canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crop {
    pub x0: i32,
    pub y0: i32,
    pub width: u32,
    pub height: u32,
}

/// Feature flags.
pub(crate) const ENABLE_NOISE: u64 = 1;
pub(crate) const ENABLE_PATCHES: u64 = 2;
pub(crate) const ENABLE_SPLINES: u64 = 0x10;
pub(crate) const USE_LF_FRAME: u64 = 0x20;
pub(crate) const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;

/// A frame's header, as written.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameHeader {
    pub frame_type: FrameType,
    pub modular: bool,
    pub flags: u64,
    /// `jpeg_upsampling` per channel (YCbCr), when the samples are YCbCr.
    pub ycbcr: Option<[u32; 3]>,
    pub upsampling: u32,
    pub ec_upsampling: Vec<u32>,
    pub group_size_shift: u32,
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    pub passes: Passes,
    pub crop: Option<Crop>,
    pub blending: Blending,
    pub ec_blending: Vec<Blending>,
    pub duration: u32,
    pub timecode: u32,
    pub is_last: bool,
    pub save_as_reference: u32,
    pub save_before_ct: Option<bool>,
    pub name: String,
    pub restoration: Restoration,
}

impl FrameHeader {
    pub(crate) fn full_frame(&self, info: &ImageInfo) -> bool {
        match self.crop {
            None => true,
            Some(c) => {
                c.x0 <= 0
                    && c.y0 <= 0
                    && i64::from(c.width) + i64::from(c.x0) >= i64::from(info.width)
                    && i64::from(c.height) + i64::from(c.y0) >= i64::from(info.height)
            }
        }
    }

    /// The decoder's default for `save_before_ct`, and whether it is sent.
    fn save_before_ct_sent(&self, info: &ImageInfo) -> (bool, bool) {
        let lf = matches!(self.frame_type, FrameType::Lf { .. });
        let can_be_referenced =
            !self.is_last && !lf && (self.duration == 0 || self.save_as_reference != 0);
        let def_false = can_be_referenced
            && self.blending.mode == BlendMode::Replace
            && self.full_frame(info)
            && self.frame_type.is_normal();
        (self.frame_type == FrameType::ReferenceOnly || def_false, lf)
    }

    pub(crate) fn check(&self, info: &ImageInfo) -> Result<()> {
        let bad = |m: String| Err(Error::InvalidInput(m));
        let ne = info.extra_channels.len();
        if self.ec_upsampling.len() != ne || self.ec_blending.len() != ne {
            return bad("one upsampling and blending per extra channel".into());
        }
        if ![1, 2, 4, 8].contains(&self.upsampling)
            || self.ec_upsampling.iter().any(|u| ![1, 2, 4, 8].contains(u))
        {
            return bad("upsampling of 1, 2, 4 or 8".into());
        }
        if self.upsampling > 1 {
            for (ec, &u) in info.extra_channels.iter().zip(&self.ec_upsampling) {
                let eff = u << ec.dim_shift;
                if eff < self.upsampling || eff > 8 {
                    return bad(format!(
                        "extra channel upsampling {u} with dim_shift {} under colour upsampling {}",
                        ec.dim_shift, self.upsampling
                    ));
                }
            }
        }
        if self.ycbcr.is_some() && info.xyb {
            return bad("YCbCr samples in an XYB image".into());
        }
        if let Some(j) = self.ycbcr
            && j.iter().any(|&v| v > 3)
        {
            return bad(format!("jpeg upsampling {j:?}"));
        }
        if let FrameType::Lf { level } = self.frame_type
            && !(1..=4).contains(&level)
        {
            return bad(format!("LF level {level}"));
        }
        if self.save_as_reference > 3 || self.blending.source > 3 {
            return bad("reference slots are 0..=3".into());
        }
        for b in std::iter::once(&self.blending).chain(&self.ec_blending) {
            if ne > 0 && b.mode.uses_alpha() && b.alpha_channel as usize >= ne {
                return bad(format!("alpha channel {}", b.alpha_channel));
            }
        }
        self.passes.check()?;
        if self.group_size_shift > 3 {
            return bad(format!("group size shift {}", self.group_size_shift));
        }
        Ok(())
    }

    pub(crate) fn write(&self, w: &mut BitWriter, info: &ImageInfo) -> Result<()> {
        self.check(info)?;
        let ne = info.extra_channels.len();
        w.bit(false); // all_default
        w.write(2, self.frame_type.code());
        w.write(1, u32::from(self.modular));
        write_u64(w, self.flags);
        if !info.xyb {
            w.bit(self.ycbcr.is_some());
        }
        let lf_frame = self.flags & USE_LF_FRAME != 0;
        if let Some(j) = self.ycbcr
            && !lf_frame
        {
            for v in j {
                w.write(2, v);
            }
        }
        if !lf_frame {
            let up = [Dist::Val(1), Dist::Val(2), Dist::Val(4), Dist::Val(8)];
            w.u32(self.upsampling, up);
            for &u in &self.ec_upsampling {
                w.u32(u, up);
            }
        }
        if self.modular {
            w.write(2, self.group_size_shift);
        }
        if !self.modular && info.xyb {
            w.write(3, self.x_qm_scale);
            w.write(3, self.b_qm_scale);
        }
        if self.frame_type != FrameType::ReferenceOnly {
            self.passes.write(w);
        }
        if let FrameType::Lf { level } = self.frame_type {
            w.u32(
                level,
                [Dist::Val(1), Dist::Val(2), Dist::Val(3), Dist::Val(4)],
            );
        } else {
            w.bit(self.crop.is_some());
        }
        if let Some(c) = self.crop {
            let d = [
                Dist::Bits(8, 0),
                Dist::Bits(11, 256),
                Dist::Bits(14, 2304),
                Dist::Bits(30, 18688),
            ];
            if self.frame_type != FrameType::ReferenceOnly {
                w.u32(pack_signed(c.x0), d);
                w.u32(pack_signed(c.y0), d);
            }
            w.u32(c.width, d);
            w.u32(c.height, d);
        }
        let full_frame = self.full_frame(info);
        if self.frame_type.is_normal() {
            self.blending.write(w, ne, full_frame);
            for b in &self.ec_blending {
                b.write(w, ne, full_frame);
            }
            if info.animation.is_some() {
                w.u32(
                    self.duration,
                    [
                        Dist::Val(0),
                        Dist::Val(1),
                        Dist::Bits(8, 0),
                        Dist::Bits(32, 0),
                    ],
                );
            }
            if info.animation.is_some_and(|a| a.have_timecodes) {
                w.write(32, self.timecode);
            }
            w.bit(self.is_last);
        }
        let lf = matches!(self.frame_type, FrameType::Lf { .. });
        if !lf && !self.is_last {
            w.write(2, self.save_as_reference);
        }
        let (sent, default) = self.save_before_ct_sent(info);
        if sent {
            w.bit(self.save_before_ct.unwrap_or(default));
        }
        write_string(w, &self.name)?;
        self.restoration.write(w, self.modular)?;
        w.u64_zero();
        Ok(())
    }

    /// The frame's size before upsampling: the crop (or image), over the LF
    /// level, over the upsampling.
    pub(crate) fn size_upsampled(&self, info: &ImageInfo) -> (usize, usize) {
        let (w, h) = match self.crop {
            Some(c) => (c.width as usize, c.height as usize),
            None => (info.width as usize, info.height as usize),
        };
        let level = match self.frame_type {
            FrameType::Lf { level } => level,
            _ => 0,
        };
        (w.div_ceil(1 << (3 * level)), h.div_ceil(1 << (3 * level)))
    }

    pub(crate) fn size(&self, info: &ImageInfo) -> (usize, usize) {
        let (w, h) = self.size_upsampled(info);
        let u = self.upsampling as usize;
        (w.div_ceil(u), h.div_ceil(u))
    }

    /// The largest chroma shifts.
    pub(crate) fn max_shifts(&self) -> (u32, u32) {
        const H: [u32; 4] = [0, 1, 1, 0];
        const V: [u32; 4] = [0, 1, 0, 1];
        match self.ycbcr {
            Some(j) => (
                j.iter().map(|&v| H[v as usize]).max().unwrap(),
                j.iter().map(|&v| V[v as usize]).max().unwrap(),
            ),
            None => (0, 0),
        }
    }

    /// Colour channel `c`'s (horizontal, vertical) shift.
    pub(crate) fn chroma_shift(&self, c: usize) -> (u32, u32) {
        const H: [u32; 4] = [0, 1, 1, 0];
        const V: [u32; 4] = [0, 1, 0, 1];
        let (mh, mv) = self.max_shifts();
        match self.ycbcr {
            Some(j) => (mh - H[j[c] as usize], mv - V[j[c] as usize]),
            None => (0, 0),
        }
    }

    pub(crate) fn group_dim(&self) -> usize {
        if self.modular {
            128 << self.group_size_shift
        } else {
            256
        }
    }

    /// The frame's size in 8x8 blocks.
    pub(crate) fn size_blocks(&self, info: &ImageInfo) -> (usize, usize) {
        let (w, h) = self.size(info);
        let (mh, mv) = self.max_shifts();
        (w.div_ceil(8 << mh) << mh, h.div_ceil(8 << mv) << mv)
    }

    pub(crate) fn groups(&self, info: &ImageInfo) -> (usize, usize) {
        let (w, h) = self.size(info);
        let gd = self.group_dim();
        (w.div_ceil(gd), h.div_ceil(gd))
    }

    pub(crate) fn lf_groups(&self, info: &ImageInfo) -> (usize, usize) {
        let (bw, bh) = self.size_blocks(info);
        let gd = self.group_dim();
        (bw.div_ceil(gd), bh.div_ceil(gd))
    }

    /// Extra channel `i`'s effective upsampling.
    pub(crate) fn ec_upsampling_effective(&self, info: &ImageInfo, i: usize) -> u32 {
        if self.upsampling > 1 {
            self.ec_upsampling[i] << info.extra_channels[i].dim_shift
        } else {
            self.ec_upsampling[i]
        }
    }
}

/// The table of contents. `order`, when given, is the sections' order in
/// the file (`order[k]`: the section at file position `k`).
pub(crate) fn write_toc(w: &mut BitWriter, sizes: &[usize], order: Option<&[usize]>) -> Result<()> {
    let n = sizes.len();
    w.bit(order.is_some());
    if let Some(order) = order {
        if order.len() != n {
            return Err(Error::InvalidInput(
                "a section order of the wrong length".into(),
            ));
        }
        // The decoder's permutation maps a section to its file position.
        let mut position = vec![usize::MAX; n];
        for (k, &s) in order.iter().enumerate() {
            if s >= n || position[s] != usize::MAX {
                return Err(Error::InvalidInput(
                    "a section order that is not a permutation".into(),
                ));
            }
            position[s] = k;
        }
        write_permutation(w, &position);
    }
    w.pad_to_byte();
    let d = [
        Dist::Bits(10, 0),
        Dist::Bits(14, 1024),
        Dist::Bits(22, 17408),
        Dist::Bits(30, 4211712),
    ];
    let max = 4211712 + (1 << 30) - 1;
    // The sections in file order.
    for k in 0..n {
        let s = order.map_or(k, |o| o[k]);
        if sizes[s] > max {
            return Err(Error::InvalidInput(format!(
                "a section of {} bytes",
                sizes[s]
            )));
        }
        w.u32(sizes[s] as u32, d);
    }
    w.pad_to_byte();
    Ok(())
}

/// The decoder's permutation context.
fn permutation_context(x: u32) -> u32 {
    super::entropy::ceil_log2(x + 1).min(7)
}

/// A permutation as its Lehmer code, entropy coded in 8 contexts.
pub(crate) fn write_permutation(w: &mut BitWriter, permutation: &[usize]) {
    let n = permutation.len();
    // Lehmer code: per position, how many remaining values are smaller.
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut code = Vec::with_capacity(n);
    for &p in permutation {
        let i = remaining.iter().position(|&r| r == p).unwrap();
        code.push(i as u32);
        remaining.remove(i);
    }
    let end = code.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
    let mut tokens = vec![Token::new(permutation_context(n as u32), end as u32)];
    let mut prev = 0;
    for &c in &code[..end] {
        tokens.push(Token::new(permutation_context(prev), c));
        prev = c;
    }
    let (code, streams) = EntropyCode::build(
        8,
        vec![Stream::new(tokens)],
        &EntropyOptions::default(),
        true,
    );
    code.write_all(w, &streams[0]);
}
