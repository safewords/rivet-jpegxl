//! The frame features sent in LfGlobal: patches (copies from a saved
//! frame), splines, and noise.

use super::bits::BitWriter;
use super::entropy::{EntropyCode, EntropyOptions, Stream, Token, pack_signed};
use crate::{Error, Result};

/// Synthesised noise: its strength at eight intensities, each in
/// 1/1024ths (0..1024).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Noise {
    pub lut: [u32; 8],
}

/// How a patch is blended onto the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchBlendMode {
    None,
    Replace,
    Add,
    Multiply,
    BlendAbove,
    BlendBelow,
    AlphaWeightedAddAbove,
    AlphaWeightedAddBelow,
}

impl PatchBlendMode {
    fn code(self) -> u32 {
        self as u32
    }

    fn uses_alpha(self) -> bool {
        matches!(
            self,
            PatchBlendMode::BlendAbove
                | PatchBlendMode::BlendBelow
                | PatchBlendMode::AlphaWeightedAddAbove
                | PatchBlendMode::AlphaWeightedAddBelow
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchBlending {
    pub mode: PatchBlendMode,
    pub alpha_channel: u32,
    pub clamp: bool,
}

/// Where a patch is placed, and how blended: one blending for the colour
/// channels, then one per extra channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchPlacement {
    pub x: u32,
    pub y: u32,
    pub blending: Vec<PatchBlending>,
}

/// A rectangle of a saved reference frame (saved before the colour
/// transform), placed one or more times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patch {
    pub reference: u32,
    pub x0: u32,
    pub y0: u32,
    pub width: u32,
    pub height: u32,
    pub placements: Vec<PatchPlacement>,
}

/// A spline as the codestream quantises it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedSpline {
    /// The starting point.
    pub start: (i32, i32),
    /// The control points after the start, as second differences.
    pub control_deltas: Vec<(i32, i32)>,
    /// The colour (X, Y, B) along the spline, 32 DCT coefficients each.
    pub color_dct: [[i32; 32]; 3],
    /// The thickness along the spline.
    pub sigma_dct: [i32; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Splines {
    pub quantization_adjustment: i32,
    pub splines: Vec<QuantizedSpline>,
}

/// A frame's LfGlobal features.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Features {
    pub patches: Vec<Patch>,
    pub splines: Option<Splines>,
    pub noise: Option<Noise>,
}

impl Features {
    pub(crate) fn flags(&self) -> u64 {
        use super::frame::{ENABLE_NOISE, ENABLE_PATCHES, ENABLE_SPLINES};
        (if self.noise.is_some() {
            ENABLE_NOISE
        } else {
            0
        }) | (if self.patches.is_empty() {
            0
        } else {
            ENABLE_PATCHES
        }) | (if self.splines.is_some() {
            ENABLE_SPLINES
        } else {
            0
        })
    }

    /// The features, in LfGlobal's order: patches, splines, noise.
    pub(crate) fn write(&self, w: &mut BitWriter, extra_channels: usize) -> Result<()> {
        if !self.patches.is_empty() {
            write_patches(w, &self.patches, extra_channels)?;
        }
        if let Some(s) = &self.splines {
            write_splines(w, s)?;
        }
        if let Some(n) = &self.noise {
            for &v in &n.lut {
                if v >= 1024 {
                    return Err(Error::InvalidInput(format!("noise strength {v}")));
                }
                w.write(10, v);
            }
        }
        Ok(())
    }
}

fn write_patches(w: &mut BitWriter, patches: &[Patch], extra_channels: usize) -> Result<()> {
    const NUM_REF_PATCH: u32 = 0;
    const REFERENCE_FRAME: u32 = 1;
    const PATCH_SIZE: u32 = 2;
    const REFERENCE_POSITION: u32 = 3;
    const POSITION: u32 = 4;
    const BLEND_MODE: u32 = 5;
    const OFFSET: u32 = 6;
    const COUNT: u32 = 7;
    const ALPHA_CHANNEL: u32 = 8;
    const CLAMP: u32 = 9;
    let stride = extra_channels + 1;
    let mut t = vec![Token::new(NUM_REF_PATCH, patches.len() as u32)];
    for p in patches {
        if p.reference > 3 || p.width == 0 || p.height == 0 || p.placements.is_empty() {
            return Err(Error::InvalidInput(format!("patch {p:?}")));
        }
        t.push(Token::new(REFERENCE_FRAME, p.reference));
        t.push(Token::new(REFERENCE_POSITION, p.x0));
        t.push(Token::new(REFERENCE_POSITION, p.y0));
        t.push(Token::new(PATCH_SIZE, p.width - 1));
        t.push(Token::new(PATCH_SIZE, p.height - 1));
        t.push(Token::new(COUNT, p.placements.len() as u32 - 1));
        let mut last: Option<(u32, u32)> = None;
        for pl in &p.placements {
            match last {
                None => {
                    t.push(Token::new(POSITION, pl.x));
                    t.push(Token::new(POSITION, pl.y));
                }
                Some((lx, ly)) => {
                    t.push(Token::signed(OFFSET, pl.x as i32 - lx as i32));
                    t.push(Token::signed(OFFSET, pl.y as i32 - ly as i32));
                }
            }
            last = Some((pl.x, pl.y));
            if pl.blending.len() != stride {
                return Err(Error::InvalidInput(format!(
                    "a patch placement with {} blendings, not {stride}",
                    pl.blending.len()
                )));
            }
            for b in &pl.blending {
                t.push(Token::new(BLEND_MODE, b.mode.code()));
                if b.mode.uses_alpha() && stride > 2 {
                    t.push(Token::new(ALPHA_CHANNEL, b.alpha_channel));
                }
                if b.mode.uses_alpha() || b.mode == PatchBlendMode::Multiply {
                    t.push(Token::new(CLAMP, u32::from(b.clamp)));
                }
            }
        }
    }
    let (code, s) = EntropyCode::build(10, vec![Stream::new(t)], &EntropyOptions::default(), true);
    code.write_all(w, &s[0]);
    Ok(())
}

fn write_splines(w: &mut BitWriter, s: &Splines) -> Result<()> {
    const QUANTIZATION_ADJUSTMENT: u32 = 0;
    const STARTING_POSITION: u32 = 1;
    const NUM_SPLINES: u32 = 2;
    const NUM_CONTROL_POINTS: u32 = 3;
    const CONTROL_POINTS: u32 = 4;
    const DCT: u32 = 5;
    if s.splines.is_empty() {
        return Err(Error::InvalidInput("splines enabled with none".into()));
    }
    let mut t = vec![Token::new(NUM_SPLINES, s.splines.len() as u32 - 1)];
    let mut last: Option<(i32, i32)> = None;
    for sp in &s.splines {
        let (x, y) = sp.start;
        match last {
            None => {
                if x < 0 || y < 0 {
                    return Err(Error::InvalidInput(
                        "the first spline starts off the frame".into(),
                    ));
                }
                t.push(Token::new(STARTING_POSITION, x as u32));
                t.push(Token::new(STARTING_POSITION, y as u32));
            }
            Some((lx, ly)) => {
                t.push(Token::new(STARTING_POSITION, pack_signed(x - lx)));
                t.push(Token::new(STARTING_POSITION, pack_signed(y - ly)));
            }
        }
        last = Some((x, y));
    }
    t.push(Token::signed(
        QUANTIZATION_ADJUSTMENT,
        s.quantization_adjustment,
    ));
    for sp in &s.splines {
        t.push(Token::new(
            NUM_CONTROL_POINTS,
            sp.control_deltas.len() as u32,
        ));
        for &(dx, dy) in &sp.control_deltas {
            t.push(Token::signed(CONTROL_POINTS, dx));
            t.push(Token::signed(CONTROL_POINTS, dy));
        }
        for c in sp.color_dct.iter().chain(std::iter::once(&sp.sigma_dct)) {
            for &v in c {
                t.push(Token::signed(DCT, v));
            }
        }
    }
    let (code, st) = EntropyCode::build(6, vec![Stream::new(t)], &EntropyOptions::default(), true);
    code.write_all(w, &st[0]);
    Ok(())
}
