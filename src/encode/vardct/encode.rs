//! A VarDCT frame: the colour channels transformed in variable-size blocks,
//! the LF (block means) modular coded per LF group, the HF coefficients
//! quantised and entropy coded per group and pass.

use super::coeffs::{
    BlockContextMap, NUM_ORDERS, ORDER_TRANSFORM, natural_order, zero_density_context,
};
use super::quant::{self, QuantEncoding, REQUIRED_SIZE_X, REQUIRED_SIZE_Y};
use super::transform::{self, TransformType};
use crate::encode::bits::BitWriter;
use crate::encode::entropy::{EntropyCode, EntropyOptions, Stream, Token, ceil_log2};
use crate::encode::features::Features;
use crate::encode::frame::FrameHeader;
use crate::encode::header::{ImageInfo, write_f16};
use crate::encode::modular::{
    self, Channel, ImageStreams, Layout, ModularCoded, ModularOptions, ModularStream,
};
use crate::{Error, Result};

/// How varblocks are chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum Strategy {
    /// One transform, tiled wherever it fits (8x8 DCTs elsewhere).
    Fixed(TransformType),
    /// Every varblock given: (block x, block y, transform), tiling the
    /// frame's blocks.
    Map(Vec<(usize, usize, TransformType)>),
    /// Larger DCTs where the picture is smooth.
    Auto,
}

/// The chroma-from-luma parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorCorrelation {
    pub color_factor: u32,
    pub base_x: f32,
    pub base_b: f32,
    pub ytox_lf: i32,
    pub ytob_lf: i32,
}

impl Default for ColorCorrelation {
    fn default() -> Self {
        ColorCorrelation {
            color_factor: 84,
            base_x: 0.0,
            base_b: 1.0,
            ytox_lf: 0,
            ytob_lf: 0,
        }
    }
}

/// VarDCT options.
#[derive(Clone, Debug, PartialEq)]
pub struct VarDctOptions {
    /// Lower is better; 1.0 is about visually lossless.
    pub distance: f32,
    pub strategy: Strategy,
    /// The 17 dequantisation tables (none: the library's).
    pub quant_tables: Option<Vec<QuantEncoding>>,
    pub block_context_map: BlockContextMap,
    /// Coefficient orders fitted to the frame rather than the natural ones.
    pub custom_orders: bool,
    /// Per-tile chroma-from-luma factors (else none).
    pub chroma_from_luma: bool,
    pub color_correlation: ColorCorrelation,
    /// The edge-preserving filter's sharpness per block, 0..=7.
    pub epf_sharpness: u8,
    /// Extra LF precision bits, 0..=3.
    pub extra_precision: u32,
    /// Overrides of the derived global scale and LF quantiser.
    pub global_scale: Option<u32>,
    pub quant_lf: Option<u32>,
    /// Vary the quantiser per block with its activity.
    pub adaptive_quantization: bool,
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    /// HF histogram sets (1..=groups), each group using one.
    pub num_histograms: u32,
    /// The LF quantisation factors (none: the defaults).
    pub lf_quant: Option<[f32; 3]>,
    /// For the LF, metadata and extra-channel modular streams.
    pub modular: ModularOptions,
    pub entropy: EntropyOptions,
}

impl Default for VarDctOptions {
    fn default() -> Self {
        VarDctOptions {
            distance: 1.0,
            strategy: Strategy::Auto,
            quant_tables: None,
            block_context_map: BlockContextMap::default(),
            custom_orders: false,
            chroma_from_luma: true,
            color_correlation: ColorCorrelation::default(),
            epf_sharpness: 4,
            extra_precision: 0,
            global_scale: None,
            quant_lf: None,
            adaptive_quantization: false,
            x_qm_scale: 3,
            b_qm_scale: 2,
            num_histograms: 1,
            lf_quant: None,
            modular: ModularOptions::default(),
            entropy: EntropyOptions::default(),
        }
    }
}

/// A VarDCT frame's samples: the three colour channels as floats (X, Y, B
/// for an XYB image; else the image's channels — Cb, Y, Cr for YCbCr),
/// each at its coded size; the extra channels as integers.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct VarDctFrame {
    pub color: Vec<Vec<f32>>,
    pub extra: Vec<Vec<i32>>,
    pub options: VarDctOptions,
}

const DEFAULT_LF_QUANT: [f32; 3] = [1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0];
const LOG_GROUP_BLOCKS: usize = 5; // 256 pixels

/// A varblock: its first block and transform.
#[derive(Clone, Copy, Debug)]
struct VarBlock {
    bx: usize,
    by: usize,
    t: TransformType,
    quant: u32,
}

/// The varblocks tiling `bw x bh` blocks.
fn layout_blocks(
    o: &VarDctOptions,
    bw: usize,
    bh: usize,
    luma: &[f32],
    stride: usize,
    is444: bool,
) -> Result<Vec<VarBlock>> {
    let mut taken = vec![false; bw * bh];
    let mut out = Vec::new();
    let mut place = |bx: usize,
                     by: usize,
                     t: TransformType,
                     taken: &mut [bool],
                     out: &mut Vec<VarBlock>|
     -> bool {
        let (cx, cy) = t.covered();
        if bx + cx > bw || by + cy > bh {
            return false;
        }
        // Not across a group (32 blocks).
        if (bx >> LOG_GROUP_BLOCKS) != ((bx + cx - 1) >> LOG_GROUP_BLOCKS)
            || (by >> LOG_GROUP_BLOCKS) != ((by + cy - 1) >> LOG_GROUP_BLOCKS)
        {
            return false;
        }
        if (0..cy).any(|y| (0..cx).any(|x| taken[(by + y) * bw + bx + x])) {
            return false;
        }
        for y in 0..cy {
            for x in 0..cx {
                taken[(by + y) * bw + bx + x] = true;
            }
        }
        out.push(VarBlock {
            bx,
            by,
            t,
            quant: 0,
        });
        true
    };
    let check_444 = |t: TransformType| -> Result<()> {
        if !is444 && t.covered() != (1, 1) {
            Err(Error::InvalidInput(format!("{t:?} with subsampled chroma")))
        } else {
            Ok(())
        }
    };
    match &o.strategy {
        Strategy::Map(list) => {
            for &(bx, by, t) in list {
                check_444(t)?;
                if !place(bx, by, t, &mut taken, &mut out) {
                    return Err(Error::InvalidInput(format!(
                        "varblock {t:?} at ({bx}, {by})"
                    )));
                }
            }
            if taken.iter().any(|&t| !t) {
                return Err(Error::InvalidInput(
                    "the varblocks leave blocks uncovered".into(),
                ));
            }
        }
        Strategy::Fixed(t) => {
            check_444(*t)?;
            let (cx, cy) = t.covered();
            for by in (0..bh).step_by(cy) {
                for bx in (0..bw).step_by(cx) {
                    place(bx, by, *t, &mut taken, &mut out);
                }
            }
        }
        Strategy::Auto => {
            if is444 {
                // Smooth regions get larger DCTs: by the spread of the
                // block means' neighbours and in-block detail.
                let detail = |bx: usize, by: usize, n: usize| -> f32 {
                    let mut lo = f32::MAX;
                    let mut hi = f32::MIN;
                    for y in by * 8..(by + n) * 8 {
                        for x in bx * 8..(bx + n) * 8 {
                            let v = luma[y * stride + x];
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                    hi - lo
                };
                let threshold = 0.004 * o.distance.max(0.1);
                for (n, t) in [(4, TransformType::Dct32), (2, TransformType::Dct16)] {
                    for by in (0..bh).step_by(n) {
                        for bx in (0..bw).step_by(n) {
                            if bx + n <= bw
                                && by + n <= bh
                                && detail(bx, by, n) < threshold * n as f32
                            {
                                place(bx, by, t, &mut taken, &mut out);
                            }
                        }
                    }
                }
            }
        }
    }
    for by in 0..bh {
        for bx in 0..bw {
            if !taken[by * bw + bx] {
                place(bx, by, TransformType::Dct8, &mut taken, &mut out);
            }
        }
    }
    // Raster order of first blocks.
    out.sort_by_key(|b| (b.by, b.bx));
    Ok(out)
}

/// A channel at its coded size, padded to whole blocks by repeating edges.
fn pad(plane: &[f32], w: usize, h: usize, pw: usize, ph: usize) -> Vec<f32> {
    let mut out = vec![0f32; pw * ph];
    for y in 0..ph {
        let sy = y.min(h - 1);
        for x in 0..pw {
            out[y * pw + x] = plane[sy * w + x.min(w - 1)];
        }
    }
    out
}

/// The decoder's reconstruction of a quantised coefficient (in steps).
#[inline]
fn dequant_bias(q: i32, c: usize, biases: &[f32; 4]) -> f32 {
    if q.abs() < 2 {
        q as f32 * biases[c]
    } else {
        q as f32 - biases[3] / q as f32
    }
}

/// The quantised value whose reconstruction is nearest `a` (in steps),
/// with a dead zone around zero.
#[inline]
fn quantize(a: f32, c: usize, biases: &[f32; 4], dead_zone: f32) -> i32 {
    if a.abs() < dead_zone {
        return 0;
    }
    let r = a.round() as i32;
    let mut best = (f32::MAX, 0);
    for q in [r - 1, r, r + 1] {
        let e = (dequant_bias(q, c, biases) - a).abs();
        if e < best.0 {
            best = (e, q);
        }
    }
    best.1
}

/// A permutation of a coefficient order's entries after `skip`, as the
/// tokens the decoder reads (`end`, then the Lehmer code).
fn permutation_tokens(perm: &[usize], skip: usize, out: &mut Vec<Token>) {
    let ctx = |x: u32| ceil_log2(x + 1).min(7);
    let n = perm.len();
    let mut remaining: Vec<usize> = (skip..n).collect();
    let mut code = Vec::with_capacity(n - skip);
    for &p in &perm[skip..] {
        let i = remaining.iter().position(|&r| r == p).unwrap();
        code.push(i as u32);
        remaining.remove(i);
    }
    let end = code.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
    out.push(Token::new(ctx(n as u32), end as u32));
    let mut prev = 0;
    for &c in &code[..end] {
        out.push(Token::new(ctx(prev), c));
        prev = c;
    }
}

/// The frame's sections in natural order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_vardct(
    f: &VarDctFrame,
    header: &FrameHeader,
    info: &ImageInfo,
    features: &Features,
    sixteen_bit: &mut bool,
) -> Result<Vec<BitWriter>> {
    let o = &f.options;
    if !(0.01..=100.0).contains(&o.distance) {
        return Err(Error::InvalidInput(format!("distance {}", o.distance)));
    }
    if o.extra_precision > 3 || o.epf_sharpness > 7 || o.x_qm_scale > 7 || o.b_qm_scale > 7 {
        return Err(Error::InvalidInput("VarDCT options out of range".into()));
    }
    if f.color.len() != 3 || f.extra.len() != info.extra_channels.len() {
        return Err(Error::InvalidInput(
            "a VarDCT frame has three colour channels and the image's extra channels".into(),
        ));
    }
    let (w, h) = header.size(info);
    let (bw, bh) = header.size_blocks(info);
    let shifts: [(u32, u32); 3] = [0, 1, 2].map(|c| header.chroma_shift(c));
    let is444 = shifts.iter().all(|&s| s == (0, 0));

    // The channels, padded to whole blocks.
    let mut planes = Vec::with_capacity(3);
    let mut pstride = [0usize; 3];
    for c in 0..3 {
        let (sx, sy) = shifts[c];
        let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
        if f.color[c].len() != cw * ch {
            return Err(Error::InvalidInput(format!(
                "colour channel {c} has {} samples, its {cw}x{ch} takes {}",
                f.color[c].len(),
                cw * ch
            )));
        }
        let (pw, ph) = ((bw >> sx) * 8, (bh >> sy) * 8);
        planes.push(pad(&f.color[c], cw, ch, pw, ph));
        pstride[c] = pw;
    }

    let mut blocks = layout_blocks(o, bw, bh, &planes[1], pstride[1], is444)?;

    // Quantisers.
    let scale = 1.12 * o.distance;
    let base_quant = 64u32;
    let global_scale = o.global_scale.unwrap_or_else(|| {
        ((65536.0 / (scale * base_quant as f32)).round() as u32).clamp(1, 73728)
    });
    let inv_global_scale = 65536.0 / global_scale as f32;
    let lf_quant = o.lf_quant.unwrap_or(DEFAULT_LF_QUANT);
    let quant_lf = o.quant_lf.unwrap_or_else(|| {
        // An LF step of about 0.0012 (Y) at distance 1.
        let want = 0.0012 * o.distance;
        ((lf_quant[1] * inv_global_scale / want).round() as u32).clamp(1, 65536)
    });
    let lf_mul = 1.0 / (1u32 << o.extra_precision) as f32;
    let lf_fac: [f32; 3] =
        [0, 1, 2].map(|c| lf_quant[c] * inv_global_scale / quant_lf as f32 * lf_mul);
    let x_dm = (1.0f32 / 1.25).powf(o.x_qm_scale as f32 - 2.0);
    let b_dm = (1.0f32 / 1.25).powf(o.b_qm_scale as f32 - 2.0);
    let dm = [x_dm, 1.0, b_dm];
    let opsin = info.opsin_inverse.unwrap_or_default();
    let biases = opsin.quant_biases;

    // Per-block activity, for adaptive quantisation.
    for b in blocks.iter_mut() {
        b.quant = base_quant;
        if o.adaptive_quantization {
            let (cx, cy) = b.t.covered();
            let p = &planes[1];
            let s = pstride[1];
            let mut sum = 0.0f32;
            for y in b.by * 8..(b.by + cy) * 8 {
                for x in b.bx * 8 + 1..(b.bx + cx) * 8 {
                    sum += (p[y * s + x] - p[y * s + x - 1]).abs();
                }
            }
            let act = sum / (cx * cy * 64) as f32;
            // Busy blocks hide error: a coarser step.
            let k = (1.0 + act * 40.0).min(2.0);
            b.quant = ((base_quant as f32 / k).round() as u32).clamp(1, 256);
        }
    }

    // The dequantisation tables.
    let tables: Vec<Vec<f32>> = (0..17)
        .map(|i| {
            let e = o
                .quant_tables
                .as_ref()
                .map_or(QuantEncoding::Library, |t| t[i].clone());
            quant::compute(&e, i)
        })
        .collect::<Result<_>>()?;
    if o.quant_tables.as_ref().is_some_and(|t| t.len() != 17) {
        return Err(Error::InvalidInput("17 quantisation tables".into()));
    }

    // Forward transforms: per channel, per varblock, coefficients; and the
    // LF image per channel (in its blocks).
    let cbw: [usize; 3] = [0, 1, 2].map(|c| bw >> shifts[c].0);
    let cbh: [usize; 3] = [0, 1, 2].map(|c| bh >> shifts[c].1);
    let mut lf: Vec<Vec<f32>> = (0..3).map(|c| vec![0f32; cbw[c] * cbh[c]]).collect();
    // coeffs[c][block index]: wide-stored coefficients (empty when the
    // channel has no block here).
    let mut coeffs: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); blocks.len()]; 3];
    for (bi, b) in blocks.iter().enumerate() {
        let (cx, cy) = b.t.covered();
        for c in [1usize, 0, 2] {
            let (sx, sy) = shifts[c];
            if ((b.bx >> sx) << sx) != b.bx || ((b.by >> sy) << sy) != b.by {
                continue;
            }
            let (sbx, sby) = (b.bx >> sx, b.by >> sy);
            let s = pstride[c];
            let mut px = vec![0f32; cx * cy * 64];
            for y in 0..cy * 8 {
                for x in 0..cx * 8 {
                    px[y * cx * 8 + x] = planes[c][(sby * 8 + y) * s + sbx * 8 + x];
                }
            }
            let (cf, lfs) = transform::forward(b.t, &px);
            for iy in 0..cy {
                for ix in 0..cx {
                    lf[c][(sby + iy) * cbw[c] + sbx + ix] = lfs[iy * cx + ix];
                }
            }
            coeffs[c][bi] = cf;
        }
    }

    // Quantised LF, chroma from luma at the LF (444 only).
    let cc = o.color_correlation;
    let y_to_x_lf = cc.base_x + cc.ytox_lf as f32 / cc.color_factor as f32;
    let y_to_b_lf = cc.base_b + cc.ytob_lf as f32 / cc.color_factor as f32;
    let mut qlf: Vec<Vec<i32>> = (0..3).map(|c| vec![0i32; lf[c].len()]).collect();
    for i in 0..lf[1].len() {
        qlf[1][i] = (lf[1][i] / lf_fac[1]).round() as i32;
    }
    for c in [0usize, 2] {
        for i in 0..lf[c].len() {
            let pred = if is444 {
                let y = qlf[1][i] as f32 * lf_fac[1];
                y * if c == 0 { y_to_x_lf } else { y_to_b_lf }
            } else {
                0.0
            };
            qlf[c][i] = ((lf[c][i] - pred) / lf_fac[c]).round() as i32;
        }
    }

    // Chroma from luma per 64x64 tile.
    let tiles_x = bw.div_ceil(8);
    let tiles_y = bh.div_ceil(8);
    let mut ytox = vec![0i32; tiles_x * tiles_y];
    let mut ytob = vec![0i32; tiles_x * tiles_y];
    if o.chroma_from_luma && is444 {
        let mut sums = vec![[0f64; 4]; tiles_x * tiles_y]; // xy, yy, by, (unused)
        for (bi, b) in blocks.iter().enumerate() {
            let t = (b.by / 8) * tiles_x + b.bx / 8;
            let (cx, cy) = b.t.covered();
            let llf = cx.max(cy) * cx.min(cy);
            let stride = cx.max(cy) * 8;
            for (k, &yv) in coeffs[1][bi].iter().enumerate() {
                let (r, col) = (k / stride, k % stride);
                if r < cx.min(cy) && col < cx.max(cy) {
                    continue; // LLF
                }
                let _ = llf;
                let yv = f64::from(yv);
                sums[t][0] += f64::from(coeffs[0][bi][k]) * yv;
                sums[t][1] += yv * yv;
                sums[t][2] += f64::from(coeffs[2][bi][k]) * yv;
            }
        }
        for t in 0..tiles_x * tiles_y {
            let [xy, yy, by, _] = sums[t];
            if yy > 1e-12 {
                let fx = ((xy / yy) as f32 - cc.base_x) * cc.color_factor as f32;
                let fb = ((by / yy) as f32 - cc.base_b) * cc.color_factor as f32;
                ytox[t] = (fx.round() as i32).clamp(-128, 127);
                ytob[t] = (fb.round() as i32).clamp(-128, 127);
            }
        }
    }

    // Quantised HF coefficients, per channel per varblock.
    let mut qcoeffs: Vec<Vec<Vec<i32>>> = vec![vec![Vec::new(); blocks.len()]; 3];
    let dead_zone = 0.55;
    for (bi, b) in blocks.iter().enumerate() {
        let (cx, cy) = b.t.covered();
        let n = cx * cy * 64;
        let table = &tables[b.t.quant_table()];
        let tile = (b.by / 8) * tiles_x + b.bx / 8;
        let x_cc = cc.base_x + ytox[tile] as f32 / cc.color_factor as f32;
        let b_cc = cc.base_b + ytob[tile] as f32 / cc.color_factor as f32;
        let step_base = inv_global_scale / b.quant as f32;
        let llf_rows = cx.min(cy);
        let llf_cols = cx.max(cy);
        let stride = llf_cols * 8;
        let is_llf = |k: usize| k / stride < llf_rows && k % stride < llf_cols;
        // Y first: its reconstruction feeds X and B.
        let mut y_rec = vec![0f32; n];
        for c in [1usize, 0, 2] {
            if coeffs[c][bi].is_empty() {
                continue;
            }
            let mut q = vec![0i32; n];
            for k in 0..n {
                if is_llf(k) {
                    continue;
                }
                let step = table[c * n + k] * step_base * dm[c];
                let mut v = coeffs[c][bi][k];
                if c != 1 && !coeffs[1][bi].is_empty() {
                    v -= y_rec[k] * if c == 0 { x_cc } else { b_cc };
                }
                q[k] = quantize(v / step, c, &biases, dead_zone);
                if c == 1 {
                    y_rec[k] = dequant_bias(q[k], 1, &biases) * step;
                }
            }
            qcoeffs[c][bi] = q;
        }
    }

    // Coefficient orders: natural, or by how often each position is
    // non-zero.
    let num_passes = header.passes.num_passes as usize;
    let mut orders: Vec<Vec<u32>> = (0..3 * NUM_ORDERS).map(|i| natural_order(i / 3)).collect();
    let mut used_orders = 0u32;
    if o.custom_orders {
        let mut counts: Vec<Vec<u64>> = (0..3 * NUM_ORDERS)
            .map(|i| vec![0; orders[i].len()])
            .collect();
        let mut present = [false; NUM_ORDERS];
        for (bi, b) in blocks.iter().enumerate() {
            let shape = b.t.shape();
            present[shape] = true;
            for c in 0..3 {
                for (k, &q) in qcoeffs[c][bi].iter().enumerate() {
                    if q != 0 {
                        counts[shape * 3 + c][k] += 1;
                    }
                }
            }
        }
        for shape in 0..NUM_ORDERS {
            if !present[shape] {
                continue;
            }
            let (cx, cy) = ORDER_TRANSFORM[shape].covered();
            let skip = cx * cy;
            for c in 0..3 {
                let nat = &orders[shape * 3 + c];
                let mut rest: Vec<u32> = nat[skip..].to_vec();
                let cnt = &counts[shape * 3 + c];
                rest.sort_by_key(|&p| std::cmp::Reverse(cnt[p as usize]));
                let mut ord = nat[..skip].to_vec();
                ord.extend(rest);
                orders[shape * 3 + c] = ord;
            }
            used_orders |= 1 << shape;
        }
    }

    // Groups and LF groups.
    let (gx_n, gy_n) = (w.div_ceil(256), h.div_ceil(256));
    let num_groups = gx_n * gy_n;
    let (lgx_n, lgy_n) = (bw.div_ceil(256), bh.div_ceil(256));
    let num_lf_groups = lgx_n * lgy_n;
    if !(1..=num_groups as u32).contains(&o.num_histograms) {
        return Err(Error::InvalidInput(format!(
            "{} histogram sets for {num_groups} groups",
            o.num_histograms
        )));
    }

    // HF tokens per pass and group.
    let bcm = &o.block_context_map;
    let num_ac = bcm.num_ac_contexts();
    let mut tokens: Vec<Vec<Vec<Token>>> = vec![vec![Vec::new(); num_groups]; num_passes];
    let group_of = |b: &VarBlock| (b.by >> LOG_GROUP_BLOCKS) * gx_n + (b.bx >> LOG_GROUP_BLOCKS);
    let group_hist = |g: usize| g % o.num_histograms as usize;
    // The decoder's per-group non-zero counts, per channel, per block.
    let mut nzeros: Vec<Vec<Vec<u32>>> =
        vec![(0..3).map(|c| vec![0u32; cbw[c] * cbh[c]]).collect(); num_passes];
    for (bi, b) in blocks.iter().enumerate() {
        let g = group_of(b);
        let (gbx, gby) = (
            (g % gx_n) << LOG_GROUP_BLOCKS,
            (g / gx_n) << LOG_GROUP_BLOCKS,
        );
        let (cx, cy) = b.t.covered();
        let num_blocks = cx * cy;
        let n = num_blocks * 64;
        let log_nb = num_blocks.ilog2() as usize;
        let shape = b.t.shape();
        // The LF bucket from the quantised LF at this block.
        let lf_idx = if bcm.num_lf_contexts() > 1 {
            let at = |c: usize| {
                let (sx, sy) = shifts[c];
                qlf[c][(b.by >> sy) * cbw[c] + (b.bx >> sx)]
            };
            bcm.lf_index([at(0), at(1), at(2)])
        } else {
            0
        };
        // Split each coefficient over the passes.
        for c in [1usize, 0, 2] {
            if qcoeffs[c][bi].is_empty() {
                continue;
            }
            let (sx, sy) = shifts[c];
            let (sbx, sby) = (b.bx >> sx, b.by >> sy);
            let mut remaining = qcoeffs[c][bi].clone();
            for p in 0..num_passes {
                let shift = header.passes.shift.get(p).copied().unwrap_or(0);
                let part: Vec<i32> = remaining.iter().map(|&v| v / (1 << shift)).collect();
                for (r, &v) in remaining.iter_mut().zip(&part) {
                    *r -= v << shift;
                }
                let order = &orders[shape * 3 + c];
                let nz_total = (num_blocks..n)
                    .filter(|&k| part[order[k] as usize] != 0)
                    .count();
                // Predicted non-zeros, from the group's blocks above and left.
                let (gx0, gy0) = (gbx >> sx, gby >> sy);
                let (lx, ly) = (sbx - gx0, sby - gy0);
                let nzc = &nzeros[p][c];
                let at = |x: usize, y: usize| nzc[(gy0 + y) * cbw[c] + gx0 + x] as usize;
                let predicted = match (lx, ly) {
                    (0, 0) => 32,
                    (0, _) => at(0, ly - 1),
                    (_, 0) => at(lx - 1, 0),
                    _ => (at(lx, ly - 1) + at(lx - 1, ly)).div_ceil(2),
                };
                let bctx = bcm.block_context(lf_idx, b.quant, shape, c);
                let hist = group_hist(g) * num_ac;
                let toks = &mut tokens[p][g];
                toks.push(Token::new(
                    (bcm.nonzero_context(predicted, bctx) + hist) as u32,
                    nz_total as u32,
                ));
                let per_block = nz_total.div_ceil(num_blocks) as u32;
                for iy in 0..cy {
                    for ix in 0..cx {
                        if sby + iy < cbh[c] && sbx + ix < cbw[c] {
                            nzeros[p][c][(sby + iy) * cbw[c] + sbx + ix] = per_block;
                        }
                    }
                }
                let offset = bcm.zero_density_offset(bctx) + hist;
                let mut left = nz_total;
                let mut prev = usize::from(nz_total <= n / 16);
                for k in num_blocks..n {
                    if left == 0 {
                        break;
                    }
                    let v = part[order[k] as usize];
                    let ctx = offset + zero_density_context(left, k, log_nb, prev);
                    toks.push(Token::signed(ctx as u32, v));
                    prev = usize::from(v != 0);
                    left -= prev;
                }
            }
        }
    }

    // HF entropy codes, one per pass, over every histogram set's contexts.
    let mut hf_codes = Vec::with_capacity(num_passes);
    let mut hf_tokens = Vec::with_capacity(num_passes);
    for pass_tokens in tokens {
        let streams: Vec<Stream> = pass_tokens.into_iter().map(Stream::new).collect();
        let (code, toks) = EntropyCode::build(
            o.num_histograms as usize * num_ac,
            streams,
            &o.entropy,
            true,
        );
        hf_codes.push(code);
        hf_tokens.push(toks);
    }

    // The modular streams: LF and metadata per LF group, raw quant tables,
    // and the extra channels.
    let lf_stream_id = |g: usize| 1 + g;
    let meta_stream_id = |g: usize| 1 + 2 * num_lf_groups + g;
    let quant_stream_id = |q: usize| 1 + 3 * num_lf_groups + q;
    let mut streams: Vec<ModularStream> = Vec::new();
    let mut counts = Vec::new();
    let lf_tag = |g: usize| (g % lgx_n, g / lgx_n);
    for g in 0..num_lf_groups {
        let (lx, ly) = lf_tag(g);
        let (ox, oy) = (lx * 256, ly * 256);
        let (rw, rh) = ((bw - ox).min(256), (bh - oy).min(256));
        // LF: Y, X, B.
        let mut chans = Vec::new();
        for c in [1usize, 0, 2] {
            let (sx, sy) = shifts[c];
            let (cw, ch) = (rw >> sx, rh >> sy);
            let mut s = Vec::with_capacity(cw * ch);
            for y in 0..ch {
                for x in 0..cw {
                    s.push(qlf[c][((oy >> sy) + y) * cbw[c] + (ox >> sx) + x]);
                }
            }
            chans.push(Channel::new(cw, ch, Some((0, 0)), s));
        }
        *sixteen_bit &= modular::fits_i16(&chans);
        streams.push(ModularStream {
            id: lf_stream_id(g),
            channels: chans,
            transforms: Vec::new(),
            header_always: false,
        });
        // Metadata.
        let (tw, th) = (rw.div_ceil(8), rh.div_ceil(8));
        let tile_at =
            |x: usize, y: usize, map: &[i32]| map[((oy / 8) + y) * tiles_x + (ox / 8) + x];
        let tx: Vec<i32> = (0..th)
            .flat_map(|y| (0..tw).map(move |x| (x, y)))
            .map(|(x, y)| tile_at(x, y, &ytox))
            .collect();
        let tb: Vec<i32> = (0..th)
            .flat_map(|y| (0..tw).map(move |x| (x, y)))
            .map(|(x, y)| tile_at(x, y, &ytob))
            .collect();
        let in_group: Vec<&VarBlock> = blocks
            .iter()
            .filter(|b| b.bx >= ox && b.bx < ox + rw && b.by >= oy && b.by < oy + rh)
            .collect();
        let count = in_group.len();
        let mut kinds = Vec::with_capacity(2 * count);
        kinds.extend(in_group.iter().map(|b| b.t as i32));
        kinds.extend(in_group.iter().map(|b| b.quant as i32 - 1));
        let epf = vec![i32::from(o.epf_sharpness); rw * rh];
        let meta = vec![
            Channel::new(tw, th, Some((3, 3)), tx),
            Channel::new(tw, th, Some((3, 3)), tb),
            Channel::new(count, 2, Some((0, 0)), kinds),
            Channel::new(rw, rh, Some((0, 0)), epf),
        ];
        *sixteen_bit &= modular::fits_i16(&meta);
        streams.push(ModularStream {
            id: meta_stream_id(g),
            channels: meta,
            transforms: Vec::new(),
            header_always: false,
        });
        counts.push((count, rw * rh));
    }
    let raw_tables: Vec<(usize, Vec<i32>)> = o
        .quant_tables
        .iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, e)| match e {
            QuantEncoding::Raw { qtable, .. } => Some((i, qtable.clone())),
            _ => None,
        })
        .collect();
    for (i, t) in &raw_tables {
        let (xw, yh) = (REQUIRED_SIZE_X[*i] * 8, REQUIRED_SIZE_Y[*i] * 8);
        let n = xw * yh;
        let chans = (0..3)
            .map(|c| Channel::new(xw, yh, Some((0, 0)), t[c * n..(c + 1) * n].to_vec()))
            .collect::<Vec<_>>();
        streams.push(ModularStream {
            id: quant_stream_id(*i),
            channels: chans,
            transforms: Vec::new(),
            header_always: false,
        });
    }
    let first_extra = streams.len();
    let num_extra = info.extra_channels.len();
    let (extra_global, extra_lf, extra_hf) = if num_extra > 0 {
        let shapes = crate::encode::encoder::channel_shapes(header, info);
        let mut ch = Vec::new();
        for (i, data) in f.extra.iter().enumerate() {
            let (cw, chh, shift) = shapes[3 + i];
            if data.len() != cw * chh {
                return Err(Error::InvalidInput(format!(
                    "extra channel {i} has {} samples, not {}",
                    data.len(),
                    cw * chh
                )));
            }
            ch.push(Channel::new(cw, chh, Some(shift), data.clone()));
        }
        *sixteen_bit &= modular::fits_i16(&ch);
        let layout = Layout {
            group_dim: 256,
            groups_x: gx_n,
            groups_y: gy_n,
            lf_groups_x: lgx_n,
            lf_groups_y: lgy_n,
            pass_shifts: (0..num_passes)
                .map(|p| header.passes.shift_bracket(p))
                .collect(),
            num_lf_groups,
        };
        let split = ImageStreams::split(ch, Vec::new(), &layout);
        let nlf = split.lf.len();
        streams.push(split.global);
        streams.extend(split.lf);
        let mut nhf = 0;
        for p in split.hf {
            nhf += p.len();
            streams.extend(p);
        }
        (true, nlf, nhf)
    } else {
        (false, 0, 0)
    };
    let (fw, fh) = match header.crop {
        Some(c) => (c.width as usize, c.height as usize),
        None => (info.width as usize, info.height as usize),
    };
    let size_limit = (1024 + fw * fh * (3 + num_extra) / 16).min(1 << 22);
    let coded = ModularCoded::new(&streams, &o.modular, size_limit);

    // LfGlobal.
    let mut global = BitWriter::new();
    features.write(&mut global, num_extra)?;
    if lf_quant == DEFAULT_LF_QUANT {
        global.bit(true);
    } else {
        global.bit(false);
        for v in lf_quant {
            write_f16(&mut global, v * 128.0)?;
        }
    }
    // QuantizerParams.
    match global_scale {
        1..=2048 => {
            global.write(2, 0);
            global.write(11, global_scale - 1);
        }
        2049..=4096 => {
            global.write(2, 1);
            global.write(11, global_scale - 2049);
        }
        4097..=8192 => {
            global.write(2, 2);
            global.write(12, global_scale - 4097);
        }
        _ => {
            global.write(2, 3);
            global.write(16, global_scale - 8193);
        }
    }
    match quant_lf {
        16 => global.write(2, 0),
        1..=32 => {
            global.write(2, 1);
            global.write(5, quant_lf - 1);
        }
        33..=256 => {
            global.write(2, 2);
            global.write(8, quant_lf - 1);
        }
        _ => {
            global.write(2, 3);
            global.write(16, quant_lf - 1);
        }
    }
    bcm.write(&mut global, &o.entropy)?;
    if cc == ColorCorrelation::default() {
        global.bit(true);
    } else {
        global.bit(false);
        match cc.color_factor {
            84 => global.write(2, 0),
            256 => global.write(2, 1),
            2..=257 => {
                global.write(2, 2);
                global.write(8, cc.color_factor - 2);
            }
            258..=65793 => {
                global.write(2, 3);
                global.write(16, cc.color_factor - 258);
            }
            _ => {
                return Err(Error::InvalidInput(format!(
                    "colour factor {}",
                    cc.color_factor
                )));
            }
        }
        if cc.base_x.abs() > 4.0 || cc.base_b.abs() > 4.0 {
            return Err(Error::InvalidInput("base correlations within 4".into()));
        }
        write_f16(&mut global, cc.base_x)?;
        write_f16(&mut global, cc.base_b)?;
        if !(-128..=127).contains(&cc.ytox_lf) || !(-128..=127).contains(&cc.ytob_lf) {
            return Err(Error::InvalidInput(
                "LF correlations within -128..=127".into(),
            ));
        }
        global.write(8, (cc.ytox_lf + 128) as u32);
        global.write(8, (cc.ytob_lf + 128) as u32);
    }
    coded.write_global_tree(&mut global);
    if extra_global {
        coded.write_stream(&mut global, first_extra);
    }

    let mut sections = vec![global];
    // LF groups: the LF, the extra channels' LF part, the metadata.
    for g in 0..num_lf_groups {
        let mut s = BitWriter::new();
        if !header_uses_lf_frame(header) {
            s.write(2, o.extra_precision);
            coded.write_stream(&mut s, 2 * g);
        }
        if extra_global {
            coded.write_stream(&mut s, first_extra + 1 + g);
        }
        let (count, area) = counts[g];
        s.write(ceil_log2(area as u32), count as u32 - 1);
        coded.write_stream(&mut s, 2 * g + 1);
        sections.push(s);
    }

    // HfGlobal.
    let mut hfg = BitWriter::new();
    match &o.quant_tables {
        None => hfg.bit(true),
        Some(t) => {
            hfg.bit(false);
            let mut raw_index = 0;
            for (i, e) in t.iter().enumerate() {
                quant::write_encoding(&mut hfg, e, i)?;
                if matches!(e, QuantEncoding::Raw { .. }) {
                    coded.write_stream(&mut hfg, 2 * num_lf_groups + raw_index);
                    raw_index += 1;
                }
            }
        }
    }
    hfg.write(ceil_log2(num_groups as u32), o.num_histograms - 1);
    for p in 0..num_passes {
        match used_orders {
            0x5f => hfg.write(2, 0),
            0x13 => hfg.write(2, 1),
            0 => hfg.write(2, 2),
            u => {
                hfg.write(2, 3);
                hfg.write(13, u);
            }
        }
        if used_orders != 0 {
            let mut t = Vec::new();
            for shape in 0..NUM_ORDERS {
                if used_orders & (1 << shape) == 0 {
                    continue;
                }
                let (cx, cy) = ORDER_TRANSFORM[shape].covered();
                let nat = natural_order(shape);
                for c in 0..3 {
                    let ord = &orders[shape * 3 + c];
                    let pos_in_nat: std::collections::HashMap<u32, usize> =
                        nat.iter().enumerate().map(|(i, &v)| (v, i)).collect();
                    let perm: Vec<usize> = ord.iter().map(|v| pos_in_nat[v]).collect();
                    permutation_tokens(&perm, cx * cy, &mut t);
                }
            }
            let (code, st) = EntropyCode::build(8, vec![Stream::new(t)], &o.entropy, true);
            code.write_all(&mut hfg, &st[0]);
        }
        hf_codes[p].write_header(&mut hfg);
    }
    sections.push(hfg);

    // The groups, per pass: the coefficients, then the extra channels.
    let extra_hf_first = first_extra + 1 + extra_lf;
    for p in 0..num_passes {
        for g in 0..num_groups {
            let mut s = BitWriter::new();
            s.write(ceil_log2(o.num_histograms), group_hist(g) as u32);
            hf_codes[p].write_tokens(&mut s, &hf_tokens[p][g]);
            if extra_global && extra_hf > 0 {
                coded.write_stream(&mut s, extra_hf_first + p * num_groups + g);
            }
            sections.push(s);
        }
    }
    if num_groups == 1 && num_passes == 1 {
        let mut all = BitWriter::new();
        for s in &sections {
            all.append_bits(s);
        }
        return Ok(vec![all]);
    }
    Ok(sections)
}

fn header_uses_lf_frame(header: &FrameHeader) -> bool {
    header.flags & crate::encode::frame::USE_LF_FRAME != 0
}
