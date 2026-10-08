//! The modular transforms, forward: what the decoder undoes. Each is
//! applied to the channel list in the order the group header lists them.

use super::Channel;
use super::predict::{Neighbours, Predictor, WeightedParams, WeightedState};
use crate::encode::bits::{BitWriter, Dist};

/// `begin_channel` and `squeeze` channel indices.
const CHANNEL_INDEX: [Dist; 4] = [
    Dist::Bits(3, 0),
    Dist::Bits(6, 8),
    Dist::Bits(10, 72),
    Dist::Bits(13, 1096),
];

/// A reversible colour transform: which of the seven operations, which of
/// the six channel orders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rct {
    pub begin_channel: u32,
    /// `operation + 7 * permutation`, 0..42; 6 is YCoCg in RGB order.
    pub rct_type: u32,
}

/// One squeeze step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SqueezeStep {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_channel: u32,
    pub num_channels: u32,
}

/// A palette: channels `begin..begin + num_channels` replaced by one index
/// channel, the colours (and deltas) in a meta channel at the front.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Palette {
    pub begin_channel: u32,
    pub num_channels: u32,
    /// Delta entries first (predicted, added to the prediction), then
    /// colours; each entry `num_channels` values.
    pub entries: Vec<Vec<i32>>,
    pub num_deltas: u32,
    /// The predictor the deltas are added to.
    pub predictor: Predictor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transform {
    Rct(Rct),
    Palette(Palette),
    /// The squeeze steps; empty for the decoder's default sequence.
    Squeeze(Vec<SqueezeStep>),
}

impl Transform {
    pub(crate) fn write(&self, w: &mut BitWriter) {
        match self {
            Transform::Rct(r) => {
                w.write(2, 0);
                w.u32(r.begin_channel, CHANNEL_INDEX);
                w.u32(
                    r.rct_type,
                    [
                        Dist::Val(6),
                        Dist::Bits(2, 0),
                        Dist::Bits(4, 2),
                        Dist::Bits(6, 10),
                    ],
                );
            }
            Transform::Palette(p) => {
                w.write(2, 1);
                w.u32(p.begin_channel, CHANNEL_INDEX);
                w.u32(
                    p.num_channels,
                    [Dist::Val(1), Dist::Val(3), Dist::Val(4), Dist::Bits(13, 1)],
                );
                w.u32(
                    p.entries.len() as u32 - p.num_deltas,
                    [
                        Dist::Bits(8, 0),
                        Dist::Bits(10, 256),
                        Dist::Bits(12, 1280),
                        Dist::Bits(16, 5376),
                    ],
                );
                w.u32(
                    p.num_deltas,
                    [
                        Dist::Val(0),
                        Dist::Bits(8, 1),
                        Dist::Bits(10, 257),
                        Dist::Bits(16, 1281),
                    ],
                );
                w.write(4, p.predictor as u32);
            }
            Transform::Squeeze(steps) => {
                w.write(2, 2);
                w.u32(
                    steps.len() as u32,
                    [
                        Dist::Val(0),
                        Dist::Bits(4, 1),
                        Dist::Bits(6, 9),
                        Dist::Bits(8, 41),
                    ],
                );
                for s in steps {
                    w.bit(s.horizontal);
                    w.bit(s.in_place);
                    w.u32(s.begin_channel, CHANNEL_INDEX);
                    w.u32(
                        s.num_channels,
                        [Dist::Val(1), Dist::Val(2), Dist::Val(3), Dist::Bits(4, 4)],
                    );
                }
            }
        }
    }

    /// Apply to `channels`, as the decoder's inverse expects.
    pub(crate) fn apply(
        &self,
        channels: &mut Vec<Channel>,
        bits_per_sample: u32,
        wp: WeightedParams,
    ) -> Result<(), String> {
        match self {
            Transform::Rct(r) => apply_rct(channels, r),
            Transform::Palette(p) => apply_palette(channels, p, bits_per_sample, wp),
            Transform::Squeeze(steps) => {
                let steps = if steps.is_empty() {
                    default_squeeze(channels)
                } else {
                    steps.clone()
                };
                for s in steps {
                    apply_squeeze(channels, &s)?;
                }
                Ok(())
            }
        }
    }
}

fn check_equal(channels: &[Channel], begin: usize, n: usize) -> Result<(), String> {
    if begin + n > channels.len() {
        return Err(format!(
            "channels {begin}..{} of {}",
            begin + n,
            channels.len()
        ));
    }
    for c in &channels[begin + 1..begin + n] {
        if (c.width, c.height, c.shift)
            != (
                channels[begin].width,
                channels[begin].height,
                channels[begin].shift,
            )
        {
            return Err("the transform's channels differ in size".into());
        }
    }
    Ok(())
}

/// The decoder's output order after an RCT: output `k` is the operation's
/// output `PERMUTATIONS[p][k]`.
const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [2, 0, 1],
    [1, 2, 0],
    [0, 2, 1],
    [1, 0, 2],
    [2, 1, 0],
];

fn apply_rct(channels: &mut [Channel], r: &Rct) -> Result<(), String> {
    let begin = r.begin_channel as usize;
    check_equal(channels, begin, 3)?;
    if r.rct_type >= 42 {
        return Err(format!("RCT type {}", r.rct_type));
    }
    let op = r.rct_type % 7;
    let perm = PERMUTATIONS[(r.rct_type / 7) as usize];
    let n = channels[begin].samples.len();
    for i in 0..n {
        let fin = [
            channels[begin].samples[i],
            channels[begin + 1].samples[i],
            channels[begin + 2].samples[i],
        ];
        // The operation's outputs, before the permutation.
        let mut o = [0i32; 3];
        for k in 0..3 {
            o[perm[k]] = fin[k];
        }
        let v = forward_rct(op, o);
        for k in 0..3 {
            channels[begin + k].samples[i] = v[k];
        }
    }
    Ok(())
}

/// The coded values whose inverse operation `op` gives `o`.
pub(crate) fn forward_rct(op: u32, o: [i32; 3]) -> [i32; 3] {
    let [o0, o1, o2] = o;
    match op {
        0 => o,
        1 => [o0, o1, o2.wrapping_sub(o0)],
        2 => [o0, o1.wrapping_sub(o0), o2],
        3 => [o0, o1.wrapping_sub(o0), o2.wrapping_sub(o0)],
        4 => [o0, o1.wrapping_sub(o0.wrapping_add(o2) >> 1), o2],
        5 => [
            o0,
            o1.wrapping_sub(o0.wrapping_add(o2) >> 1),
            o2.wrapping_sub(o0),
        ],
        6 => {
            // YCoCg-R from (R, G, B).
            let co = o0.wrapping_sub(o2);
            let t = o2.wrapping_add(co >> 1);
            let cg = o1.wrapping_sub(t);
            let y = t.wrapping_add(cg >> 1);
            [y, co, cg]
        }
        _ => unreachable!(),
    }
}

/// The squeeze steps the decoder uses when the transform lists none.
pub(crate) fn default_squeeze(channels: &[Channel]) -> Vec<SqueezeStep> {
    let meta = channels.iter().take_while(|c| c.shift.is_none()).count();
    let (mut w, mut h) = (channels[meta].width, channels[meta].height);
    let nc = channels.len() - meta;
    let mut steps = Vec::new();
    if nc > 2 && (channels[meta + 1].width, channels[meta + 1].height) == (w, h) {
        let s = SqueezeStep {
            horizontal: true,
            in_place: false,
            begin_channel: meta as u32 + 1,
            num_channels: 2,
        };
        if w > 1 {
            steps.push(s);
        }
        if h > 1 {
            steps.push(SqueezeStep {
                horizontal: false,
                ..s
            });
        }
    }
    const MAX_FIRST_PREVIEW_SIZE: usize = 8;
    let s = SqueezeStep {
        horizontal: false,
        in_place: true,
        begin_channel: meta as u32,
        num_channels: nc as u32,
    };
    if w <= h && h > MAX_FIRST_PREVIEW_SIZE {
        steps.push(s);
        h = h.div_ceil(2);
    }
    while w > MAX_FIRST_PREVIEW_SIZE || h > MAX_FIRST_PREVIEW_SIZE {
        if w > MAX_FIRST_PREVIEW_SIZE {
            steps.push(SqueezeStep {
                horizontal: true,
                ..s
            });
            w = w.div_ceil(2);
        }
        if h > MAX_FIRST_PREVIEW_SIZE {
            steps.push(s);
            h = h.div_ceil(2);
        }
    }
    steps
}

/// The decoder's smoothing term for a pair, from the previous pair's
/// second value, this pair's average and the next pair's.
fn tendency(b: i64, a: i64, n: i64) -> i64 {
    let mut diff = 0;
    if b >= a && a >= n {
        diff = (4 * b - 3 * n - a + 6) / 12;
        if diff - (diff & 1) > 2 * (b - a) {
            diff = 2 * (b - a) + 1;
        }
        if diff + (diff & 1) > 2 * (a - n) {
            diff = 2 * (a - n);
        }
    } else if b <= a && a <= n {
        diff = (4 * b - 3 * n - a - 6) / 12;
        if diff + (diff & 1) < 2 * (b - a) {
            diff = 2 * (b - a) - 1;
        }
        if diff - (diff & 1) < 2 * (a - n) {
            diff = 2 * (a - n);
        }
    }
    diff
}

/// Squeeze a line of samples into averages and residuals.
fn squeeze_line(line: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let pairs = line.len() / 2;
    let navg = line.len().div_ceil(2);
    let mut avg = vec![0i32; navg];
    let mut diffs = vec![0i64; pairs];
    for x in 0..pairs {
        let (a, b) = (i64::from(line[2 * x]), i64::from(line[2 * x + 1]));
        let diff = a - b;
        avg[x] = (a - diff / 2) as i32;
        diffs[x] = diff;
    }
    if navg > pairs {
        avg[navg - 1] = line[line.len() - 1];
    }
    let mut res = vec![0i32; pairs];
    let mut prev_b = i64::from(avg.first().copied().unwrap_or(0));
    for x in 0..pairs {
        let next = if x + 1 < navg { avg[x + 1] } else { avg[x] };
        let t = tendency(prev_b, i64::from(avg[x]), i64::from(next));
        res[x] = (diffs[x] - t) as i32;
        prev_b = i64::from(line[2 * x + 1]);
    }
    (avg, res)
}

fn apply_squeeze(channels: &mut Vec<Channel>, s: &SqueezeStep) -> Result<(), String> {
    let begin = s.begin_channel as usize;
    let end = begin + s.num_channels as usize;
    if end > channels.len() || s.num_channels == 0 {
        return Err(format!("squeeze of channels {begin}..{end}"));
    }
    if channels[begin].shift.is_none() != channels[end - 1].shift.is_none() {
        return Err("a squeeze mixes meta and image channels".into());
    }
    if channels[begin].shift.is_none() && !s.in_place {
        return Err("a squeeze of meta channels must be in place".into());
    }
    let insert_at = if s.in_place { end } else { channels.len() };
    let mut residuals = Vec::new();
    for c in begin..end {
        let ch = &channels[c];
        if ch.width == 0 || ch.height == 0 {
            return Err("a squeeze of an empty channel".into());
        }
        let (w, h) = (ch.width, ch.height);
        let shift = ch.shift.map(|(sx, sy)| {
            if s.horizontal {
                (sx + 1, sy)
            } else {
                (sx, sy + 1)
            }
        });
        let (avg, res) = if s.horizontal {
            let (aw, rw) = (w.div_ceil(2), w / 2);
            let mut avg = Vec::with_capacity(aw * h);
            let mut res = Vec::with_capacity(rw * h);
            for y in 0..h {
                let (a, r) = squeeze_line(&ch.samples[y * w..(y + 1) * w]);
                avg.extend(a);
                res.extend(r);
            }
            (
                Channel::new(aw, h, shift, avg),
                Channel::new(rw, h, shift, res),
            )
        } else {
            let (ah, rh) = (h.div_ceil(2), h / 2);
            let mut avg = vec![0; w * ah];
            let mut res = vec![0; w * rh];
            let mut column = vec![0; h];
            for x in 0..w {
                for y in 0..h {
                    column[y] = ch.samples[y * w + x];
                }
                let (a, r) = squeeze_line(&column);
                for (y, v) in a.into_iter().enumerate() {
                    avg[y * w + x] = v;
                }
                for (y, v) in r.into_iter().enumerate() {
                    res[y * w + x] = v;
                }
            }
            (
                Channel::new(w, ah, shift, avg),
                Channel::new(w, rh, shift, res),
            )
        };
        channels[c] = avg;
        residuals.push(res);
    }
    for (i, r) in residuals.into_iter().enumerate() {
        channels.insert(insert_at + i, r);
    }
    Ok(())
}

/// The decoder's implicit palette entry for an index past the explicit
/// ones (`index - size`) or below zero, for channel `c` of a palette at
/// `bits` per sample.
pub(crate) fn implicit_entry(index: i64, size: usize, c: usize, bits: u32) -> i32 {
    const DELTA_PALETTE: [[i32; 3]; 72] = [
        [0, 0, 0],
        [4, 4, 4],
        [11, 0, 0],
        [0, 0, -13],
        [0, -12, 0],
        [-10, -10, -10],
        [-18, -18, -18],
        [-27, -27, -27],
        [-18, -18, 0],
        [0, 0, -32],
        [-32, 0, 0],
        [-37, -37, -37],
        [0, -32, -32],
        [24, 24, 45],
        [50, 50, 50],
        [-45, -24, -24],
        [-24, -45, -45],
        [0, -24, -24],
        [-34, -34, 0],
        [-24, 0, -24],
        [-45, -45, -24],
        [64, 64, 64],
        [-32, 0, -32],
        [0, -32, 0],
        [-32, 0, 32],
        [-24, -45, -24],
        [45, 24, 45],
        [24, -24, -45],
        [-45, -24, 24],
        [80, 80, 80],
        [64, 0, 0],
        [0, 0, -64],
        [0, -64, -64],
        [-24, -24, 45],
        [96, 96, 96],
        [64, 64, 0],
        [45, -24, -24],
        [34, -34, 0],
        [112, 112, 112],
        [24, -45, -45],
        [45, 45, -24],
        [0, -32, 32],
        [24, -24, 45],
        [0, 96, 96],
        [45, -24, 24],
        [24, -45, -24],
        [-24, -45, 24],
        [0, -64, 0],
        [96, 0, 0],
        [128, 128, 128],
        [64, 0, 64],
        [144, 144, 144],
        [96, 96, 0],
        [-36, -36, 36],
        [45, -24, -45],
        [45, -45, -24],
        [0, 0, -96],
        [0, 128, 128],
        [0, 96, 0],
        [45, 24, -45],
        [-128, 0, 0],
        [24, -45, 24],
        [-45, 24, -45],
        [64, 0, -64],
        [64, -64, -64],
        [96, 0, 96],
        [45, -45, 24],
        [24, 45, -45],
        [64, 64, -64],
        [128, 128, 0],
        [0, 0, -128],
        [-24, 45, -45],
    ];
    if c >= 3 {
        return 0;
    }
    let bits = bits.min(24) as usize;
    if index < 0 {
        let idx = ((-(index + 1)) as usize) % 143;
        let mut v = DELTA_PALETTE[(idx + 1) >> 1][c] * if idx & 1 == 0 { -1 } else { 1 };
        if bits > 8 {
            v *= 1 << (bits - 8);
        }
        return v;
    }
    let scale = |v: usize| ((v * ((1usize << bits) - 1)) / 4) as i32;
    let cube = index as usize - size;
    if cube < 64 {
        let shifted = cube >> (c * 2);
        scale(shifted % 4) + (1 << (bits as isize - 3).max(0))
    } else {
        let i = (cube - 64) % 125;
        let v = match c {
            0 => i,
            1 => i / 5,
            _ => i / 25,
        };
        scale(v % 5)
    }
}

fn apply_palette(
    channels: &mut Vec<Channel>,
    p: &Palette,
    bits: u32,
    wp: WeightedParams,
) -> Result<(), String> {
    let begin = p.begin_channel as usize;
    let nc = p.num_channels as usize;
    check_equal(channels, begin, nc)?;
    if p.entries.iter().any(|e| e.len() != nc) || (p.num_deltas as usize) > p.entries.len() {
        return Err("palette entries of the wrong length".into());
    }
    let size = p.entries.len();
    let nd = p.num_deltas as usize;
    let (w, h) = (channels[begin].width, channels[begin].height);
    let mut index = vec![0i32; w * h];

    // Colours: an exact match among the explicit colours, else the
    // implicit cube.
    let mut lookup: std::collections::HashMap<Vec<i32>, i32> = Default::default();
    for (i, e) in p.entries.iter().enumerate().skip(nd) {
        lookup.entry(e.clone()).or_insert(i as i32);
    }
    let implicit_colour = |colour: &[i32]| -> Option<i32> {
        if nc > 3 && colour[3..].iter().any(|&v| v != 0) {
            return None;
        }
        (0..64 + 125).find_map(|k| {
            let idx = (size + k) as i64;
            (0..nc)
                .all(|c| implicit_entry(idx, size, c, bits) == colour[c])
                .then_some(idx as i32)
        })
    };

    if p.predictor == Predictor::Zero || nd == 0 {
        // Zero predictor: a delta entry is its own value.
        let mut colour = vec![0i32; nc];
        for i in 0..w * h {
            for c in 0..nc {
                colour[c] = channels[begin + c].samples[i];
            }
            index[i] = if let Some(&k) = lookup.get(&colour) {
                k
            } else if let Some(k) = (0..nd).find(|&k| p.entries[k] == colour) {
                k as i32
            } else if let Some(k) = implicit_colour(&colour) {
                k
            } else {
                return Err(format!("colour {colour:?} is not in the palette"));
            };
        }
    } else {
        // Deltas are added to a prediction from the decoded output, per
        // channel; a sample takes a delta only if it fits every channel.
        let mut wps: Vec<Option<WeightedState>> = (0..nc)
            .map(|_| (p.predictor == Predictor::Weighted).then(|| WeightedState::new(wp, w)))
            .collect();
        let rows = |c: usize, y: usize| &channels[begin + c].samples[y * w..(y + 1) * w];
        let mut colour = vec![0i32; nc];
        let mut preds = vec![0i64; nc];
        for y in 0..h {
            for x in 0..w {
                for c in 0..nc {
                    colour[c] = rows(c, y)[x];
                    let row = rows(c, y);
                    let top = if y > 0 { rows(c, y - 1) } else { &[][..] };
                    let toptop = if y > 1 { rows(c, y - 2) } else { &[][..] };
                    let n = Neighbours::at(row, top, toptop, x, y);
                    preds[c] = match &mut wps[c] {
                        Some(s) => s.predict(x, y & 1, &n).0,
                        None => p.predictor.predict(&n, 0),
                    };
                }
                let delta_fits =
                    |e: &[i32]| (0..nc).all(|c| preds[c] + i64::from(e[c]) == i64::from(colour[c]));
                let k = if let Some(&k) = lookup.get(&colour) {
                    k
                } else if let Some(k) = (0..nd).find(|&k| delta_fits(&p.entries[k])) {
                    k as i32
                } else if let Some(k) = implicit_colour(&colour) {
                    k
                } else if let Some(k) = (1..=143i64).find(|&k| {
                    let e: Vec<i32> = (0..nc).map(|c| implicit_entry(-k, size, c, bits)).collect();
                    delta_fits(&e)
                }) {
                    -(k as i32)
                } else {
                    return Err(format!("colour {colour:?} is not in the palette"));
                };
                index[y * w + x] = k;
                for c in 0..nc {
                    if let Some(s) = &mut wps[c] {
                        s.update(colour[c], x, y & 1);
                    }
                }
            }
        }
    }

    let mut meta = Vec::with_capacity(size * nc);
    for c in 0..nc {
        for e in &p.entries {
            meta.push(e[c]);
        }
    }
    let shift = channels[begin].shift;
    channels.drain(begin + 1..begin + nc);
    channels[begin] = Channel::new(w, h, shift, index);
    channels.insert(0, Channel::new(size, nc, None, meta));
    Ok(())
}
