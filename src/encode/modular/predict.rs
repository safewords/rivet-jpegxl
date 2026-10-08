//! Prediction as the decoder does it: the neighbourhood of a sample (with
//! the edge rules), the 14 predictors, the self-correcting weighted
//! predictor, and the properties the MA tree splits on.

/// The predictors, as the tree codes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Predictor {
    Zero = 0,
    West = 1,
    North = 2,
    AverageWestNorth = 3,
    Select = 4,
    Gradient = 5,
    Weighted = 6,
    NorthEast = 7,
    NorthWest = 8,
    WestWest = 9,
    AverageWestNorthWest = 10,
    AverageNorthNorthWest = 11,
    AverageNorthNorthEast = 12,
    AverageAll = 13,
}

impl Predictor {
    pub const ALL: [Predictor; 14] = [
        Predictor::Zero,
        Predictor::West,
        Predictor::North,
        Predictor::AverageWestNorth,
        Predictor::Select,
        Predictor::Gradient,
        Predictor::Weighted,
        Predictor::NorthEast,
        Predictor::NorthWest,
        Predictor::WestWest,
        Predictor::AverageWestNorthWest,
        Predictor::AverageNorthNorthWest,
        Predictor::AverageNorthNorthEast,
        Predictor::AverageAll,
    ];

    /// The prediction from a neighbourhood (`wp`: the weighted predictor's,
    /// used only by `Weighted`).
    #[inline(always)]
    pub fn predict(self, n: &Neighbours, wp: i64) -> i64 {
        let (left, top, topleft, topright) = (
            i64::from(n.left),
            i64::from(n.top),
            i64::from(n.topleft),
            i64::from(n.topright),
        );
        match self {
            Predictor::Zero => 0,
            Predictor::West => left,
            Predictor::North => top,
            Predictor::AverageWestNorth => (left + top) / 2,
            Predictor::Select => {
                let p = left + top - topleft;
                if (p - left).abs() < (p - top).abs() {
                    left
                } else {
                    top
                }
            }
            Predictor::Gradient => clamped_gradient(left, top, topleft),
            Predictor::Weighted => wp,
            Predictor::NorthEast => topright,
            Predictor::NorthWest => topleft,
            Predictor::WestWest => i64::from(n.leftleft),
            Predictor::AverageWestNorthWest => (left + topleft) / 2,
            Predictor::AverageNorthNorthWest => (top + topleft) / 2,
            Predictor::AverageNorthNorthEast => (top + topright) / 2,
            Predictor::AverageAll => {
                (6 * top - 2 * i64::from(n.toptop)
                    + 7 * left
                    + i64::from(n.leftleft)
                    + i64::from(n.toprightright)
                    + 3 * topright
                    + 8)
                    / 16
            }
        }
    }
}

#[inline(always)]
pub fn clamped_gradient(left: i64, top: i64, topleft: i64) -> i64 {
    let (lo, hi) = (left.min(top), left.max(top));
    if topleft < lo {
        hi
    } else if topleft > hi {
        lo
    } else {
        left + top - topleft
    }
}

/// A sample's neighbours, as the decoder fills them in at the edges.
#[derive(Clone, Copy, Debug, Default)]
pub struct Neighbours {
    pub left: i32,
    pub top: i32,
    pub toptop: i32,
    pub topleft: i32,
    pub topright: i32,
    pub leftleft: i32,
    pub toprightright: i32,
}

impl Neighbours {
    /// For `row[x]`, with `top` / `toptop` the rows above (empty when
    /// there are none).
    #[inline(always)]
    pub fn at(row: &[i32], top: &[i32], toptop: &[i32], x: usize, y: usize) -> Self {
        let w = row.len();
        let left = if x > 0 {
            row[x - 1]
        } else if y > 0 {
            top[0]
        } else {
            0
        };
        let up = if y > 0 { top[x] } else { left };
        let topleft = if x > 0 && y > 0 { top[x - 1] } else { left };
        let topright = if x + 1 < w && y > 0 { top[x + 1] } else { up };
        let leftleft = if x > 1 { row[x - 2] } else { left };
        let toptop = if y > 1 { toptop[x] } else { up };
        let toprightright = if x + 2 < w && y > 0 {
            top[x + 2]
        } else {
            topright
        };
        Neighbours {
            left,
            top: up,
            toptop,
            topleft,
            topright,
            leftleft,
            toprightright,
        }
    }
}

/// The weighted predictor's parameters (the group header's
/// `WeightedHeader`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightedParams {
    pub p1c: u32,
    pub p2c: u32,
    pub p3c: [u32; 5],
    pub w: [u32; 4],
}

impl Default for WeightedParams {
    fn default() -> Self {
        WeightedParams {
            p1c: 16,
            p2c: 10,
            p3c: [7, 7, 7, 0, 0],
            w: [0xd, 0xc, 0xc, 0xc],
        }
    }
}

impl WeightedParams {
    /// The header: `all_default`, or every field.
    pub(crate) fn write(&self, w: &mut crate::encode::bits::BitWriter) {
        if *self == WeightedParams::default() {
            w.bit(true);
            return;
        }
        w.bit(false);
        w.write(5, self.p1c);
        w.write(5, self.p2c);
        for &c in &self.p3c {
            w.write(5, c);
        }
        for &k in &self.w {
            w.write(4, k);
        }
    }
}

const PRED_EXTRA_BITS: i64 = 3;
const PREDICTION_ROUND: i64 = ((1 << PRED_EXTRA_BITS) >> 1) - 1;
const DIVLOOKUP: [u32; 64] = {
    let mut t = [0u32; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = (1 << 24) / (i as u32 + 1);
        i += 1;
    }
    t
};

/// The weighted predictor's state over one channel, as the decoder keeps
/// it: two rows of errors.
pub struct WeightedState {
    params: WeightedParams,
    xsize: usize,
    prediction: [i64; 4],
    pred: i64,
    pred_errors: Vec<[u32; 4]>,
    error: Vec<i32>,
}

impl WeightedState {
    pub fn new(params: WeightedParams, xsize: usize) -> Self {
        let n = (xsize + 1) * 2;
        WeightedState {
            params,
            xsize,
            prediction: [0; 4],
            pred: 0,
            pred_errors: vec![[0; 4]; n],
            error: vec![0; n],
        }
    }

    fn rows(&self, y: usize) -> (usize, usize) {
        if y & 1 != 0 {
            (0, self.xsize + 1)
        } else {
            (self.xsize + 1, 0)
        }
    }

    /// The prediction for `(x, y)`, and property 15 (the largest recent
    /// error).
    #[inline]
    pub fn predict(&mut self, x: usize, y: usize, n: &Neighbours) -> (i64, i32) {
        let (cur, prev) = self.rows(y);
        let ne_x = if x + 1 < self.xsize { x + 1 } else { x };
        let nw_x = x.saturating_sub(1);
        let (en, ene, enw) = (
            self.pred_errors[prev + x],
            self.pred_errors[prev + ne_x],
            self.pred_errors[prev + nw_x],
        );
        let mut weights = [0u32; 4];
        for k in 0..4 {
            let err = en[k].wrapping_add(ene[k]).wrapping_add(enw[k]);
            let shift = (u64::from(err) + 1).ilog2().saturating_sub(5);
            let div = DIVLOOKUP[(err >> shift) as usize];
            weights[k] = 4 + ((self.params.w[k] * div) >> shift);
        }
        let te_w = i64::from(self.error[cur + x]);
        let te_n = i64::from(self.error[prev + 1 + x]);
        let te_nw = i64::from(self.error[prev + 1 + nw_x]);
        let te_ne = i64::from(self.error[prev + 1 + ne_x]);
        let sum_wn = te_n + te_w;
        let mut p = te_w;
        for e in [te_n, te_nw, te_ne] {
            if e.abs() > p.abs() {
                p = e;
            }
        }
        let bits = |v: i32| i64::from(v) << PRED_EXTRA_BITS;
        let (nn_, w_, ne_, nw_, tt_) = (
            bits(n.top),
            bits(n.left),
            bits(n.topright),
            bits(n.topleft),
            bits(n.toptop),
        );
        let c = &self.params;
        let p0 = w_ + ne_ - nn_;
        let p1 = nn_ - (((sum_wn + te_ne) * i64::from(c.p1c)) >> 5);
        let p2 = w_ - (((sum_wn + te_nw) * i64::from(c.p2c)) >> 5);
        let p3 = nn_
            - ((te_nw * i64::from(c.p3c[0])
                + te_n * i64::from(c.p3c[1])
                + te_ne * i64::from(c.p3c[2])
                + (tt_ - nn_) * i64::from(c.p3c[3])
                + (nw_ - w_) * i64::from(c.p3c[4]))
                >> 5);
        let total: u64 = weights.iter().map(|&w| u64::from(w)).sum();
        let log_weight = total.ilog2();
        let ws: [i64; 4] = weights.map(|w| i64::from(w >> (log_weight - 4)));
        let weight_sum: i64 = ws.iter().sum();
        let preds = [p0, p1, p2, p3];
        let mut sum = (weight_sum >> 1) - 1;
        for k in 0..4 {
            sum += ws[k] * preds[k];
        }
        let mut pred = (sum * i64::from(DIVLOOKUP[(weight_sum - 1) as usize])) >> 24;
        if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
            let mx = w_.max(ne_.max(nn_));
            let mn = w_.min(ne_.min(nn_));
            pred = mn.max(mx.min(pred));
        }
        self.prediction = preds;
        self.pred = pred;
        ((pred + PREDICTION_ROUND) >> PRED_EXTRA_BITS, p as i32)
    }

    /// After `(x, y)`'s value is known.
    #[inline]
    pub fn update(&mut self, value: i32, x: usize, y: usize) {
        let (cur, prev) = self.rows(y);
        let v = i64::from(value) << PRED_EXTRA_BITS;
        self.error[cur + x + 1] = (self.pred - v) as i32;
        let errs = self
            .prediction
            .map(|p| (((p - v).abs() + PREDICTION_ROUND) >> PRED_EXTRA_BITS) as u32);
        self.pred_errors[cur + x] = errs;
        let pe = &mut self.pred_errors[prev + x + 1];
        for k in 0..4 {
            pe[k] = pe[k].wrapping_add(errs[k]);
        }
    }
}

/// Properties 0..16 that do not refer to other channels.
pub const NUM_NONREF_PROPERTIES: usize = 16;
/// Properties each earlier channel of the same shape adds.
pub const PROPERTIES_PER_REFERENCE: usize = 4;

/// Fill properties 2..=14 for a sample (0, 1 and 15 are the caller's);
/// `last_9` carries property 9 along the row (0 at its start).
#[inline(always)]
pub fn local_properties(props: &mut [i32], n: &Neighbours, x: usize, y: usize, last_9: &mut i32) {
    props[2] = y as i32;
    props[3] = x as i32;
    props[4] = n.top.wrapping_abs();
    props[5] = n.left.wrapping_abs();
    props[6] = n.top;
    props[7] = n.left;
    props[8] = n.left.wrapping_sub(*last_9);
    props[9] = n.left.wrapping_add(n.top).wrapping_sub(n.topleft);
    *last_9 = props[9];
    props[10] = n.left.wrapping_sub(n.topleft);
    props[11] = n.topleft.wrapping_sub(n.top);
    props[12] = n.top.wrapping_sub(n.topright);
    props[13] = n.top.wrapping_sub(n.toptop);
    props[14] = n.left.wrapping_sub(n.leftleft);
}

/// The reference properties of row `y` for `channel`: per column, four per
/// earlier channel of the same size and shift, nearest first, `count` (a
/// multiple of 4) in all, zeros where there are fewer channels.
pub fn reference_row(
    channels: &[super::Channel],
    channel: usize,
    y: usize,
    count: usize,
) -> Vec<i32> {
    let me = &channels[channel];
    let width = me.width;
    let mut out = vec![0i32; width * count];
    if count == 0 {
        return out;
    }
    let mut offset = 0;
    for i in 0..channel {
        if offset >= count {
            break;
        }
        let j = channel - i - 1;
        let other = &channels[j];
        if (other.width, other.height, other.shift) != (me.width, me.height, me.shift) {
            continue;
        }
        let cur = &other.samples[y * width..(y + 1) * width];
        let prev = (y > 0).then(|| &other.samples[(y - 1) * width..y * width]);
        for x in 0..width {
            let v = cur[x];
            let left = if x > 0 { cur[x - 1] } else { 0 };
            let top = prev.map_or(left, |p| p[x]);
            let topleft = match prev {
                Some(p) if x > 0 => p[x - 1],
                _ => left,
            };
            let predicted = clamped_gradient(i64::from(left), i64::from(top), i64::from(topleft));
            let d = i64::from(v) - predicted;
            let r = &mut out[x * count + offset..];
            r[0] = v.wrapping_abs();
            r[1] = v;
            r[2] = d.wrapping_abs() as i32;
            r[3] = d as i32;
        }
        offset += PROPERTIES_PER_REFERENCE;
    }
    out
}
