//! The meta-adaptive (MA) tree: learned from samples, sent breadth first as
//! the decoder reads it, walked per sample to give its predictor, offset,
//! multiplier and context.

use std::collections::VecDeque;

use super::predict::{NUM_NONREF_PROPERTIES, PROPERTIES_PER_REFERENCE, Predictor};
use crate::encode::entropy::{HybridUint, Token, pack_signed};

/// A node, as the decoder holds it: `left` and `right` index the node list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    /// Samples whose `property` is above `value` go to `left`.
    Split {
        property: u32,
        value: i32,
        left: usize,
        right: usize,
    },
    Leaf {
        predictor: Predictor,
        offset: i32,
        multiplier: u32,
        /// The leaf's context: its place among the leaves, breadth first.
        id: u32,
    },
}

/// A tree in the decoder's order.
#[derive(Clone, Debug)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

/// A tree as built: nested.
#[derive(Clone, Debug)]
pub enum Built {
    Split {
        property: u32,
        value: i32,
        left: Box<Built>,
        right: Box<Built>,
    },
    Leaf(Predictor),
}

const SPLIT_VALUE_CONTEXT: u32 = 0;
const PROPERTY_CONTEXT: u32 = 1;
const PREDICTOR_CONTEXT: u32 = 2;
const OFFSET_CONTEXT: u32 = 3;
const MULTIPLIER_LOG_CONTEXT: u32 = 4;
const MULTIPLIER_BITS_CONTEXT: u32 = 5;
pub const TREE_CONTEXTS: usize = 6;

impl Tree {
    /// One leaf.
    pub fn single(predictor: Predictor) -> Self {
        Tree::from_built(&Built::Leaf(predictor))
    }

    /// Breadth first, as the decoder reads it.
    pub fn from_built(root: &Built) -> Self {
        let mut nodes = Vec::new();
        let mut queue = VecDeque::from([root]);
        let mut leaves = 0;
        // Children of node i are placed after everything queued before
        // them, which is breadth-first order.
        let mut pending: Vec<&Built> = Vec::new();
        while let Some(b) = queue.pop_front() {
            pending.push(b);
            if let Built::Split { left, right, .. } = b {
                queue.push_back(left);
                queue.push_back(right);
            }
        }
        // Assign indices: in BFS order, a split's children are the next two
        // unassigned nodes after all earlier splits' children.
        let mut next_child = 1;
        for b in &pending {
            match b {
                Built::Split {
                    property, value, ..
                } => {
                    nodes.push(Node::Split {
                        property: *property,
                        value: *value,
                        left: next_child,
                        right: next_child + 1,
                    });
                    next_child += 2;
                }
                Built::Leaf(p) => {
                    nodes.push(Node::Leaf {
                        predictor: *p,
                        offset: 0,
                        multiplier: 1,
                        id: leaves,
                    });
                    leaves += 1;
                }
            }
        }
        Tree { nodes }
    }

    pub fn num_leaves(&self) -> usize {
        self.nodes.len().div_ceil(2)
    }

    /// The decoder's property count: the largest split property plus one.
    pub fn num_properties(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                Node::Split { property, .. } => Some(*property as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0)
            + 1
    }

    /// How many reference properties the decoder computes for this tree.
    pub fn num_reference_properties(&self) -> usize {
        self.num_properties()
            .saturating_sub(NUM_NONREF_PROPERTIES)
            .next_multiple_of(PROPERTIES_PER_REFERENCE)
    }

    /// Whether any leaf predicts with the weighted predictor or any split
    /// reads its property (15).
    pub fn uses_weighted(&self) -> bool {
        self.nodes.iter().any(|n| match n {
            Node::Split { property, .. } => *property == 15,
            Node::Leaf { predictor, .. } => *predictor == Predictor::Weighted,
        })
    }

    #[inline]
    pub fn walk(&self, props: &[i32]) -> usize {
        let mut i = 0;
        loop {
            match self.nodes[i] {
                Node::Split {
                    property,
                    value,
                    left,
                    right,
                } => {
                    i = if props[property as usize] > value {
                        left
                    } else {
                        right
                    };
                }
                Node::Leaf { .. } => return i,
            }
        }
    }

    /// The tree's tokens in its six contexts.
    pub fn tokens(&self) -> Vec<Token> {
        let mut out = Vec::with_capacity(self.nodes.len() * 5);
        for n in &self.nodes {
            match *n {
                Node::Split {
                    property, value, ..
                } => {
                    out.push(Token::new(PROPERTY_CONTEXT, property + 1));
                    out.push(Token::new(SPLIT_VALUE_CONTEXT, pack_signed(value)));
                }
                Node::Leaf {
                    predictor,
                    offset,
                    multiplier,
                    ..
                } => {
                    let log = multiplier.trailing_zeros().min(30);
                    let bits = (multiplier >> log) - 1;
                    out.push(Token::new(PROPERTY_CONTEXT, 0));
                    out.push(Token::new(PREDICTOR_CONTEXT, predictor as u32));
                    out.push(Token::new(OFFSET_CONTEXT, pack_signed(offset)));
                    out.push(Token::new(MULTIPLIER_LOG_CONTEXT, log));
                    out.push(Token::new(MULTIPLIER_BITS_CONTEXT, bits));
                }
            }
        }
        out
    }
}

/// How a tree is learned.
#[derive(Clone, Debug)]
pub struct LearnOptions {
    /// The predictors a leaf may use.
    pub predictors: Vec<Predictor>,
    /// Bits a split must save to be made.
    pub split_cost: f64,
    /// The most leaves.
    pub max_leaves: usize,
    /// The most cut points tried per property.
    pub max_cuts: usize,
}

/// Samples: per property, its values; per predictor, the residuals.
pub struct Samples {
    pub properties: Vec<u32>,
    pub values: Vec<Vec<i32>>,
    pub predictors: Vec<Predictor>,
    pub residuals: Vec<Vec<i64>>,
    /// Each sample stands for this many samples (for sampled images).
    pub weight: f64,
}

impl Samples {
    pub fn new(properties: Vec<u32>, predictors: Vec<Predictor>) -> Self {
        Samples {
            values: vec![Vec::new(); properties.len()],
            residuals: vec![Vec::new(); predictors.len()],
            properties,
            predictors,
            weight: 1.0,
        }
    }

    pub fn len(&self) -> usize {
        self.residuals.first().map_or(0, Vec::len)
    }
}

const TOKENS: usize = 64;
const SPLIT: HybridUint = HybridUint::new(4, 2, 0);

/// Token and raw-bit count of a residual.
#[inline]
fn token_of(r: i64) -> (u8, u8) {
    let v = pack_signed(r.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
    let s = SPLIT.split(v);
    ((s.token as usize).min(TOKENS - 1) as u8, s.nbits as u8)
}

/// Bits to code a histogram (entropy plus raw bits).
fn cost(hist: &[u64; TOKENS], raw: u64) -> f64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut bits = raw as f64;
    for &c in hist {
        if c > 0 {
            bits += c as f64 * (t / c as f64).log2();
        }
    }
    bits
}

/// Learn a tree from `samples`.
pub fn learn(samples: &Samples, options: &LearnOptions) -> Built {
    let n = samples.len();
    if n == 0 || samples.predictors.is_empty() {
        return Built::Leaf(
            options
                .predictors
                .first()
                .copied()
                .unwrap_or(Predictor::Gradient),
        );
    }
    let np = samples.predictors.len();
    // Tokens per predictor per sample.
    let tokens: Vec<Vec<(u8, u8)>> = samples
        .residuals
        .iter()
        .map(|rs| rs.iter().map(|&r| token_of(r)).collect())
        .collect();

    // Cut points per property: values at quantiles of the samples.
    let cuts: Vec<Vec<i32>> = samples
        .values
        .iter()
        .map(|vals| {
            let mut sorted = vals.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() <= 1 {
                return Vec::new();
            }
            // Cut after a value: everything above goes left.
            let candidates = &sorted[..sorted.len() - 1];
            if candidates.len() <= options.max_cuts {
                return candidates.to_vec();
            }
            let mut by_count = vals.clone();
            by_count.sort_unstable();
            let mut out: Vec<i32> = (1..=options.max_cuts)
                .map(|k| by_count[(k * (n - 1)) / (options.max_cuts + 1)])
                .filter(|v| v < sorted.last().unwrap())
                .collect();
            out.dedup();
            out
        })
        .collect();
    // Each sample's bin per property: how many cuts lie below its value.
    let bins: Vec<Vec<u16>> = samples
        .values
        .iter()
        .zip(&cuts)
        .map(|(vals, cs)| {
            vals.iter()
                .map(|v| cs.partition_point(|c| c < v) as u16)
                .collect()
        })
        .collect();

    let mut leaves = 1;
    let mut indices: Vec<u32> = (0..n as u32).collect();
    let ctx = Context {
        samples,
        options,
        tokens: &tokens,
        cuts: &cuts,
        bins: &bins,
        np,
    };
    let mut ranges = vec![(i32::MIN, i32::MAX); samples.properties.len()];
    build(&ctx, &mut indices[..], &mut leaves, &mut ranges)
}

struct Context<'a> {
    samples: &'a Samples,
    options: &'a LearnOptions,
    tokens: &'a [Vec<(u8, u8)>],
    cuts: &'a [Vec<i32>],
    bins: &'a [Vec<u16>],
    np: usize,
}

/// The best predictor for `indices`, and its cost.
fn best_predictor(ctx: &Context, indices: &[u32]) -> (usize, f64) {
    (0..ctx.np)
        .map(|p| {
            let mut h = [0u64; TOKENS];
            let mut raw = 0;
            for &i in indices {
                let (t, b) = ctx.tokens[p][i as usize];
                h[t as usize] += 1;
                raw += u64::from(b);
            }
            (p, cost(&h, raw))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

fn build(
    ctx: &Context,
    indices: &mut [u32],
    leaves: &mut usize,
    ranges: &mut Vec<(i32, i32)>,
) -> Built {
    let (best_p, node_cost) = best_predictor(ctx, indices);
    let leaf = Built::Leaf(ctx.samples.predictors[best_p]);
    if *leaves >= ctx.options.max_leaves || indices.len() < 2 {
        return leaf;
    }

    // The best split: per property, histograms per bin per predictor, then
    // every cut from the top.
    let mut best: Option<(f64, usize, usize)> = None; // (cost, property, cut)
    for (q, cuts) in ctx.cuts.iter().enumerate() {
        if cuts.is_empty() {
            continue;
        }
        let (lo, hi) = ranges[q];
        let nb = cuts.len() + 1;
        let mut hist = vec![[0u64; TOKENS]; nb * ctx.np];
        let mut raw = vec![0u64; nb * ctx.np];
        let mut count = vec![0usize; nb];
        for &i in indices.iter() {
            let b = ctx.bins[q][i as usize] as usize;
            count[b] += 1;
            for p in 0..ctx.np {
                let (t, r) = ctx.tokens[p][i as usize];
                hist[b * ctx.np + p][t as usize] += 1;
                raw[b * ctx.np + p] += u64::from(r);
            }
        }
        // Left: bins above the cut. Sweep the cut downwards.
        let mut left_h = vec![[0u64; TOKENS]; ctx.np];
        let mut left_r = vec![0u64; ctx.np];
        let mut total_h = vec![[0u64; TOKENS]; ctx.np];
        let mut total_r = vec![0u64; ctx.np];
        for b in 0..nb {
            for p in 0..ctx.np {
                for t in 0..TOKENS {
                    total_h[p][t] += hist[b * ctx.np + p][t];
                }
                total_r[p] += raw[b * ctx.np + p];
            }
        }
        let mut left_n = 0;
        for cut in (0..cuts.len()).rev() {
            // Bin cut + 1 moves left: values above cuts[cut].
            let b = cut + 1;
            left_n += count[b];
            for p in 0..ctx.np {
                for t in 0..TOKENS {
                    left_h[p][t] += hist[b * ctx.np + p][t];
                }
                left_r[p] += raw[b * ctx.np + p];
            }
            if left_n == 0 || left_n == indices.len() {
                continue;
            }
            let v = cuts[cut];
            if v < lo || v >= hi {
                continue;
            }
            let lc = (0..ctx.np)
                .map(|p| cost(&left_h[p], left_r[p]))
                .fold(f64::INFINITY, f64::min);
            let rc = (0..ctx.np)
                .map(|p| {
                    let mut h = total_h[p];
                    for t in 0..TOKENS {
                        h[t] -= left_h[p][t];
                    }
                    cost(&h, total_r[p] - left_r[p])
                })
                .fold(f64::INFINITY, f64::min);
            let c = lc + rc;
            if best.is_none_or(|(bc, _, _)| c < bc) {
                best = Some((c, q, cut));
            }
        }
    }
    let Some((split_cost, q, cut)) = best else {
        return leaf;
    };
    let gain = (node_cost - split_cost) * ctx.samples.weight;
    if gain <= ctx.options.split_cost {
        return leaf;
    }
    let value = ctx.cuts[q][cut];
    // Partition: left (above) first.
    let mut mid = 0;
    for i in 0..indices.len() {
        if ctx.samples.values[q][indices[i] as usize] > value {
            indices.swap(i, mid);
            mid += 1;
        }
    }
    *leaves += 1;
    let (left_idx, right_idx) = indices.split_at_mut(mid);
    let old = ranges[q];
    ranges[q] = (value + 1, old.1);
    let left = build(ctx, left_idx, leaves, ranges);
    ranges[q] = (old.0, value);
    let right = build(ctx, right_idx, leaves, ranges);
    ranges[q] = old;
    Built::Split {
        property: ctx.samples.properties[q],
        value,
        left: Box::new(left),
        right: Box::new(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breadth_first_with_leaf_ids_in_order() {
        let t = Built::Split {
            property: 0,
            value: 0,
            left: Box::new(Built::Split {
                property: 2,
                value: 5,
                left: Box::new(Built::Leaf(Predictor::West)),
                right: Box::new(Built::Leaf(Predictor::North)),
            }),
            right: Box::new(Built::Leaf(Predictor::Zero)),
        };
        let tree = Tree::from_built(&t);
        assert_eq!(tree.nodes.len(), 5);
        assert!(matches!(
            tree.nodes[1],
            Node::Split {
                left: 3,
                right: 4,
                ..
            }
        ));
        assert!(matches!(
            tree.nodes[2],
            Node::Leaf {
                predictor: Predictor::Zero,
                id: 0,
                ..
            }
        ));
        assert!(matches!(
            tree.nodes[3],
            Node::Leaf {
                predictor: Predictor::West,
                id: 1,
                ..
            }
        ));
        let mut props = [0i32; 16];
        props[0] = 1;
        props[2] = 9;
        assert_eq!(tree.walk(&props), 3);
        props[0] = 0;
        assert_eq!(tree.walk(&props), 2);
    }

    #[test]
    fn learning_separates_what_differs() {
        // Property 0 tells two populations apart: one smooth, one noisy.
        let mut s = Samples::new(vec![0, 2], vec![Predictor::Zero, Predictor::West]);
        for i in 0..4000 {
            let c = i % 2;
            s.values[0].push(c);
            s.values[1].push(i);
            let r = if c == 0 {
                0
            } else {
                ((i * 7919) % 200) as i64 - 100
            };
            s.residuals[0].push(r);
            s.residuals[1].push(r);
        }
        let opts = LearnOptions {
            predictors: vec![Predictor::Zero],
            split_cost: 16.0,
            max_leaves: 64,
            max_cuts: 16,
        };
        match learn(&s, &opts) {
            Built::Split {
                property: 0,
                value: 0,
                ..
            } => {}
            other => panic!("{other:?}"),
        }
    }
}
