//! The modular image: each channel's samples, predicted from their
//! neighbours, the residuals entropy coded in contexts a meta-adaptive (MA)
//! tree picks.
//!
//! The tree this encoder writes splits on the channel only, so each channel
//! has its own predictor and its own histogram. The predictor is the one of
//! West, North and the clamped gradient that codes the channel smallest.

use super::bits::BitWriter;
use super::entropy::{EntropyCode, Histograms, pack_signed};
use super::headers::GROUP_DIM;

/// A channel's samples, row by row.
pub(super) struct Plane {
    pub width: usize,
    pub height: usize,
    pub samples: Vec<i32>,
}

impl Plane {
    fn row(&self, rect: &Rect, y: usize) -> &[i32] {
        let start = (rect.y + y) * self.width + rect.x;
        &self.samples[start..start + rect.width]
    }
}

/// A part of the image: a group, or all of it.
#[derive(Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

/// The predictors this encoder chooses from, as the tree codes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Predictor {
    West = 1,
    North = 2,
    Gradient = 5,
}

impl Predictor {
    const ALL: [Predictor; 3] = [Predictor::Gradient, Predictor::West, Predictor::North];

    /// The prediction for `row[x]`, with the decoder's edge rules: off the
    /// left edge, West is the sample above (or 0 on the first row); off the
    /// top, North and North-west are West.
    #[inline(always)]
    fn predict(self, row: &[i32], top: Option<&[i32]>, x: usize) -> i32 {
        let left = if x > 0 {
            row[x - 1]
        } else {
            top.map_or(0, |t| t[0])
        };
        let (north, north_west) = match top {
            Some(t) => (t[x], if x > 0 { t[x - 1] } else { left }),
            None => (left, left),
        };
        match self {
            Predictor::West => left,
            Predictor::North => north,
            Predictor::Gradient => {
                let gradient = i64::from(left) + i64::from(north) - i64::from(north_west);
                let (lo, hi) = (left.min(north), left.max(north));
                gradient.clamp(i64::from(lo), i64::from(hi)) as i32
            }
        }
    }
}

/// Every residual of `plane` within `rect`, predicted by `predictor`, in
/// the order they are coded.
fn for_each_residual(plane: &Plane, rect: &Rect, predictor: Predictor, mut f: impl FnMut(i32)) {
    for y in 0..rect.height {
        let row = plane.row(rect, y);
        let top = (y > 0).then(|| plane.row(rect, y - 1));
        for x in 0..rect.width {
            f(row[x] - predictor.predict(row, top, x));
        }
    }
}

/// The predictor that codes `plane` smallest, by an estimate of the bits
/// its residuals take: the token entropy plus the raw bits.
fn choose_predictor(plane: &Plane) -> Predictor {
    let whole = Rect {
        x: 0,
        y: 0,
        width: plane.width,
        height: plane.height,
    };
    let cost = |predictor: Predictor| {
        let mut counts = [0u64; 64];
        let mut raw_bits = 0u64;
        for_each_residual(plane, &whole, predictor, |r| {
            let token = super::entropy::tokenize(pack_signed(r));
            counts[(token.symbol as usize).min(63)] += 1;
            raw_bits += u64::from(token.nbits);
        });
        let total = counts.iter().sum::<u64>() as f64;
        let entropy: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| c as f64 * (total / c as f64).log2())
            .sum();
        entropy + raw_bits as f64
    };
    Predictor::ALL
        .into_iter()
        .map(|p| (cost(p), p))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, p)| p)
        .unwrap()
}

/// A node of the MA tree.
enum Node {
    /// Samples whose `property` is above `value` go to `left`.
    Split {
        property: u32,
        value: i32,
        left: Box<Node>,
        right: Box<Node>,
    },
    Leaf {
        predictor: Predictor,
        channel: usize,
    },
}

/// A tree giving channels `first..=last` a leaf each, splitting on the
/// channel index (property 0).
fn channel_tree(first: usize, last: usize, predictors: &[Predictor]) -> Node {
    if first == last {
        return Node::Leaf {
            predictor: predictors[first],
            channel: first,
        };
    }
    let mid = (first + last) / 2;
    Node::Split {
        property: 0,
        value: mid as i32,
        left: Box::new(channel_tree(mid + 1, last, predictors)),
        right: Box::new(channel_tree(first, mid, predictors)),
    }
}

/// The tree's contexts for the tree itself.
const SPLIT_VALUE_CONTEXT: usize = 0;
const PROPERTY_CONTEXT: usize = 1;
const PREDICTOR_CONTEXT: usize = 2;
const OFFSET_CONTEXT: usize = 3;
const MULTIPLIER_LOG_CONTEXT: usize = 4;
const MULTIPLIER_BITS_CONTEXT: usize = 5;
const TREE_CONTEXTS: usize = 6;

/// The tree, as the values it is coded with in its contexts (breadth
/// first, as the decoder reads it), and each channel's data context (its
/// leaf's place among the leaves, in the same order).
fn flatten(tree: &Node, channels: usize) -> (Vec<(usize, u32)>, Vec<usize>) {
    let mut symbols = Vec::new();
    let mut context_of = vec![0; channels];
    let mut leaves = 0;
    let mut queue = std::collections::VecDeque::from([tree]);
    while let Some(node) = queue.pop_front() {
        match node {
            Node::Split {
                property,
                value,
                left,
                right,
            } => {
                symbols.push((PROPERTY_CONTEXT, property + 1));
                symbols.push((SPLIT_VALUE_CONTEXT, pack_signed(*value)));
                queue.push_back(left);
                queue.push_back(right);
            }
            Node::Leaf { predictor, channel } => {
                symbols.push((PROPERTY_CONTEXT, 0));
                symbols.push((PREDICTOR_CONTEXT, *predictor as u32));
                symbols.push((OFFSET_CONTEXT, pack_signed(0)));
                symbols.push((MULTIPLIER_LOG_CONTEXT, 0));
                symbols.push((MULTIPLIER_BITS_CONTEXT, 0));
                context_of[*channel] = leaves;
                leaves += 1;
            }
        }
    }
    (symbols, context_of)
}

/// The image's channels as the frame codes them, ready to write.
pub(super) struct ModularImage {
    planes: Vec<Plane>,
    predictors: Vec<Predictor>,
    tree_symbols: Vec<(usize, u32)>,
    context_of: Vec<usize>,
    tree_code: EntropyCode,
    data_code: EntropyCode,
    groups: Vec<Rect>,
}

impl ModularImage {
    /// `planes`: the colour channels then the extra ones, all one size.
    pub(super) fn new(planes: Vec<Plane>) -> Self {
        let (width, height) = (planes[0].width, planes[0].height);
        let predictors: Vec<Predictor> = planes.iter().map(choose_predictor).collect();
        let tree = channel_tree(0, planes.len() - 1, &predictors);
        let (tree_symbols, context_of) = flatten(&tree, planes.len());

        let mut tree_histograms = Histograms::new(TREE_CONTEXTS);
        for &(context, value) in &tree_symbols {
            tree_histograms.add(context, value);
        }
        let tree_code = EntropyCode::new(&tree_histograms);

        let dim = GROUP_DIM as usize;
        let groups: Vec<Rect> = (0..height.div_ceil(dim))
            .flat_map(|gy| {
                (0..width.div_ceil(dim)).map(move |gx| Rect {
                    x: gx * dim,
                    y: gy * dim,
                    width: dim.min(width - gx * dim),
                    height: dim.min(height - gy * dim),
                })
            })
            .collect();

        let leaves = predictors.len();
        let mut data_histograms = Histograms::new(leaves);
        for rect in &groups {
            for (c, plane) in planes.iter().enumerate() {
                for_each_residual(plane, rect, predictors[c], |r| {
                    data_histograms.add(context_of[c], pack_signed(r));
                });
            }
        }
        let data_code = EntropyCode::new(&data_histograms);

        ModularImage {
            planes,
            predictors,
            tree_symbols,
            context_of,
            tree_code,
            data_code,
            groups,
        }
    }

    /// Whether the image is one group, coded entirely in the global section.
    pub(super) fn single_group(&self) -> bool {
        self.groups.len() == 1
    }

    pub(super) fn groups(&self) -> usize {
        self.groups.len()
    }

    /// The LfGlobal section: the default LF dequantisation, the global
    /// tree and its histograms, the global image's header, and — for a
    /// one-group image — every sample.
    pub(super) fn write_global(&self, w: &mut BitWriter) {
        w.bit(true); // LfQuant: all_default
        w.bit(true); // a global tree
        self.tree_code.write_header(w);
        for &(context, value) in &self.tree_symbols {
            self.tree_code.write(w, context, value);
        }
        self.data_code.write_header(w);
        write_group_header(w);
        if self.single_group() {
            self.write_samples(w, &self.groups[0]);
        }
    }

    /// Group `index`'s section (of a many-group image).
    pub(super) fn write_group(&self, w: &mut BitWriter, index: usize) {
        write_group_header(w);
        self.write_samples(w, &self.groups[index]);
    }

    fn write_samples(&self, w: &mut BitWriter, rect: &Rect) {
        for (c, plane) in self.planes.iter().enumerate() {
            let context = self.context_of[c];
            for_each_residual(plane, rect, self.predictors[c], |r| {
                self.data_code.write(w, context, pack_signed(r));
            });
        }
    }
}

/// A modular stream's header: the global tree, the weighted predictor's
/// default parameters, no transforms.
fn write_group_header(w: &mut BitWriter) {
    w.bit(true); // use_global_tree
    w.bit(true); // WeightedHeader: all_default
    w.write(2, 0); // no transforms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_is_clamped_to_its_neighbours() {
        let top = [10, 50];
        let row = [40, 0];
        // left 40, north 50, north-west 10: 80, clamped to 50.
        assert_eq!(Predictor::Gradient.predict(&row, Some(&top), 1), 50);
        // The first column predicts from the sample above.
        assert_eq!(Predictor::West.predict(&row, Some(&top), 0), 10);
        assert_eq!(Predictor::Gradient.predict(&row, None, 0), 0);
    }

    #[test]
    fn channel_tree_gives_each_channel_a_leaf() {
        let predictors = [
            Predictor::West,
            Predictor::North,
            Predictor::Gradient,
            Predictor::West,
        ];
        let tree = channel_tree(0, 3, &predictors);
        let (_, contexts) = flatten(&tree, 4);
        let mut sorted = contexts.clone();
        sorted.sort();
        assert_eq!(sorted, [0, 1, 2, 3]);
    }
}
