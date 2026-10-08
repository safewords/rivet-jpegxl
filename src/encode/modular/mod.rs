//! Modular coding: channels of integers, transformed (RCT, palette,
//! squeeze), split into the frame's sections, each sample predicted from
//! its neighbours with the residual coded in a context an MA tree picks.
//!
//! A frame's modular streams — the image's global part, its LF groups and
//! groups, and (for VarDCT) the LF coefficients, the block metadata and
//! custom quantisation tables — share one global tree and its histograms,
//! or each has a local tree of its own.

pub(crate) mod predict;
pub(crate) mod transforms;
pub(crate) mod tree;

use crate::encode::bits::{BitWriter, Dist};
use crate::encode::entropy::{EntropyCode, EntropyOptions, Stream, Token};
use predict::{Neighbours, WeightedState};
pub use predict::{Predictor, WeightedParams};
pub use transforms::{Palette, Rct, SqueezeStep, Transform};
use tree::{LearnOptions, Node, Samples, TREE_CONTEXTS, Tree};

/// A channel of samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Channel {
    pub width: usize,
    pub height: usize,
    /// The channel's (horizontal, vertical) subsampling shift; none for a
    /// meta channel (a palette).
    pub shift: Option<(u32, u32)>,
    pub samples: Vec<i32>,
    /// A squeeze residual (detail that lossy coding may quantise).
    pub residual: bool,
}

impl Channel {
    pub fn new(width: usize, height: usize, shift: Option<(u32, u32)>, samples: Vec<i32>) -> Self {
        debug_assert_eq!(samples.len(), width * height);
        Channel {
            width,
            height,
            shift,
            samples,
            residual: false,
        }
    }

    fn is_meta_or_small(&self, group_dim: usize) -> bool {
        self.shift.is_none() || (self.width <= group_dim && self.height <= group_dim)
    }

    /// The part at (`x`, `y`) of at most `w` by `h`.
    fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Channel {
        if x >= self.width || y >= self.height {
            return Channel::new(0, 0, self.shift, Vec::new());
        }
        let w = w.min(self.width - x);
        let h = h.min(self.height - y);
        let mut samples = Vec::with_capacity(w * h);
        for r in y..y + h {
            samples.extend_from_slice(&self.samples[r * self.width + x..r * self.width + x + w]);
        }
        let mut c = Channel::new(w, h, self.shift, samples);
        c.residual = self.residual;
        c
    }
}

/// How the MA tree is made.
#[derive(Clone, Debug, PartialEq)]
pub enum TreeMode {
    /// One leaf, this predictor.
    Fixed(Predictor),
    /// Learned from the samples.
    Learn,
}

/// How modular streams are coded.
#[derive(Clone, Debug, PartialEq)]
pub struct ModularOptions {
    pub tree: TreeMode,
    /// The predictors learning chooses among.
    pub predictors: Vec<Predictor>,
    /// The properties learning may split on (16 and up: earlier channels').
    pub properties: Vec<u32>,
    /// The weighted predictor's parameters.
    pub weighted: WeightedParams,
    /// A tree per stream instead of one shared.
    pub local_trees: bool,
    /// The most samples learning looks at.
    pub max_samples: usize,
    /// Bits a split must save.
    pub split_cost: f64,
    pub max_leaves: usize,
    pub entropy: EntropyOptions,
}

impl Default for ModularOptions {
    fn default() -> Self {
        ModularOptions {
            tree: TreeMode::Learn,
            predictors: vec![
                Predictor::Gradient,
                Predictor::West,
                Predictor::North,
                Predictor::AverageWestNorth,
                Predictor::Select,
                Predictor::Zero,
            ],
            properties: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            weighted: WeightedParams::default(),
            local_trees: false,
            max_samples: 1 << 18,
            split_cost: 48.0,
            max_leaves: 256,
            entropy: EntropyOptions::default(),
        }
    }
}

/// A modular stream: its id (property 1), its channels as coded, the
/// transforms its header lists (already applied to `channels`).
pub(crate) struct ModularStream {
    pub id: usize,
    pub channels: Vec<Channel>,
    pub transforms: Vec<Transform>,
    /// Write the group header even when there is nothing to decode (the
    /// global image's, read whenever the frame has modular channels).
    pub header_always: bool,
}

impl ModularStream {
    fn is_empty(&self) -> bool {
        self.channels.iter().all(|c| c.width == 0 || c.height == 0)
    }
}

/// Visit every sample of a stream as the decoder computes it: the channel
/// index, position, properties (with `references` reference properties),
/// neighbourhood, the weighted prediction (when `wp`), and its value.
fn visit(
    stream: &ModularStream,
    references: usize,
    wp: Option<WeightedParams>,
    mut f: impl FnMut(usize, &[i32], &Neighbours, i64, i32),
) {
    let mut props = vec![0i32; predict::NUM_NONREF_PROPERTIES + references];
    props[1] = stream.id as i32;
    for (c, ch) in stream.channels.iter().enumerate() {
        if ch.width == 0 || ch.height == 0 {
            continue;
        }
        props[0] = c as i32;
        let w = ch.width;
        let mut state = wp.map(|p| WeightedState::new(p, w));
        for y in 0..ch.height {
            let refs = predict::reference_row(&stream.channels, c, y, references);
            let row = &ch.samples[y * w..(y + 1) * w];
            let top = if y > 0 {
                &ch.samples[(y - 1) * w..y * w]
            } else {
                &[][..]
            };
            let toptop = if y > 1 {
                &ch.samples[(y - 2) * w..(y - 1) * w]
            } else {
                &[][..]
            };
            let mut last_9 = 0;
            for x in 0..w {
                let n = Neighbours::at(row, top, toptop, x, y);
                predict::local_properties(&mut props, &n, x, y, &mut last_9);
                let (wp_pred, wp_prop) = match &mut state {
                    Some(s) => s.predict(x, y, &n),
                    None => (0, 0),
                };
                props[15] = wp_prop;
                if references > 0 {
                    props[predict::NUM_NONREF_PROPERTIES..]
                        .copy_from_slice(&refs[x * references..(x + 1) * references]);
                }
                f(c, &props, &n, wp_pred, row[x]);
                if let Some(s) = &mut state {
                    s.update(row[x], x, y);
                }
            }
        }
    }
}

/// The reference properties `properties` needs.
fn references_for(properties: &[u32]) -> usize {
    properties
        .iter()
        .map(|&p| p as usize + 1)
        .max()
        .unwrap_or(0)
        .saturating_sub(predict::NUM_NONREF_PROPERTIES)
        .next_multiple_of(predict::PROPERTIES_PER_REFERENCE)
}

/// Samples of `streams` for learning.
fn sample(streams: &[&ModularStream], options: &ModularOptions) -> Samples {
    let mut s = Samples::new(options.properties.clone(), options.predictors.clone());
    let total: usize = streams
        .iter()
        .flat_map(|st| &st.channels)
        .map(|c| c.width * c.height)
        .sum();
    let keep = if total > options.max_samples {
        options.max_samples as f64 / total as f64
    } else {
        1.0
    };
    s.weight = 1.0 / keep;
    let threshold = (keep * f64::from(u32::MAX)) as u64;
    let mut rng = 0x2545_f491u64;
    let needs_wp =
        options.predictors.contains(&Predictor::Weighted) || options.properties.contains(&15);
    let refs = references_for(&options.properties);
    for st in streams {
        visit(
            st,
            refs,
            needs_wp.then_some(options.weighted),
            |_, props, n, wp, v| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if (rng >> 32) > threshold {
                    return;
                }
                for (k, &p) in options.properties.iter().enumerate() {
                    s.values[k].push(props[p as usize]);
                }
                for (k, pr) in options.predictors.iter().enumerate() {
                    s.residuals[k].push(i64::from(v) - pr.predict(n, wp));
                }
            },
        );
    }
    s
}

fn make_tree(streams: &[&ModularStream], options: &ModularOptions, size_limit: usize) -> Tree {
    match &options.tree {
        TreeMode::Fixed(p) => Tree::single(*p),
        TreeMode::Learn => {
            let samples = sample(streams, options);
            let learn = LearnOptions {
                predictors: options.predictors.clone(),
                split_cost: options.split_cost,
                max_leaves: options.max_leaves.min(size_limit.div_ceil(2)).max(1),
                max_cuts: 32,
            };
            Tree::from_built(&tree::learn(&samples, &learn))
        }
    }
}

/// Route every sample of `streams` through `tree`: per stream, per sample,
/// its leaf (node index) and residual against the leaf's predictor. Then
/// give each leaf the offset and multiplier its residuals allow.
fn route(
    tree: &mut Tree,
    streams: &[&ModularStream],
    weighted: WeightedParams,
) -> Vec<Vec<(u32, i64)>> {
    let refs = tree.num_reference_properties();
    let wp = tree.uses_weighted().then_some(weighted);
    let routed: Vec<Vec<(u32, i64)>> = streams
        .iter()
        .map(|st| {
            let mut out = Vec::new();
            visit(st, refs, wp, |_, props, n, wp_pred, v| {
                let leaf = tree.walk(props);
                let Node::Leaf { predictor, .. } = tree.nodes[leaf] else {
                    unreachable!()
                };
                out.push((leaf as u32, i64::from(v) - predictor.predict(n, wp_pred)));
            });
            out
        })
        .collect();

    // Per leaf: the first residual, and the gcd of the differences from it.
    let mut first: Vec<Option<i64>> = vec![None; tree.nodes.len()];
    let mut gcd = vec![0u64; tree.nodes.len()];
    for &(leaf, r) in routed.iter().flatten() {
        let l = leaf as usize;
        match first[l] {
            None => first[l] = Some(r),
            Some(f) => gcd[l] = gcd_u64(gcd[l], (r - f).unsigned_abs()),
        }
    }
    for (l, node) in tree.nodes.iter_mut().enumerate() {
        if let Node::Leaf {
            offset, multiplier, ..
        } = node
            && let Some(f) = first[l]
        {
            let m = gcd[l];
            if m == 0 {
                // Every residual the same: the offset says it all.
                if let Ok(o) = i32::try_from(f) {
                    *offset = o;
                }
            } else if m > 1 && m <= u64::from(u32::MAX) {
                let mut o = f.rem_euclid(m as i64);
                if o > (m as i64) / 2 {
                    o -= m as i64;
                }
                *offset = o as i32;
                *multiplier = m as u32;
            }
        }
    }
    routed
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd_u64(b, a % b) }
}

/// The tokens of routed residuals.
fn tokens(tree: &Tree, routed: &[(u32, i64)]) -> Vec<Token> {
    routed
        .iter()
        .map(|&(leaf, r)| {
            let Node::Leaf {
                offset,
                multiplier,
                id,
                ..
            } = tree.nodes[leaf as usize]
            else {
                unreachable!()
            };
            let q = (r - i64::from(offset)) / i64::from(multiplier);
            Token::signed(id, q as i32)
        })
        .collect()
}

/// A tree and the codes for it and for the data it routes.
struct CodedTree {
    tree: Tree,
    tree_code: EntropyCode,
    tree_tokens: Vec<Token>,
    data_code: EntropyCode,
}

impl CodedTree {
    fn write(&self, w: &mut BitWriter) {
        self.tree_code.write_all(w, &self.tree_tokens);
        self.data_code.write_header(w);
    }
}

fn code_tree(tree: Tree, entropy: &EntropyOptions) -> (EntropyCode, Vec<Token>, Tree) {
    let (code, mut streams) = EntropyCode::build(
        TREE_CONTEXTS,
        vec![Stream::new(tree.tokens())],
        entropy,
        true,
    );
    (code, streams.remove(0), tree)
}

/// A frame's modular streams, coded.
pub(crate) struct ModularCoded {
    global: Option<CodedTree>,
    streams: Vec<CodedStream>,
}

struct CodedStream {
    header: BitWriter,
    local: Option<CodedTree>,
    tokens: Vec<Token>,
    write: bool,
    use_global: bool,
}

impl ModularCoded {
    /// Code `streams`. `size_limit` caps the global tree's nodes as the
    /// decoder does.
    pub(crate) fn new(
        streams: &[ModularStream],
        options: &ModularOptions,
        size_limit: usize,
    ) -> Self {
        let local = options.local_trees;
        let global_streams: Vec<&ModularStream> = if local {
            Vec::new()
        } else {
            streams.iter().filter(|s| !s.is_empty()).collect()
        };

        let mut global = None;
        let mut global_tokens: Vec<Vec<Token>> = Vec::new();
        if !global_streams.is_empty() {
            let mut tree = make_tree(&global_streams, options, size_limit);
            let routed = route(&mut tree, &global_streams, options.weighted);
            let token_streams: Vec<Stream> = routed
                .iter()
                .zip(&global_streams)
                .map(|(r, st)| Stream {
                    tokens: tokens(&tree, r),
                    width: st.channels.iter().map(|c| c.width).max().unwrap_or(0) as u32,
                })
                .collect();
            let (data_code, toks) =
                EntropyCode::build(tree.num_leaves(), token_streams, &options.entropy, true);
            global_tokens = toks;
            let (tree_code, tree_tokens, tree) = code_tree(tree, &options.entropy);
            global = Some(CodedTree {
                tree,
                tree_code,
                tree_tokens,
                data_code,
            });
        }

        let mut next_global = global_tokens.into_iter();
        let coded = streams
            .iter()
            .map(|st| {
                let empty = st.is_empty();
                let mut header = BitWriter::new();
                let use_global = !local;
                header.bit(use_global);
                options.weighted.write(&mut header);
                write_transforms(&mut header, &st.transforms);
                if empty {
                    return CodedStream {
                        header,
                        local: None,
                        tokens: Vec::new(),
                        write: st.header_always,
                        use_global,
                    };
                }
                if use_global {
                    return CodedStream {
                        header,
                        local: None,
                        tokens: next_global.next().unwrap(),
                        write: true,
                        use_global,
                    };
                }
                let samples: usize = st.channels.iter().map(|c| c.width * c.height).sum();
                let limit = (1024 + samples).min(1 << 20);
                let one = [st];
                let mut tree = make_tree(&one, options, limit);
                let routed = route(&mut tree, &one, options.weighted);
                let (data_code, mut toks) = EntropyCode::build(
                    tree.num_leaves(),
                    vec![Stream {
                        tokens: tokens(&tree, &routed[0]),
                        width: st.channels.iter().map(|c| c.width).max().unwrap_or(0) as u32,
                    }],
                    &options.entropy,
                    true,
                );
                let (tree_code, tree_tokens, tree) = code_tree(tree, &options.entropy);
                CodedStream {
                    header,
                    local: Some(CodedTree {
                        tree,
                        tree_code,
                        tree_tokens,
                        data_code,
                    }),
                    tokens: toks.remove(0),
                    write: true,
                    use_global,
                }
            })
            .collect();
        ModularCoded {
            global,
            streams: coded,
        }
    }

    /// The global tree, as LfGlobal sends it: a flag, the tree and the data
    /// histograms.
    pub(crate) fn write_global_tree(&self, w: &mut BitWriter) {
        match &self.global {
            None => w.bit(false),
            Some(t) => {
                w.bit(true);
                t.write(w);
            }
        }
    }

    /// Stream `i`: its header (unless it has nothing to decode), any local
    /// tree, its data.
    pub(crate) fn write_stream(&self, w: &mut BitWriter, i: usize) {
        let s = &self.streams[i];
        if !s.write {
            return;
        }
        w.append_bits(&s.header);
        if s.tokens.is_empty() {
            return;
        }
        if let Some(local) = &s.local {
            local.write(w);
            local.data_code.write_tokens(w, &s.tokens);
        } else {
            debug_assert!(s.use_global);
            self.global
                .as_ref()
                .unwrap()
                .data_code
                .write_tokens(w, &s.tokens);
        }
    }

    /// The global tree, if any.
    #[allow(dead_code)]
    pub(crate) fn global_tree(&self) -> Option<&Tree> {
        self.global.as_ref().map(|g| &g.tree)
    }
}

fn write_transforms(w: &mut BitWriter, transforms: &[Transform]) {
    w.u32(
        transforms.len() as u32,
        [
            Dist::Val(0),
            Dist::Val(1),
            Dist::Bits(4, 2),
            Dist::Bits(8, 18),
        ],
    );
    for t in transforms {
        t.write(w);
    }
}

/// Where a frame's modular sections fall.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    pub group_dim: usize,
    pub groups_x: usize,
    pub groups_y: usize,
    pub lf_groups_x: usize,
    pub lf_groups_y: usize,
    /// Per pass, the range of shifts its sections carry.
    pub pass_shifts: Vec<(u32, u32)>,
    pub num_lf_groups: usize,
}

impl Layout {
    pub(crate) fn num_groups(&self) -> usize {
        self.groups_x * self.groups_y
    }

    /// Stream ids, as the decoder numbers them.
    pub(crate) fn lf_stream_id(&self, g: usize) -> usize {
        1 + self.num_lf_groups + g
    }

    pub(crate) fn hf_stream_id(&self, pass: usize, g: usize) -> usize {
        1 + 3 * self.num_lf_groups + 17 + self.num_groups() * pass + g
    }
}

/// The image's channels after its global transforms, split into the
/// sections the decoder reads them from: the global stream (leading meta or
/// small channels), each LF group's (shifts of 3 and up), each pass's
/// groups'.
pub(crate) struct ImageStreams {
    pub global: ModularStream,
    pub lf: Vec<ModularStream>,
    /// `[pass][group]`.
    pub hf: Vec<Vec<ModularStream>>,
}

impl ImageStreams {
    pub(crate) fn split(coded: Vec<Channel>, transforms: Vec<Transform>, layout: &Layout) -> Self {
        let gd = layout.group_dim;
        let lead = coded.iter().take_while(|c| c.is_meta_or_small(gd)).count();
        let rest = &coded[lead..];
        let min_shift = |c: &Channel| c.shift.map(|(a, b)| a.min(b));
        let lf_channels: Vec<&Channel> = rest
            .iter()
            .filter(|c| min_shift(c).is_some_and(|s| s >= 3))
            .collect();
        let lf_dim = gd * 8;
        let lf = (0..layout.lf_groups_x * layout.lf_groups_y)
            .map(|g| {
                let (gx, gy) = (g % layout.lf_groups_x, g / layout.lf_groups_x);
                ModularStream {
                    id: layout.lf_stream_id(g),
                    channels: lf_channels
                        .iter()
                        .map(|c| {
                            let (sx, sy) = c.shift.unwrap();
                            let (dx, dy) = (lf_dim >> sx, lf_dim >> sy);
                            c.crop(gx * dx, gy * dy, dx, dy)
                        })
                        .collect(),
                    transforms: Vec::new(),
                    header_always: false,
                }
            })
            .collect();
        let hf = layout
            .pass_shifts
            .iter()
            .enumerate()
            .map(|(pass, &(lo, hi))| {
                let channels: Vec<&Channel> = rest
                    .iter()
                    .filter(|c| min_shift(c).is_some_and(|s| lo <= s && s <= hi))
                    .collect();
                (0..layout.num_groups())
                    .map(|g| {
                        let (gx, gy) = (g % layout.groups_x, g / layout.groups_x);
                        ModularStream {
                            id: layout.hf_stream_id(pass, g),
                            channels: channels
                                .iter()
                                .map(|c| {
                                    let (sx, sy) = c.shift.unwrap();
                                    let (dx, dy) = (gd >> sx, gd >> sy);
                                    c.crop(gx * dx, gy * dy, dx, dy)
                                })
                                .collect(),
                            transforms: Vec::new(),
                            header_always: false,
                        }
                    })
                    .collect()
            })
            .collect();
        let mut global_channels = coded;
        global_channels.truncate(lead);
        ImageStreams {
            global: ModularStream {
                id: 0,
                channels: global_channels,
                transforms,
                header_always: true,
            },
            lf,
            hf,
        }
    }
}

/// Round each squeeze residual to a multiple of a step: `base` for the
/// finest detail, halving with each coarser level (never below 1). The
/// decoder reconstructs from the rounded residuals; leaf multipliers then
/// code them for what they are.
pub(crate) fn quantize_residuals(channels: &mut [Channel], base: f32) {
    for c in channels.iter_mut().filter(|c| c.residual) {
        let (sx, sy) = c.shift.unwrap_or((0, 0));
        let level = (sx + sy).max(1) - 1;
        let q = (f64::from(base) / f64::from(1u32 << level.min(30)))
            .round()
            .max(1.0) as i64;
        if q > 1 {
            for v in c.samples.iter_mut() {
                let r = (*v as f64 / q as f64).round() as i64 * q;
                *v = r.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            }
        }
    }
}

/// The largest absolute sample of `channels` (to choose the decoder's
/// 16-bit storage).
pub(crate) fn fits_i16(channels: &[Channel]) -> bool {
    channels
        .iter()
        .flat_map(|c| &c.samples)
        .all(|&v| i16::try_from(v).is_ok())
}
