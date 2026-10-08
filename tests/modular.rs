//! Modular coding, feature by feature, each output decoded by jxl-rs and
//! compared sample for sample.

use jpegxl::encode::{
    LosslessOptions, ModularOptions, Palette, Predictor, Rct, Samples, SqueezeStep, Transform,
    TreeMode, WeightedParams, encode_lossless_with,
};
use jpegxl::{Channels, Pixels};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn picture(w: u32, h: u32, channels: Channels, max: u32, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed | 1);
    let n = channels.count() as u32;
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..n {
                let smooth = (x * (c + 2) * 3 + y * (5 + c)) % (max + 1);
                let edge = if (x / 7 + y / 5) % 2 == 0 { max / 3 } else { 0 };
                let noise = (rng.next() % 5) as u32;
                out.push((smooth + edge + noise).min(max));
            }
        }
    }
    out
}

fn check_u8(
    w: u32,
    h: u32,
    channels: Channels,
    pixels: &[u8],
    options: &LosslessOptions,
    what: &str,
) -> usize {
    let jxl = encode_lossless_with(w, h, channels, Samples::U8(pixels), options)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let image = jpegxl::decode(&jxl).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        image.pixels == Pixels::U8(pixels.to_vec()),
        "{what}: the samples differ"
    );
    jxl.len()
}

fn check_u16(
    w: u32,
    h: u32,
    channels: Channels,
    pixels: &[u16],
    options: &LosslessOptions,
    what: &str,
) {
    let jxl = encode_lossless_with(w, h, channels, Samples::U16(pixels), options)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let image = jpegxl::decode(&jxl).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        image.pixels == Pixels::U16(pixels.to_vec()),
        "{what}: the samples differ"
    );
}

fn u8s(v: Vec<u32>) -> Vec<u8> {
    v.into_iter().map(|x| x as u8).collect()
}

#[test]
fn every_predictor() {
    let pixels = u8s(picture(37, 23, Channels::Rgba, 255, 5));
    for p in [
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
    ] {
        let options = LosslessOptions {
            modular: ModularOptions {
                tree: TreeMode::Fixed(p),
                ..Default::default()
            },
            ..Default::default()
        };
        check_u8(37, 23, Channels::Rgba, &pixels, &options, &format!("{p:?}"));
        // Narrow and short channels too (the decoder's small-channel path).
        check_u8(
            3,
            2,
            Channels::Rgb,
            &pixels[..18],
            &options,
            &format!("{p:?} 3x2"),
        );
    }
}

#[test]
fn learned_trees_on_every_property() {
    let pixels = u8s(picture(70, 50, Channels::Rgb, 255, 6));
    // All properties, three earlier channels' references, every predictor.
    let options = LosslessOptions {
        modular: ModularOptions {
            properties: (0..28).collect(),
            predictors: vec![
                Predictor::Gradient,
                Predictor::Weighted,
                Predictor::West,
                Predictor::North,
                Predictor::AverageAll,
                Predictor::Select,
            ],
            split_cost: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    check_u8(70, 50, Channels::Rgb, &pixels, &options, "all properties");
}

#[test]
fn custom_weighted_parameters() {
    let pixels = u8s(picture(40, 40, Channels::Gray, 255, 8));
    let options = LosslessOptions {
        modular: ModularOptions {
            tree: TreeMode::Fixed(Predictor::Weighted),
            weighted: WeightedParams {
                p1c: 3,
                p2c: 31,
                p3c: [1, 2, 3, 4, 5],
                w: [1, 15, 7, 0],
            },
            ..Default::default()
        },
        ..Default::default()
    };
    check_u8(
        40,
        40,
        Channels::Gray,
        &pixels,
        &options,
        "weighted parameters",
    );
}

#[test]
fn local_trees() {
    let pixels = u8s(picture(300, 280, Channels::Rgb, 255, 9));
    let options = LosslessOptions {
        modular: ModularOptions {
            local_trees: true,
            ..Default::default()
        },
        ..Default::default()
    };
    check_u8(300, 280, Channels::Rgb, &pixels, &options, "local trees");
}

#[test]
fn every_rct() {
    let pixels = u8s(picture(20, 9, Channels::Rgba, 255, 10));
    for rct_type in 0..42 {
        let options = LosslessOptions {
            transforms: vec![Transform::Rct(Rct {
                begin_channel: 0,
                rct_type,
            })],
            ..Default::default()
        };
        check_u8(
            20,
            9,
            Channels::Rgba,
            &pixels,
            &options,
            &format!("RCT {rct_type}"),
        );
    }
    // On the last three channels of four, at 16 bits (17-bit residues).
    let pixels: Vec<u16> = picture(20, 9, Channels::Rgba, 65535, 11)
        .into_iter()
        .map(|v| v as u16)
        .collect();
    for rct_type in [6, 13, 41] {
        let options = LosslessOptions {
            transforms: vec![Transform::Rct(Rct {
                begin_channel: 1,
                rct_type,
            })],
            ..Default::default()
        };
        check_u16(
            20,
            9,
            Channels::Rgba,
            &pixels,
            &options,
            &format!("16-bit RCT {rct_type}"),
        );
    }
}

/// A picture of few colours.
fn few_colours(w: u32, h: u32) -> Vec<u8> {
    let colours = [
        [10u8, 20, 30],
        [200, 100, 0],
        [0, 0, 0],
        [255, 255, 255],
        [64, 128, 64],
    ];
    (0..w * h)
        .flat_map(|i| colours[((i % w) / 3 + (i / w) / 4) as usize % 5])
        .collect()
}

#[test]
fn palettes() {
    let (w, h) = (33, 21);
    let pixels = few_colours(w, h);
    let colours = vec![
        vec![10, 20, 30],
        vec![200, 100, 0],
        vec![0, 0, 0],
        vec![255, 255, 255],
        vec![64, 128, 64],
    ];
    let plain = LosslessOptions {
        transforms: vec![Transform::Palette(Palette {
            begin_channel: 0,
            num_channels: 3,
            entries: colours.clone(),
            num_deltas: 0,
            predictor: Predictor::Zero,
        })],
        ..Default::default()
    };
    check_u8(w, h, Channels::Rgb, &pixels, &plain, "palette");

    // Some colours left out: the implicit cube has black and white.
    let partial = LosslessOptions {
        transforms: vec![Transform::Palette(Palette {
            begin_channel: 0,
            num_channels: 3,
            entries: vec![colours[0].clone(), colours[1].clone(), colours[4].clone()],
            num_deltas: 0,
            predictor: Predictor::Zero,
        })],
        ..Default::default()
    };
    check_u8(w, h, Channels::Rgb, &pixels, &partial, "implicit colours");

    // A one-channel palette on each channel in turn, and after an RCT.
    let single = LosslessOptions {
        transforms: vec![
            Transform::Rct(Rct {
                begin_channel: 0,
                rct_type: 6,
            }),
            Transform::Palette(Palette {
                begin_channel: 1,
                num_channels: 1,
                entries: {
                    // Every value channel 1 takes after the RCT.
                    let mut values: Vec<i32> = Vec::new();
                    for c in pixels.chunks(3) {
                        let (r, g, b) = (i32::from(c[0]), i32::from(c[1]), i32::from(c[2]));
                        let co = r - b;
                        if !values.contains(&co) {
                            values.push(co);
                        }
                        let _ = g;
                    }
                    values.into_iter().map(|v| vec![v]).collect()
                },
                num_deltas: 0,
                predictor: Predictor::Zero,
            }),
        ],
        ..Default::default()
    };
    check_u8(
        w,
        h,
        Channels::Rgb,
        &pixels,
        &single,
        "channel palette after RCT",
    );
}

#[test]
fn delta_palettes() {
    // A smooth picture: deltas from the prediction cover most samples.
    let (w, h) = (24u32, 16u32);
    let pixels: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8]))
        .collect();
    let mut entries: Vec<Vec<i32>> = Vec::new();
    // Deltas: every per-channel step this picture takes from any predictor
    // would be too many; give a handful, the rest fall to colours.
    for d in [[4, 0, 2], [0, 4, 2], [0, 0, 0], [4, 4, 4]] {
        entries.push(d.to_vec());
    }
    let mut colours: Vec<Vec<i32>> = pixels
        .chunks(3)
        .map(|c| c.iter().map(|&v| i32::from(v)).collect())
        .collect();
    colours.sort();
    colours.dedup();
    entries.extend(colours);
    for predictor in [
        Predictor::West,
        Predictor::North,
        Predictor::Gradient,
        Predictor::Weighted,
        Predictor::AverageAll,
    ] {
        let options = LosslessOptions {
            transforms: vec![Transform::Palette(Palette {
                begin_channel: 0,
                num_channels: 3,
                entries: entries.clone(),
                num_deltas: 4,
                predictor,
            })],
            ..Default::default()
        };
        check_u8(
            w,
            h,
            Channels::Rgb,
            &pixels,
            &options,
            &format!("delta palette {predictor:?}"),
        );
    }
}

#[test]
fn squeezes() {
    let pixels = u8s(picture(45, 30, Channels::Rgba, 255, 12));
    let default = LosslessOptions {
        transforms: vec![Transform::Squeeze(Vec::new())],
        ..Default::default()
    };
    check_u8(45, 30, Channels::Rgba, &pixels, &default, "default squeeze");
    let explicit = LosslessOptions {
        transforms: vec![Transform::Squeeze(vec![
            SqueezeStep {
                horizontal: true,
                in_place: true,
                begin_channel: 0,
                num_channels: 4,
            },
            SqueezeStep {
                horizontal: false,
                in_place: false,
                begin_channel: 1,
                num_channels: 2,
            },
            SqueezeStep {
                horizontal: false,
                in_place: true,
                begin_channel: 0,
                num_channels: 1,
            },
        ])],
        ..Default::default()
    };
    check_u8(
        45,
        30,
        Channels::Rgba,
        &pixels,
        &explicit,
        "explicit squeeze",
    );
    // Big enough that squeezed channels land in LF groups and groups.
    let pixels = u8s(picture(600, 520, Channels::Rgb, 255, 13));
    check_u8(
        600,
        520,
        Channels::Rgb,
        &pixels,
        &default,
        "squeezed, many groups",
    );
    // Squeeze after RCT and palette-free: the usual lossy-modular chain.
    let chain = LosslessOptions {
        transforms: vec![
            Transform::Rct(Rct {
                begin_channel: 0,
                rct_type: 6,
            }),
            Transform::Squeeze(Vec::new()),
        ],
        ..Default::default()
    };
    check_u8(600, 520, Channels::Rgb, &pixels, &chain, "RCT then squeeze");
}

#[test]
fn learning_beats_a_fixed_predictor() {
    let pixels = u8s(picture(128, 128, Channels::Rgb, 255, 14));
    let fixed = LosslessOptions {
        modular: ModularOptions {
            tree: TreeMode::Fixed(Predictor::Gradient),
            ..Default::default()
        },
        ..Default::default()
    };
    let a = check_u8(128, 128, Channels::Rgb, &pixels, &fixed, "fixed");
    let b = check_u8(
        128,
        128,
        Channels::Rgb,
        &pixels,
        &LosslessOptions::default(),
        "learned",
    );
    assert!(b < a, "learned {b} >= fixed {a}");
}
