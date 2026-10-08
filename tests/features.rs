//! Patches, splines and noise, read back by jxl-rs.

#![allow(clippy::needless_range_loop)]

mod common;

use common::{RawOptions, decode_raw, plane, to_int};
use jpegxl::encode::{
    Encoder, Features, Frame, FrameContent, FrameOptions, FrameType, ImageInfo, ModularFrame,
    ModularOptions, Noise, Patch, PatchBlendMode, PatchBlending, PatchPlacement, QuantizedSpline,
    Splines,
};

fn modular(color: Vec<Vec<i32>>) -> FrameContent {
    FrameContent::Modular(ModularFrame {
        color,
        extra: vec![],
        options: ModularOptions::default(),
        transforms: Vec::new(),
    })
}

#[test]
fn patches_copy_from_a_saved_frame() {
    let (w, h) = (32usize, 24usize);
    let reference: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c + 40)).collect();
    let base: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
    let replace = PatchBlending {
        mode: PatchBlendMode::Replace,
        alpha_channel: 0,
        clamp: false,
    };
    let add = PatchBlending {
        mode: PatchBlendMode::Add,
        alpha_channel: 0,
        clamp: false,
    };
    let patches = vec![
        Patch {
            reference: 1,
            x0: 3,
            y0: 2,
            width: 6,
            height: 5,
            placements: vec![
                PatchPlacement {
                    x: 0,
                    y: 0,
                    blending: vec![replace],
                },
                PatchPlacement {
                    x: 20,
                    y: 10,
                    blending: vec![replace],
                },
                PatchPlacement {
                    x: 7,
                    y: 18,
                    blending: vec![replace],
                },
            ],
        },
        Patch {
            reference: 1,
            x0: 10,
            y0: 10,
            width: 4,
            height: 4,
            placements: vec![PatchPlacement {
                x: 26,
                y: 1,
                blending: vec![add],
            }],
        },
    ];
    let mut e = Encoder::new(ImageInfo::new(w as u32, h as u32)).unwrap();
    e.add_frame(Frame {
        options: FrameOptions {
            frame_type: FrameType::ReferenceOnly,
            save_as_reference: 1,
            save_before_ct: Some(true),
            ..Default::default()
        },
        content: modular(reference.clone()),
    });
    e.add_frame(Frame {
        options: FrameOptions {
            features: Features {
                patches: patches.clone(),
                ..Default::default()
            },
            ..Default::default()
        },
        content: modular(base.clone()),
    });
    let jxl = e.finish().unwrap();
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    assert_eq!(d.frames.len(), 1);
    let mut expect = base.clone();
    for p in &patches {
        for pl in &p.placements {
            for dy in 0..p.height as usize {
                for dx in 0..p.width as usize {
                    let to = (pl.y as usize + dy) * w + pl.x as usize + dx;
                    let from = (p.y0 as usize + dy) * w + p.x0 as usize + dx;
                    for c in 0..3 {
                        expect[c][to] = match pl.blending[0].mode {
                            PatchBlendMode::Replace => reference[c][from],
                            _ => expect[c][to] + reference[c][from],
                        };
                    }
                }
            }
        }
    }
    for i in 0..w * h {
        for c in 0..3 {
            assert_eq!(
                to_int(d.frames[0].color[i * 3 + c], 8),
                i64::from(expect[c][i]),
                "sample {i} channel {c}"
            );
        }
    }
}

#[test]
fn splines_draw() {
    let (w, h) = (48usize, 40usize);
    let mut color_dct = [[0i32; 32]; 3];
    color_dct[1][0] = 80;
    color_dct[0][0] = 10;
    let mut sigma_dct = [0i32; 32];
    sigma_dct[0] = 40;
    let splines = Splines {
        quantization_adjustment: 0,
        splines: vec![
            QuantizedSpline {
                start: (5, 5),
                control_deltas: vec![(10, 4), (2, 3), (-1, 1)],
                color_dct,
                sigma_dct,
            },
            QuantizedSpline {
                start: (30, 8),
                control_deltas: vec![(0, 10)],
                color_dct,
                sigma_dct,
            },
        ],
    };
    let flat = vec![vec![100; w * h]; 3];
    let mut e = Encoder::new(ImageInfo::new(w as u32, h as u32)).unwrap();
    e.add_frame(Frame {
        options: FrameOptions {
            features: Features {
                splines: Some(splines),
                ..Default::default()
            },
            ..Default::default()
        },
        content: modular(flat),
    });
    let jxl = e.finish().unwrap();
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    let changed = d.frames[0]
        .color
        .iter()
        .filter(|&&v| (v - 100.0 / 255.0).abs() > 1e-3)
        .count();
    assert!(changed > 20, "the splines changed {changed} samples");
}

#[test]
fn noise_parameters() {
    let (w, h) = (16usize, 16usize);
    for lut in [[0u32; 8], [10, 50, 100, 200, 300, 500, 800, 1023]] {
        let mut e = Encoder::new(ImageInfo::new(w as u32, h as u32)).unwrap();
        e.add_frame(Frame {
            options: FrameOptions {
                features: Features {
                    noise: Some(Noise { lut }),
                    ..Default::default()
                },
                ..Default::default()
            },
            content: modular(vec![vec![120; w * h]; 3]),
        });
        let jxl = e.finish().unwrap();
        decode_raw(&jxl, &RawOptions::default()).unwrap_or_else(|e| panic!("{lut:?}: {e}"));
    }
}
