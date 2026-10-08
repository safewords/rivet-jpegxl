//! Frame features, each encoded and read back by jxl-rs.

mod common;

use common::{RawOptions, decode_raw, plane, to_int};
use jpegxl::encode::{
    BlendMode, Blending, Crop, Encoder, ExtraChannel, ExtraChannelKind, Frame, FrameContent,
    FrameOptions, FrameType, ImageInfo, ModularFrame, ModularOptions, Passes, Restoration,
    SampleFormat, Transform, UpsamplingWeights,
};

fn modular(color: Vec<Vec<i32>>, extra: Vec<Vec<i32>>) -> FrameContent {
    FrameContent::Modular(ModularFrame {
        color,
        extra,
        options: ModularOptions::default(),
        transforms: Vec::new(),
    })
}

fn encode(info: ImageInfo, frames: Vec<Frame>) -> Vec<u8> {
    let mut e = Encoder::new(info).unwrap();
    for f in frames {
        e.add_frame(f);
    }
    e.finish().unwrap()
}

fn f(v: i32) -> f32 {
    v as f32 / 255.0
}

#[test]
fn crops_and_every_blend_mode() {
    let (w, h) = (24usize, 18usize);
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.extra_channels
        .push(ExtraChannel::alpha(SampleFormat::Int(8)));
    let base: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
    let base_alpha = plane(w, h, 255, 9);
    let crops = [
        Crop {
            x0: 5,
            y0: 3,
            width: 10,
            height: 7,
        },
        Crop {
            x0: -4,
            y0: -2,
            width: 12,
            height: 9,
        },
        Crop {
            x0: 18,
            y0: 14,
            width: 9,
            height: 8,
        },
    ];
    for crop in crops {
        for mode in [
            BlendMode::Replace,
            BlendMode::Add,
            BlendMode::Blend,
            BlendMode::AlphaWeightedAdd,
            BlendMode::Multiply,
        ] {
            let (cw, ch) = (crop.width as usize, crop.height as usize);
            let top: Vec<Vec<i32>> = (0..3).map(|c| plane(cw, ch, 255, c + 20)).collect();
            let top_alpha = plane(cw, ch, 255, 31);
            let blending = Blending {
                mode,
                alpha_channel: 0,
                clamp: false,
                source: 1,
            };
            let alpha_blending = Blending {
                mode: if mode == BlendMode::Blend {
                    BlendMode::Blend
                } else {
                    BlendMode::Replace
                },
                alpha_channel: 0,
                clamp: false,
                source: 1,
            };
            let frames = vec![
                Frame {
                    options: FrameOptions {
                        save_as_reference: 1,
                        is_last: Some(false),
                        ..Default::default()
                    },
                    content: modular(base.clone(), vec![base_alpha.clone()]),
                },
                Frame {
                    options: FrameOptions {
                        crop: Some(crop),
                        blending,
                        ec_blending: vec![alpha_blending],
                        ..Default::default()
                    },
                    content: modular(top.clone(), vec![top_alpha.clone()]),
                },
            ];
            let jxl = encode(info.clone(), frames);
            let d = decode_raw(&jxl, &RawOptions::default())
                .unwrap_or_else(|e| panic!("{mode:?} {crop:?}: {e}"));
            assert_eq!(d.frames.len(), 1);
            let out = &d.frames[0];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let (cx, cy) = (x as i32 - crop.x0, y as i32 - crop.y0);
                    let inside = cx >= 0 && cy >= 0 && (cx as usize) < cw && (cy as usize) < ch;
                    let bg_a = f(base_alpha[i]);
                    for c in 0..3 {
                        let bg = f(base[c][i]);
                        let expect = if !inside {
                            bg
                        } else {
                            let j = cy as usize * cw + cx as usize;
                            let fg = f(top[c][j]);
                            let fa = f(top_alpha[j]);
                            match mode {
                                BlendMode::Replace => fg,
                                BlendMode::Add => fg + bg,
                                BlendMode::Multiply => fg * bg,
                                BlendMode::AlphaWeightedAdd => bg + fg * fa,
                                BlendMode::Blend => {
                                    let a = fa + bg_a * (1.0 - fa);
                                    if a.abs() < 1e-9 {
                                        0.0
                                    } else {
                                        (fg * fa + bg * bg_a * (1.0 - fa)) / a
                                    }
                                }
                            }
                        };
                        let got = out.color[i * 3 + c];
                        assert!(
                            (got - expect).abs() < 1e-4,
                            "{mode:?} {crop:?} ({x},{y}) channel {c}: {got} != {expect}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn reference_only_and_skip_progressive_frames() {
    let (w, h) = (16usize, 12usize);
    let info = ImageInfo::new(w as u32, h as u32);
    let a: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 100, c)).collect();
    let b: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 100, c + 5)).collect();
    let frames = vec![
        // Saved to slot 2, never shown.
        Frame {
            options: FrameOptions {
                frame_type: FrameType::ReferenceOnly,
                save_as_reference: 2,
                ..Default::default()
            },
            content: modular(a.clone(), vec![]),
        },
        // Added onto slot 2.
        Frame {
            options: FrameOptions {
                frame_type: FrameType::SkipProgressive,
                blending: Blending {
                    mode: BlendMode::Add,
                    source: 2,
                    ..Default::default()
                },
                ..Default::default()
            },
            content: modular(b.clone(), vec![]),
        },
    ];
    let jxl = encode(info, frames);
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    assert_eq!(d.frames.len(), 1);
    for i in 0..w * h {
        for c in 0..3 {
            assert_eq!(
                to_int(d.frames[0].color[i * 3 + c], 8),
                i64::from(a[c][i] + b[c][i])
            );
        }
    }
}

#[test]
fn upsampling() {
    let (w, h) = (37usize, 21usize);
    for up in [2u32, 4, 8] {
        for custom in [false, true] {
            let mut info = ImageInfo::new(w as u32, h as u32);
            info.extra_channels.push(ExtraChannel {
                kind: ExtraChannelKind::Depth,
                format: SampleFormat::Int(8),
                dim_shift: 1,
                name: String::new(),
            });
            if custom {
                // The default kernels, nudged.
                info.upsampling_weights = UpsamplingWeights {
                    weights2: Some([
                        -0.017, -0.0345, -0.0402, -0.0292, -0.0062, 0.1411, 0.289, 0.0028, -0.0161,
                        0.5666, 0.0378, -0.0199, -0.0314, -0.0119, -0.0021,
                    ]),
                    weights4: None,
                    weights8: None,
                };
            }
            let options = FrameOptions {
                upsampling: up,
                ec_upsampling: vec![up.max(2) / 2 * if up == 8 { 1 } else { 1 }],
                ..Default::default()
            };
            let e = Encoder::new(info.clone()).unwrap();
            let sizes = e.channel_sizes(&options).unwrap();
            let (cw, ch) = sizes[0];
            assert_eq!((cw, ch), (w.div_ceil(up as usize), h.div_ceil(up as usize)));
            // A constant picture upsamples to itself.
            let color = vec![vec![128; cw * ch]; 3];
            let extra = vec![vec![77; sizes[3].0 * sizes[3].1]];
            let jxl = encode(
                info,
                vec![Frame {
                    options,
                    content: modular(color, extra),
                }],
            );
            let d = decode_raw(&jxl, &RawOptions::default())
                .unwrap_or_else(|e| panic!("upsampling {up} custom {custom}: {e}"));
            assert_eq!(d.frames[0].header.size, (w, h));
            if !custom {
                for v in &d.frames[0].color {
                    assert!((v - f(128)).abs() < 2e-3, "upsampling {up}: {v}");
                }
                for v in &d.frames[0].extra[0] {
                    assert!((v - f(77)).abs() < 2e-3, "upsampling {up}: {v}");
                }
            }
        }
    }
}

#[test]
fn passes_section_order_and_group_sizes() {
    let (w, h) = (300usize, 200usize);
    let info = ImageInfo::new(w as u32, h as u32);
    let color: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
    let check = |options: FrameOptions, transforms: Vec<Transform>, what: &str| {
        let jxl = encode(
            info.clone(),
            vec![Frame {
                options,
                content: FrameContent::Modular(ModularFrame {
                    color: color.clone(),
                    extra: vec![],
                    options: ModularOptions::default(),
                    transforms,
                }),
            }],
        );
        let d = decode_raw(&jxl, &RawOptions::default()).unwrap_or_else(|e| panic!("{what}: {e}"));
        for i in 0..w * h {
            for c in 0..3 {
                assert_eq!(
                    to_int(d.frames[0].color[i * 3 + c], 8),
                    i64::from(color[c][i]),
                    "{what}"
                );
            }
        }
    };
    for shift in 0..4 {
        check(
            FrameOptions {
                group_size_shift: shift,
                ..Default::default()
            },
            vec![],
            &format!("group size shift {shift}"),
        );
    }
    // Squeezed, so channels fall in different passes.
    let passes = Passes {
        num_passes: 3,
        shift: vec![0, 0],
        downsample: vec![4, 2],
        last_pass: vec![0, 1],
    };
    check(
        FrameOptions {
            passes: passes.clone(),
            group_size_shift: 0,
            ..Default::default()
        },
        vec![Transform::Squeeze(Vec::new())],
        "three passes",
    );
    // The sections in reverse.
    let groups = w.div_ceil(128) * h.div_ceil(128);
    let n = 2 + 1 + groups * 3;
    check(
        FrameOptions {
            passes,
            group_size_shift: 0,
            section_order: Some((0..n).rev().collect()),
            ..Default::default()
        },
        vec![Transform::Squeeze(Vec::new())],
        "reversed sections",
    );
}

#[test]
fn ycbcr_samples() {
    let (w, h) = (20usize, 14usize);
    // JPEG sampling factors: the full-resolution channel (Y) is marked.
    for j in [[0u32, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]] {
        let info = ImageInfo::new(w as u32, h as u32);
        let options = FrameOptions {
            ycbcr: Some(j),
            ..Default::default()
        };
        let e = Encoder::new(info.clone()).unwrap();
        let sizes = e.channel_sizes(&options).unwrap();
        // Samples centred as JPEG's are: Y less 128, chroma about 0.
        // Neutral chroma: gray out.
        let y = plane(sizes[1].0, sizes[1].1, 255, 4);
        let color = vec![
            vec![0; sizes[0].0 * sizes[0].1],
            y.iter().map(|v| v - 128).collect(),
            vec![0; sizes[2].0 * sizes[2].1],
        ];
        let jxl = encode(
            info,
            vec![Frame {
                options,
                content: modular(color, vec![]),
            }],
        );
        let d = decode_raw(&jxl, &RawOptions::default()).unwrap_or_else(|e| panic!("{j:?}: {e}"));
        for i in 0..w * h {
            for c in 0..3 {
                let got = d.frames[0].color[i * 3 + c];
                assert!(
                    (got - f(y[i])).abs() < 0.01,
                    "{j:?} sample {i}: {got} vs {}",
                    f(y[i])
                );
            }
        }
    }
}

#[test]
fn restoration_filters() {
    let (w, h) = (40usize, 30usize);
    let info = ImageInfo::new(w as u32, h as u32);
    let filters = [
        Restoration::DEFAULT,
        Restoration {
            gaborish: Some(Some([0.1, 0.05, 0.1, 0.05, 0.1, 0.05])),
            ..Restoration::NONE
        },
        Restoration {
            epf_iters: 3,
            epf_weights: Some(([30.0, 4.0, 3.0], 0.4, 0.5)),
            epf_sigma: Some((0.5, 0.8, 6.0, 0.6)),
            epf_sigma_for_modular: 2.0,
            ..Restoration::NONE
        },
        Restoration {
            epf_iters: 1,
            ..Restoration::NONE
        },
    ];
    for r in filters {
        let jxl = encode(
            info.clone(),
            vec![Frame {
                options: FrameOptions {
                    restoration: r.clone(),
                    ..Default::default()
                },
                content: modular(vec![vec![90; w * h]; 3], vec![]),
            }],
        );
        let d = decode_raw(&jxl, &RawOptions::default()).unwrap_or_else(|e| panic!("{r:?}: {e}"));
        for v in &d.frames[0].color {
            assert!((v - f(90)).abs() < 2e-3, "{r:?}: {v}");
        }
    }
}

#[test]
fn timecodes_and_save_before_colour_transform() {
    let (w, h) = (8usize, 8usize);
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.animation = Some(jpegxl::encode::Animation {
        tps_numerator: 10,
        tps_denominator: 1,
        num_loops: 1,
        have_timecodes: true,
    });
    let frames = (0..3)
        .map(|k| Frame {
            options: FrameOptions {
                duration: 1,
                timecode: 0x0102_0300 + k,
                save_as_reference: if k < 2 { 1 } else { 0 },
                save_before_ct: if k == 0 { Some(true) } else { None },
                ..Default::default()
            },
            content: modular(
                (0..3).map(|c| plane(w, h, 255, c + u64::from(k))).collect(),
                vec![],
            ),
        })
        .collect();
    let jxl = encode(info, frames);
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    assert_eq!(d.frames.len(), 3);
}
