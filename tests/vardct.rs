//! VarDCT, read back by jxl-rs and scored against the source.

#![allow(clippy::needless_range_loop)]

mod common;

use common::plane;
use jpegxl::encode::{
    Encoder, Frame, FrameContent, FrameOptions, ImageInfo, OpsinInverse, Strategy, TransformType,
    VarDctFrame, VarDctOptions, Xyb, srgb_to_linear,
};
use jpegxl::{Pixels, SampleType};

fn rgb(w: usize, h: usize) -> Vec<u8> {
    let p: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c + 3)).collect();
    (0..w * h)
        .flat_map(|i| [p[0][i] as u8, p[1][i] as u8, p[2][i] as u8])
        .collect()
}

fn xyb_planes(pixels: &[u8], w: usize, h: usize) -> Vec<Vec<f32>> {
    let xyb = Xyb::new(&OpsinInverse::default(), 255.0);
    let mut out = vec![vec![0f32; w * h]; 3];
    for i in 0..w * h {
        let lin = [0, 1, 2].map(|c| srgb_to_linear(f32::from(pixels[i * 3 + c]) / 255.0));
        let v = xyb.from_linear(lin);
        for c in 0..3 {
            out[c][i] = v[c];
        }
    }
    out
}

fn decode_u8(jxl: &[u8]) -> Vec<u8> {
    let image = jpegxl::decode_with(
        jxl,
        &jpegxl::DecodeOptions {
            sample_type: SampleType::U8,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    match image.pixels {
        Pixels::U8(p) => p,
        _ => unreachable!(),
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

fn encode(w: usize, h: usize, pixels: &[u8], options: VarDctOptions) -> Vec<u8> {
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.xyb = true;
    let mut e = Encoder::new(info).unwrap();
    e.add_frame(Frame {
        options: FrameOptions::default(),
        content: FrameContent::VarDct(VarDctFrame {
            color: xyb_planes(pixels, w, h),
            extra: vec![],
            options,
        }),
    });
    e.finish().unwrap()
}

#[test]
fn dct8_decodes() {
    let (w, h) = (64usize, 48usize);
    let pixels = rgb(w, h);
    let jxl = encode(
        w,
        h,
        &pixels,
        VarDctOptions {
            strategy: Strategy::Fixed(TransformType::Dct8),
            chroma_from_luma: false,
            ..Default::default()
        },
    );
    let back = decode_u8(&jxl);
    let q = psnr(&pixels, &back);
    eprintln!("DCT8: {} bytes, PSNR {q:.2}", jxl.len());
    assert!(q > 30.0, "PSNR {q}");
}

fn fine() -> VarDctOptions {
    VarDctOptions {
        distance: 0.05,
        ..Default::default()
    }
}

#[test]
fn every_transform() {
    let (w, h) = (300usize, 270usize);
    let pixels = rgb(w, h);
    for t in TransformType::ALL {
        let jxl = encode(
            w,
            h,
            &pixels,
            VarDctOptions {
                strategy: Strategy::Fixed(t),
                ..fine()
            },
        );
        let q = psnr(&pixels, &decode_u8(&jxl));
        eprintln!("{t:?}: {} bytes, PSNR {q:.2}", jxl.len());
        assert!(q > 28.0, "{t:?}: PSNR {q}");
    }
}

#[test]
fn auto_strategy_and_adaptive_quantization() {
    let (w, h) = (128usize, 96usize);
    // A smooth gradient with a sharp square: large DCTs and small ones.
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let edge = (40..70).contains(&x) && (30..60).contains(&y);
            let v = if edge { 230 } else { (x + y) as u8 };
            [v, (x * 2) as u8, (y * 2) as u8]
        })
        .collect();
    for adaptive in [false, true] {
        let jxl = encode(
            w,
            h,
            &pixels,
            VarDctOptions {
                adaptive_quantization: adaptive,
                ..fine()
            },
        );
        let q = psnr(&pixels, &decode_u8(&jxl));
        assert!(q > 38.0, "adaptive {adaptive}: PSNR {q}");
    }
}

#[test]
fn custom_orders_histograms_and_contexts() {
    use jpegxl::encode::BlockContextMap;
    let (w, h) = (300usize, 270usize);
    let pixels = rgb(w, h);
    let mut bcm = BlockContextMap {
        lf_thresholds: [vec![-5, 3], vec![10], vec![]],
        qf_thresholds: vec![30, 70],
        context_map: Vec::new(),
    };
    let size = 3 * 13 * 3 * 3 * 2;
    bcm.context_map = (0..size).map(|i| (i % 11) as u8).collect();
    for (orders, hist, bcm) in [
        (true, 1, BlockContextMap::default()),
        (false, 4, BlockContextMap::default()),
        (true, 2, bcm),
    ] {
        let jxl = encode(
            w,
            h,
            &pixels,
            VarDctOptions {
                custom_orders: orders,
                num_histograms: hist,
                block_context_map: bcm.clone(),
                adaptive_quantization: true,
                ..fine()
            },
        );
        let q = psnr(&pixels, &decode_u8(&jxl));
        assert!(q > 38.0, "orders {orders} histograms {hist}: PSNR {q}");
    }
}

#[test]
fn sweep() {
    let (w, h) = (64usize, 48usize);
    let pixels = rgb(w, h);
    for d in [4.0, 1.0, 0.3, 0.1, 0.03] {
        for cfl in [false, true] {
            let jxl = encode(
                w,
                h,
                &pixels,
                VarDctOptions {
                    distance: d,
                    strategy: Strategy::Fixed(TransformType::Dct8),
                    chroma_from_luma: cfl,
                    ..Default::default()
                },
            );
            let back = decode_u8(&jxl);
            eprintln!(
                "d={d} cfl={cfl}: {} bytes, PSNR {:.2}",
                jxl.len(),
                psnr(&pixels, &back)
            );
        }
    }
}

#[test]
fn custom_quant_tables_of_every_encoding() {
    use jpegxl::encode::{Bands, QuantEncoding};
    let (w, h) = (300usize, 270usize);
    let pixels = rgb(w, h);
    let bands = |a: f32, b: f32, c: f32| {
        Bands::from_array(&[
            [a, -0.5, -0.5, -0.4],
            [b, -0.3, -0.3, -0.3],
            [c, -1.0, -0.5, -0.5],
        ])
    };
    let x8 = |n: usize| 3 * 64 * n;
    let mut tables = vec![QuantEncoding::Library; 17];
    tables[0] = QuantEncoding::Dct {
        params: bands(3000.0, 600.0, 500.0),
    };
    tables[1] = QuantEncoding::Identity {
        xyb_weights: [
            [280.0, 3000.0, 3000.0],
            [60.0, 800.0, 800.0],
            [18.0, 200.0, 200.0],
        ],
    };
    tables[2] = QuantEncoding::Dct2 {
        xyb_weights: [
            [3800.0, 2500.0, 1200.0, 600.0, 480.0, 300.0],
            [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
            [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
        ],
    };
    tables[3] = QuantEncoding::Dct4 {
        params: bands(2200.0, 392.0, 112.0),
        xyb_mul: [[1.0, 1.5], [1.0, 1.0], [0.8, 1.0]],
    };
    tables[4] = QuantEncoding::Raw {
        qtable: (0..x8(4)).map(|i| 10 + (i % 7) as i32).collect(),
        qtable_den: 0.0002,
    };
    tables[9] = QuantEncoding::Dct4x8 {
        params: bands(2200.0, 764.0, 527.0),
        xyb_mul: [1.0, 1.2, 0.9],
    };
    tables[10] = QuantEncoding::Afv {
        params4x8: bands(2200.0, 764.0, 527.0),
        params4x4: bands(2200.0, 392.0, 112.0),
        weights: [
            [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
            [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
            [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
        ],
    };
    tables[11] = QuantEncoding::Dct {
        params: bands(24000.0, 8400.0, 4500.0),
    };
    for t in [
        TransformType::Dct8,
        TransformType::Identity,
        TransformType::Dct2,
        TransformType::Dct4,
        TransformType::Dct16,
        TransformType::Dct4x8,
        TransformType::Afv2,
        TransformType::Dct64,
    ] {
        let jxl = encode(
            w,
            h,
            &pixels,
            VarDctOptions {
                strategy: Strategy::Fixed(t),
                quant_tables: Some(tables.clone()),
                ..fine()
            },
        );
        let q = psnr(&pixels, &decode_u8(&jxl));
        assert!(q > 28.0, "{t:?}: PSNR {q}");
    }
}

#[test]
fn passes_extra_channels_and_parameters() {
    use jpegxl::encode::{ColorCorrelation, ExtraChannel, Passes, Restoration, SampleFormat};
    let (w, h) = (300usize, 270usize);
    let pixels = rgb(w, h);
    let alpha: Vec<i32> = plane(w, h, 255, 9);
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.xyb = true;
    info.extra_channels
        .push(ExtraChannel::alpha(SampleFormat::Int(8)));
    let cases: Vec<(FrameOptions, VarDctOptions)> = vec![
        (
            FrameOptions {
                passes: Passes {
                    num_passes: 3,
                    shift: vec![2, 1],
                    downsample: vec![],
                    last_pass: vec![],
                },
                ..Default::default()
            },
            fine(),
        ),
        (
            FrameOptions {
                restoration: Restoration::DEFAULT,
                ..Default::default()
            },
            fine(),
        ),
        (
            FrameOptions::default(),
            VarDctOptions {
                color_correlation: ColorCorrelation {
                    color_factor: 200,
                    base_x: 0.1,
                    base_b: 0.9,
                    ytox_lf: 3,
                    ytob_lf: -4,
                },
                lf_quant: Some([1.0 / 2048.0, 1.0 / 1024.0, 1.0 / 512.0]),
                extra_precision: 2,
                x_qm_scale: 1,
                b_qm_scale: 4,
                epf_sharpness: 7,
                ..fine()
            },
        ),
    ];
    for (k, (frame, options)) in cases.into_iter().enumerate() {
        let mut e = Encoder::new(info.clone()).unwrap();
        e.add_frame(Frame {
            options: frame,
            content: FrameContent::VarDct(VarDctFrame {
                color: xyb_planes(&pixels, w, h),
                extra: vec![alpha.clone()],
                options,
            }),
        });
        let jxl = e.finish().unwrap();
        let image = jpegxl::decode_with(
            &jxl,
            &jpegxl::DecodeOptions {
                sample_type: SampleType::U8,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("case {k}: {e}"));
        let Pixels::U8(p) = image.pixels else {
            unreachable!()
        };
        let rgb_back: Vec<u8> = p.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
        let alpha_back: Vec<i32> = p.chunks(4).map(|c| i32::from(c[3])).collect();
        let q = psnr(&pixels, &rgb_back);
        assert!(q > 35.0, "case {k}: PSNR {q}");
        assert_eq!(alpha_back, alpha, "case {k}: the alpha is lossless");
    }
}

#[test]
fn ycbcr_vardct() {
    let (w, h) = (96usize, 64usize);
    let pixels = rgb(w, h);
    for j in [[0u32, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]] {
        let info = ImageInfo::new(w as u32, h as u32);
        let options = FrameOptions {
            ycbcr: Some(j),
            ..Default::default()
        };
        let e0 = Encoder::new(info.clone()).unwrap();
        let sizes = e0.channel_sizes(&options).unwrap();
        // JPEG's YCbCr, centred; chroma averaged down.
        let at = |x: usize, y: usize, c: usize| {
            f32::from(pixels[(y.min(h - 1) * w + x.min(w - 1)) * 3 + c]) / 255.0
        };
        let conv = |x: usize, y: usize| {
            let (r, g, b) = (at(x, y, 0), at(x, y, 1), at(x, y, 2));
            let yy = 0.299 * r + 0.587 * g + 0.114 * b;
            [
                -0.168736 * r - 0.331264 * g + 0.5 * b,
                yy - 128.0 / 255.0,
                0.5 * r - 0.418688 * g - 0.081312 * b,
            ]
        };
        let mut color = Vec::new();
        for (c, &(cw, ch)) in sizes[..3].iter().enumerate() {
            let (fx, fy) = (w.div_ceil(cw), h.div_ceil(ch));
            let mut plane = vec![0f32; cw * ch];
            for y in 0..ch {
                for x in 0..cw {
                    let mut s = 0.0;
                    for dy in 0..fy {
                        for dx in 0..fx {
                            s += conv(x * fx + dx, y * fy + dy)[c];
                        }
                    }
                    plane[y * cw + x] = s / (fx * fy) as f32;
                }
            }
            color.push(plane);
        }
        let mut e = Encoder::new(info).unwrap();
        e.add_frame(Frame {
            options,
            content: FrameContent::VarDct(VarDctFrame {
                color,
                extra: vec![],
                options: VarDctOptions {
                    strategy: Strategy::Fixed(TransformType::Dct8),
                    chroma_from_luma: false,
                    ..fine()
                },
            }),
        });
        let jxl = e.finish().unwrap();
        let q = psnr(&pixels, &decode_u8(&jxl));
        assert!(q > 25.0, "{j:?}: PSNR {q}");
    }
}

#[test]
fn lf_frames() {
    use jpegxl::encode::{FrameType, ModularFrame};
    let (w, h) = (256usize, 200usize);
    let pixels = rgb(w, h);
    let planes = xyb_planes(&pixels, w, h);
    // The LF: 8x8 means, coded as a modular XYB frame at 1/8 size.
    let (lw, lh) = (w.div_ceil(8), h.div_ceil(8));
    let scale = [1.0f32 / 65536.0, 1.0 / 16384.0, 1.0 / 16384.0];
    let mean = |c: usize, bx: usize, by: usize| {
        let mut s = 0.0;
        for y in by * 8..by * 8 + 8 {
            for x in bx * 8..bx * 8 + 8 {
                s += planes[c][y.min(h - 1) * w + x.min(w - 1)];
            }
        }
        s / 64.0
    };
    let mut lf = vec![vec![0i32; lw * lh]; 3];
    for by in 0..lh {
        for bx in 0..lw {
            let (x, y, b) = (mean(0, bx, by), mean(1, bx, by), mean(2, bx, by));
            lf[0][by * lw + bx] = (y / scale[1]).round() as i32;
            lf[1][by * lw + bx] = (x / scale[0]).round() as i32;
            lf[2][by * lw + bx] = ((b - y) / scale[2]).round() as i32;
        }
    }
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.xyb = true;
    let mut e = Encoder::new(info).unwrap();
    e.add_frame(Frame {
        options: FrameOptions {
            frame_type: FrameType::Lf { level: 1 },
            ..Default::default()
        },
        content: FrameContent::Modular(ModularFrame {
            lf_quant: Some(scale),
            ..ModularFrame::new(lf, vec![])
        }),
    });
    e.add_frame(Frame {
        options: FrameOptions {
            use_lf_frame: true,
            ..Default::default()
        },
        content: FrameContent::VarDct(VarDctFrame {
            color: planes,
            extra: vec![],
            options: fine(),
        }),
    });
    let jxl = e.finish().unwrap();
    let q = psnr(&pixels, &decode_u8(&jxl));
    assert!(q > 38.0, "PSNR {q}");
}

#[test]
fn rgb_vardct_crops_upsampling_and_features() {
    use jpegxl::encode::{Crop, Features, Noise, QuantizedSpline, Splines};
    let (w, h) = (120usize, 90usize);
    let pixels = rgb(w, h);
    let lin: Vec<Vec<f32>> = (0..3)
        .map(|c| {
            (0..w * h)
                .map(|i| f32::from(pixels[i * 3 + c]) / 255.0)
                .collect()
        })
        .collect();
    // RGB samples (no XYB): the colour channels as they are.
    let info = ImageInfo::new(w as u32, h as u32);
    let mut e = Encoder::new(info.clone()).unwrap();
    e.add_frame(Frame {
        options: FrameOptions::default(),
        content: FrameContent::VarDct(VarDctFrame {
            color: lin.clone(),
            extra: vec![],
            options: VarDctOptions {
                distance: 0.02,
                chroma_from_luma: false,
                ..Default::default()
            },
        }),
    });
    let q = psnr(&pixels, &decode_u8(&e.finish().unwrap()));
    assert!(q > 30.0, "RGB VarDCT: PSNR {q}");

    // An XYB image: a cropped VarDCT frame over a base, upsampled, with noise
    // and a spline. A smooth picture, which upsampling keeps.
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [(x * 2) as u8, (y * 2) as u8, (x + y) as u8]
        })
        .collect();
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.xyb = true;
    let planes = xyb_planes(&pixels, w, h);
    let mut e = Encoder::new(info).unwrap();
    e.add_frame(Frame {
        options: FrameOptions {
            is_last: Some(false),
            save_as_reference: 1,
            ..Default::default()
        },
        content: FrameContent::VarDct(VarDctFrame {
            color: planes.clone(),
            extra: vec![],
            options: fine(),
        }),
    });
    let crop = Crop {
        x0: 10,
        y0: 6,
        width: 50,
        height: 40,
    };
    let sub: Vec<Vec<f32>> = (0..3)
        .map(|c| {
            let mut v = Vec::new();
            for y in 0..20 {
                for x in 0..25 {
                    v.push(planes[c][(6 + y * 2) * w + 10 + x * 2]);
                }
            }
            v
        })
        .collect();
    let mut dct = [[0i32; 32]; 3];
    dct[1][0] = 30;
    let mut sigma = [0i32; 32];
    sigma[0] = 20;
    e.add_frame(Frame {
        options: FrameOptions {
            crop: Some(crop),
            upsampling: 2,
            blending: jpegxl::encode::Blending {
                source: 1,
                ..Default::default()
            },
            features: Features {
                noise: Some(Noise {
                    lut: [5, 10, 20, 30, 30, 20, 10, 5],
                }),
                splines: Some(Splines {
                    quantization_adjustment: 0,
                    splines: vec![QuantizedSpline {
                        start: (2, 2),
                        control_deltas: vec![(5, 3)],
                        color_dct: dct,
                        sigma_dct: sigma,
                    }],
                }),
                ..Default::default()
            },
            ..Default::default()
        },
        content: FrameContent::VarDct(VarDctFrame {
            color: sub,
            extra: vec![],
            options: fine(),
        }),
    });
    let back = decode_u8(&e.finish().unwrap());
    // Outside the crop, the base frame shows through.
    let outside: Vec<usize> = (0..w * h)
        .filter(|&i| (i % w) < 10 || (i / w) < 6)
        .collect();
    let a: Vec<u8> = outside
        .iter()
        .flat_map(|&i| pixels[i * 3..i * 3 + 3].to_vec())
        .collect();
    let b: Vec<u8> = outside
        .iter()
        .flat_map(|&i| back[i * 3..i * 3 + 3].to_vec())
        .collect();
    let q = psnr(&a, &b);
    assert!(q > 35.0, "around the crop: PSNR {q}");
}

/// 8x8 means of a plane.
fn means(p: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (lw, lh) = (w.div_ceil(8), h.div_ceil(8));
    let mut out = vec![0f32; lw * lh];
    for by in 0..lh {
        for bx in 0..lw {
            let mut s = 0.0;
            for y in by * 8..by * 8 + 8 {
                for x in bx * 8..bx * 8 + 8 {
                    s += p[y.min(h - 1) * w + x.min(w - 1)];
                }
            }
            out[by * lw + bx] = s / 64.0;
        }
    }
    (out, lw, lh)
}

#[test]
fn two_lf_levels() {
    use jpegxl::encode::{FrameType, ModularFrame};
    let (w, h) = (520usize, 300usize);
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [(x / 3) as u8, (y / 2) as u8, ((x + y) / 4) as u8]
        })
        .collect();
    let planes = xyb_planes(&pixels, w, h);
    // Level 1: the 8x8 means; level 2: their 8x8 means.
    let lf1: Vec<Vec<f32>> = (0..3).map(|c| means(&planes[c], w, h).0).collect();
    let (l1w, l1h) = (w.div_ceil(8), h.div_ceil(8));
    let lf2: Vec<Vec<f32>> = (0..3).map(|c| means(&lf1[c], l1w, l1h).0).collect();
    let (l2w, l2h) = (l1w.div_ceil(8), l1h.div_ceil(8));
    let scale = [1.0f32 / 65536.0, 1.0 / 16384.0, 1.0 / 16384.0];
    let mut q2 = vec![vec![0i32; l2w * l2h]; 3];
    for i in 0..l2w * l2h {
        q2[0][i] = (lf2[1][i] / scale[1]).round() as i32;
        q2[1][i] = (lf2[0][i] / scale[0]).round() as i32;
        q2[2][i] = ((lf2[2][i] - lf2[1][i]) / scale[2]).round() as i32;
    }
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.xyb = true;
    let mut e = Encoder::new(info).unwrap();
    e.add_frame(Frame {
        options: FrameOptions {
            frame_type: FrameType::Lf { level: 2 },
            ..Default::default()
        },
        content: FrameContent::Modular(ModularFrame {
            lf_quant: Some(scale),
            ..ModularFrame::new(q2, vec![])
        }),
    });
    e.add_frame(Frame {
        options: FrameOptions {
            frame_type: FrameType::Lf { level: 1 },
            use_lf_frame: true,
            ..Default::default()
        },
        content: FrameContent::VarDct(VarDctFrame {
            color: lf1,
            extra: vec![],
            options: fine(),
        }),
    });
    e.add_frame(Frame {
        options: FrameOptions {
            use_lf_frame: true,
            ..Default::default()
        },
        content: FrameContent::VarDct(VarDctFrame {
            color: planes,
            extra: vec![],
            options: fine(),
        }),
    });
    let q = psnr(&pixels, &decode_u8(&e.finish().unwrap()));
    assert!(q > 38.0, "PSNR {q}");
}

#[test]
fn custom_opsin_and_upsampling_kernels() {
    use jpegxl::encode::UpsamplingWeights;
    let (w, h) = (64usize, 48usize);
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [(x * 3) as u8, (y * 4) as u8, 100]
        })
        .collect();
    let mut opsin = OpsinInverse {
        opsin_biases: [-0.003; 3],
        ..Default::default()
    };
    opsin.inverse_matrix[0] *= 1.01;
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.xyb = true;
    info.opsin_inverse = Some(opsin);
    let mut w4 = vec![0.0f32; 55];
    w4[10] = 0.2;
    w4[11] = 0.3;
    info.upsampling_weights = UpsamplingWeights {
        weights2: None,
        weights4: Some(w4),
        weights8: Some(vec![0.01; 210]),
    };
    let xyb = Xyb::new(&opsin, 255.0);
    let mut color = vec![vec![0f32; w * h]; 3];
    for i in 0..w * h {
        let lin = [0, 1, 2].map(|c| srgb_to_linear(f32::from(pixels[i * 3 + c]) / 255.0));
        let v = xyb.from_linear(lin);
        for c in 0..3 {
            color[c][i] = v[c];
        }
    }
    let mut e = Encoder::new(info).unwrap();
    e.add_frame(Frame {
        options: FrameOptions::default(),
        content: FrameContent::VarDct(VarDctFrame {
            color,
            extra: vec![],
            options: fine(),
        }),
    });
    let q = psnr(&pixels, &decode_u8(&e.finish().unwrap()));
    assert!(q > 38.0, "PSNR {q}");
}

#[test]
fn encode_lossy_layouts() {
    let (w, h) = (90usize, 70usize);
    let smooth = |c: usize, i: usize| -> u32 {
        let (x, y) = (i % w, i / w);
        ((x * (c + 2) + y * 3) % 256) as u32
    };
    for channels in [
        jpegxl::Channels::Gray,
        jpegxl::Channels::GrayAlpha,
        jpegxl::Channels::Rgb,
        jpegxl::Channels::Rgba,
    ] {
        let n = channels.count();
        let pixels: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                (0..n).map(move |c| {
                    if channels.has_alpha() && c == n - 1 {
                        (i % 251) as u8
                    } else {
                        smooth(c, i) as u8
                    }
                })
            })
            .collect();
        let jxl = jpegxl::encode_lossy(
            w as u32,
            h as u32,
            channels,
            jpegxl::Samples::U8(&pixels),
            0.5,
        )
        .unwrap();
        let image = jpegxl::decode_with(
            &jxl,
            &jpegxl::DecodeOptions {
                sample_type: SampleType::U8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(image.channels, channels);
        let Pixels::U8(back) = image.pixels else {
            unreachable!()
        };
        let color_of = |v: &[u8]| -> Vec<u8> {
            v.chunks(n)
                .flat_map(|p| p[..if channels.has_alpha() { n - 1 } else { n }].to_vec())
                .collect()
        };
        let q = psnr(&color_of(&pixels), &color_of(&back));
        assert!(q > 34.0, "{channels:?}: PSNR {q}");
        if channels.has_alpha() {
            let a: Vec<u8> = pixels.chunks(n).map(|p| p[n - 1]).collect();
            let b: Vec<u8> = back.chunks(n).map(|p| p[n - 1]).collect();
            assert_eq!(a, b, "{channels:?}: alpha");
        }
    }
}
