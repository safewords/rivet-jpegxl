//! XYB and lossy modular coding, and per-group transforms, read back by
//! jxl-rs.

#![allow(clippy::needless_range_loop)]

mod common;

use common::plane;
use jpegxl::encode::{
    Encoder, Frame, FrameContent, FrameOptions, ImageInfo, ModularFrame, OpsinInverse, Rct,
    Transform, Xyb, srgb_to_linear,
};
use jpegxl::{Pixels, SampleType};

fn decode_u8(jxl: &[u8]) -> Vec<u8> {
    let image = jpegxl::decode_with(
        jxl,
        &jpegxl::DecodeOptions {
            sample_type: SampleType::U8,
            ..Default::default()
        },
    )
    .unwrap();
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

fn rgb(w: usize, h: usize) -> Vec<u8> {
    let p: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c + 3)).collect();
    (0..w * h)
        .flat_map(|i| [p[0][i] as u8, p[1][i] as u8, p[2][i] as u8])
        .collect()
}

#[test]
fn xyb_modular() {
    let (w, h) = (48usize, 32usize);
    let pixels = rgb(w, h);
    let xyb = Xyb::new(&OpsinInverse::default(), 255.0);
    let scale = [1.0f32 / 8192.0, 1.0 / 1024.0, 1.0 / 1024.0];
    let mut channels = vec![vec![0i32; w * h]; 3];
    for i in 0..w * h {
        let lin = [0, 1, 2].map(|c| srgb_to_linear(f32::from(pixels[i * 3 + c]) / 255.0));
        let [x, y, b] = xyb.from_linear(lin);
        // Coded as Y, X, B - Y.
        channels[0][i] = (y / scale[1]).round() as i32;
        channels[1][i] = (x / scale[0]).round() as i32;
        channels[2][i] = ((b - y) / scale[2]).round() as i32;
    }
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.xyb = true;
    let mut e = Encoder::new(info).unwrap();
    e.add_frame(Frame {
        options: FrameOptions::default(),
        content: FrameContent::Modular(ModularFrame {
            lf_quant: Some(scale),
            ..ModularFrame::new(channels, vec![])
        }),
    });
    let jxl = e.finish().unwrap();
    let back = decode_u8(&jxl);
    let q = psnr(&pixels, &back);
    assert!(q > 40.0, "PSNR {q}");
}

#[test]
fn lossy_squeeze() {
    let (w, h) = (96usize, 64usize);
    let pixels = rgb(w, h);
    let color: Vec<Vec<i32>> = (0..3)
        .map(|c| (0..w * h).map(|i| i32::from(pixels[i * 3 + c])).collect())
        .collect();
    let mut sizes = Vec::new();
    for q in [None, Some(2.0), Some(8.0), Some(32.0)] {
        let mut e = Encoder::new(ImageInfo::new(w as u32, h as u32)).unwrap();
        e.add_frame(Frame {
            options: FrameOptions::default(),
            content: FrameContent::Modular(ModularFrame {
                transforms: vec![
                    Transform::Rct(Rct {
                        begin_channel: 0,
                        rct_type: 6,
                    }),
                    Transform::Squeeze(Vec::new()),
                ],
                residual_quantization: q,
                ..ModularFrame::new(color.clone(), vec![])
            }),
        });
        let jxl = e.finish().unwrap();
        let back = decode_u8(&jxl);
        let p = psnr(&pixels, &back);
        match q {
            None => assert_eq!(back, pixels),
            Some(q) => assert!(p > 45.0 - f64::from(q).log2() * 4.0, "step {q}: PSNR {p}"),
        }
        sizes.push(jxl.len());
    }
    assert!(sizes.windows(2).all(|s| s[1] < s[0]), "{sizes:?}");
}

#[test]
fn group_transforms() {
    let (w, h) = (300usize, 270usize);
    let pixels = rgb(w, h);
    let color: Vec<Vec<i32>> = (0..3)
        .map(|c| (0..w * h).map(|i| i32::from(pixels[i * 3 + c])).collect())
        .collect();
    for group_transforms in [
        vec![Transform::Rct(Rct {
            begin_channel: 0,
            rct_type: 6,
        })],
        vec![Transform::Squeeze(Vec::new())],
    ] {
        let mut e = Encoder::new(ImageInfo::new(w as u32, h as u32)).unwrap();
        e.add_frame(Frame {
            options: FrameOptions::default(),
            content: FrameContent::Modular(ModularFrame {
                group_transforms: group_transforms.clone(),
                ..ModularFrame::new(color.clone(), vec![])
            }),
        });
        let jxl = e.finish().unwrap();
        assert_eq!(decode_u8(&jxl), pixels, "{group_transforms:?}");
    }
}
