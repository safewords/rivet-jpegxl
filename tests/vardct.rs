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
