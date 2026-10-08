//! The image header's features, each encoded and read back by jxl-rs.

#![allow(clippy::needless_range_loop)]

mod common;

use common::{RawOptions, decode_raw, plane, to_int};
use jpegxl::encode::{
    Animation, Chromaticity, ColorEncoding, ColorSpace, ColorSpec, Encoder, ExtraChannel,
    ExtraChannelKind, Frame, FrameContent, FrameOptions, ImageInfo, ModularFrame, ModularOptions,
    Primaries, RenderingIntent, SampleFormat, ToneMapping, TransferFunction, WhitePoint,
    float_to_format_bits,
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

fn still(info: ImageInfo, color: Vec<Vec<i32>>, extra: Vec<Vec<i32>>) -> Vec<u8> {
    encode(
        info,
        vec![Frame {
            options: FrameOptions::default(),
            content: modular(color, extra),
        }],
    )
}

#[test]
fn integer_bit_depths() {
    let (w, h) = (19usize, 11usize);
    for bits in [1, 2, 3, 5, 8, 9, 10, 12, 14, 16, 20, 24] {
        let max = ((1u64 << bits) - 1) as u32;
        let color: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, max, c + 1)).collect();
        let mut info = ImageInfo::new(w as u32, h as u32);
        info.format = SampleFormat::Int(bits);
        let jxl = still(info, color.clone(), vec![]);
        let d =
            decode_raw(&jxl, &RawOptions::default()).unwrap_or_else(|e| panic!("{bits} bits: {e}"));
        let f = &d.frames[0];
        for i in 0..w * h {
            for c in 0..3 {
                let got = to_int(f.color[i * 3 + c], bits);
                assert_eq!(
                    got,
                    i64::from(color[c][i]),
                    "{bits} bits, sample {i} channel {c}"
                );
            }
        }
    }
}

#[test]
fn float_formats() {
    let (w, h) = (16usize, 9usize);
    let mut rng = common::Rng(5);
    for (bits, exp) in [(32, 8), (16, 5), (16, 8), (24, 7), (11, 5), (20, 6)] {
        let mant = bits - exp - 1;
        // Values the format holds exactly: a few mantissa bits, a modest
        // exponent, both signs, zero, and HDR values above 1.
        let mut values = Vec::new();
        for i in 0..w * h * 3 {
            let m = (rng.next() % (1 << mant.min(6))) as f32 / (1 << mant.min(6)) as f32;
            let e = (rng.next() % 6) as i32 - 3;
            let s = if i % 7 == 0 { -1.0 } else { 1.0 };
            values.push(if i % 11 == 0 {
                0.0
            } else {
                s * (1.0 + m) * 2f32.powi(e)
            });
        }
        let ints: Vec<i32> = values
            .iter()
            .map(|&v| {
                float_to_format_bits(v, bits, exp).unwrap_or_else(|| panic!("{v} in {bits}/{exp}"))
            })
            .collect();
        let color: Vec<Vec<i32>> = (0..3)
            .map(|c| ints[c * w * h..(c + 1) * w * h].to_vec())
            .collect();
        let mut info = ImageInfo::new(w as u32, h as u32);
        info.format = SampleFormat::Float {
            bits,
            exponent_bits: exp,
        };
        info.color = ColorSpec::Encoding(ColorEncoding::LINEAR_SRGB);
        let jxl = still(info, color, vec![]);
        let d = decode_raw(&jxl, &RawOptions::default())
            .unwrap_or_else(|e| panic!("{bits}/{exp}: {e}"));
        let f = &d.frames[0];
        for i in 0..w * h {
            for c in 0..3 {
                assert_eq!(
                    f.color[i * 3 + c],
                    values[c * w * h + i],
                    "{bits}/{exp} sample {i} channel {c}"
                );
            }
        }
    }
}

#[test]
fn extra_channels_of_every_kind() {
    let (w, h) = (23usize, 17usize);
    let kinds = [
        ExtraChannelKind::Alpha { associated: false },
        ExtraChannelKind::Depth,
        ExtraChannelKind::SelectionMask,
        ExtraChannelKind::Black,
        ExtraChannelKind::Cfa(1),
        ExtraChannelKind::Cfa(40),
        ExtraChannelKind::Thermal,
        ExtraChannelKind::Reserved(0),
        ExtraChannelKind::Reserved(7),
        ExtraChannelKind::Unknown,
        ExtraChannelKind::Optional,
    ];
    let formats = [
        SampleFormat::Int(8),
        SampleFormat::Int(16),
        SampleFormat::Int(1),
        SampleFormat::Int(12),
    ];
    let mut info = ImageInfo::new(w as u32, h as u32);
    let mut extra = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        let format = formats[i % formats.len()];
        info.extra_channels.push(ExtraChannel {
            kind: *kind,
            format,
            dim_shift: 0,
            name: if i % 2 == 0 {
                format!("channel {i}")
            } else {
                String::new()
            },
        });
        extra.push(plane(w, h, (1u32 << format.bits()) - 1, i as u64 + 10));
    }
    let color: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
    let jxl = still(info.clone(), color.clone(), extra.clone());
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    assert_eq!(d.basic.extra_channels.len(), kinds.len());
    let f = &d.frames[0];
    for (k, e) in extra.iter().enumerate() {
        let bits = info.extra_channels[k].format.bits();
        for i in 0..w * h {
            assert_eq!(
                to_int(f.extra[k][i], bits),
                i64::from(e[i]),
                "channel {k} sample {i}"
            );
        }
    }
    for i in 0..w * h {
        for c in 0..3 {
            assert_eq!(to_int(f.color[i * 3 + c], 8), i64::from(color[c][i]));
        }
    }
}

#[test]
fn spot_colours_and_premultiplied_alpha_decode() {
    let (w, h) = (12usize, 8usize);
    let mut info = ImageInfo::new(w as u32, h as u32);
    info.extra_channels.push(ExtraChannel {
        kind: ExtraChannelKind::Alpha { associated: true },
        format: SampleFormat::Int(8),
        dim_shift: 0,
        name: "alpha".into(),
    });
    info.extra_channels.push(ExtraChannel {
        kind: ExtraChannelKind::SpotColor([0.25, 0.5, 1.0, 0.75]),
        format: SampleFormat::Int(8),
        dim_shift: 0,
        name: "spot".into(),
    });
    let alpha = vec![255; w * h];
    let spot = plane(w, h, 255, 3);
    let color: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
    let jxl = still(info, color, vec![alpha, spot.clone()]);
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    assert_eq!(d.basic.extra_channels.len(), 2);
    assert!(d.basic.extra_channels[0].alpha_associated);
    for i in 0..w * h {
        assert_eq!(to_int(d.frames[0].extra[1][i], 8), i64::from(spot[i]));
    }
}

#[test]
fn colour_encodings() {
    let (w, h) = (8usize, 8usize);
    let encodings = [
        ColorEncoding::SRGB,
        ColorEncoding::LINEAR_SRGB,
        ColorEncoding::DISPLAY_P3,
        ColorEncoding::BT2100_PQ,
        ColorEncoding::BT2100_HLG,
        ColorEncoding {
            transfer: TransferFunction::Gamma(4545500),
            ..ColorEncoding::SRGB
        },
        ColorEncoding {
            transfer: TransferFunction::Bt709,
            intent: RenderingIntent::Perceptual,
            ..ColorEncoding::SRGB
        },
        ColorEncoding {
            transfer: TransferFunction::Dci,
            white_point: WhitePoint::Dci,
            primaries: Primaries::P3,
            intent: RenderingIntent::Absolute,
            color_space: ColorSpace::Rgb,
        },
        ColorEncoding {
            white_point: WhitePoint::E,
            intent: RenderingIntent::Saturation,
            ..ColorEncoding::SRGB
        },
        ColorEncoding {
            white_point: WhitePoint::Custom(Chromaticity::from_f64(0.3, 0.32)),
            primaries: Primaries::Custom {
                red: Chromaticity::from_f64(0.68, 0.32),
                green: Chromaticity::from_f64(0.265, 0.69),
                blue: Chromaticity::from_f64(0.15, -0.06),
            },
            ..ColorEncoding::SRGB
        },
        ColorEncoding::GRAY,
        ColorEncoding {
            transfer: TransferFunction::Linear,
            white_point: WhitePoint::Custom(Chromaticity::from_f64(0.31, 0.33)),
            ..ColorEncoding::GRAY
        },
    ];
    for enc in encodings {
        let mut info = ImageInfo::new(w as u32, h as u32);
        info.color = ColorSpec::Encoding(enc);
        let n = if enc.color_space == ColorSpace::Gray {
            1
        } else {
            3
        };
        let color: Vec<Vec<i32>> = (0..n).map(|c| plane(w, h, 255, c as u64)).collect();
        let jxl = still(info, color.clone(), vec![]);
        let d = decode_raw(&jxl, &RawOptions::default()).unwrap_or_else(|e| panic!("{enc:?}: {e}"));
        let text = match &d.embedded {
            jpegxl::jxl::api::JxlColorProfile::Simple(e) => format!("{e:?}"),
            jpegxl::jxl::api::JxlColorProfile::Icc(_) => panic!("{enc:?}: an ICC profile"),
        };
        if enc.color_space == ColorSpace::Gray {
            assert!(text.contains("Grayscale"), "{enc:?}: {text}");
        }
        for i in 0..w * h {
            for c in 0..n {
                assert_eq!(
                    to_int(d.frames[0].color[i * n + c], 8),
                    i64::from(color[c][i]),
                    "{enc:?}"
                );
            }
        }
    }
}

/// An ICC profile jxl-rs itself writes for a colour encoding.
fn sample_profile() -> Vec<u8> {
    let mut info = ImageInfo::new(8, 8);
    info.color = ColorSpec::Encoding(ColorEncoding::DISPLAY_P3);
    let jxl = still(info, (0..3).map(|c| plane(8, 8, 255, c)).collect(), vec![]);
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    d.embedded
        .try_as_icc()
        .expect("an ICC profile")
        .into_owned()
}

#[test]
fn icc_profiles() {
    let profile = sample_profile();
    assert!(profile.len() > 128);
    let (w, h) = (10usize, 6usize);
    // As given; with its tags in another order; and a made-up one.
    let mut odd = profile.clone();
    odd[44..48].copy_from_slice(b"ZZZZ");
    let mut random: Vec<u8> = (0..300u32).map(|i| (i * 131 % 256) as u8).collect();
    random[..4].copy_from_slice(&300u32.to_be_bytes());
    for (k, p) in [profile, odd, random].into_iter().enumerate() {
        let mut info = ImageInfo::new(w as u32, h as u32);
        info.color = ColorSpec::Icc {
            profile: p.clone(),
            gray: false,
        };
        let color: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
        let jxl = still(info, color, vec![]);
        let mut input = &jxl[..];
        // The header alone is enough to read the profile back.
        let mut decoder =
            jpegxl::jxl::api::JxlDecoder::<jpegxl::jxl::api::states::Initialized>::new(
                Default::default(),
            );
        let decoder = loop {
            match decoder.process(&mut input, None) {
                Ok(jpegxl::jxl::api::ProcessingResult::Complete { result }) => break result,
                Ok(jpegxl::jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    decoder = fallback
                }
                Err(e) => panic!("profile {k}: {e}"),
            }
        };
        match decoder.embedded_color_profile() {
            jpegxl::jxl::api::JxlColorProfile::Icc(got) => assert_eq!(got, &p, "profile {k}"),
            _ => panic!("profile {k}: not an ICC profile"),
        }
    }
}

#[test]
fn orientation_and_intrinsic_size() {
    let (w, h) = (6usize, 4usize);
    let color: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
    for orientation in 1..=8u32 {
        let mut info = ImageInfo::new(w as u32, h as u32);
        info.orientation = orientation;
        info.intrinsic_size = Some((60, 40));
        let jxl = still(info, color.clone(), vec![]);
        let d = decode_raw(
            &jxl,
            &RawOptions {
                adjust_orientation: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.basic.orientation as u32, orientation);
        let transposed = orientation >= 5;
        let size = d.frames[0].header.size;
        assert_eq!(
            size,
            if transposed { (h, w) } else { (w, h) },
            "orientation {orientation}"
        );
    }
}

#[test]
fn tone_mapping() {
    let mut info = ImageInfo::new(4, 4);
    info.tone_mapping = ToneMapping {
        intensity_target: 4000.0,
        min_nits: 0.05,
        relative_to_max_display: true,
        linear_below: 0.5,
    };
    info.color = ColorSpec::Encoding(ColorEncoding::BT2100_PQ);
    info.format = SampleFormat::Int(12);
    let jxl = still(info, (0..3).map(|c| plane(4, 4, 4095, c)).collect(), vec![]);
    let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
    let t = &d.basic.tone_mapping;
    assert_eq!(t.intensity_target, 4000.0);
    assert!((t.min_nits - 0.05).abs() < 1e-4);
    assert!(t.relative_to_max_display);
    assert_eq!(t.linear_below, 0.5);
}

#[test]
fn preview() {
    let (w, h) = (40usize, 32usize);
    for (pw, ph) in [(16usize, 8usize), (13, 7), (40, 30)] {
        let mut info = ImageInfo::new(w as u32, h as u32);
        info.preview = Some((pw as u32, ph as u32));
        let mut e = Encoder::new(info).unwrap();
        let preview: Vec<Vec<i32>> = (0..3).map(|c| plane(pw, ph, 255, c + 7)).collect();
        e.set_preview(Frame {
            options: FrameOptions::default(),
            content: modular(preview.clone(), vec![]),
        })
        .unwrap();
        let main: Vec<Vec<i32>> = (0..3).map(|c| plane(w, h, 255, c)).collect();
        e.add_frame(Frame {
            options: FrameOptions::default(),
            content: modular(main.clone(), vec![]),
        });
        let jxl = e.finish().unwrap();
        let d = decode_raw(
            &jxl,
            &RawOptions {
                skip_preview: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.basic.preview_size, Some((pw, ph)));
        assert_eq!(d.frames.len(), 2, "{pw}x{ph}");
        assert_eq!(d.frames[0].header.size, (pw, ph));
        for i in 0..pw * ph {
            assert_eq!(
                to_int(d.frames[0].color[i * 3], 8),
                i64::from(preview[0][i])
            );
        }
        for i in 0..w * h {
            assert_eq!(
                to_int(d.frames[1].color[i * 3 + 2], 8),
                i64::from(main[2][i])
            );
        }
        // And skipped.
        let d = decode_raw(&jxl, &RawOptions::default()).unwrap();
        assert_eq!(d.frames.len(), 1);
    }
}

#[test]
fn animation() {
    let (w, h) = (9usize, 7usize);
    for (tps_n, tps_d, loops, timecodes) in [
        (100, 1, 0, false),
        (30000, 1001, 3, true),
        (7, 3, 70000, false),
    ] {
        let mut info = ImageInfo::new(w as u32, h as u32);
        info.animation = Some(Animation {
            tps_numerator: tps_n,
            tps_denominator: tps_d,
            num_loops: loops,
            have_timecodes: timecodes,
        });
        let durations = [1u32, 0, 300, 70000];
        let frames: Vec<Frame> = durations
            .iter()
            .enumerate()
            .map(|(k, &d)| Frame {
                options: FrameOptions {
                    duration: d,
                    timecode: k as u32 * 1000,
                    name: format!("frame {k}"),
                    ..Default::default()
                },
                content: modular(
                    (0..3)
                        .map(|c| plane(w, h, 255, (k * 3 + c) as u64))
                        .collect(),
                    vec![],
                ),
            })
            .collect();
        let jxl = encode(info, frames);
        let d = decode_raw(
            &jxl,
            &RawOptions {
                coalescing: false,
                ..Default::default()
            },
        )
        .unwrap();
        let a = d.basic.animation.as_ref().unwrap();
        assert_eq!(
            (
                a.tps_numerator,
                a.tps_denominator,
                a.num_loops,
                a.have_timecodes
            ),
            (tps_n, tps_d, loops, timecodes)
        );
        // A frame of no duration that is not the last is not shown.
        let shown: Vec<usize> = (0..4).filter(|&k| durations[k] > 0 || k == 3).collect();
        assert_eq!(d.frames.len(), shown.len());
        for (f, &k) in d.frames.iter().zip(&shown) {
            assert_eq!(f.header.name, format!("frame {k}"));
            let expect = f64::from(durations[k]) * 1000.0 * f64::from(tps_d) / f64::from(tps_n);
            assert!(
                (f.header.duration.unwrap() - expect).abs() < 1e-6,
                "{:?} {expect}",
                f.header.duration
            );
        }
    }
}
