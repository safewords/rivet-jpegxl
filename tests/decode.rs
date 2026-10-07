//! Decoding the fixtures: small files from jxl-rs's own test data
//! (BSD-3-Clause, `tests/data/LICENSE.jxl-rs`).

use jpegxl::{
    Channels, Color, DecodeOptions, Decoder, Error, Limits, Pixels, SampleType, Wrapping,
};

const RGB: &[u8] = include_bytes!("data/3x3_srgb_lossless.jxl");
const RGBA: &[u8] = include_bytes!("data/3x3a_srgb_lossless.jxl");
const RGB16: &[u8] = include_bytes!("data/image_integration_rgb16.jxl");
const EXIF: &[u8] = include_bytes!("data/exif.jxl");
const GRAY: &[u8] = include_bytes!("data/vardct_grayscale_unused_channel.jxl");
const ANIMATION: &[u8] = include_bytes!("data/5_frames_numbered_jxli.jxl");

fn with(sample_type: SampleType) -> DecodeOptions {
    DecodeOptions {
        sample_type,
        ..Default::default()
    }
}

#[test]
fn an_8_bit_rgb_still_probes_and_decodes() {
    assert!(jpegxl::is_jxl(RGB));
    let info = jpegxl::probe(RGB).unwrap();
    assert_eq!((info.width, info.height), (3, 3));
    assert_eq!((info.bits_per_sample, info.float), (8, false));
    assert!(!info.has_alpha && !info.gray && !info.cmyk);
    assert_eq!(info.orientation, 1);
    assert!(info.animation.is_none());
    // This file's transfer is a pure gamma (1/2.2), not the sRGB curve.
    assert_eq!(info.color, Color::Gamma(0.45455));
    assert!(
        info.icc_profile.is_some(),
        "the pixels' colour space, as ICC"
    );

    let image = jpegxl::decode(RGB).unwrap();
    assert_eq!(
        (image.width, image.height, image.channels),
        (3, 3, Channels::Rgb)
    );
    let Pixels::U8(p) = &image.pixels else {
        panic!("8-bit pixels for an 8-bit file: {:?}", image.pixels)
    };
    assert_eq!(p.len(), 27);
    assert!(p.iter().any(|&s| s != 0));
}

#[test]
fn alpha_is_kept() {
    let info = jpegxl::probe(RGBA).unwrap();
    assert!(info.has_alpha);
    let image = jpegxl::decode(RGBA).unwrap();
    assert_eq!(image.channels, Channels::Rgba);
    assert_eq!(image.pixels.len(), 3 * 3 * 4);
}

#[test]
fn sixteen_bits_come_out_as_sixteen_or_as_asked() {
    let info = jpegxl::probe(RGB16).unwrap();
    assert!(info.bits_per_sample > 8);
    let image = jpegxl::decode(RGB16).unwrap();
    assert_eq!(image.channels, Channels::Rgb);
    assert_eq!(
        image.pixels,
        Pixels::U16(vec![u16::MAX, 0, 0, 32768, u16::MAX, 0]),
        "exact samples"
    );

    let image = jpegxl::decode_with(RGB16, &with(SampleType::U8)).unwrap();
    assert_eq!(image.pixels, Pixels::U8(vec![255, 0, 0, 128, 255, 0]));

    let image = jpegxl::decode_with(RGB16, &with(SampleType::F32)).unwrap();
    let Pixels::F32(p) = image.pixels else {
        panic!("float")
    };
    let want = [1.0, 0.0, 0.0, 0.5, 1.0, 0.0];
    for (got, want) in p.iter().zip(want) {
        assert!((got - want).abs() < 1e-3, "{p:?}");
    }
}

#[test]
fn gray_stays_gray_unless_asked() {
    let info = jpegxl::probe(GRAY).unwrap();
    assert!(info.gray);
    let image = jpegxl::decode(GRAY).unwrap();
    assert!(image.channels.is_gray(), "{:?}", image.channels);
    let rgb = jpegxl::decode_with(
        GRAY,
        &DecodeOptions {
            gray_to_rgb: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rgb.channels.count(), image.channels.count() + 2);
    assert_eq!(
        rgb.pixels.len() / rgb.channels.count(),
        image.pixels.len() / image.channels.count()
    );
}

#[test]
fn the_container_and_its_exif_are_read() {
    let info = jpegxl::probe(EXIF).unwrap();
    assert_eq!(info.wrapping, Wrapping::Container);
    let exif = info.exif.expect("the Exif box");
    assert!(
        exif.starts_with(b"II*\0") || exif.starts_with(b"MM\0*"),
        "the TIFF header: {:?}",
        &exif[..exif.len().min(8)]
    );
    jpegxl::decode(EXIF).unwrap();
}

#[test]
fn an_animation_decodes_frame_by_frame() {
    let decoder = Decoder::new(ANIMATION).unwrap();
    let info = decoder.info().clone();
    assert!(info.animation.is_some());
    let frames: Vec<_> = decoder.frames().collect::<jpegxl::Result<_>>().unwrap();
    assert_eq!(frames.len(), 5);
    let mut at = 0.0;
    for f in &frames {
        assert_eq!((f.image.width, f.image.height), (info.width, info.height));
        assert_eq!(f.timestamp_ms, at);
        at += f.duration_ms;
    }
    // Numbered frames: no two the same.
    assert!(
        frames
            .windows(2)
            .all(|w| w[0].image.pixels != w[1].image.pixels)
    );
    // The first frame alone is what `decode` makes.
    assert_eq!(decoder.decode().unwrap(), frames[0].image);
}

#[test]
fn one_thread_and_many_make_the_same_pixels() {
    let one = Decoder::with_options(
        ANIMATION,
        DecodeOptions {
            threads: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let many = Decoder::with_options(
        ANIMATION,
        DecodeOptions {
            threads: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let a: Vec<_> = one.frames().map(|f| f.unwrap().image).collect();
    let b: Vec<_> = many.frames().map(|f| f.unwrap().image).collect();
    assert_eq!(a, b);
}

#[test]
fn limits_and_bad_input_are_errors_not_panics() {
    let small = DecodeOptions {
        limits: Limits {
            max_pixels: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        jpegxl::decode_with(RGB, &small),
        Err(Error::LimitExceeded(_))
    ));
    let one_frame = DecodeOptions {
        limits: Limits {
            max_frames: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let decoder = Decoder::with_options(ANIMATION, one_frame).unwrap();
    let frames: Vec<_> = decoder.frames().collect();
    assert!(frames[0].is_ok());
    assert!(matches!(frames[1], Err(Error::LimitExceeded(_))));

    assert!(matches!(
        jpegxl::probe(b"\x89PNG\r\n\x1a\n"),
        Err(Error::NotJpegXl)
    ));
    assert!(!jpegxl::is_jxl(b"\xFF\xD8\xFF"));
    for cut in 2..RGB.len() {
        assert!(jpegxl::decode(&RGB[..cut]).is_err(), "cut at {cut}");
    }
    for cut in (2..ANIMATION.len()).step_by(97) {
        let _ = jpegxl::decode(&ANIMATION[..cut]);
    }
    let mut flipped = ANIMATION.to_vec();
    for i in (40..flipped.len()).step_by(13) {
        flipped[i] ^= 0x5A;
    }
    let _ = jpegxl::decode(&flipped);
}

#[test]
fn orientation_can_be_left_to_the_caller() {
    let info = jpegxl::probe(RGB).unwrap();
    assert_eq!(
        info.stored_dims(),
        (info.width, info.height),
        "orientation 1"
    );
    let raw = jpegxl::decode_with(
        RGB,
        &DecodeOptions {
            apply_orientation: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(raw, jpegxl::decode(RGB).unwrap(), "upright already");
}
