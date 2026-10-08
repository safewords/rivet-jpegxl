//! The container, read back by jxl-rs.

#![allow(clippy::needless_range_loop)]

use jpegxl::encode::{Container, MetadataBox, encode_lossless, wrap};
use jpegxl::{Channels, Pixels, Samples, Wrapping};

fn picture() -> (Vec<u8>, Vec<u8>) {
    let pixels: Vec<u8> = (0..40 * 30 * 3).map(|i| (i * 7 % 251) as u8).collect();
    let jxl = encode_lossless(40, 30, Channels::Rgb, Samples::U8(&pixels)).unwrap();
    (pixels, jxl)
}

fn tiff() -> Vec<u8> {
    let mut t = b"MM\0\x2a\0\0\0\x08".to_vec();
    t.extend_from_slice(&[0, 0]);
    t
}

#[test]
fn layouts_decode_to_the_same_picture() {
    let (pixels, codestream) = picture();
    let n = codestream.len();
    let layouts = [
        Container::default(),
        Container {
            level: Some(5),
            ..Default::default()
        },
        Container {
            level: Some(10),
            boxes: vec![
                MetadataBox::exif(tiff()),
                MetadataBox::xmp(b"<x:xmpmeta/>".to_vec()),
                MetadataBox {
                    kind: *b"jumb",
                    data: vec![1, 2, 3],
                    brotli: false,
                    after_codestream: true,
                },
            ],
            ..Default::default()
        },
        Container {
            split_at: Some(vec![1, n / 3, n / 2]),
            ..Default::default()
        },
        Container {
            split_at: Some(vec![10, n / 2]),
            jxlp_out_of_order: true,
            ..Default::default()
        },
        Container {
            large_sizes: true,
            open_last_box: true,
            ..Default::default()
        },
        Container {
            split_at: Some(vec![n / 2]),
            open_last_box: true,
            boxes: vec![MetadataBox::exif(tiff())],
            ..Default::default()
        },
    ];
    for (k, c) in layouts.iter().enumerate() {
        let file = wrap(&codestream, c).unwrap();
        assert!(jpegxl::is_jxl(&file));
        let info = jpegxl::probe(&file).unwrap_or_else(|e| panic!("layout {k}: {e}"));
        assert_eq!(info.wrapping, Wrapping::Container);
        let image = jpegxl::decode(&file).unwrap_or_else(|e| panic!("layout {k}: {e}"));
        assert!(image.pixels == Pixels::U8(pixels.clone()), "layout {k}");
        if c.boxes
            .iter()
            .any(|b| &b.kind == b"Exif" && !b.after_codestream)
        {
            assert_eq!(info.exif.as_deref(), Some(&tiff()[..]), "layout {k}");
        }
    }
}

#[cfg(feature = "brotli")]
#[test]
fn brotli_wrapped_boxes() {
    let (pixels, codestream) = picture();
    let mut exif = MetadataBox::exif(tiff());
    exif.brotli = true;
    let file = wrap(
        &codestream,
        &Container {
            boxes: vec![exif],
            ..Default::default()
        },
    )
    .unwrap();
    let info = jpegxl::probe(&file).unwrap();
    assert_eq!(info.exif.as_deref(), Some(&tiff()[..]));
    assert!(jpegxl::decode(&file).unwrap().pixels == Pixels::U8(pixels));
}
