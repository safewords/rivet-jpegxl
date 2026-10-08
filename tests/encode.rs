//! The lossless encoder, checked by decoding its output with jxl-rs and
//! comparing every sample.

use jpegxl::{Channels, Pixels, Samples, encode_lossless};

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

const ALL_CHANNELS: [Channels; 4] = [
    Channels::Gray,
    Channels::GrayAlpha,
    Channels::Rgb,
    Channels::Rgba,
];

/// A picture with smooth gradients, edges and some noise: every predictor
/// has something to do.
fn picture(width: u32, height: u32, channels: Channels, max: u32, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed | 1);
    let count = channels.count();
    let mut out = Vec::with_capacity(width as usize * height as usize * count);
    for y in 0..height {
        for x in 0..width {
            for c in 0..count as u32 {
                let smooth = (x * (c + 1) * 3 + y * 5) % (max + 1);
                let edge = if (x / 7 + y / 5) % 2 == 0 { max / 3 } else { 0 };
                let noise = (rng.next() % 9) as u32;
                out.push((smooth + edge + noise).min(max));
            }
        }
    }
    out
}

fn round_trip_u8(width: u32, height: u32, channels: Channels, pixels: Vec<u8>) {
    let jxl = encode_lossless(width, height, channels, Samples::U8(&pixels))
        .unwrap_or_else(|e| panic!("{width}x{height} {channels:?}: {e}"));
    let image =
        jpegxl::decode(&jxl).unwrap_or_else(|e| panic!("{width}x{height} {channels:?}: {e}"));
    assert_eq!((image.width, image.height), (width, height));
    assert_eq!(image.channels, channels);
    assert_eq!(image.bits_per_sample, 8);
    assert!(
        image.pixels == Pixels::U8(pixels),
        "{width}x{height} {channels:?}: the samples differ"
    );
}

fn round_trip_u16(width: u32, height: u32, channels: Channels, pixels: Vec<u16>) {
    let jxl = encode_lossless(width, height, channels, Samples::U16(&pixels))
        .unwrap_or_else(|e| panic!("{width}x{height} {channels:?}: {e}"));
    let image =
        jpegxl::decode(&jxl).unwrap_or_else(|e| panic!("{width}x{height} {channels:?}: {e}"));
    assert_eq!((image.width, image.height), (width, height));
    assert_eq!(image.channels, channels);
    assert_eq!(image.bits_per_sample, 16);
    assert!(
        image.pixels == Pixels::U16(pixels),
        "{width}x{height} {channels:?}: the samples differ"
    );
}

#[test]
fn every_layout_at_8_bits() {
    for channels in ALL_CHANNELS {
        for (w, h) in [(1, 1), (3, 3), (17, 5), (64, 64), (255, 1), (1, 200)] {
            let pixels = picture(w, h, channels, 255, u64::from(w * h))
                .into_iter()
                .map(|v| v as u8)
                .collect();
            round_trip_u8(w, h, channels, pixels);
        }
    }
}

#[test]
fn every_layout_at_16_bits() {
    for channels in ALL_CHANNELS {
        for (w, h) in [(1, 1), (9, 13), (100, 40)] {
            let pixels = picture(w, h, channels, 65535, 7)
                .into_iter()
                .map(|v| (v * 211 % 65536) as u16)
                .collect();
            round_trip_u16(w, h, channels, pixels);
        }
    }
}

#[test]
fn many_groups() {
    // 256-pixel groups: 3x2 of them, the last row and column partial; and
    // past one 2048-pixel LF group across.
    for (w, h, channels) in [(600, 300, Channels::Rgba), (2100, 260, Channels::Gray)] {
        let pixels = picture(w, h, channels, 255, 99)
            .into_iter()
            .map(|v| v as u8)
            .collect();
        round_trip_u8(w, h, channels, pixels);
    }
}

#[test]
fn extremes() {
    // One value everywhere: single-symbol codes.
    round_trip_u8(40, 30, Channels::Rgb, vec![200; 40 * 30 * 3]);
    round_trip_u16(10, 10, Channels::GrayAlpha, vec![65535; 200]);
    // White noise: every token, the largest residuals.
    let mut rng = Rng(12345);
    let noise: Vec<u16> = (0..70 * 50 * 4).map(|_| rng.next() as u16).collect();
    round_trip_u16(70, 50, Channels::Rgba, noise);
    let noise: Vec<u8> = (0..300 * 3).map(|_| rng.next() as u8).collect();
    round_trip_u8(300, 1, Channels::Rgb, noise);
    // Alternating extremes.
    let checker: Vec<u8> = (0..32 * 32)
        .map(|i| if (i % 32 + i / 32) % 2 == 0 { 0 } else { 255 })
        .collect();
    round_trip_u8(32, 32, Channels::Gray, checker);
}

#[test]
fn smooth_pictures_compress() {
    let (w, h) = (256u32, 256u32);
    let pixels: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [x as u8, y as u8, ((x + y) / 2) as u8]))
        .collect();
    let jxl = encode_lossless(w, h, Channels::Rgb, Samples::U8(&pixels)).unwrap();
    // Prefix codes spend at least a bit on each sample whose context sees
    // more than one residual: about a bit a sample here, against eight.
    assert!(
        jxl.len() < pixels.len() / 7,
        "{} bytes for {} samples",
        jxl.len(),
        pixels.len()
    );
    assert_eq!(jpegxl::decode(&jxl).unwrap().pixels, Pixels::U8(pixels));
}

#[test]
fn the_header_says_what_was_encoded() {
    let jxl = encode_lossless(5, 4, Channels::GrayAlpha, Samples::U16(&[300; 40])).unwrap();
    assert!(jpegxl::is_jxl(&jxl));
    let info = jpegxl::probe(&jxl).unwrap();
    assert_eq!((info.width, info.height), (5, 4));
    assert!(info.gray && info.has_alpha && !info.float);
    assert_eq!(info.bits_per_sample, 16);
    assert_eq!(info.color, jpegxl::Color::Srgb);
    assert_eq!(info.wrapping, jpegxl::Wrapping::Codestream);
}

#[test]
fn bad_input_is_an_error() {
    for (w, h, n) in [(0, 4, 0), (4, 0, 0), (4, 4, 47), (4, 4, 49)] {
        let samples = vec![0u8; n];
        assert!(matches!(
            encode_lossless(w, h, Channels::Rgb, Samples::U8(&samples)),
            Err(jpegxl::Error::InvalidInput(_))
        ));
    }
}

#[test]
fn every_entropy_coder_setting() {
    use jpegxl::{EntropyOptions, LosslessOptions, Lz77Mode, ModularOptions, encode_lossless_with};
    let mut rng = Rng(77);
    let mut pictures: Vec<(u32, u32, Channels, Vec<u8>)> = vec![
        (
            40,
            30,
            Channels::Rgb,
            picture(40, 30, Channels::Rgb, 255, 3)
                .into_iter()
                .map(|v| v as u8)
                .collect(),
        ),
        (
            300,
            270,
            Channels::Rgba,
            picture(300, 270, Channels::Rgba, 255, 4)
                .into_iter()
                .map(|v| v as u8)
                .collect(),
        ),
        (16, 16, Channels::Gray, vec![9; 256]),
    ];
    // Repeats for LZ77 to find.
    let tile: Vec<u8> = (0..64 * 3).map(|_| rng.next() as u8).collect();
    let repeated: Vec<u8> = (0..64 * 64)
        .flat_map(|i| {
            [
                tile[(i % 64) * 3],
                tile[(i % 64) * 3 + 1],
                tile[(i % 64) * 3 + 2],
            ]
        })
        .collect();
    pictures.push((64, 64, Channels::Rgb, repeated));
    for ans in [false, true] {
        for lz77 in [Lz77Mode::Off, Lz77Mode::Rle, Lz77Mode::Full] {
            for clustering in [false, true] {
                for optimize_uint in [false, true] {
                    let options = LosslessOptions {
                        modular: ModularOptions {
                            entropy: EntropyOptions {
                                ans,
                                lz77,
                                clustering,
                                max_histograms: if clustering { 2 } else { 256 },
                                optimize_uint,
                            },
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    for (w, h, channels, pixels) in &pictures {
                        let jxl =
                            encode_lossless_with(*w, *h, *channels, Samples::U8(pixels), &options)
                                .unwrap_or_else(|e| panic!("{options:?}: {e}"));
                        let image = jpegxl::decode(&jxl)
                            .unwrap_or_else(|e| panic!("{options:?} {w}x{h} {channels:?}: {e}"));
                        assert!(
                            image.pixels == Pixels::U8(pixels.clone()),
                            "{options:?} {w}x{h} {channels:?}: samples differ"
                        );
                    }
                }
            }
        }
    }
}
