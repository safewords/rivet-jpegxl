# rivet-jpegxl

[![CI](https://github.com/safewords/rivet-jpegxl/actions/workflows/ci.yml/badge.svg)](https://github.com/safewords/rivet-jpegxl/actions/workflows/ci.yml)

**JPEG XL decoding** with a small, typed API, over
[jxl-rs](https://github.com/libjxl/jxl-rs) — the JPEG XL project's own
pure-Rust decoder — and a **JPEG XL encoder** of this crate's own that
writes everything the decoder reads, lossless and lossy.
No C, no system libraries, no build script.

Written for the **[rivet](https://github.com/safewords/rivet)** transcoder,
where it is JPEG XL input to the still-image path (`rivet image in.jxl`,
`mode=image`). It gives jxl-rs's type-state decoder the shape of rivet's
other image crates: `probe`, `decode`, `frames`, an `Info`, `Limits`, a named
`Error`.

Published as `rivet-jpegxl`; **imported as `jpegxl`** (`use jpegxl::…`). One
dependency, `jxl`.

```toml
[dependencies]
jpegxl = { package = "rivet-jpegxl", git = "https://github.com/safewords/rivet-jpegxl", branch = "develop" }
```

## Use

```rust,no_run
# fn main() -> jpegxl::Result<()> {
let data = std::fs::read("in.jxl").unwrap();

// The header alone.
let info = jpegxl::probe(&data)?;
println!("{}x{} {} bits{}", info.width, info.height, info.bits_per_sample,
         if info.float { " float" } else { "" });

// A still, or an animation's first frame, at the file's own precision.
let image = jpegxl::decode(&data)?;
match &image.pixels {
    jpegxl::Pixels::U8(p) => println!("8-bit {:?}, {} samples", image.channels, p.len()),
    jpegxl::Pixels::U16(p) => println!("16-bit, {} samples", p.len()),
    jpegxl::Pixels::F32(p) => println!("float, {} samples", p.len()),
}

// Every frame of an animation, as asked for.
let decoder = jpegxl::Decoder::new(&data)?;
for frame in decoder.frames() {
    let frame = frame?;
    println!("at {} ms for {} ms", frame.timestamp_ms, frame.duration_ms);
}

// 8-bit RGB(A) whatever the file, on four threads.
let options = jpegxl::DecodeOptions {
    sample_type: jpegxl::SampleType::U8,
    gray_to_rgb: true,
    threads: 4,
    ..Default::default()
};
let rgb = jpegxl::decode_with(&data, &options)?;

// Lossless encoding: the same samples back from any decoder.
let pixels = vec![0u8; 64 * 48 * 4];
let jxl = jpegxl::encode_lossless(64, 48, jpegxl::Channels::Rgba, jpegxl::Samples::U8(&pixels))?;
# Ok(()) }
```

| | |
|---|---|
| **Pixels** | Interleaved, rows top to bottom, no padding, straight alpha. `SampleType::Auto` (the default) keeps the file's precision: `U8` up to 8 bits a sample, `U16` (full range) above, `F32` for a float (usually HDR) file; or ask for one. Gray stays gray unless `gray_to_rgb`. CMYK files come out as RGB. |
| **Colour** | The pixels are in the colour space `Image::icc_profile` describes: the embedded profile, or one made from the codestream's colour encoding. `Info::color` names the common encodings (sRGB, linear, gamma, Display P3, BT.2100 PQ / HLG) for a caller without a colour manager. `Info::intensity_target` is the HDR peak. |
| **Orientation** | Applied by default (`apply_orientation`): the picture comes out upright, and `Info::width` / `height` are the upright size; `Info::orientation` says what it was, `stored_dims` the size as coded. |
| **Metadata** | `Info::exif` (an `Exif` box before the codestream; the TIFF header on), `Info::wrapping` (bare codestream or container). A Brotli-compressed box (`brob`) needs the `brotli` feature. |
| **Animation** | `Info::animation` (tick rate, loop count); `Decoder::frames` decodes each frame composited on the canvas, with its duration and timestamp. |
| **Limits** | `Limits::max_pixels` (default 2^28) is checked from the header before anything is allocated, and fed to jxl-rs's own check; `max_frames` bounds `frames`. |
| **Threads** | `threads` (0: one per core) runs jxl-rs's parallel work on scoped std threads; the pixels are the same whatever the count. |

## Encoding

`jpegxl::encode` writes everything jxl-rs reads — [PARITY.md](PARITY.md)
lists each feature with the API that writes it and the test that checks it.

- **Short ways:** `encode_lossless` (gray / RGB, ± alpha, 8 or 16 bits,
  every sample exact) and `encode_lossy` (VarDCT in XYB at a distance, the
  alpha exact).
- **`Encoder`:** an `ImageInfo` (any integer or float sample format, extra
  channels of every kind, named colour encodings or ICC profiles,
  orientation, preview, animation, tone mapping, custom XYB and upsampling
  kernels) and any number of frames — modular or VarDCT, regular, LF,
  reference-only or skip-progressive, cropped, blended, saved to reference
  slots, upsampled, YCbCr with chroma subsampling, in progressive passes,
  with Gaborish and EPF, patches, splines and noise.
- **Modular:** all 14 predictors and the weighted predictor, MA trees
  learned on every property, RCT, palettes (with deltas and the implicit
  entries), squeeze (lossless, or lossy with quantised residuals), global
  and per-group transforms and trees.
- **VarDCT:** all 27 block transforms, chosen per region or given;
  per-block quantisers; every dequantisation-table encoding; chroma from
  luma; custom coefficient orders and block context maps; LF frames.
- **Entropy coding:** ANS or prefix codes, LZ77, histogram clustering.
- **Container:** `wrap` puts a codestream in `jxlc` or `jxlp` boxes with
  Exif / XMP / JUMBF / any metadata boxes, Brotli-wrapped if asked.

```rust,no_run
# fn main() -> jpegxl::Result<()> {
let rgba = vec![0u8; 640 * 480 * 4];
let exact = jpegxl::encode_lossless(640, 480, jpegxl::Channels::Rgba, jpegxl::Samples::U8(&rgba))?;
let small = jpegxl::encode_lossy(640, 480, jpegxl::Channels::Rgba, jpegxl::Samples::U8(&rgba), 1.0)?;
# Ok(()) }
```

The encoder favours exactness and coverage over speed and size: its
choices (tree learning, block selection, quantisation) are simple, and
libjxl makes smaller files.

## How it is checked

`cargo test`: the jxl-rs test files in `tests/data` — 8-bit RGB and RGBA,
16-bit RGB to exact samples (and as 8-bit and float), grayscale, a container
with Exif, a five-frame animation decoded frame by frame (and the same on one
thread and many), the limits, and truncated or damaged input as errors rather
than panics — plus the colour-encoding and Exif parsing. jxl-rs is tested
against the JPEG XL conformance suite in its own repository.

The encoder is checked against jxl-rs, feature by feature (PARITY.md): each
test encodes, decodes with jxl-rs and compares — sample for sample where the
coding is lossless, by PSNR where it is lossy. Its VarDCT transforms are
checked, forwards and back, against jxl-rs's own inverse transforms.

## Licence

This crate, the encoder included: the Open Encoding Attribution License 1.0
([LICENSE.md](LICENSE.md)).
jxl-rs and the test files in `tests/data`: BSD-3-Clause, Copyright (c) the
JPEG XL Project Authors ([tests/data/LICENSE.jxl-rs](tests/data/LICENSE.jxl-rs)),
with Google's royalty-free patent grant for JPEG XL
([PATENTS](https://github.com/libjxl/jxl-rs/blob/main/PATENTS)). See
[NOTICE](NOTICE).
