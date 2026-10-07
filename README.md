# rivet-jpegxl

[![CI](https://github.com/safewords/rivet-jpegxl/actions/workflows/ci.yml/badge.svg)](https://github.com/safewords/rivet-jpegxl/actions/workflows/ci.yml)

**JPEG XL decoding** with a small, typed API, over
[jxl-rs](https://github.com/libjxl/jxl-rs) — the JPEG XL project's own
pure-Rust decoder. No C, no system libraries, no build script.

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

Decode only: jxl-rs is a decoder.

## How it is checked

`cargo test`: the jxl-rs test files in `tests/data` — 8-bit RGB and RGBA,
16-bit RGB to exact samples (and as 8-bit and float), grayscale, a container
with Exif, a five-frame animation decoded frame by frame (and the same on one
thread and many), the limits, and truncated or damaged input as errors rather
than panics — plus the colour-encoding and Exif parsing. jxl-rs is tested
against the JPEG XL conformance suite in its own repository.

## Licence

This crate: the Open Encoding Attribution License 1.0 ([LICENSE.md](LICENSE.md)).
jxl-rs and the test files in `tests/data`: BSD-3-Clause, Copyright (c) the
JPEG XL Project Authors ([tests/data/LICENSE.jxl-rs](tests/data/LICENSE.jxl-rs)),
with Google's royalty-free patent grant for JPEG XL
([PATENTS](https://github.com/libjxl/jxl-rs/blob/main/PATENTS)). See
[NOTICE](NOTICE).
