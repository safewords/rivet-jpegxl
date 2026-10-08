# Encoder–decoder parity

Every part of the JPEG XL codestream and container that jxl-rs (the
decoder this crate wraps) reads, the encoder (`jpegxl::encode`) can write.
Each row names the API that writes it and the test that writes it, decodes
it with jxl-rs and checks the result: sample for sample where the coding is
lossless, by PSNR where it is lossy.

## Entropy coding

| Feature | Encoder | Test |
|---|---|---|
| Prefix codes: simple (1–4 symbols, both 4-symbol shapes), complex with repeat codes | `EntropyOptions { ans: false }` | `encode::every_entropy_coder_setting` |
| ANS: one-symbol, two-symbol, evenly spread and general distributions (precision 0–13, RLE) | `EntropyOptions { ans: true }` (default) | `encode::every_entropy_coder_setting` |
| Hybrid-uint configurations | `optimize_uint` | `encode::every_entropy_coder_setting` |
| LZ77: runs, copies, 2-D special distances | `Lz77Mode::{Rle, Full}` | `encode::every_entropy_coder_setting` |
| Context maps: simple, entropy coded, move-to-front | `clustering`, `max_histograms` | `encode::every_entropy_coder_setting` |

## Image header

| Feature | Encoder | Test |
|---|---|---|
| Size (small, ratio, plain), all-default metadata | `ImageInfo` | `header::*` |
| Integer samples 1–31 bits | `SampleFormat::Int` | `header::integer_bit_depths`, `header::widest_integers` |
| Float samples (any exponent / mantissa split) | `SampleFormat::Float` | `header::float_formats` |
| Extra channels: alpha (straight, premultiplied), depth, spot colour, selection mask, black, CFA, thermal, reserved 0–7, unknown, optional; names, bit depths, `dim_shift` | `ExtraChannel` | `header::extra_channels_of_every_kind`, `header::spot_colours_and_premultiplied_alpha_decode`, `frames::upsampling` |
| Colour encodings: RGB / gray, D65 / E / DCI / custom white, sRGB / BT.2100 / P3 / custom primaries, every transfer function and gamma, every intent | `ColorSpec::Encoding` | `header::colour_encodings` |
| ICC profiles (header prediction, tag list, XYZ and common-type commands) | `ColorSpec::Icc` | `header::icc_profiles` |
| Orientation 1–8, intrinsic size | `ImageInfo::orientation`, `intrinsic_size` | `header::orientation_and_intrinsic_size` |
| Preview frame | `ImageInfo::preview`, `Encoder::set_preview` | `header::preview` |
| Animation (tick rate, loops, timecodes) | `ImageInfo::animation` | `header::animation`, `frames::timecodes_and_save_before_colour_transform` |
| Tone mapping | `ImageInfo::tone_mapping` | `header::tone_mapping` |
| Custom opsin inverse, custom 2x / 4x / 8x upsampling kernels | `opsin_inverse`, `upsampling_weights` | `vardct::custom_opsin_and_upsampling_kernels`, `frames::upsampling` |
| 16-bit modular storage flag | chosen from the samples | every test |

## Frames

| Feature | Encoder | Test |
|---|---|---|
| Regular, LF (levels 1–4), reference-only, skip-progressive frames | `FrameOptions::frame_type` | `frames::reference_only_and_skip_progressive_frames`, `vardct::lf_frames`, `vardct::two_lf_levels` |
| Crops (on and off the canvas) | `FrameOptions::crop` | `frames::crops_and_every_blend_mode` |
| Blending: replace, add, blend, alpha-weighted add, multiply; per extra channel; source slots; clamp | `Blending` | `frames::crops_and_every_blend_mode` |
| Reference slots, save before colour transform | `save_as_reference`, `save_before_ct` | `frames::*`, `features::patches_copy_from_a_saved_frame` |
| Durations, timecodes, names, is-last | `FrameOptions` | `header::animation` |
| Upsampling 2 / 4 / 8, extra-channel upsampling | `upsampling`, `ec_upsampling` | `frames::upsampling`, `vardct::rgb_vardct_crops_upsampling_and_features` |
| YCbCr samples, every chroma subsampling (modular and VarDCT) | `FrameOptions::ycbcr` | `frames::ycbcr_samples`, `vardct::ycbcr_vardct` |
| Progressive passes (shifts, downsampling brackets) | `Passes` | `frames::passes_section_order_and_group_sizes`, `vardct::passes_extra_channels_and_parameters` |
| Group sizes 128–1024 | `group_size_shift` | `frames::passes_section_order_and_group_sizes` |
| Permuted table of contents | `section_order` | `frames::passes_section_order_and_group_sizes` |
| Restoration filters: Gaborish (default or custom weights), EPF (iterations, sharpness LUT, channel scales, sigmas) | `Restoration` | `frames::restoration_filters`, `vardct::passes_extra_channels_and_parameters` |
| Patches (every blend mode) | `Features::patches` | `features::patches_copy_from_a_saved_frame` |
| Splines | `Features::splines` | `features::splines_draw`, `vardct::rgb_vardct_crops_upsampling_and_features` |
| Noise | `Features::noise` | `features::noise_parameters` |
| Adaptive LF smoothing on or off | `skip_adaptive_lf_smoothing` | `vardct::*` |

## Modular

| Feature | Encoder | Test |
|---|---|---|
| All 14 predictors | `TreeMode::Fixed`, learned trees | `modular::every_predictor` |
| Weighted predictor, custom parameters | `WeightedParams` | `modular::custom_weighted_parameters` |
| MA trees on all 16 properties and reference properties; leaf offsets and multipliers | `TreeMode::Learn`, `properties` | `modular::learned_trees_on_every_property`, `lossy_modular::lossy_squeeze` |
| Global and per-stream trees | `local_trees` | `modular::local_trees` |
| RCT, all 42 | `Transform::Rct` | `modular::every_rct` |
| Palette: colours, deltas with any predictor, implicit cube and delta entries, channel palettes | `Transform::Palette` | `modular::palettes`, `modular::delta_palettes` |
| Squeeze, default and explicit | `Transform::Squeeze` | `modular::squeezes` |
| Per-group transforms | `ModularFrame::group_transforms` | `lossy_modular::group_transforms` |
| Global / LF-group / per-pass group sections | automatic | `modular::squeezes`, `frames::passes_section_order_and_group_sizes` |
| XYB modular with LF quantisation factors | `ImageInfo::xyb`, `ModularFrame::lf_quant` | `lossy_modular::xyb_modular` |

## VarDCT

| Feature | Encoder | Test |
|---|---|---|
| All 27 transforms (DCT 2x2 to 256x256, identity, 4x8 / 8x4, AFV 0–3) | `Strategy::{Fixed, Map, Auto}` | `vardct::every_transform` (inverse and forward checked against jxl-rs's transforms in the unit tests) |
| Quantiser: global scale, LF quantiser, per-block quantisers, x / b matrix scales | `VarDctOptions` | `vardct::auto_strategy_and_adaptive_quantization`, `vardct::passes_extra_channels_and_parameters` |
| Dequantisation tables: library, identity, DCT2, DCT4, DCT4x8, AFV, banded DCT, raw (modular coded) | `quant_tables` | `vardct::custom_quant_tables_of_every_encoding` |
| LF: extra precision, custom LF quantisation, chroma-from-luma at the LF | `extra_precision`, `lf_quant`, `ColorCorrelation` | `vardct::passes_extra_channels_and_parameters` |
| Chroma from luma: colour factor, base correlations, per-tile maps | `chroma_from_luma`, `ColorCorrelation` | `vardct::passes_extra_channels_and_parameters` |
| Block context maps: default and custom (LF and quantiser thresholds) | `BlockContextMap` | `vardct::custom_orders_histograms_and_contexts` |
| Coefficient orders: natural and custom | `custom_orders` | `vardct::custom_orders_histograms_and_contexts` |
| Several HF histogram sets | `num_histograms` | `vardct::custom_orders_histograms_and_contexts` |
| EPF sharpness map | `epf_sharpness` | `vardct::passes_extra_channels_and_parameters` |
| XYB and non-XYB (RGB, YCbCr) colour | `ImageInfo::xyb`, `ycbcr` | `vardct::rgb_vardct_crops_upsampling_and_features`, `vardct::ycbcr_vardct` |
| Extra channels alongside VarDCT | `VarDctFrame::extra` | `vardct::passes_extra_channels_and_parameters` |

## Container

| Feature | Encoder | Test |
|---|---|---|
| Signature and `ftyp` (versions 0, 1) | `wrap` | `container::*` |
| `jxlc`; `jxlp` in order and out of order | `Container::split_at`, `jxlp_out_of_order` | `container::layouts_decode_to_the_same_picture` |
| Level box | `Container::level` | `container::layouts_decode_to_the_same_picture` |
| Metadata boxes (Exif, XMP, JUMBF, any type), before or after the codestream | `MetadataBox` | `container::layouts_decode_to_the_same_picture` |
| 64-bit and open-ended box sizes | `large_sizes`, `open_last_box` | `container::layouts_decode_to_the_same_picture` |
| `brob` (Brotli-wrapped) boxes | `MetadataBox::brotli` | `container::brotli_wrapped_boxes` (`--features brotli`) |

## Not applicable

jxl-rs does not reconstruct JPEG files from JPEG XL, so the `jbrd` box is
only a box to it; `MetadataBox` writes one if asked.

The encoder also writes the XYB and unknown colour spaces and the unknown
transfer function by name (`ColorSpace::{Xyb, Unknown}`,
`TransferFunction::Unknown`), which the format allows; jxl-rs reads them
only alongside an ICC profile, as it has no output profile for them.
