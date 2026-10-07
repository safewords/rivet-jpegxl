//! The codestream's headers: the image header (size, metadata, colour
//! encoding) and a frame's header and table of contents.

use super::bits::{BitWriter, Dist};

/// A U32 field coding a dimension less one.
const DIMENSION: [Dist; 4] = [
    Dist::Bits(9, 0),
    Dist::Bits(13, 0),
    Dist::Bits(18, 0),
    Dist::Bits(30, 0),
];
/// `bits_per_sample` of an integer image.
const BITS_PER_SAMPLE: [Dist; 4] = [Dist::Val(8), Dist::Val(10), Dist::Val(12), Dist::Bits(6, 1)];
/// The number of extra channels.
const EXTRA_CHANNELS: [Dist; 4] = [
    Dist::Val(0),
    Dist::Val(1),
    Dist::Bits(4, 2),
    Dist::Bits(12, 1),
];
/// A TOC entry: a section's size in bytes.
const SECTION_SIZE: [Dist; 4] = [
    Dist::Bits(10, 0),
    Dist::Bits(14, 1024),
    Dist::Bits(22, 17408),
    Dist::Bits(30, 4211712),
];
/// `upsampling`, `num_passes`, a blending mode: the first option, 1 or 0.
const FIRST: u32 = 0;

/// What the image header says about the picture.
pub(super) struct ImageHeader {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u32,
    pub gray: bool,
    pub alpha: bool,
}

impl ImageHeader {
    /// The signature, `SizeHeader`, `ImageMetadata` and the default
    /// `CustomTransformData`, padded to a byte as the first frame needs.
    pub(super) fn write(&self, w: &mut BitWriter) {
        w.write(8, 0xff);
        w.write(8, 0x0a);

        // SizeHeader: not `small`, height, no aspect ratio, width.
        w.bit(false);
        w.u32(self.height - 1, DIMENSION);
        w.write(3, 0);
        w.u32(self.width - 1, DIMENSION);

        // ImageMetadata.
        w.bit(false); // all_default
        w.bit(false); // extra_fields: upright, no preview or animation
        self.write_bit_depth(w);
        w.bit(self.bits_per_sample <= 12); // modular_16bit_sufficient
        w.u32(u32::from(self.alpha), EXTRA_CHANNELS);
        if self.alpha {
            self.write_alpha_info(w);
        }
        w.bit(false); // xyb_encoded: the samples are coded as they are
        self.write_color_encoding(w);
        w.u64_zero(); // extensions

        w.bit(true); // CustomTransformData: all_default
        w.pad_to_byte();
    }

    fn write_bit_depth(&self, w: &mut BitWriter) {
        w.bit(false); // float
        w.u32(self.bits_per_sample, BITS_PER_SAMPLE);
    }

    /// Straight alpha at the image's depth.
    fn write_alpha_info(&self, w: &mut BitWriter) {
        if self.bits_per_sample == 8 {
            w.bit(true); // all_default: 8-bit straight alpha, unnamed
            return;
        }
        w.bit(false); // all_default
        w.enumeration(0); // Alpha
        self.write_bit_depth(w);
        w.write(2, 0); // dim_shift 0
        w.empty_string(); // name
        w.bit(false); // alpha_associated
    }

    /// sRGB, or gray with the sRGB transfer curve and D65 white.
    fn write_color_encoding(&self, w: &mut BitWriter) {
        if !self.gray {
            w.bit(true); // all_default: sRGB
            return;
        }
        w.bit(false); // all_default
        w.bit(false); // want_icc
        w.enumeration(1); // colour space: gray
        w.enumeration(1); // white point: D65
        w.bit(false); // have_gamma
        w.enumeration(13); // transfer function: sRGB
        w.enumeration(1); // rendering intent: relative
    }
}

/// A lone, full-canvas, modular frame: no transforms of its own, no
/// restoration filters, `group_size_shift` 1 (256-pixel groups).
pub(super) fn write_frame_header(w: &mut BitWriter, extra_channels: u32) {
    w.bit(false); // all_default
    w.write(2, 0); // frame_type: regular
    w.write(1, 1); // encoding: modular
    w.u64_zero(); // flags
    w.bit(false); // do_ycbcr
    w.write(2, FIRST); // upsampling 1
    for _ in 0..extra_channels {
        w.write(2, FIRST); // ec_upsampling 1
    }
    w.write(2, GROUP_SIZE_SHIFT);
    w.write(2, FIRST); // num_passes 1
    w.bit(false); // have_crop
    w.write(2, FIRST); // blending: replace (and covering the canvas, no source)
    for _ in 0..extra_channels {
        w.write(2, FIRST);
    }
    w.bit(true); // is_last
    w.empty_string(); // name

    // RestorationFilter: neither Gaborish nor the edge-preserving filter,
    // which would change the lossless samples.
    w.bit(false); // all_default
    w.bit(false); // gab
    w.write(2, 0); // epf_iters
    w.u64_zero(); // extensions

    w.u64_zero(); // extensions
}

/// `group_size_shift`: groups of `128 << 1` pixels.
const GROUP_SIZE_SHIFT: u32 = 1;
/// A group's side.
pub(super) const GROUP_DIM: u32 = 128 << GROUP_SIZE_SHIFT;

/// The table of contents: each section's size, in order, then padding.
pub(super) fn write_toc(w: &mut BitWriter, section_sizes: &[usize]) {
    w.bit(false); // not permuted
    w.pad_to_byte();
    for &size in section_sizes {
        w.u32(
            u32::try_from(size).expect("a section over 1 GiB"),
            SECTION_SIZE,
        );
    }
    w.pad_to_byte();
}
