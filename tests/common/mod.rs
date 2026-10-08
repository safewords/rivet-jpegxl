//! Decoding with jxl-rs directly, every channel as f32, for checking what
//! the encoder wrote.

#![allow(dead_code)]

use jpegxl::jxl::api::{
    Endianness, JxlBasicInfo, JxlColorProfile, JxlColorType, JxlDataFormat, JxlDecoder,
    JxlDecoderOptions, JxlFrameHeader, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};

pub struct Decoded {
    pub basic: JxlBasicInfo,
    pub embedded: JxlColorProfile,
    pub frames: Vec<DecodedFrame>,
}

pub struct DecodedFrame {
    pub header: JxlFrameHeader,
    /// Interleaved colour samples.
    pub color: Vec<f32>,
    pub extra: Vec<Vec<f32>>,
}

pub struct RawOptions {
    pub coalescing: bool,
    pub skip_preview: bool,
    pub adjust_orientation: bool,
}

impl Default for RawOptions {
    fn default() -> Self {
        RawOptions {
            coalescing: true,
            skip_preview: true,
            adjust_orientation: false,
        }
    }
}

fn f32_bytes(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: f32 has no invalid bit patterns and u8 no alignment needs.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4) }
}

pub fn decode_raw(data: &[u8], o: &RawOptions) -> Result<Decoded, String> {
    let mut options = JxlDecoderOptions::default();
    options.coalescing = o.coalescing;
    options.skip_preview = o.skip_preview;
    options.adjust_orientation = o.adjust_orientation;
    let mut input = data;
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder: JxlDecoder<states::WithImageInfo> = loop {
        match decoder
            .process(&mut input, None)
            .map_err(|e| e.to_string())?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err("truncated header".into());
                }
                decoder = fallback;
            }
        }
    };
    let basic = decoder.basic_info().clone();
    let embedded = decoder.embedded_color_profile().clone();
    let gray = match &embedded {
        JxlColorProfile::Simple(e) => format!("{e:?}").starts_with("Grayscale"),
        JxlColorProfile::Icc(p) => p.len() > 20 && &p[16..20] == b"GRAY",
    };
    let color_type = if gray {
        JxlColorType::Grayscale
    } else {
        JxlColorType::Rgb
    };
    let nc = if gray { 1 } else { 3 };
    let ne = basic.extra_channels.len();
    let f32_format = JxlDataFormat::F32 {
        endianness: Endianness::native(),
    };
    decoder
        .set_pixel_format(JxlPixelFormat {
            color_type,
            color_data_format: Some(f32_format.clone()),
            extra_channel_format: vec![Some(f32_format); ne],
        })
        .map_err(|e| e.to_string())?;

    let mut frames = Vec::new();
    loop {
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|e| e.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("truncated frame header".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let mut header = frame.frame_header();
        // jxl-rs reports the image's size for the preview frame.
        if frames.is_empty()
            && !o.skip_preview
            && let Some(p) = basic.preview_size
        {
            header.size = p;
        }
        let (w, h) = header.size;
        let mut color = vec![0f32; w * h * nc];
        let mut extra = vec![vec![0f32; w * h]; ne];
        let next = {
            let mut buffers = vec![JxlOutputBuffer::new(f32_bytes(&mut color), h, w * nc * 4)];
            for e in extra.iter_mut() {
                buffers.push(JxlOutputBuffer::new(f32_bytes(e), h, w * 4));
            }
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|e| e.to_string())?
                {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("truncated frame".into());
                        }
                        frame = fallback;
                    }
                }
            }
        };
        frames.push(DecodedFrame {
            header,
            color,
            extra,
        });
        decoder = next;
        if !decoder.has_more_frames() {
            break;
        }
    }
    Ok(Decoded {
        basic,
        embedded,
        frames,
    })
}

/// Integer samples of `bits` as the decoder's floats give them back.
pub fn to_int(v: f32, bits: u32) -> i64 {
    (f64::from(v) * ((1u64 << bits) - 1) as f64).round() as i64
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// A smooth, edged, noisy plane of values up to `max`.
pub fn plane(w: usize, h: usize, max: u32, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as u32, (i / w) as u32);
            let smooth = (x * 3 + y * 5 + seed as u32 * 7) % (max + 1);
            let edge = if (x / 7 + y / 5) % 2 == 0 { max / 3 } else { 0 };
            ((smooth + edge + (rng.next() % 5) as u32).min(max)) as i32
        })
        .collect()
}
