//! What can go wrong.

use std::fmt;

/// A JPEG XL decoding or encoding error.
#[derive(Debug)]
pub enum Error {
    /// The data is not a JPEG XL file: neither the bare codestream signature
    /// (`FF 0A`) nor the container's (`JXL ` box).
    NotJpegXl,
    /// The data breaks the JPEG XL syntax, as jxl-rs reported it.
    Bitstream(jxl::error::Error),
    /// The file ends before the picture does.
    Truncated,
    /// A valid file this crate does not decode, named.
    Unsupported(String),
    /// The picture is larger than the [`Limits`](crate::Limits) allow.
    LimitExceeded(String),
    /// A picture the encoder was given does not add up (its size, its
    /// sample count).
    InvalidInput(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotJpegXl => {
                f.write_str("not a JPEG XL file (no codestream or container signature)")
            }
            Error::Bitstream(e) => write!(f, "invalid JPEG XL data: {e}"),
            Error::Truncated => f.write_str("the JPEG XL file ends before its picture does"),
            Error::Unsupported(m) => write!(f, "unsupported JPEG XL feature: {m}"),
            Error::LimitExceeded(m) => write!(f, "JPEG XL picture over the decoder's limits: {m}"),
            Error::InvalidInput(m) => write!(f, "cannot encode the picture: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Bitstream(e) => Some(e),
            _ => None,
        }
    }
}

impl From<jxl::error::Error> for Error {
    fn from(e: jxl::error::Error) -> Self {
        match e {
            // jxl-rs's own size check, run from the limits passed down.
            jxl::error::Error::ImageSizeTooLarge(w, h) => {
                Error::LimitExceeded(format!("a {w}x{h} picture"))
            }
            e => Error::Bitstream(e),
        }
    }
}

/// `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
