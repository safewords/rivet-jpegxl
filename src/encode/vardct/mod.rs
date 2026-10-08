//! VarDCT: lossy coding of the colour channels in variable-size DCT blocks.

pub(crate) mod coeffs;
pub(crate) mod encode;
mod library;
pub(crate) mod quant;
pub(crate) mod transform;
