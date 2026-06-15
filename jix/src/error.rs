//! Error type for the `jix` crate.
//!
//! [`Error`] is the single error type returned by all fallible operations in this crate.
//! It pairs an [`ErrorKind`] for programmatic branching with a human-readable message for
//! diagnostics and logging.

use core::fmt;
use std::borrow::Cow;
use std::ops::Range;

use crate::dtype::Dtype;
use crate::{Dimension, NDIM_MAX};

/// Error type for all operations in this crate.
///
/// An error consists of an [`ErrorKind`] for programmatic matching and a human-readable
/// message for diagnostics. Use [`Error::kind`] to branch on the failure category and
/// [`Error::message`] (or the [`Display`](fmt::Display) impl) for a descriptive string.
pub struct Error(Box<ErrorRepr>);
struct ErrorRepr {
    kind: ErrorKind,
    msg: Cow<'static, str>,
}
impl Error {
    /// Create a new [`Error`] with the given [`ErrorKind`] and a human-readable `message`.
    pub fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        let msg = message.into();
        Self(Box::new(ErrorRepr { kind, msg }))
    }
}
impl Error {
    /// Get the error kind.
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// Get the error message.
    pub fn message(&self) -> &str {
        &self.0.msg
    }

    pub(crate) fn io(err: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, format!("IO error: {err}"))
    }
}

/// Categorises errors returned by this crate.
///
/// Match on this to distinguish failure modes programmatically; the human-readable
/// description is available via [`Error::message`] or the [`Display`](fmt::Display) impl.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Index out of bounds, slice start > end, nd-index with wrong number of dimensions, etc
    InvalidIndex,
    /// Buffer size is too small for the requested operation
    InvalidBufferSize,
    /// Number of dimensions exceeds max, see [`NDIM_MAX`]
    TooManyDimensions,
    /// Shape operation (e.g. reshape, permute_axes, concat) is invalid for the given shape and arguments
    InvalidShapeOperation,
    /// Unsupported dtype by operation, or incorrect dtype when accessing array data, etc
    UnsupportedDtype,
    /// Invalid argument. Generic error.
    InvalidArgument,

    /// I/O error while reading/writing archive file, or invalid file path, etc
    Io,
    /// Archive file (serialized array) is invalid or corrupted,
    /// (e.g. missing required metadata, invalid metadata values, invalid layout, etc)
    InvalidArchive,
    /// Compression/decompression failure, or invalid compression params/metadata
    CodecError,
}

impl fmt::Debug for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Error")
            .field("kind", &self.kind())
            .field("message", &self.message())
            .finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message().fmt(fmt)
    }
}
impl std::error::Error for Error {}
pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err(crate::Error::new(crate::ErrorKind::$kind, format!($($arg)*)))
    };
}
macro_rules! ensure {
    ($cond:expr, $kind:ident, $($arg:tt)*) => {
        if !$cond {
            crate::error::bail!($kind, $($arg)*);
        }
    };
}
pub(crate) use {bail, ensure};

#[inline(always)]
pub(crate) fn check_ndim<D: Dimension>(ndim: usize) -> Result<()> {
    if let Some(expected) = D::NDIM {
        ensure!(
            ndim == expected,
            TooManyDimensions,
            "ndim {ndim} does not match expected {expected} for this dimension type"
        );
    } else {
        ensure!(
            ndim <= NDIM_MAX,
            TooManyDimensions,
            "Too many dimensions: {ndim} (max={NDIM_MAX})"
        );
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn check_dtype(actual: &Dtype, expected: &Dtype) -> Result<()> {
    ensure!(
        actual == expected,
        UnsupportedDtype,
        "expected dtype {expected} but got {actual}",
    );
    Ok(())
}

#[inline(always)]
pub(crate) fn check_get_range(shape: &[u64], index: &[Range<u64>]) -> Result<()> {
    ensure!(
        shape.len() == index.len(),
        InvalidIndex,
        "Index has different number of dimensions {} than shape {}",
        index.len(),
        shape.len()
    );
    for (dim, (&dim_size, range)) in shape.iter().zip(index).enumerate() {
        ensure!(
            range.start <= range.end,
            InvalidIndex,
            "Index range {range:?} has start greater than end at dimension {dim}"
        );
        ensure!(
            range.end <= dim_size,
            InvalidIndex,
            "Index range {range:?} out of bounds for shape {shape:?} at dimension {dim}"
        );
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn check_get_buffer_size(
    index: &[Range<u64>],
    dtype: &Dtype,
    buf: &mut [u8],
) -> Result<usize> {
    let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
    let required_size = nitems * dtype.itemsize() as usize;
    let buf_len = buf.len();
    ensure!(
        buf_len == required_size,
        InvalidBufferSize,
        "Unexpected buffer size {buf_len} requested index {index:?} with dtype {dtype} (required size: {required_size})"
    );
    ensure!(
        (buf.as_ptr() as usize).is_multiple_of(dtype.alignment().as_usize()),
        InvalidArgument,
        "Buffer pointer is not aligned to required alignment {} for dtype {dtype}",
        dtype.alignment(),
    );
    Ok(nitems)
}
