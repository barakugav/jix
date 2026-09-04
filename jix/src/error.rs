//! Error type for the `jix` crate.
//!
//! [`Error`] is the single error type returned by all fallible operations in this crate.
//! It pairs an [`ErrorKind`] for programmatic branching with a human-readable message for
//! diagnostics and logging.

use core::fmt;
use std::borrow::Cow;
use std::ops::Range;

use crate::dtype::Dtype;
use crate::{Dimension, IterExt, NDIM_MAX};

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

/// Categories errors returned by this crate.
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

macro_rules! error {
    ($kind:ident, $($arg:tt)*) => {
        crate::Error::new(crate::ErrorKind::$kind, format!($($arg)*))
    };
}
macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err(crate::error::error!($kind, $($arg)*))
    };
}
macro_rules! ensure {
    ($cond:expr, $kind:ident, $($arg:tt)*) => {
        if !$cond {
            crate::error::bail!($kind, $($arg)*);
        }
    };
}
pub(crate) use {bail, ensure, error};

#[inline]
pub(crate) fn check_ndim<D: Dimension>(ndim: usize) -> Result<()> {
    #[cold]
    #[inline(never)]
    fn check_dim_fail<D: Dimension>(ndim: usize) -> Result<()> {
        if let Some(expected) = D::NDIM {
            bail!(
                TooManyDimensions,
                "ndim {ndim} does not match expected {expected} for this dimension type"
            );
        } else {
            bail!(
                TooManyDimensions,
                "Too many dimensions: {ndim} (max={NDIM_MAX})"
            );
        }
    }
    if let Some(expected) = D::NDIM {
        if ndim != expected {
            return check_dim_fail::<D>(ndim);
        }
    } else if ndim > NDIM_MAX {
        return check_dim_fail::<D>(ndim);
    }
    Ok(())
}

#[track_caller]
#[inline]
pub(crate) fn assert_dim<D: Dimension>(ndim: usize) {
    #[track_caller]
    #[cold]
    #[inline(never)]
    fn assert_dim_fail<D: Dimension>(ndim: usize) -> ! {
        if let Some(expected) = D::NDIM {
            panic!("ndim {ndim} does not match expected {expected} for this dimension type");
        } else {
            panic!("Too many dimensions: {ndim} (max={NDIM_MAX})");
        }
    }
    if let Some(expected) = D::NDIM {
        if ndim != expected {
            assert_dim_fail::<D>(ndim);
        }
    } else if ndim > NDIM_MAX {
        assert_dim_fail::<D>(ndim);
    }
}

#[inline(always)]
pub(crate) fn check_dtype(actual: &Dtype, expected: &Dtype) -> Result<()> {
    if actual != expected {
        #[cold]
        #[inline(never)]
        fn fail_check_dtype(actual: &Dtype, expected: &Dtype) -> Result<()> {
            bail!(
                UnsupportedDtype,
                "expected dtype {expected} but got {actual}"
            );
        }
        return fail_check_dtype(actual, expected);
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn check_dtype_size_nonzero(dtype: &Dtype) -> Result<()> {
    if dtype.itemsize() == 0 {
        #[cold]
        #[inline(never)]
        fn fail_check_dtype_size_nonzero(dtype: &Dtype) -> Result<()> {
            bail!(
                UnsupportedDtype,
                "arrays of a zero-sized dtype are not supported: {dtype}"
            );
        }
        return fail_check_dtype_size_nonzero(dtype);
    }
    Ok(())
}

#[inline]
pub(crate) fn check_shape_overflow(shape: &[u64], itemsize: u64) -> Result<()> {
    let product = shape.iter().cloned().chain([itemsize]).try_product();
    if product.is_none() {
        #[cold]
        #[inline(never)]
        fn shape_overflow_fail(shape: &[u64], itemsize: u64) -> Result<()> {
            bail!(
                InvalidShapeOperation,
                "shape has overflowed u64 product with itemsize {itemsize}: {shape:?}"
            );
        }
        return shape_overflow_fail(shape, itemsize);
    }
    Ok(())
}

#[inline]
pub(crate) fn check_get_range(shape: &[u64], index: &[Range<u64>]) -> Result<()> {
    if shape.len() != index.len() {
        #[cold]
        #[inline(never)]
        fn get_range_fail_ndim(shape_ndim: usize, index_ndim: usize) -> Result<()> {
            bail!(
                InvalidIndex,
                "Index has different number of dimensions {index_ndim} than shape {shape_ndim}"
            )
        }
        return get_range_fail_ndim(shape.len(), index.len());
    }
    for (dim, (&dim_size, range)) in shape.iter().zip(index).enumerate() {
        #[cold]
        #[inline(never)]
        fn check_range_fail(dim: usize, range: &Range<u64>, dim_size: u64) -> Result<()> {
            bail!(
                InvalidIndex,
                "Index range {range:?} out of bounds for shape dimension {dim} with size {dim_size}"
            )
        }
        if range.start > range.end || range.end > dim_size {
            return check_range_fail(dim, range, dim_size);
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn check_get_buffer_size(
    index: &[Range<u64>],
    dtype: &Dtype,
    buf: &mut [u8],
) -> Result<usize> {
    let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
    let required_size = nitems * dtype.itemsize() as usize;
    let buf_len = buf.len();

    if buf_len != required_size {
        #[cold]
        #[inline(never)]
        fn buffer_size_fail(
            buf_len: usize,
            required_size: usize,
            index: &[Range<u64>],
            dtype: &Dtype,
        ) -> Result<usize> {
            bail!(
                InvalidBufferSize,
                "Unexpected buffer size {buf_len} requested index {index:?} with dtype {dtype} (required size: {required_size})"
            );
        }
        return buffer_size_fail(buf_len, required_size, index, dtype);
    }
    Ok(nitems)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtyped;
    use crate::{Dim, DimDyn};

    #[test]
    fn check_ndim_rejects_mismatch() {
        assert!(check_ndim::<Dim<2>>(2).is_ok());
        assert!(check_ndim::<DimDyn>(NDIM_MAX).is_ok());
        let static_mismatch = check_ndim::<Dim<2>>(3).unwrap_err();
        assert_eq!(static_mismatch.kind(), ErrorKind::TooManyDimensions);
        let too_many = check_ndim::<DimDyn>(NDIM_MAX + 1).unwrap_err();
        assert_eq!(too_many.kind(), ErrorKind::TooManyDimensions);
    }

    #[test]
    #[should_panic(expected = "does not match expected")]
    fn assert_dim_rejects_static_mismatch() {
        assert_dim::<Dim<2>>(3);
    }

    #[test]
    #[should_panic(expected = "Too many dimensions")]
    fn assert_dim_rejects_too_many() {
        assert_dim::<DimDyn>(NDIM_MAX + 1);
    }

    #[test]
    fn check_shape_overflow_catches_u64_product() {
        assert!(check_shape_overflow(&[1000, 1000], 4).is_ok());
        let err = check_shape_overflow(&[u64::MAX, 2], 4).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidShapeOperation);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn check_get_range_validates_ndim_and_bounds() {
        assert!(check_get_range(&[2, 3], &[0..2, 0..3]).is_ok());
        for bad in [&[0..2][..], &[0..2, 0..4], &[2..0, 0..3]] {
            assert_eq!(
                check_get_range(&[2, 3], bad).unwrap_err().kind(),
                ErrorKind::InvalidIndex,
                "{bad:?}"
            );
        }
    }

    #[test]
    fn check_get_buffer_size_requires_the_exact_size() {
        let dtype = i32::DTYPE;
        let index = [0..2, 0..3];
        let mut exact = [0u8; 24];
        assert_eq!(
            check_get_buffer_size(&index, &dtype, &mut exact).unwrap(),
            6
        );
        let mut short = [0u8; 8];
        let err = check_get_buffer_size(&index, &dtype, &mut short).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidBufferSize);
    }
}
