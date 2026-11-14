//! Error type for the `jix` crate.
//!
//! [`Error`] is the single error type returned by all fallible operations in this crate.
//! It pairs an [`ErrorKind`] for programmatic branching with a human-readable message for
//! diagnostics and logging.

use core::fmt;
use std::borrow::Cow;
use std::ops::Range;

use crate::dtype::Dtype;
use crate::storage::Index;
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
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}
impl Error {
    /// Create a new [`Error`] with the given [`ErrorKind`] and a human-readable `message`.
    pub fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        let msg = message.into();
        Self(Box::new(ErrorRepr {
            kind,
            msg,
            source: None,
        }))
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
        let mut e = Self::new(ErrorKind::Io, format!("IO error: {err}"));
        e.0.source = Some(Box::new(err));
        e
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
    /// Shape operation (e.g. reshape, permute_dims, concat) is invalid for the given shape and arguments
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
    /// An integer element-wise operation hit a result it can't represent (e.g. division by zero,
    /// an integer raised to a negative integer power, etc).
    Arithmetic,
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
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source.as_deref().map(|e| e as _)
    }
}
pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

macro_rules! error {
    ($kind:ident, $($arg:tt)*) => {
        crate::Error::new(crate::error::ErrorKind::$kind, format!($($arg)*))
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

/// Check that `ndim` does not exceed [`NDIM_MAX`].
#[inline]
pub(crate) fn check_ndim(ndim: usize) -> Result<()> {
    #[cold]
    #[inline(never)]
    fn check_ndim_fail(ndim: usize) -> Result<()> {
        bail!(
            TooManyDimensions,
            "Too many dimensions: {ndim} (max={NDIM_MAX})"
        );
    }
    if ndim > NDIM_MAX {
        return check_ndim_fail(ndim);
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
    let product = shape.iter().cloned().chain([itemsize]).try_shape_product();
    if product.is_none() || shape.iter().any(|&d| d > i64::MAX as u64) {
        #[cold]
        #[inline(never)]
        fn shape_overflow_fail(shape: &[u64], itemsize: u64) -> Result<()> {
            bail!(
                InvalidShapeOperation,
                "shape {shape:?} with itemsize {itemsize} overflows: dims must be at most \
                 i64::MAX and the byte size must fit in u64"
            );
        }
        return shape_overflow_fail(shape, itemsize);
    }
    Ok(())
}

#[cold]
#[inline(never)]
fn get_range_fail_ndim(shape_ndim: usize, index_ndim: usize) -> Result<()> {
    bail!(
        InvalidIndex,
        "Index has different number of dimensions {index_ndim} than shape {shape_ndim}"
    )
}

#[cold]
#[inline(never)]
fn check_range_fail(dim: usize, range: Range<u64>, dim_size: u64) -> Result<()> {
    bail!(
        InvalidIndex,
        "Index range {range:?} out of bounds for shape dimension {dim} with size {dim_size}"
    )
}

/// Checks `ranges`, read from an array of `shape`, before splitting them into an [`Index`]: one
/// range per dimension, within it and not starting after it ends, of a size that fits in `usize`.
#[inline]
pub(crate) fn check_ranges(shape: &[u64], ranges: &[Range<u64>]) -> Result<()> {
    check_ndim(ranges.len())?;
    if shape.len() != ranges.len() {
        return get_range_fail_ndim(shape.len(), ranges.len());
    }
    for (dim, (range, &dim_size)) in ranges.iter().zip(shape).enumerate() {
        if range.start > range.end || range.end > dim_size {
            return check_range_fail(dim, range.clone(), dim_size);
        }
    }
    ensure!(
        ranges
            .iter()
            .all(|r| usize::try_from(r.end - r.start).is_ok()),
        InvalidShapeOperation,
        "read region {ranges:?} does not fit in memory"
    );
    Ok(())
}

#[inline]
pub(crate) fn check_get_range(shape: &[u64], index: &Index<'_>) -> Result<()> {
    if shape.len() != index.ndim() {
        return get_range_fail_ndim(shape.len(), index.ndim());
    }
    for (dim, &dim_size) in shape.iter().enumerate() {
        if index.end(dim) > dim_size {
            return check_range_fail(dim, index.start()[dim]..index.end(dim), dim_size);
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn check_get_range_ndim(shape: &[u64], index: &Index<'_>) -> Result<()> {
    if shape.len() != index.ndim() {
        return get_range_fail_ndim(shape.len(), index.ndim());
    }
    Ok(())
}

#[inline]
pub(crate) fn check_get_buffer_size(
    index: &Index<'_>,
    dtype: &Dtype,
    buf: &mut [u8],
) -> Result<()> {
    let required_size = index
        .size()
        .iter()
        .map(|&n| n as u64)
        .chain([dtype.itemsize() as u64])
        .try_shape_product();
    let buf_len = buf.len();

    if Some(buf_len as u64) != required_size {
        #[cold]
        #[inline(never)]
        fn buffer_size_fail(
            buf_len: usize,
            required_size: u64,
            index: &Index<'_>,
            dtype: &Dtype,
        ) -> Result<()> {
            bail!(
                InvalidBufferSize,
                "Unexpected buffer size {buf_len} requested index {index:?} with dtype {dtype} (required size: {required_size})"
            );
        }
        return buffer_size_fail(buf_len, required_size.unwrap_or(u64::MAX), index, dtype);
    }
    Ok(())
}

#[inline]
pub(crate) fn check_region_size(index: &Index<'_>, itemsize: usize) -> Result<()> {
    if cfg!(target_pointer_width = "32") {
        let bytes = index
            .size()
            .iter()
            .map(|&n| n.max(1) as u64)
            .chain([itemsize as u64])
            .try_shape_product();
        ensure!(
            bytes.is_some_and(|b| b <= isize::MAX as u64),
            InvalidShapeOperation,
            "read region {index:?} with itemsize {itemsize} does not fit in memory"
        );
    }
    Ok(())
}

/// An arithmetic error an integer element-wise operation can hit while computing its output: a
/// result with no integer value. It always fails the read, with [`ErrorKind::Arithmetic`].
///
/// Float operations report no errors: their `inf` and `NaN` results carry them.
/// Integer overflow wraps (two's complement), also `MIN / -1`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
#[repr(u8)]
pub(crate) enum ArithmeticErrorKind {
    /// Integer division or remainder by zero.
    DivideByZero = 1,
    /// An integer base raised to a negative integer exponent.
    NegativePower = 2,
}

/// The arithmetic errors an element-wise computation hit.
///
/// Scalar kernels that can hit an error, like [`scalar::Div`](crate::scalar::Div), take a
/// `&mut ArithmeticStatus`: rather than panicking, they produce a fallback value for the element
/// and set the status to error. After the computation, [`into_result`](Self::into_result) turns
/// the status to `Ok` or an error with [`ErrorKind::Arithmetic`].
///
/// ```
/// use jix::scalar::Div;
/// use jix::error::{ArithmeticStatus, ErrorKind};
///
/// let mut status = ArithmeticStatus::default();
/// assert_eq!(7i32.div(2, &mut status), 3);
/// assert!(status.into_result().is_ok());
/// assert_eq!(7i32.div(0, &mut status), 0);
/// assert_eq!(status.into_result().unwrap_err().kind(), ErrorKind::Arithmetic);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArithmeticStatus(u8);
impl ArithmeticStatus {
    /// Create a new `ArithmeticStatus` with no errors set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `error` if `cond` holds.
    #[inline(always)]
    pub(crate) fn set_if(&mut self, error: ArithmeticErrorKind, cond: bool) {
        self.0 |= error as u8 * cond as u8;
    }

    /// Whether `error` is set.
    #[inline(always)]
    pub(crate) fn contains(self, error: ArithmeticErrorKind) -> bool {
        (self.0 & error as u8) != 0
    }

    /// Whether no error is set.
    #[inline(always)]
    pub(crate) fn is_ok(self) -> bool {
        self.0 == 0
    }

    /// `Ok` if no error is set, else an [`ErrorKind::Arithmetic`] error describing the first one.
    #[inline]
    pub fn into_result(self) -> Result<()> {
        #[cold]
        #[inline(never)]
        fn fail(status: ArithmeticStatus) -> Result<()> {
            let message = if status.contains(ArithmeticErrorKind::DivideByZero) {
                "integer division or modulo by zero"
            } else if status.contains(ArithmeticErrorKind::NegativePower) {
                "integers to negative integer powers are not allowed"
            } else {
                "unknown arithmetic error"
            };
            Err(Error::new(ErrorKind::Arithmetic, message))
        }
        if !self.is_ok() {
            return fail(self);
        }
        Ok(())
    }
}
impl std::ops::BitOrAssign for ArithmeticStatus {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl From<ArithmeticErrorKind> for ArithmeticStatus {
    #[inline(always)]
    fn from(error: ArithmeticErrorKind) -> Self {
        Self(error as u8)
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use crate::dtype::Dtyped;
    use crate::{Dim, DimDyn};

    #[test]
    fn check_ndim_rejects_too_many() {
        assert!(check_ndim(0).is_ok());
        assert!(check_ndim(NDIM_MAX).is_ok());
        let too_many = check_ndim(NDIM_MAX + 1).unwrap_err();
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
        assert!(check_shape_overflow(&[i64::MAX as u64], 1).is_ok());
        let err = check_shape_overflow(&[i64::MAX as u64 + 1], 1).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidShapeOperation);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn check_get_range_validates_ndim_and_bounds() {
        assert!(check_ranges(&[2, 3], &[0..2, 0..3]).is_ok());
        for bad in [&[0..2][..], &[0..2, 0..4], &[2..1, 0..3]] {
            let kind = check_ranges(&[2, 3], bad).unwrap_err().kind();
            assert_eq!(kind, ErrorKind::InvalidIndex, "{bad:?}");
        }

        let check = |start, size| check_get_range(&[2, 3], &Index::new(start, size).unwrap());
        assert!(check(&[0, 0], &[2, 3]).is_ok());
        for (start, size) in [(&[0][..], &[2][..]), (&[0, 0], &[2, 4]), (&[0, 1], &[2, 3])] {
            assert_eq!(
                check(start, size).unwrap_err().kind(),
                ErrorKind::InvalidIndex,
                "{start:?} {size:?}"
            );
        }
    }

    #[test]
    #[cfg(target_pointer_width = "32")]
    fn read_region_must_fit_in_memory_on_32_bit() {
        let index = |size| Index::new(&[0, 0][..size.len()], size).unwrap();
        assert!(check_region_size(index(&[1 << 30, 1]), 1).is_ok());
        assert!(check_region_size(index(&[1 << 30, 2]), 1).is_err());
        assert!(check_ranges(&[1 << 33], &[0..1 << 32]).is_err());
    }

    #[test]
    fn check_get_buffer_size_requires_the_exact_size() {
        let dtype = i32::DTYPE;
        let index = Index::new(&[0, 0], &[2, 3]).unwrap();
        let mut exact = [0u8; 24];
        assert!(check_get_buffer_size(&index, &dtype, &mut exact).is_ok());
        let mut short = [0u8; 8];
        let err = check_get_buffer_size(&index, &dtype, &mut short).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidBufferSize);
    }
}
