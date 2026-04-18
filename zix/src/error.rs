use core::fmt;
use std::borrow::Cow;
use std::ops::Range;

use crate::dtype::Dtype;
use crate::NDIM_MAX;

pub struct Error(Box<ErrorRepr>);
struct ErrorRepr {
    kind: ErrorKind,
    msg: Cow<'static, str>,
}
impl Error {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        let msg = message.into();
        Self(Box::new(ErrorRepr { kind, msg }))
    }
}
impl Error {
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    fn message(&self) -> &str {
        &self.0.msg
    }

    pub(crate) fn io(err: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, format!("IO error: {err}"))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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
pub type Result<T, E = Error> = std::result::Result<T, E>;

macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err(crate::error::Error::new(crate::error::ErrorKind::$kind, format!($($arg)*)));
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

pub(crate) fn check_ndim(ndim: usize) -> Result<()> {
    ensure!(
        ndim <= NDIM_MAX,
        TooManyDimensions,
        "Too many dimensions: {ndim} (max={NDIM_MAX})"
    );
    Ok(())
}

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
            range.start < dim_size && range.end <= dim_size,
            InvalidIndex,
            "Index range {range:?} out of bounds for shape {shape:?} at dimension {dim}"
        );
    }
    Ok(())
}

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
        "Buffer size {buf_len} is too small for requested index {index:?} with dtype {dtype:?} (required size: {required_size})"
    );
    ensure!(
        (buf.as_ptr() as usize) % dtype.alignment() as usize == 0,
        InvalidArgument,
        "Buffer pointer is not aligned to required alignment {} for dtype {:?}",
        dtype.alignment(),
        dtype
    );
    Ok(nitems)
}
