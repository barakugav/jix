use jix_core::codec::{Codec, EncoderParams};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::util::{IntoPyResult, OrKwargs};

/// Parameters controlling the block layout and codec configuration of an array.
///
/// `ArrayParams` groups two independent sets of configuration:
///
/// - **Block layout** - the nd-block shape used to divide the array into independently
///   compressed blocks. A good block layout is critical for performance and should match the
///   access pattern of your workload.
/// - **Codec** - compression settings used when writing and reading blocks. The defaults
///   (Zstd level 3 with byte shuffling, block sized to fit in the L1 data cache) are
///   suitable for most workloads.
///
/// All fields are optional. Fields left unset are filled in automatically using cache-size
/// heuristics or inherited from the source array when copying.
///
/// Functions that accept a `params` argument (such as [`jix.compact()`][jix.compact] and [`jix.copy()`][jix.copy])
/// also accept a plain `dict` as a shorthand - any key omitted from the dict uses its
/// default value:
///
/// ```python
/// # These are equivalent:
/// jix.compact(data, params=jix.ArrayParams(block_shape=[64, 64]))
/// jix.compact(data, params={"block_shape": [64, 64]})
/// ```
///
/// Note:
///     **On construction** (e.g. [`jix.compact()`][jix.compact]): data is split into blocks according to the
///     block layout, and each block is compressed with the codec settings.
///     **On copy** (e.g. [`jix.copy()`][jix.copy]): a new compressed array is created, inheriting any
///     unset fields from the source array's storage. After shape-changing operations
///     (`reshape`, `permute_axes`, etc.) the inherited block layout may not suit the new
///     shape - consider passing explicit params to [`jix.copy()`][jix.copy] after such ops.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     data = np.zeros((1024, 1024), dtype=np.float32)
///
///     # Store with a block shape tuned for row access
///     a = jix.compact(data, params=jix.ArrayParams(block_shape=[1, 1024]))
///
///     # After a transpose, pin the block shape for the new layout
///     b = jix.copy(a.T, params={"block_shape": [1024, 1]})
///     ```
#[gen_stub_pyclass]
#[pyclass(module = "jix", frozen)]
pub struct ArrayParams(pub(crate) jix_core::ArrayParams);
impl ArrayParams {
    pub(crate) fn resolve(
        py: Python<'_>,
        params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
    ) -> PyResult<jix_core::ArrayParams> {
        match params {
            None => Ok(jix_core::ArrayParams::default()),
            Some(OrKwargs::Value(param)) => Ok(param.get().0.clone()),
            Some(OrKwargs::Kwargs(mut kwargs)) => {
                macro_rules! extract_arg {
                    ($key:expr, $ty:ty) => {
                        kwargs
                            .remove($key)
                            .map(|v| {
                                v.bind(py).extract::<$ty>().map_err(|e| {
                                    PyTypeError::new_err(format!(
                                        "{} must be of type {}: {e}",
                                        $key,
                                        stringify!($ty)
                                    ))
                                })
                            })
                            .transpose()
                    };
                }

                let params = ArrayParams::new(
                    extract_arg!("block_shape", Vec<u32>)?,
                    extract_arg!("block_shape_tag", Vec<String>)?,
                    extract_arg!("block_size_hint", u64)?,
                    extract_arg!("preferred_read_shape", Vec<u32>)?,
                    extract_arg!("preferred_read_size_hint", u64)?,
                    extract_arg!("codec", String)?,
                    extract_arg!("compression_level", u32)?,
                    extract_arg!("filters", Vec<String>)?,
                )?;
                if !kwargs.is_empty() {
                    return Err(PyTypeError::new_err(format!(
                        "Unexpected ArrayParams kwargs: {}",
                        kwargs.into_keys().collect::<Vec<_>>().join(", ")
                    )));
                }
                Ok(params.0.clone())
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ArrayParams {
    /// Creates an `ArrayParams` with the given settings; all arguments are keyword-only and optional.
    ///
    /// Any field left as `None` is filled in automatically when the params are applied: block
    /// shape defaults to a size that fits in the L1 data cache, codec defaults to Zstd level 3
    /// with byte shuffling.
    ///
    /// Args:
    ///     block_shape: Explicit storage block shape, as a list of integers (one per
    ///         dimension). When set, array data is divided into nd-blocks of exactly this shape
    ///         (each dimension is clamped to the array boundary). Choosing a block shape that
    ///         matches your access pattern is the most important tuning knob: if you always read
    ///         row slices, a block shape of `[1, <row_length>]` avoids decompressing neighboring
    ///         rows. When not set, the shape is auto-computed to fit approximately
    ///         `block_size_hint` bytes.
    ///     block_shape_tag: Per-dimension constraint on how `block_shape` is scaled when a
    ///         downstream operation auto-computes a new block shape. One string per dimension:
    ///         `"fixed"` pins the block size exactly (the default when `block_shape` is set by
    ///         the user); `"multiple-of"` allows scaling up while keeping it a multiple of the
    ///         given value; `"any"` allows free choice (used when an op makes the original size
    ///         irrelevant, e.g. a broadcast dimension). Requires `block_shape` to also be set.
    ///         Length must equal the number of dimensions.
    ///     block_size_hint: Target block size in bytes, used when auto-computing or scaling the
    ///         block shape for dimensions that are not `"fixed"`. Ignored when all dimensions
    ///         are `"fixed"`. Defaults to the L1 data cache size.
    ///     preferred_read_shape: Recommended region size to request in a single read, as a list
    ///         of integers (one per dimension). Reads that cover a region of approximately this
    ///         shape avoid decompressing unnecessary blocks. Typically larger than `block_shape`
    ///         and targets the L2 cache. When not set, auto-computed from
    ///         `preferred_read_size_hint`.
    ///     preferred_read_size_hint: Target size in bytes for the preferred read region, used
    ///         when auto-computing `preferred_read_shape`. Defaults to the L2 cache size.
    ///     codec: Compression algorithm applied to each block. Currently the only accepted
    ///         value is `"zstd"`. Defaults to `"zstd"` when left unset.
    ///     compression_level: Compression level passed to the codec. For Zstd the valid range
    ///         is 1-22; higher values compress more but are slower to encode. Defaults to 3.
    ///     filters: List of filters applied to the raw block bytes *before* compression.
    ///         Filters improve the compression ratio for typed numeric data: `"byte-shuffle"`
    ///         groups bytes by significance (e.g. all high bytes together, then all low bytes);
    ///         `"bit-shuffle"` groups bits across elements. Defaults to `["byte-shuffle"]`.
    ///
    /// Returns:
    ///     A new [`jix.ArrayParams`][jix.ArrayParams] with the specified settings.
    #[new]
    #[pyo3(signature = (
        *,
        block_shape=None,
        block_shape_tag=None,
        block_size_hint=None,
        preferred_read_shape=None,
        preferred_read_size_hint=None,
        codec=None,
        compression_level=None,
        filters=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_shape: Option<Vec<u32>>,
        #[gen_stub(override_type(type_repr="typing.Optional[typing.Sequence[typing.Literal['fixed', 'multiple-of', 'any']]]", imports=("typing")))]
        block_shape_tag: Option<Vec<String>>,
        block_size_hint: Option<u64>,
        preferred_read_shape: Option<Vec<u32>>,
        preferred_read_size_hint: Option<u64>,
        #[gen_stub(override_type(type_repr="typing.Optional[typing.Literal['zstd']]", imports=("typing")))]
        codec: Option<String>,
        compression_level: Option<u32>,
        #[gen_stub(override_type(type_repr="typing.Optional[typing.Literal['byte-shuffle', 'bit-shuffle']]", imports=("typing")))]
        filters: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut params = jix_core::ArrayParams::default();
        if let Some(block_shape) = block_shape {
            params.block_shape(&block_shape);
        }
        if let Some(block_shape_tag) = block_shape_tag {
            let block_shape_tag = block_shape_tag
                .iter()
                .map(|s| match s.as_str() {
                    "fixed" => Ok(jix_core::storage::BlockShapeTag::Fixed),
                    "multiple-of" => Ok(jix_core::storage::BlockShapeTag::MultipleOf),
                    "any" => Ok(jix_core::storage::BlockShapeTag::Any),
                    _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid block_shape_tag: {s}"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?;
            params.block_shape_tag(&block_shape_tag);
        }
        if let Some(block_size_hint) = block_size_hint {
            params.block_size_hint(block_size_hint);
        }
        if let Some(preferred_read_shape) = preferred_read_shape {
            params.preferred_read_shape(&preferred_read_shape);
        }
        if let Some(preferred_read_size_hint) = preferred_read_size_hint {
            params.preferred_read_size_hint(preferred_read_size_hint);
        }

        if codec.is_some() || compression_level.is_some() || filters.is_some() {
            let mut encoder_params = EncoderParams::default();
            if let Some(codec) = codec {
                match codec.as_str() {
                    "zstd" => {
                        encoder_params.codec(Codec::Zstd);
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported codec: {codec}"
                        )));
                    }
                }
            }
            if let Some(compression_level) = compression_level {
                encoder_params.level(compression_level).into_py_result()?;
            }
            if let Some(filters) = filters {
                let filters = filters
                    .into_iter()
                    .map(|filter| match filter.as_str() {
                        "byte-shuffle" => Ok(jix_core::codec::Filter::ByteShuffle),
                        "bit-shuffle" => Ok(jix_core::codec::Filter::BitShuffle),
                        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported filter: {filter}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                encoder_params.filters(&filters).into_py_result()?;
            }
            params.encoder_params(encoder_params);
        }

        Ok(Self(params))
    }
}
