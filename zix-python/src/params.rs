use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::util::OrKwargs;

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
/// Functions that accept a `params` argument (such as `zix.compact()` and `zix.copy()`)
/// also accept a plain `dict` as a shorthand - any key omitted from the dict uses its
/// default value:
///
/// ```python,ignore
/// # These are equivalent:
/// zix.compact(data, params=zix.ArrayParams(block_shape=[64, 64]))
/// zix.compact(data, params={"block_shape": [64, 64]})
/// ```
///
/// # When params are applied
///
/// - **On construction** (e.g. `zix.compact()`): data is split into blocks according to the
///   block layout, and each block is compressed with the codec settings.
/// - **On copy** (e.g. `zix.copy()`): a new compressed array is created, inheriting any
///   unset fields from the source array's storage. After shape-changing operations
///   (`reshape`, `permute_axes`, etc.) the inherited block layout may not suit the new
///   shape - consider passing explicit params to `zix.copy()` after such ops.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// data = np.zeros((1024, 1024), dtype=np.float32)
///
/// # Store with a block shape tuned for row access
/// a = zix.compact(data, params=zix.ArrayParams(block_shape=[1, 1024]))
///
/// # After a transpose, pin the block shape for the new layout
/// b = zix.copy(a.T, params={"block_shape": [1024, 1]})
/// ```
#[gen_stub_pyclass]
#[pyclass(module = "zix", frozen)]
pub struct ArrayParams(pub(crate) zix_core::ArrayParams);
impl ArrayParams {
    pub(crate) fn resolve(
        py: Python<'_>,
        params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
    ) -> PyResult<zix_core::ArrayParams> {
        match params {
            None => Ok(zix_core::ArrayParams::default()),
            Some(OrKwargs::Value(param)) => Ok(param.get().0.clone()),
            Some(OrKwargs::Kwargs(mut kwargs)) => {
                let block_shape = kwargs
                    .remove("block_shape")
                    .map(|v| {
                        v.bind(py).extract::<Vec<u32>>().map_err(|_| {
                            PyTypeError::new_err("block_shape must be a list of integers")
                        })
                    })
                    .transpose()?;
                let block_shape_tag = kwargs
                    .remove("block_shape_tag")
                    .map(|v| {
                        v.bind(py).extract::<Vec<String>>().map_err(|_| {
                            PyTypeError::new_err("block_shape_tag must be a list of strings")
                        })
                    })
                    .transpose()?;
                let block_size_hint = kwargs
                    .remove("block_size_hint")
                    .map(|v| {
                        v.bind(py)
                            .extract::<u64>()
                            .map_err(|_| PyTypeError::new_err("block_size_hint must be an integer"))
                    })
                    .transpose()?;
                let preferred_read_shape = kwargs
                    .remove("preferred_read_shape")
                    .map(|v| {
                        v.bind(py).extract::<Vec<u32>>().map_err(|_| {
                            PyTypeError::new_err("preferred_read_shape must be a list of integers")
                        })
                    })
                    .transpose()?;
                let preferred_read_size_hint = kwargs
                    .remove("preferred_read_size_hint")
                    .map(|v| {
                        v.bind(py).extract::<u64>().map_err(|_| {
                            PyTypeError::new_err("preferred_read_size_hint must be an integer")
                        })
                    })
                    .transpose()?;
                if !kwargs.is_empty() {
                    return Err(PyTypeError::new_err(format!(
                        "Unexpected ArrayParams kwargs: {}",
                        kwargs.into_keys().collect::<Vec<_>>().join(", ")
                    )));
                }
                let params = ArrayParams::new(
                    block_shape,
                    block_shape_tag,
                    block_size_hint,
                    preferred_read_shape,
                    preferred_read_size_hint,
                )?;
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
    /// **`block_shape`** - explicit storage block shape, as a list of integers (one per
    /// dimension). When set, array data is divided into nd-blocks of exactly this shape (each
    /// dimension is clamped to the array boundary). Choosing a block shape that matches your
    /// access pattern is the most important tuning knob: if you always read row slices, a block
    /// shape of `[1, <row_length>]` avoids decompressing neighboring rows. When not set, the
    /// shape is auto-computed to fit approximately `block_size_hint` bytes.
    ///
    /// **`block_shape_tag`** - per-dimension constraint on how `block_shape` is scaled when
    /// a downstream operation auto-computes a new block shape. One string per dimension:
    /// - `"fixed"` - the block size for this dimension is pinned exactly; shape-changing ops
    ///   preserve it as-is. This is the default when `block_shape` is set by the user.
    /// - `"multiple-of"` - the block size may be scaled up, but must remain a multiple of the
    ///   given value. Used internally when an op constrains granularity without fixing the size.
    /// - `"any"` - the block size for this dimension can be freely chosen. Used internally when
    ///   an op makes the original size irrelevant (e.g. a broadcast dimension).
    ///
    /// Requires `block_shape` to also be set. Length must equal the number of dimensions.
    ///
    /// **`block_size_hint`** - target block size in bytes, used when auto-computing or scaling
    /// the block shape for dimensions that are not `"fixed"`. Ignored when all dimensions are
    /// `"fixed"`. Defaults to the L1 data cache size.
    ///
    /// **`preferred_read_shape`** - recommended region size to request in a single read, as a
    /// list of integers (one per dimension). Reads that cover a region of approximately this
    /// shape avoid decompressing unnecessary blocks. It is typically larger than `block_shape`
    /// and targets the L2 cache. When not set, auto-computed from `preferred_read_size_hint`.
    ///
    /// **`preferred_read_size_hint`** - target size in bytes for the preferred read region,
    /// used when auto-computing `preferred_read_shape`. Defaults to the L2 cache size.
    #[new]
    #[pyo3(signature = (
        *,
        block_shape=None,
        block_shape_tag=None,
        block_size_hint=None,
        preferred_read_shape=None,
        preferred_read_size_hint=None
    ))]
    pub fn new(
        block_shape: Option<Vec<u32>>,
        #[gen_stub(override_type(type_repr="typing.Optional[typing.Sequence[typing.Literal['fixed', 'multiple-of', 'any']]]", imports=("typing")))]
        block_shape_tag: Option<Vec<String>>,
        block_size_hint: Option<u64>,
        preferred_read_shape: Option<Vec<u32>>,
        preferred_read_size_hint: Option<u64>,
    ) -> PyResult<Self> {
        let mut params = zix_core::ArrayParams::default();
        if let Some(block_shape) = block_shape {
            params.block_shape(&block_shape);
        }
        if let Some(block_shape_tag) = block_shape_tag {
            let block_shape_tag = block_shape_tag
                .iter()
                .map(|s| match s.as_str() {
                    "fixed" => Ok(zix_core::storage::BlockShapeTag::Fixed),
                    "multiple-of" => Ok(zix_core::storage::BlockShapeTag::MultipleOf),
                    "any" => Ok(zix_core::storage::BlockShapeTag::Any),
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
        Ok(Self(params))
    }
}
