use pyo3::prelude::*;
use zix_core::Array as ZixArray;
use zix_core::ArrayStorage;

use crate::codec::ReadContext;
use crate::util::{IntoPyResult, OrKwargs};
use crate::{Array, ArrayParams};

/// Copies the data of an array into a new compact array by compressing it into new blocks.
///
/// The primary use of `copy` is to materialize a lazy operation chain. A `zix.Array` can
/// wrap an arbitrary lazy computation - for example the result of `a * 2.0 + b`. Reads to
/// such lazy arrays always perform the whole computation pipeline on the fly, which is very
/// flexible but can be inefficient for repeated access. Calling `copy` breaks the lazy
/// chain and materializes the result as a standalone compact array.
///
/// In contrast to "simple" views such as unary element-wise operations, lazy ops that change
/// the shape of the array (e.g. `reshape`, `broadcast`, `permute_axes`) can cause block
/// boundaries to no longer align with the logical layout of the array, causing reads to
/// decompress excess data. Calling `copy` on the result of such an operation re-encodes the
/// data with a freshly derived block shape that matches the new layout. The block shape is
/// automatically derived using a heuristic that aims to preserve user choices, but it is not
/// perfect - pass explicit `params` after shape-changing ops when you know the access pattern.
///
/// Codec settings (compression level, filters, etc.) are inherited from the source storage.
///
/// `params` controls the block layout and codec of the output. Accepts a `zix.ArrayParams`
/// instance or a plain `dict` (e.g. `{"block_shape": [64, 64]}`). Any field not set is
/// inherited from the source array's storage. See `zix.ArrayParams` for details.
///
/// `context` is an optional `zix.ReadContext` to reuse when decoding the source array.
/// When omitted, a context is created internally. See `zix.ReadContext` for details.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// a = zix.compact(np.array([[1.5, 2.0], [3.14, 6.17]], dtype=np.float32))
/// # Materialize an arithmetic pipeline
/// b = (a * 7.399) \    # Array<Mul<Compact, Scalar<f32>>> (lazy views, rust internal types)
///     .floor() \       # Array<Floor<Mul<Compact, Scalar<f32>>>>
///     .copy()          # Array<Compact> - materialize the pipeline
///
/// # After a shape-changing op, pin the block shape explicitly
/// c = zix.copy(a.T, params={"block_shape": [2, 1]})
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    *,
    params=None,
    context=None,
))]
pub fn copy<'py>(
    array: &Bound<'py, Array>,
    params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
    context: Option<&Bound<'_, ReadContext>>,
) -> PyResult<Bound<'py, Array>> {
    copy_impl_with(array.py(), &array.get().arr, params, context)
}
pub(crate) fn copy_impl<'py, S>(py: Python<'py>, array: &ZixArray<S>) -> PyResult<Bound<'py, Array>>
where
    S: ArrayStorage + Sync,
    S::ElementType: 'static,
    S::Dimension: 'static,
{
    copy_impl_with(py, array, None, None)
}

pub(crate) fn copy_impl_with<'py, S>(
    py: Python<'py>,
    array: &ZixArray<S>,
    params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
    context: Option<&Bound<'_, ReadContext>>,
) -> PyResult<Bound<'py, Array>>
where
    S: ArrayStorage + Sync,
    S::ElementType: 'static,
    S::Dimension: 'static,
{
    let params = ArrayParams::resolve(py, params)?;
    let context = context.map(|ctx| ctx.get());

    let array = py.detach::<PyResult<_>, _>(|| {
        let context_guard;
        let context = match context {
            Some(ctx) => {
                context_guard = ctx.lock();
                &*context_guard
            }
            None => &array.read_ctx(),
        };

        let ret = array.copy_with(params, context).into_py_result()?;
        Ok(Array::from_core(ret.to_type_dyn().to_dim_dyn().into_any()))
    })?;
    Bound::new(py, array)
}
