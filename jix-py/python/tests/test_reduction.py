"""
Property tests for reduction ops (max, min, argmax, argmin, sum, product, mean, var, std, all, any).
Mirrors the test block in jix/src/ops/reduction.rs.

Python-specific coverage beyond the Rust tests:
  - negative axis values (e.g. axis=-1 for last axis)
  - list and tuple axis inputs for multi-axis reductions
  - axis=None (reduce all axes)
  - keepdims=True / False
  - dtype promotion parity with numpy for sum/product/mean/var/std (the output dtype of
    each reduction must match numpy exactly; see test_output_dtype_matches_numpy)
"""

import numpy as np
import pytest
import jix
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays
from hypothesis.strategies import DataObject
from tests_util import (
    any_element_strategy,
    assert_array_matches,
    complexes,
    floats,
    ints,
    logical_op_element_strategy,
    op_safe_element_strategy,
    uints,
)

# ---------------------------------------------------------------------------
# Shape strategies for reductions (no zero-length dims - Rust mirrors this)
# ---------------------------------------------------------------------------


def _reduction_shape_strategy():
    return st.one_of(
        st.lists(st.integers(1, 100), min_size=1, max_size=1),
        st.lists(st.integers(100, 1000), min_size=1, max_size=1),
        st.lists(st.integers(1, 16), min_size=2, max_size=2),
        st.lists(st.integers(16, 37), min_size=2, max_size=2),
        st.lists(st.integers(1, 12), min_size=3, max_size=3),
        st.lists(st.integers(1, 8), min_size=4, max_size=4),
        st.lists(st.integers(1, 4), min_size=1, max_size=8),
    )


def _reduction_shape_strategy_small():
    """Tiny shapes for product to keep accumulator from overflowing."""
    return st.one_of(
        st.lists(st.integers(1, 4), min_size=1, max_size=1),
        st.lists(st.integers(1, 2), min_size=2, max_size=2),
    )


# ---------------------------------------------------------------------------
# Axis strategies - Python-specific: int/list/tuple, positive and negative
# ---------------------------------------------------------------------------


@st.composite
def _axes_strategy(draw, ndim):
    """
    Return axis as None, int, list[int], or tuple[int].
    Values may be negative. None means reduce all axes.
    """
    if ndim == 0:
        return None
    # ~25% chance: reduce all (None)
    if draw(st.integers(0, 3)) == 0:
        return None
    # Pick n unique positive axes then optionally negate each
    n = draw(st.integers(1, ndim))
    pos_axes = sorted(draw(st.lists(st.integers(0, ndim - 1), min_size=n, max_size=n, unique=True)))
    axes = [ax - ndim if draw(st.booleans()) else ax for ax in pos_axes]
    # Vary the container type
    if len(axes) == 1:
        form = draw(st.integers(0, 2))
        if form == 0:
            return axes[0]  # bare int
        if form == 1:
            return list(axes)
        return tuple(axes)
    return tuple(axes) if draw(st.booleans()) else list(axes)


@st.composite
def _single_axis_strategy(draw, ndim):
    """Single int axis (possibly negative). For argmax/argmin."""
    ax = draw(st.integers(0, ndim - 1))
    return ax - ndim if draw(st.booleans()) else ax


# ---------------------------------------------------------------------------
# Composite array + axis strategies
# ---------------------------------------------------------------------------


@st.composite
def _carray_reduction(draw, dtype, element_st, shape_st=None):
    """Yields (np_array, jix_array, axis, keepdims)."""
    shape = tuple(draw(shape_st or _reduction_shape_strategy()))
    ndim = len(shape)
    np_a = draw(np_arrays(dtype=dtype, shape=shape, elements=element_st))
    block_shape = draw(st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim))
    # A block dim may not exceed its array dim (validation requires 1 <= block <= max(shape, 1)).
    block_shape = [min(b, max(s, 1)) for b, s in zip(block_shape, shape)]
    za = jix.compact(np_a, params={"block_shape": block_shape})
    axis = draw(_axes_strategy(ndim))
    keepdims = draw(st.booleans())
    return np_a, za, axis, keepdims


@st.composite
def _carray_single_axis_reduction(draw, dtype, element_st):
    """Yields (np_array, jix_array, single_int_axis, keepdims)."""
    shape = tuple(draw(_reduction_shape_strategy()))
    ndim = len(shape)
    np_a = draw(np_arrays(dtype=dtype, shape=shape, elements=element_st))
    block_shape = draw(st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim))
    # A block dim may not exceed its array dim (validation requires 1 <= block <= max(shape, 1)).
    block_shape = [min(b, max(s, 1)) for b, s in zip(block_shape, shape)]
    za = jix.compact(np_a, params={"block_shape": block_shape})
    axis = draw(_single_axis_strategy(ndim))
    keepdims = draw(st.booleans())
    return np_a, za, axis, keepdims


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _np_axis(axis):
    """numpy requires axis as int, tuple, or None - not list."""
    return tuple(axis) if isinstance(axis, list) else axis


def _out_dtype(dtype):
    """Output dtype jix uses for reductions (always 64-bit)."""
    if np.issubdtype(dtype, np.signedinteger):
        return np.int64
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.uint64
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    return np.float64


def _sum_ref(np_a, axis, keepdims, dtype):
    return np.sum(np_a.astype(_out_dtype(dtype)), axis=_np_axis(axis), keepdims=keepdims)


def _prod_ref(np_a, axis, keepdims, dtype):
    return np.prod(np_a.astype(_out_dtype(dtype)), axis=_np_axis(axis), keepdims=keepdims)


def _element_st(dtype):
    """op_safe for floats (no NaN in output), any for integers/bool."""
    if np.issubdtype(dtype, np.floating):
        return op_safe_element_strategy(dtype)
    return any_element_strategy(dtype)


# ---------------------------------------------------------------------------
# Dtype lists
# ---------------------------------------------------------------------------

# float16 reductions now accumulate in native f16 (see commit "Preset float dtype in
# reductions, upcast integers"), so for sum/product/mean their values can drift far from a
# high-precision reference under catastrophic cancellation or overflow. The adversarial
# property tests for those ops therefore cover f32/f64 only; float16 value parity for the
# common (well-behaved) case is checked in test_common_case_matches_numpy, and float16
# output dtype is checked exhaustively in test_output_dtype_matches_numpy. var/std keep f16
# in the property tests because they accumulate the mean in f64 internally and stay precise.
_WIDE_FLOATS = [np.float32, np.float64]


# ---------------------------------------------------------------------------
# max / min - any strategy; op_safe for floats to avoid NaN-ignoring semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ints + uints + floats + [np.bool_])
@given(st.data())
def test_max(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, _element_st(dtype)), label="array")
    assert_array_matches(
        jix.max(za, axis=axis, keepdims=keepdims),
        np.max(np_a, axis=_np_axis(axis), keepdims=keepdims),
        data=data,
    )


@pytest.mark.parametrize("dtype", ints + uints + floats + [np.bool_])
@given(st.data())
def test_min(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, _element_st(dtype)), label="array")
    assert_array_matches(
        jix.min(za, axis=axis, keepdims=keepdims),
        np.min(np_a, axis=_np_axis(axis), keepdims=keepdims),
        data=data,
    )


# ---------------------------------------------------------------------------
# argmax / argmin - single axis only; output dtype is u64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ints + uints + floats + [np.bool_])
@given(st.data())
def test_argmax(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_single_axis_reduction(dtype, _element_st(dtype)), label="array")
    result = jix.argmax(za, axis=axis, keepdims=keepdims)
    expected = np.argmax(np_a, axis=_np_axis(axis), keepdims=keepdims).astype(np.uint64)
    assert_array_matches(result, expected, data=data)


@pytest.mark.parametrize("dtype", ints + uints + floats + [np.bool_])
@given(st.data())
def test_argmin(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_single_axis_reduction(dtype, _element_st(dtype)), label="array")
    result = jix.argmin(za, axis=axis, keepdims=keepdims)
    expected = np.argmin(np_a, axis=_np_axis(axis), keepdims=keepdims).astype(np.uint64)
    assert_array_matches(result, expected, data=data)


# ---------------------------------------------------------------------------
# sum - op_safe to avoid large accumulations; wrapping for integers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ints + uints + _WIDE_FLOATS + complexes + [np.bool_])
@given(st.data())
def test_sum(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, op_safe_element_strategy(dtype)), label="array")
    result = jix.sum(za, axis=axis, keepdims=keepdims)
    rtol = 0.0 if np.issubdtype(dtype, np.integer) else 1e-3
    assert_array_matches(result, _sum_ref(np_a, axis, keepdims, dtype), data=data, rtol=rtol)


# ---------------------------------------------------------------------------
# product - small shapes to limit accumulation depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ints + uints + _WIDE_FLOATS + complexes)
@given(st.data())
def test_product(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(
        _carray_reduction(
            dtype,
            op_safe_element_strategy(dtype),
            shape_st=_reduction_shape_strategy_small(),
        ),
        label="array",
    )
    result = jix.product(za, axis=axis, keepdims=keepdims)
    rtol = 0.0 if np.issubdtype(dtype, np.integer) else 1e-3
    assert_array_matches(result, _prod_ref(np_a, axis, keepdims, dtype), data=data, rtol=rtol)


# ---------------------------------------------------------------------------
# mean - floats and complex only (Python binding does not accept integers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ints + uints + _WIDE_FLOATS + complexes + [np.bool_])
@given(st.data())
def test_mean(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, op_safe_element_strategy(dtype)), label="array")
    assert_array_matches(
        jix.mean(za, axis=axis, keepdims=keepdims),
        np.mean(np_a.astype(_out_dtype(dtype)), axis=_np_axis(axis), keepdims=keepdims),
        data=data,
        rtol=1e-3,
    )


# ---------------------------------------------------------------------------
# var / std - f32, f64 only; ddof=0 (population) for property tests
#   (ddof=1 can produce NaN for size-1 reductions, tested separately below)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ints + uints + floats + [np.bool_])
@given(st.data())
def test_var(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, op_safe_element_strategy(dtype)), label="array")
    assert_array_matches(
        jix.var(za, axis=axis, keepdims=keepdims, ddof=0.0),
        np.var(np_a.astype(np.float64), axis=_np_axis(axis), keepdims=keepdims, ddof=0),
        data=data,
        rtol=1e-3,
        # atol covers near-zero cases where the true variance is 0 but numpy accumulates
        # a tiny FP error (e.g. all-equal elements like [-99.97, -99.97, -99.97]).
        atol=1e-10,
    )


@pytest.mark.parametrize("dtype", ints + uints + floats + [np.bool_])
@given(st.data())
def test_std(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, op_safe_element_strategy(dtype)), label="array")
    assert_array_matches(
        jix.std(za, axis=axis, keepdims=keepdims, ddof=0.0),
        np.std(np_a.astype(np.float64), axis=_np_axis(axis), keepdims=keepdims, ddof=0),
        data=data,
        rtol=1e-3,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# all / any - logical_op_strategy (includes zeros for true-branch coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        # np.int8, np.int16, np.int32, np.int64,  # not supported
        # np.uint8, np.uint16, np.uint32, np.uint64,  # not supported
        # np.float16, np.float32, np.float64,  # not supported
        np.bool_,
    ],
)
@given(st.data())
def test_all(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, logical_op_element_strategy(dtype)), label="array")
    assert_array_matches(
        jix.all(za, axis=axis, keepdims=keepdims),
        np.all(np_a, axis=_np_axis(axis), keepdims=keepdims),
        data=data,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        # np.int8, np.int16, np.int32, np.int64,  # not supported
        # np.uint8, np.uint16, np.uint32, np.uint64,  # not supported
        # np.float16, np.float32, np.float64,  # not supported
        np.bool_,
    ],
)
@given(st.data())
def test_any(dtype: np.dtype, data: DataObject):
    np_a, za, axis, keepdims = data.draw(_carray_reduction(dtype, logical_op_element_strategy(dtype)), label="array")
    assert_array_matches(
        jix.any(za, axis=axis, keepdims=keepdims),
        np.any(np_a, axis=_np_axis(axis), keepdims=keepdims),
        data=data,
    )


# ---------------------------------------------------------------------------
# Handcrafted tests for Python-specific axis API
# ---------------------------------------------------------------------------


def test_axis_negative():
    """Negative axis values are accepted and normalized."""
    d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    za = jix.compact(d)
    np.testing.assert_array_equal(jix.sum(za, axis=-1).numpy(), d.sum(axis=-1))
    np.testing.assert_array_equal(jix.sum(za, axis=-2).numpy(), d.sum(axis=-2))
    np.testing.assert_array_equal(jix.max(za, axis=-1).numpy(), d.max(axis=-1))


def test_axis_none_reduces_all():
    """axis=None reduces over all axes, returning a scalar."""
    d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    za = jix.compact(d)
    assert jix.sum(za).numpy()[()] == 21
    assert jix.max(za).numpy()[()] == 6
    assert jix.min(za).numpy()[()] == 1


def test_axis_list_and_tuple():
    """axis=[0,1] and axis=(0,1) reduce multiple axes simultaneously."""
    d = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    za = jix.compact(d)
    np.testing.assert_array_equal(jix.sum(za, axis=[0, 2]).numpy(), d.sum(axis=(0, 2)))
    np.testing.assert_array_equal(jix.sum(za, axis=(0, 2)).numpy(), d.sum(axis=(0, 2)))
    np.testing.assert_array_equal(jix.max(za, axis=(1, 2)).numpy(), d.max(axis=(1, 2)))


def test_keepdims():
    """keepdims=True preserves the reduced axis as size 1."""
    d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    za = jix.compact(d)

    result = jix.sum(za, axis=0, keepdims=True)
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result.numpy(), [[5, 7, 9]])

    result = jix.sum(za, axis=None, keepdims=True)
    assert result.shape == (1, 1)
    assert result.numpy()[(0, 0)] == 21


def test_argmax_argmin_axis():
    """argmax / argmin return u64 indices; negative axis normalized correctly."""
    d = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int32)
    za = jix.compact(d)
    np.testing.assert_array_equal(jix.argmax(za, axis=1).numpy(), [1, 2])
    np.testing.assert_array_equal(jix.argmax(za, axis=-1).numpy(), [1, 2])
    np.testing.assert_array_equal(jix.argmin(za, axis=0).numpy(), [0, 1, 0])
    np.testing.assert_array_equal(jix.argmin(za, axis=-2).numpy(), [0, 1, 0])
    assert jix.argmax(za, axis=1).dtype == np.dtype("uint64")


def test_argmax_1d_axis_none():
    """For 1-D arrays, axis=None is equivalent to axis=0."""
    d = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32)
    za = jix.compact(d)
    assert jix.argmax(za).numpy()[()] == np.uint64(5)
    assert jix.argmin(za).numpy()[()] == np.uint64(1)


def test_var_std_ddof1():
    """var/std with ddof=1 (sample variance) on known data."""
    d = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32)
    za = jix.compact(d)
    assert abs(jix.var(za).numpy()[()] - 4.0) < 1e-4
    assert abs(jix.var(za, ddof=1.0).numpy()[()] - np.var(d, ddof=1)) < 1e-3
    assert abs(jix.std(za).numpy()[()] - 2.0) < 1e-4
    assert abs(jix.std(za, ddof=1.0).numpy()[()] - np.std(d, ddof=1)) < 1e-3


# ---------------------------------------------------------------------------
# Dtype promotion parity with numpy
#
# The commit "Preset float dtype in reductions, upcast integers" changed sum/product/mean/
# var/std so that float and complex inputs keep their own width instead of always widening
# to 64-bit, while integers still upcast. The result is that jix's reduction output dtype
# now matches numpy exactly for every integer, unsigned, float, and complex input. These
# tests pin that mapping.
#
# The only intentional divergences (verified against numpy and documented in the op
# docstrings) both involve bool:
#   - sum(bool)     -> jix u64, numpy int64  (values equal; signedness differs)
#   - product(bool) -> unsupported in jix,   numpy would upcast to int64
# They are covered by test_bool_sum_dtype_divergence / test_product_bool_unsupported and
# excluded from the exact-parity checks below.
# ---------------------------------------------------------------------------

# name -> (jix fn, numpy fn)
_REDUCE_FN = {
    "sum": (jix.sum, np.sum),
    "product": (jix.product, np.prod),
    "mean": (jix.mean, np.mean),
    "var": (jix.var, np.var),
    "std": (jix.std, np.std),
}

# Input dtypes whose reduction output dtype must match numpy exactly, per op.
_DTYPE_MATCH = {
    "sum": ints + uints + floats + complexes,  # bool: divergence, see below
    "product": ints + uints + floats + complexes,  # bool: unsupported, see below
    "mean": ints + uints + floats + complexes + [np.bool_],
    "var": ints + uints + floats + complexes + [np.bool_],
    "std": ints + uints + floats + complexes + [np.bool_],
}

_DTYPE_MATCH_CASES = [(fname, dtype) for fname, dtypes in _DTYPE_MATCH.items() for dtype in dtypes]


def _dtype_sample(dtype: np.dtype) -> np.ndarray:
    """A small (3, 4) array of the given dtype for exercising reduction code paths."""
    if dtype == np.bool_:
        return np.array(
            [[True, False, True, True], [False, True, True, False], [True, True, False, True]],
            dtype=np.bool_,
        )
    if np.issubdtype(dtype, np.complexfloating):
        base = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
        return (base + 1j * base[::-1]).astype(dtype)
    return np.arange(1, 13, dtype=dtype).reshape(3, 4)


@pytest.mark.parametrize(
    "func_name,dtype",
    _DTYPE_MATCH_CASES,
    ids=[f"{f}-{np.dtype(d).name}" for f, d in _DTYPE_MATCH_CASES],
)
@pytest.mark.parametrize("axis,keepdims", [(None, False), (None, True), (0, True), (1, False)])
def test_output_dtype_matches_numpy(func_name: str, dtype: np.dtype, axis, keepdims: bool):
    """jix's reduction output dtype matches numpy exactly across all code paths."""
    jix_fn, np_fn = _REDUCE_FN[func_name]
    np_a = _dtype_sample(dtype)
    za = jix.compact(np_a)
    jix_out = jix_fn(za, axis=axis, keepdims=keepdims)
    np_out = np_fn(np_a, axis=axis, keepdims=keepdims)
    assert jix_out.dtype == np_out.dtype, (
        f"{func_name}({np.dtype(dtype).name}) axis={axis} keepdims={keepdims}: "
        f"jix -> {jix_out.dtype}, numpy -> {np_out.dtype}"
    )


# Well-behaved (moderate magnitude, no cancellation/overflow) inputs where jix and numpy
# agree value-wise even for the reduced-precision dtypes. Edge cases are allowed to diverge.
_COMMON_CASE_ARRAYS = {
    np.float16: np.array([1.0, 2.5, 3.0, 0.5, 4.25, 2.0, 1.5, 3.75, 2.25, 1.0], dtype=np.float16),
    np.float32: np.array([1.1, 2.5, 3.0, 0.5, 4.25, 2.0, 1.5, 3.75, 2.25, 1.0], dtype=np.float32),
    np.float64: np.array([1.1, 2.5, 3.0, 0.5, 4.25, 2.0, 1.5, 3.75, 2.25, 1.0], dtype=np.float64),
    np.complex64: np.array([1 + 0.5j, 2 + 1.5j, 3 + 2j, 0.5 + 1j, 4 + 0.25j], dtype=np.complex64),
    np.complex128: np.array([1 + 0.5j, 2 + 1.5j, 3 + 2j, 0.5 + 1j, 4 + 0.25j], dtype=np.complex128),
    np.int32: np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32),
    np.uint8: np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.uint8),
}


@pytest.mark.parametrize("dtype", list(_COMMON_CASE_ARRAYS), ids=[np.dtype(d).name for d in _COMMON_CASE_ARRAYS])
@pytest.mark.parametrize("func_name", list(_REDUCE_FN))
def test_common_case_matches_numpy(func_name: str, dtype: np.dtype):
    """For common-case inputs jix reductions match numpy in both dtype and value (isclose)."""
    jix_fn, np_fn = _REDUCE_FN[func_name]
    np_a = _COMMON_CASE_ARRAYS[dtype]
    za = jix.compact(np_a)
    jix_out = jix_fn(za)
    np_out = np_fn(np_a)
    assert jix_out.dtype == np_out.dtype
    np.testing.assert_allclose(
        np.asarray(jix_out.numpy()).astype(np.complex128),
        np.asarray(np_out).astype(np.complex128),
        rtol=1e-2,
        atol=1e-3,
    )


def test_bool_sum_dtype_divergence():
    """sum(bool): jix uses u64, numpy uses int64. Values (the true count) match; dtype does not."""
    np_a = np.array([True, False, True, True, False], dtype=np.bool_)
    za = jix.compact(np_a)
    jix_out = jix.sum(za)
    np_out = np.sum(np_a)
    assert jix_out.dtype == np.dtype("uint64")
    assert jix_out.dtype != np_out.dtype  # numpy promotes bool -> int64 instead
    assert int(jix_out.numpy()[()]) == int(np_out) == 3


def test_product_bool_unsupported():
    """product(bool) is not supported by jix (numpy would upcast bool -> int64)."""
    za = jix.compact(np.array([True, False, True], dtype=np.bool_))
    with pytest.raises(RuntimeError):
        jix.product(za)
