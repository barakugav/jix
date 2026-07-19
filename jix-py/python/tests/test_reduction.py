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
            return axes[0]
        if form == 1:
            return list(axes)
        return tuple(axes)
    return tuple(axes) if draw(st.booleans()) else list(axes)


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

# float16 reductions accumulate in native f16, so for sum/product/mean their values can drift
# far from a high-precision reference under catastrophic cancellation or overflow. The
# adversarial property tests for those ops therefore cover f32/f64 only; float16 value parity
# for the common (well-behaved) case is checked in test_common_case_matches_numpy, and
# float16 output dtype is checked exhaustively in test_output_dtype_matches_numpy. var/std
# keep f16 in the property tests because they accumulate the mean in f64 internally and stay
# precise.
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


def test_min_concrete():
    # Edge inputs: dtype min/max (int8/uint8), float NaN/inf/-inf (min propagates NaN like
    # numpy.min, not numpy.nanmin - see reduction.rs docs), and bool. Non-default block
    # shapes cross block boundaries.
    d_i8 = np.array([[-128, 0, 127], [5, -100, 3]], dtype=np.int8)
    za = jix.compact(d_i8, params={"block_shape": [1, 2]})
    for axis in (0, 1, None):
        assert_array_matches(jix.min(za, axis=axis), np.min(d_i8, axis=_np_axis(axis)))

    d_u8 = np.array([[0, 255, 3], [7, 1, 254]], dtype=np.uint8)
    zb = jix.compact(d_u8, params={"block_shape": [2, 1]})
    for axis in (0, 1, None):
        assert_array_matches(jix.min(zb, axis=axis), np.min(d_u8, axis=_np_axis(axis)))

    d_f = np.array([[-1.5, 0.0, float("inf")], [float("nan"), 2.5, float("-inf")]], dtype=np.float64)
    zc = jix.compact(d_f, params={"block_shape": [1, 3]})
    for axis in (0, 1, None):
        assert_array_matches(jix.min(zc, axis=axis), np.min(d_f, axis=_np_axis(axis)))

    d_b = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    zd = jix.compact(d_b, params={"block_shape": [1, 1]})
    for axis in (0, 1, None):
        assert_array_matches(jix.min(zd, axis=axis), np.min(d_b, axis=_np_axis(axis)))


# ---------------------------------------------------------------------------
# argmax / argmin - single axis only; output dtype is u64
# ---------------------------------------------------------------------------


def test_argmax_concrete():
    # Tie: verify jix returns the FIRST index of the max, matching numpy.argmax.
    d = np.array([[3, 5, 5, 1], [5, 2, 5, 4]], dtype=np.int32)
    za = jix.compact(d, params={"block_shape": [1, 2]})
    for axis in (0, 1):
        assert_array_matches(jix.argmax(za, axis=axis), np.argmax(d, axis=_np_axis(axis)).astype(np.uint64))
    assert jix.argmax(za, axis=1).numpy().tolist() == [1, 0]  # first '5' in each row

    # axis=None is only valid for 1-D arrays (equivalent to axis=0); tie included.
    d_1d = np.array([2, 7, 7, 1, 7], dtype=np.int32)
    zb = jix.compact(d_1d, params={"block_shape": [2]})
    assert_array_matches(jix.argmax(zb, axis=None), np.argmax(d_1d).astype(np.uint64))
    assert jix.argmax(zb, axis=None).numpy()[()] == 1  # first occurrence of the max (7)

    # dtype min/max edges: int8, uint8.
    d_i8 = np.array([[-128, 127, 0], [127, -128, 5]], dtype=np.int8)
    zc = jix.compact(d_i8, params={"block_shape": [2, 1]})
    for axis in (0, 1):
        assert_array_matches(jix.argmax(zc, axis=axis), np.argmax(d_i8, axis=_np_axis(axis)).astype(np.uint64))

    d_u8 = np.array([[0, 255, 3], [255, 1, 254]], dtype=np.uint8)
    zd = jix.compact(d_u8, params={"block_shape": [1, 3]})
    for axis in (0, 1):
        assert_array_matches(jix.argmax(zd, axis=axis), np.argmax(d_u8, axis=_np_axis(axis)).astype(np.uint64))

    # float, NaN-free: jix's NaN-index semantics diverge from numpy.argmax by design (see
    # reduction.rs docs), so NaN is intentionally excluded from this numpy cross-check.
    d_f = np.array([[-1.5, 2.5, 2.5], [2.5, 0.0, -1.5]], dtype=np.float64)
    ze = jix.compact(d_f, params={"block_shape": [1, 2]})
    for axis in (0, 1):
        assert_array_matches(jix.argmax(ze, axis=axis), np.argmax(d_f, axis=_np_axis(axis)).astype(np.uint64))

    d_b = np.array([[False, True, True], [True, False, False]], dtype=np.bool_)
    zf = jix.compact(d_b, params={"block_shape": [1, 1]})
    for axis in (0, 1):
        assert_array_matches(jix.argmax(zf, axis=axis), np.argmax(d_b, axis=_np_axis(axis)).astype(np.uint64))


def test_argmin_concrete():
    # Tie: verify jix returns the FIRST index of the min, matching numpy.argmin.
    d = np.array([[3, 1, 1, 5], [1, 4, 1, 2]], dtype=np.int32)
    za = jix.compact(d, params={"block_shape": [1, 2]})
    for axis in (0, 1):
        assert_array_matches(jix.argmin(za, axis=axis), np.argmin(d, axis=_np_axis(axis)).astype(np.uint64))
    assert jix.argmin(za, axis=1).numpy().tolist() == [1, 0]  # first '1' in each row

    # axis=None is only valid for 1-D arrays (equivalent to axis=0); tie included.
    d_1d = np.array([5, 0, 0, 3, 0], dtype=np.int32)
    zb = jix.compact(d_1d, params={"block_shape": [2]})
    assert_array_matches(jix.argmin(zb, axis=None), np.argmin(d_1d).astype(np.uint64))
    assert jix.argmin(zb, axis=None).numpy()[()] == 1  # first occurrence of the min (0)

    # dtype min/max edges: int8, uint8.
    d_i8 = np.array([[-128, 127, 0], [127, -128, 5]], dtype=np.int8)
    zc = jix.compact(d_i8, params={"block_shape": [2, 1]})
    for axis in (0, 1):
        assert_array_matches(jix.argmin(zc, axis=axis), np.argmin(d_i8, axis=_np_axis(axis)).astype(np.uint64))

    d_u8 = np.array([[0, 255, 3], [255, 1, 254]], dtype=np.uint8)
    zd = jix.compact(d_u8, params={"block_shape": [1, 3]})
    for axis in (0, 1):
        assert_array_matches(jix.argmin(zd, axis=axis), np.argmin(d_u8, axis=_np_axis(axis)).astype(np.uint64))

    # float, NaN-free: jix's NaN-index semantics diverge from numpy.argmin by design (see
    # reduction.rs docs), so NaN is intentionally excluded from this numpy cross-check.
    d_f = np.array([[-1.5, -1.5, 2.5], [2.5, 0.0, -1.5]], dtype=np.float64)
    ze = jix.compact(d_f, params={"block_shape": [1, 2]})
    for axis in (0, 1):
        assert_array_matches(jix.argmin(ze, axis=axis), np.argmin(d_f, axis=_np_axis(axis)).astype(np.uint64))

    d_b = np.array([[True, False, False], [False, True, True]], dtype=np.bool_)
    zf = jix.compact(d_b, params={"block_shape": [1, 1]})
    for axis in (0, 1):
        assert_array_matches(jix.argmin(zf, axis=axis), np.argmin(d_b, axis=_np_axis(axis)).astype(np.uint64))


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


def test_product_concrete():
    # Tiny shapes keep the accumulator from overflowing. Mix of zero and non-zero, positive
    # and negative, across signed/unsigned ints, wide floats, and complex.
    cases = [
        (np.int8, np.array([-4, 4, 0, -1], dtype=np.int8)),
        (np.int64, np.array([[-100, 5], [100, -1]], dtype=np.int64)),
        (np.uint8, np.array([4, 2, 3, 1], dtype=np.uint8)),
        (np.uint64, np.array([[30, 0], [15, 2]], dtype=np.uint64)),
        (np.float32, np.array([[-2.5, 4.0], [3.0, -1.5]], dtype=np.float32)),
        (np.float64, np.array([[-100.0, 0.0], [50.5, -1.5]], dtype=np.float64)),
        (np.complex64, np.array([[-2 + 1j, 3 - 1j], [1 + 1j, 2 - 2j]], dtype=np.complex64)),
        (np.complex128, np.array([[-50 + 25j, 0 + 0j], [10 - 10j, 1 + 1j]], dtype=np.complex128)),
    ]
    for dtype, np_a in cases:
        za = jix.compact(np_a, params={"block_shape": [1] * np_a.ndim})
        rtol = 0.0 if np.issubdtype(dtype, np.integer) else 1e-3
        axes = (0, None) if np_a.ndim == 1 else (0, 1, None)
        for axis in axes:
            result = jix.product(za, axis=axis)
            assert_array_matches(result, _prod_ref(np_a, axis, False, dtype), rtol=rtol)


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


# Covers ints/uints (upcast to float64), f16/f32/f64, and bool. The all-equal-elements case
# (near-zero true variance) exercises the atol below; 0 is present in every case.
_VAR_STD_CASES = [
    np.array([[-100, 0, 100], [50, -50, 0]], dtype=np.int32),
    np.array([[0, 30, 15], [30, 0, 10]], dtype=np.uint32),
    np.array([[-100.0, 0.0, 100.0], [50.0, -50.0, 25.0]], dtype=np.float16),
    np.array([[-99.97, -99.97, -99.97], [1.5, -1.5, 0.0]], dtype=np.float32),
    np.array([[-99.97, -99.97, -99.97], [1.5, -1.5, 0.0]], dtype=np.float64),
    np.array([[True, False, True], [False, False, True]], dtype=np.bool_),
]


def test_var_concrete():
    for np_a in _VAR_STD_CASES:
        za = jix.compact(np_a, params={"block_shape": [1, 2]})
        for axis in (0, 1, None):
            assert_array_matches(
                jix.var(za, axis=axis, ddof=0.0),
                np.var(np_a.astype(np.float64), axis=_np_axis(axis), ddof=0),
                rtol=1e-3,
                # atol covers near-zero cases where the true variance is 0 but numpy
                # accumulates a tiny FP error (e.g. all-equal elements like
                # [-99.97, -99.97, -99.97]).
                atol=1e-10,
            )


def test_std_concrete():
    for np_a in _VAR_STD_CASES:
        za = jix.compact(np_a, params={"block_shape": [1, 2]})
        for axis in (0, 1, None):
            assert_array_matches(
                jix.std(za, axis=axis, ddof=0.0),
                np.std(np_a.astype(np.float64), axis=_np_axis(axis), ddof=0),
                rtol=1e-3,
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# all / any - logical_op_strategy (includes zeros for true-branch coverage)
# ---------------------------------------------------------------------------


def test_all_concrete():
    # bool is the only supported dtype. Mixed rows/columns plus an all-True array exercise
    # both the True and False outcomes on every axis.
    d = np.array([[True, True, False], [True, True, True], [False, False, False]], dtype=np.bool_)
    za = jix.compact(d, params={"block_shape": [1, 2]})
    for axis in (0, 1, None):
        assert_array_matches(jix.all(za, axis=axis), np.all(d, axis=_np_axis(axis)))

    d_all_true = np.array([[True, True], [True, True]], dtype=np.bool_)
    zb = jix.compact(d_all_true, params={"block_shape": [1, 1]})
    for axis in (0, 1, None):
        assert_array_matches(jix.all(zb, axis=axis), np.all(d_all_true, axis=_np_axis(axis)))


def test_any_concrete():
    # bool is the only supported dtype. Mixed rows/columns plus an all-False array exercise
    # both the True and False outcomes on every axis.
    d = np.array([[False, False, True], [False, False, False], [True, True, True]], dtype=np.bool_)
    za = jix.compact(d, params={"block_shape": [1, 2]})
    for axis in (0, 1, None):
        assert_array_matches(jix.any(za, axis=axis), np.any(d, axis=_np_axis(axis)))

    d_all_false = np.array([[False, False], [False, False]], dtype=np.bool_)
    zb = jix.compact(d_all_false, params={"block_shape": [1, 1]})
    for axis in (0, 1, None):
        assert_array_matches(jix.any(zb, axis=axis), np.any(d_all_false, axis=_np_axis(axis)))


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
# jix's reduction output dtype matches numpy exactly for every integer, unsigned, float, and
# complex input: sum/product/mean/var/std keep float and complex inputs at their own width
# while integers upcast. These tests pin that mapping.
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
