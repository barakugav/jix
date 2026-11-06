"""
Arithmetic errors of element-wise operations: integer division by zero and a negative
integer power raise an ArithmeticError when the array is read, while float operations give inf and
NaN, without raising or warning. Mirrors the tests in jix/src/ops/op1.rs and jix/src/ops/op2.rs.
"""

import io
import warnings
from functools import partial

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import carrays2_mixed_strategy, comparable_element_strategy, ints, op_safe_element_strategy, uints

import jix

MESSAGE = "integer division or modulo by zero"

# The integer divisions as (function name, rounding mode), with their results for a non-zero
# divisor.
_INT_DIVISIONS = {
    ("divide", "trunc"): lambda x, y: abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1),
    ("divide", "floor"): lambda x, y: x // y,
    ("divide", "euclid"): lambda x, y: (x - x % abs(y)) // y,
    ("remainder", "trunc"): lambda x, y: abs(x) % abs(y) * (1 if x >= 0 else -1),
    ("remainder", "floor"): lambda x, y: x % y,
    ("remainder", "euclid"): lambda x, y: x % abs(y),
}


def _int_div(op, a, b):
    name, rounding_mode = op
    return getattr(jix, name)(a, b, rounding_mode=rounding_mode)


def _int_div_ref(op, np_a, np_b):
    """`op` of each pair of non-zero divisor, wrapping (`MIN // -1` is `MIN`)."""
    out = np.zeros_like(np_a)
    for idx in np.ndindex(np_a.shape):
        x, y = int(np_a[idx]), int(np_b[idx])
        out[idx] = np.array(_INT_DIVISIONS[op](x, y)).astype(np_a.dtype)
    return out


def _arrays(dtype):
    a = np.arange(1000).astype(dtype) % 100
    b = (np.arange(1000) % 7 + 1).astype(dtype)
    b[300] = 0
    b[-1] = 0
    return a, b


@pytest.mark.parametrize("op", _INT_DIVISIONS, ids="-".join)
@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_zero_raises(op, dtype):
    a, b = _arrays(dtype)
    q = _int_div(op, jix.compact(a), jix.compact(b))
    with pytest.raises(ArithmeticError, match=MESSAGE) as e:
        q.numpy()
    assert type(e.value) is ArithmeticError  # not the ZeroDivisionError subclass
    # Each zero divisor alone: in the SIMD bulk part of the loop, and in its scalar tail.
    for part in [slice(None, -1), slice(301, None)]:
        with pytest.raises(ArithmeticError, match=MESSAGE):
            q[part]
    # Only the elements read count.
    np.testing.assert_array_equal(q[:300], _int_div_ref(op, a[:300], b[:300]))


@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_scalar_zero_raises(dtype):
    za = jix.compact(np.arange(10).astype(dtype))
    for q in [za // 0, za % 0, jix.divide(za, 0, rounding_mode="trunc"), jix.remainder(za, 0, rounding_mode="euclid")]:
        with pytest.raises(ArithmeticError, match=MESSAGE):
            q.numpy()


def test_errors_raise_when_read(tmp_path):
    # Operations are lazy: building `q` does not raise, each read of it does.
    q = jix.compact(np.array([6, 7], dtype=np.int32)) // jix.compact(np.array([2, 0], dtype=np.int32))
    with pytest.raises(ArithmeticError, match=MESSAGE):
        q.numpy()
    with pytest.raises(ArithmeticError, match=MESSAGE):
        jix.compact(q)
    with pytest.raises(ArithmeticError, match=MESSAGE):
        jix.write_array(q, tmp_path / "q.jix")
    assert not (tmp_path / "q.jix").exists()
    with pytest.raises(ArithmeticError, match=MESSAGE):
        jix.write_array(q, io.BytesIO())
    np.testing.assert_array_equal(q[:1], [3])


@pytest.mark.parametrize("op", _INT_DIVISIONS, ids="-".join)
@pytest.mark.parametrize("dtype", ints + uints)
@settings(deadline=None)  # the reference is a Python loop over up to thousands of elements
@given(st.data())
def test_divide_by_zero_random(op, dtype, data: DataObject):
    # Zero divisors in about a third of the elements, random shapes and block shapes.
    (np_a, za), (np_b, zb) = data.draw(
        carrays2_mixed_strategy(
            dtype,
            dtype,
            element_st_a=op_safe_element_strategy(dtype),
            element_st_b=comparable_element_strategy(dtype),
        ),
        label="arrays",
    )
    if (np_b == 0).any():
        with pytest.raises(ArithmeticError, match=MESSAGE):
            _int_div(op, za, zb).numpy()
    else:
        np.testing.assert_array_equal(_int_div(op, za, zb).numpy(), _int_div_ref(op, np_a, np_b))


@pytest.mark.parametrize("op", _INT_DIVISIONS, ids="-".join)
@pytest.mark.parametrize("dtype", ints)
def test_min_by_minus_one_wraps(op, dtype):
    # As PyTorch: the quotients wrap to `MIN`, and the remainders are `0`, exact.
    info = np.iinfo(dtype)
    a = np.array([info.min, 7, info.max], dtype=dtype)
    b = np.full(3, -1, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _int_div(op, jix.compact(a), jix.compact(b)).numpy()
    np.testing.assert_array_equal(result, _int_div_ref(op, a, b))


@pytest.mark.parametrize("dtype", ints)
def test_power_negative_exponent_raises(dtype):
    a = np.array([2, 3, 4, -5], dtype=dtype)
    b = np.array([3, -1, 0, 3], dtype=dtype)
    q = jix.power(jix.compact(a), jix.compact(b))
    assert q.dtype == dtype
    # An error as numpy, though an ArithmeticError and when the array is read.
    with pytest.raises(ValueError, match="negative integer powers"):
        np.power(a, b)
    with pytest.raises(ArithmeticError, match="integers to negative integer powers are not allowed") as e:
        q.numpy()
    assert type(e.value) is ArithmeticError
    # Only the elements read count.
    np.testing.assert_array_equal(q[2:], [1, -125])


@pytest.mark.parametrize("dtype", ints + uints)
def test_power_wraps_like_numpy(dtype):
    signed = np.issubdtype(dtype, np.signedinteger)
    a = (np.arange(-50, 50) if signed else np.arange(100)).astype(dtype)
    b = (np.arange(100) % 70).astype(dtype if signed else np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = jix.power(jix.compact(a), jix.compact(b)).numpy()
    np.testing.assert_array_equal(result, np.power(a, b))


@pytest.mark.parametrize("dtype", ints + uints)
def test_arithmetic_wraps_like_numpy(dtype):
    info = np.iinfo(dtype)
    edges = np.array([info.min, info.min + 1, 0, 1, info.max // 2 + 1, info.max], dtype=dtype)
    a = np.resize(np.repeat(edges, len(edges)), 1000)
    b = np.resize(np.tile(edges, len(edges)), 1000)
    za, zb = jix.compact(a), jix.compact(b)
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal((za + zb).numpy(), a + b)
        np.testing.assert_array_equal((za - zb).numpy(), a - b)
        np.testing.assert_array_equal((za * zb).numpy(), a * b)
        np.testing.assert_array_equal(jix.square(za).numpy(), np.square(a))
        np.testing.assert_array_equal(jix.absolute(za).numpy(), np.absolute(a))
        if np.issubdtype(dtype, np.signedinteger):
            np.testing.assert_array_equal(jix.negative(za).numpy(), np.negative(a))


# Operands for the float tests: every pair of these.
_FLOAT_EDGES = [0.0, -0.0, 1.0, -2.5, 1e30, -1e-30, np.inf, -np.inf, np.nan, "max", "-max", "tiny"]


def _float_edges(dtype):
    info = np.finfo(dtype)
    special = {"max": info.max, "-max": -info.max, "tiny": info.smallest_subnormal}
    with np.errstate(over="ignore"):
        return np.array([special.get(x, x) for x in _FLOAT_EDGES], dtype=dtype)


def _read_quietly(array):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return array.numpy()


_FLOAT_BINARY_OPS = [
    pytest.param(jix.divide, id="divide"),
    *[
        pytest.param(partial(getattr(jix, name), rounding_mode=mode), id=f"{name}-{mode}")
        for name in ["divide", "remainder"]
        for mode in ["trunc", "floor", "euclid"]
    ],
    pytest.param(jix.power, id="power"),
]


@pytest.mark.parametrize("jix_fn", _FLOAT_BINARY_OPS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_binary_ops_do_not_raise(jix_fn, dtype):
    # Every pair of edge values after ordinary ones: some in the SIMD bulk part of the loop, the
    # last ones in its scalar tail.
    edges = _float_edges(dtype)
    a = np.concatenate([np.full(300, 1.5, dtype=dtype), np.repeat(edges, len(edges))])
    b = np.concatenate([np.full(300, 3.0, dtype=dtype), np.tile(edges, len(edges))])
    result = _read_quietly(jix_fn(jix.compact(a), jix.compact(b)))
    assert np.isnan(result).any()


# Past the domains and overflow thresholds of the float functions.
_FUNCTION_EDGES = [0.5, -1.0, 1.0000001, -1.0000001, 11.1, 88.8, 709.8, -745.2]

_FUNCTIONS = [
    (jix.sqrt, np.sqrt),
    (jix.log, np.log),
    (jix.log2, np.log2),
    (jix.log10, np.log10),
    (jix.exp, np.exp),
    (jix.sin, np.sin),
    (jix.cos, np.cos),
    (jix.tan, np.tan),
    (jix.asin, np.arcsin),
    (jix.acos, np.arccos),
]


@pytest.mark.parametrize(("jix_fn", "np_fn"), _FUNCTIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_functions_do_not_raise(jix_fn, np_fn, dtype):
    with np.errstate(over="ignore"):
        edges = np.concatenate([_float_edges(dtype), np.array(_FUNCTION_EDGES, dtype=dtype)])
    x = np.concatenate([np.full(300, 0.5, dtype=dtype), edges])
    result = _read_quietly(jix_fn(jix.compact(x)))
    with np.errstate(all="ignore"):
        expected = np_fn(x)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))


def test_float_divide_by_zero():
    a = jix.compact(np.array([1.0, 0.0, -6.0, 6.0]))
    b = jix.compact(np.array([0.0, 0.0, 0.0, 3.0]))
    np.testing.assert_array_equal(_read_quietly(a / b), [np.inf, np.nan, -np.inf, 2.0])
    np.testing.assert_array_equal(_read_quietly(a // b), [np.inf, np.nan, -np.inf, 2.0])
    np.testing.assert_array_equal(_read_quietly(a % b), [np.nan, np.nan, np.nan, 0.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_power_zero_to_minus_inf(dtype):
    # IEEE 754's `inf`, as numpy's power.
    a = jix.compact(np.array([0.0, -0.0, 0.0], dtype=dtype))
    b = jix.compact(np.array([-np.inf, -np.inf, np.inf], dtype=dtype))
    np.testing.assert_array_equal(_read_quietly(jix.power(a, b)), [np.inf, np.inf, 0.0])
