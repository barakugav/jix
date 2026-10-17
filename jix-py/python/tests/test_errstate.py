"""
Arithmetic error handling under jix.errstate: integer and float division, sqrt, integer power and
wrapping integer arithmetic. Mirrors the tests in jix/src/ops/op1.rs and jix/src/ops/op2.rs.
"""

import threading
import warnings

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from tests_util import carrays2_mixed_strategy, comparable_element_strategy, ints, op_safe_element_strategy, uints

import jix

MESSAGE = "divide by zero encountered"
DEFAULT_MODES = {"divide": "raise", "over": "warn", "under": "warn", "invalid": "raise"}


def _modes(**kinds):
    return {**DEFAULT_MODES, **kinds}


# The integer divisions, with their results for a non-zero divisor.
_INT_DIVISIONS = {
    jix.divide_trunc: lambda x, y: abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1),
    jix.divide_floor: lambda x, y: x // y,
    jix.divide_euclid: lambda x, y: (x - x % abs(y)) // y,
    jix.remainder_trunc: lambda x, y: abs(x) % abs(y) * (1 if x >= 0 else -1),
    jix.remainder_floor: lambda x, y: x % y,
    jix.remainder_euclid: lambda x, y: x % abs(y),
}
_QUOTIENTS = [jix.divide_trunc, jix.divide_floor, jix.divide_euclid]


def _int_div_ref(jix_fn, np_a, np_b):
    """`jix_fn` of each pair as numpy: `0` for a zero divisor, and wrapping (`MIN // -1` is `MIN`)."""
    out = np.zeros_like(np_a)
    for idx in np.ndindex(np_a.shape):
        x, y = int(np_a[idx]), int(np_b[idx])
        if y != 0:
            out[idx] = np.array(_INT_DIVISIONS[jix_fn](x, y)).astype(np_a.dtype)
    return out


def _arrays(dtype):
    a = np.arange(1000).astype(dtype) % 100
    b = (np.arange(1000) % 7 + 1).astype(dtype)
    b[300] = 0
    b[-1] = 0
    return a, b


@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_zero_warn(dtype):
    a, b = _arrays(dtype)
    za, zb = jix.compact(a), jix.compact(b)
    with jix.errstate(divide="warn"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = (za // zb).numpy()
    np.testing.assert_array_equal(result, _int_div_ref(jix.divide_floor, a, b))
    # One warning per read, however many elements hit it.
    assert [(x.category, str(x.message)) for x in w] == [(RuntimeWarning, MESSAGE)]


@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_zero_raises_by_default(dtype):
    a, b = _arrays(dtype)
    za, zb = jix.compact(a), jix.compact(b)
    assert jix.geterr() == DEFAULT_MODES
    with pytest.raises(FloatingPointError, match=MESSAGE):
        (za // zb).numpy()
    # Only the elements read count.
    np.testing.assert_array_equal((za // zb)[:300], _int_div_ref(jix.divide_floor, a[:300], b[:300]))
    with jix.errstate(divide="raise"), pytest.raises(FloatingPointError, match=MESSAGE):
        (za // zb).numpy()


@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_zero_ignore(dtype):
    a, b = _arrays(dtype)
    za, zb = jix.compact(a), jix.compact(b)
    with jix.errstate(divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal((za // zb).numpy(), _int_div_ref(jix.divide_floor, a, b))


@pytest.mark.parametrize("dtype", ints)
def test_min_floor_divide_minus_one_overflows(dtype):
    # `MIN`, and an overflow, as numpy.
    info = np.iinfo(dtype)
    a = np.array([info.min, 7, info.max], dtype=dtype)
    b = np.array([-1, -1, -1], dtype=dtype)
    q = jix.compact(a) // jix.compact(b)
    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        result = q.numpy()
    np.testing.assert_array_equal(result, np.array([info.min, -7, -info.max], dtype=dtype))
    with pytest.warns(RuntimeWarning, match="overflow encountered in floor_divide"):
        np.testing.assert_array_equal(a // b, result)
    with jix.errstate(over="raise"), pytest.raises(FloatingPointError, match="overflow encountered"):
        q.numpy()


def test_compact_and_write_array_report(tmp_path):
    q = jix.compact(np.array([6, 7], dtype=np.int32)) // jix.compact(np.array([2, 0], dtype=np.int32))
    with jix.errstate(divide="warn"):
        with pytest.warns(RuntimeWarning, match=MESSAGE):
            np.testing.assert_array_equal(jix.compact(q).numpy(), [3, 0])
        with pytest.warns(RuntimeWarning, match=MESSAGE):
            jix.write_array(q, tmp_path / "q.jix")
    np.testing.assert_array_equal(jix.read_array(tmp_path / "q.jix").numpy(), [3, 0])

    with pytest.raises(FloatingPointError, match=MESSAGE):
        jix.compact(q)
    with pytest.raises(FloatingPointError, match=MESSAGE):
        jix.write_array(q, tmp_path / "q2.jix")


def test_modes_apply_at_read_time():
    # Operations are lazy: the modes in effect when the array is read apply.
    za = jix.compact(np.array([6, 7], dtype=np.int32))
    zb = jix.compact(np.array([2, 0], dtype=np.int32))
    with jix.errstate(divide="warn"):
        q = za // zb
    with pytest.raises(FloatingPointError, match=MESSAGE):
        q.numpy()
    with jix.errstate(divide="warn"), pytest.warns(RuntimeWarning, match=MESSAGE):
        q.numpy()


def test_errstate_as_function():
    assert jix.geterr() == DEFAULT_MODES
    try:
        # Called without `with`, the modes change until changed again.
        jix.errstate(divide="warn")
        assert jix.geterr() == _modes(divide="warn")
        jix.errstate(all="ignore")
        assert jix.geterr() == _modes(divide="ignore", over="ignore", under="ignore", invalid="ignore")
        # An explicit kind takes precedence over `all`.
        jix.errstate(all="raise", over="warn")
        assert jix.geterr() == _modes(divide="raise", over="warn", under="raise", invalid="raise")
        jix.errstate(under="ignore", invalid="warn")
        assert jix.geterr() == _modes(divide="raise", over="warn", under="ignore", invalid="warn")
    finally:
        jix.errstate(**DEFAULT_MODES)
    assert jix.geterr() == DEFAULT_MODES

    for bad in ["call", "print", "log", "Raise", ""]:
        for kind in ["all", "divide", "over", "under", "invalid"]:
            with pytest.raises(ValueError, match="invalid error mode"):
                jix.errstate(**{kind: bad})
    assert jix.geterr() == DEFAULT_MODES


def test_errstate_restores():
    with jix.errstate(divide="warn"):
        assert jix.geterr() == _modes(divide="warn")
        with jix.errstate(all="ignore"):
            assert jix.geterr() == _modes(divide="ignore", over="ignore", under="ignore", invalid="ignore")
        assert jix.geterr() == _modes(divide="warn")
    assert jix.geterr() == DEFAULT_MODES

    with pytest.raises(KeyError), jix.errstate(divide="ignore"):
        raise KeyError
    assert jix.geterr() == DEFAULT_MODES


def test_modes_are_per_thread():
    seen = []
    with jix.errstate(divide="warn"):
        thread = threading.Thread(target=lambda: seen.append(jix.geterr()))
        thread.start()
        thread.join()
    assert seen == [DEFAULT_MODES]


@pytest.mark.parametrize("jix_fn", _INT_DIVISIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", ints + uints)
@given(st.data())
def test_divide_by_zero_random(jix_fn, dtype, data: DataObject):
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
    expected = _int_div_ref(jix_fn, np_a, np_b)
    with jix.errstate(divide="ignore"):
        np.testing.assert_array_equal(jix_fn(za, zb).numpy(), expected)
    with jix.errstate(divide="raise"):
        if (np_b == 0).any():
            with pytest.raises(FloatingPointError, match=MESSAGE):
                jix_fn(za, zb).numpy()
        else:
            np.testing.assert_array_equal(jix_fn(za, zb).numpy(), expected)


@pytest.mark.parametrize("jix_fn", _INT_DIVISIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", ints)
def test_min_by_minus_one(jix_fn, dtype):
    # As numpy: the quotients are `MIN`, an overflow, and the remainders `0`, exact.
    info = np.iinfo(dtype)
    a = np.array([info.min, 7, info.max], dtype=dtype)
    b = np.full(3, -1, dtype=dtype)
    result = jix_fn(jix.compact(a), jix.compact(b))
    expected = _int_div_ref(jix_fn, a, b)
    if jix_fn in _QUOTIENTS:
        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            np.testing.assert_array_equal(result.numpy(), expected)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            np.testing.assert_array_equal(result.numpy(), expected)


# Operands for the float error tests: every pair of these.
_FLOAT_EDGES = [0.0, -0.0, 1.0, -2.5, 1e30, -1e-30, np.inf, -np.inf, np.nan, "max", "-max", "tiny"]
_KINDS = {"divide by zero": "divide", "overflow": "over", "invalid": "invalid"}


def _float_edges(dtype):
    info = np.finfo(dtype)
    special = {"max": info.max, "-max": -info.max, "tiny": info.smallest_subnormal}
    with np.errstate(over="ignore"):
        return np.array([special.get(x, x) for x in _FLOAT_EDGES], dtype=dtype)


def _error_kind(f, *args):
    """The kind of the `FloatingPointError` `f(*args)` raises, or None."""
    try:
        f(*args)
    except FloatingPointError as e:
        return next(kind for word, kind in _KINDS.items() if word in str(e))
    return None


def _errors_match_numpy(jix_op, np_op, *operands):
    """Each operand tuple in one element of an array of ordinary values, in the SIMD bulk part of
    the loop and in its scalar tail: jix reports the error numpy reports, ignoring underflow,
    which jix does not detect."""
    n = 1000
    for i in [300, n - 1]:
        arrays = []
        for x, fill in zip(operands, [1.5, 3.0]):
            a = np.full(n, fill, dtype=x.dtype)
            a[i] = x
            arrays.append(a)
        with np.errstate(all="raise", under="ignore"):
            expected = _error_kind(np_op, *arrays)
        with jix.errstate(all="raise"):
            actual = _error_kind(lambda *a: jix_op(*map(jix.compact, a)).numpy(), *arrays)
        assert actual == expected, (operands, i)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_divide_errors_match_numpy(dtype):
    edges = _float_edges(dtype)
    for x in edges:
        for y in edges:
            _errors_match_numpy(lambda a, b: a / b, np.divide, x, y)


# numpy has only `floor_divide`, whose errors are those of the three.
@pytest.mark.parametrize("jix_fn", _QUOTIENTS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_rounded_divide_errors_match_numpy(jix_fn, dtype):
    edges = _float_edges(dtype)
    for x in edges:
        for y in edges:
            _errors_match_numpy(jix_fn, np.floor_divide, x, y)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_sqrt_errors_match_numpy(dtype):
    for x in _float_edges(dtype):
        _errors_match_numpy(jix.sqrt, np.sqrt, x)


def test_float_divide_modes():
    a = jix.compact(np.array([1.0, 0.0, 6.0]))
    b = jix.compact(np.array([0.0, 0.0, 3.0]))
    q = a / b
    with pytest.raises(FloatingPointError, match=MESSAGE):
        q.numpy()
    with jix.errstate(divide="ignore"), pytest.raises(FloatingPointError, match="invalid"):
        q.numpy()
    with jix.errstate(divide="warn", invalid="warn"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = q.numpy()
    assert sorted(str(x.message) for x in w) == [MESSAGE, "invalid operation encountered"]
    np.testing.assert_array_equal(result, [np.inf, np.nan, 2.0])


@pytest.mark.parametrize("dtype", ints)
def test_power_negative_exponent(dtype):
    a = np.array([2, 3, 4, -5], dtype=dtype)
    b = np.array([3, -1, 0, 3], dtype=dtype)
    q = jix.power(jix.compact(a), jix.compact(b))
    assert q.dtype == dtype
    # numpy raises a ValueError whatever its modes, jix follows `invalid`.
    with pytest.raises(ValueError, match="negative integer powers"):
        np.power(a, b)
    with pytest.raises(FloatingPointError, match="invalid operation encountered"):
        q.numpy()
    with jix.errstate(invalid="ignore"):
        np.testing.assert_array_equal(q.numpy(), [8, 0, 1, -125])
    with jix.errstate(invalid="warn"), pytest.warns(RuntimeWarning, match="invalid operation encountered"):
        q.numpy()


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
    with np.errstate(all="raise"), jix.errstate(all="raise"):
        np.testing.assert_array_equal((za + zb).numpy(), a + b)
        np.testing.assert_array_equal((za - zb).numpy(), a - b)
        np.testing.assert_array_equal((za * zb).numpy(), a * b)
        np.testing.assert_array_equal(jix.square(za).numpy(), np.square(a))
        np.testing.assert_array_equal(jix.absolute(za).numpy(), np.absolute(a))
        if np.issubdtype(dtype, np.signedinteger):
            np.testing.assert_array_equal(jix.negative(za).numpy(), np.negative(a))
