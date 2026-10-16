"""
Arithmetic error handling: integer division by zero under jix.seterr / jix.errstate.
Mirrors the div-by-zero tests in jix/src/ops/op2.rs.
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

MESSAGE = "divide by zero encountered in Div"


def _trunc_div_ref(np_a, np_b):
    """`a // b` truncating toward zero, `0` for a zero divisor, `MIN // -1` wrapping."""
    out = np.zeros_like(np_a)
    for idx in np.ndindex(np_a.shape):
        x, y = int(np_a[idx]), int(np_b[idx])
        if y != 0:
            q = abs(x) // abs(y)
            q = q if (x < 0) == (y < 0) else -q
            out[idx] = np.array(q).astype(np_a.dtype)  # wraps MIN // -1
    return out


def _arrays(dtype):
    a = np.arange(1000).astype(dtype) % 100
    b = (np.arange(1000) % 7 + 1).astype(dtype)
    b[300] = 0
    b[-1] = 0
    return a, b


@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_zero_warns_by_default(dtype):
    a, b = _arrays(dtype)
    za, zb = jix.compact(a), jix.compact(b)
    assert jix.geterr() == {"divide": "warn"}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = (za // zb).numpy()
    np.testing.assert_array_equal(result, _trunc_div_ref(a, b))
    # One warning per read, however many elements hit it.
    assert [(x.category, str(x.message)) for x in w] == [(RuntimeWarning, MESSAGE)]


@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_zero_raise(dtype):
    a, b = _arrays(dtype)
    za, zb = jix.compact(a), jix.compact(b)
    with jix.errstate(divide="raise"):
        with pytest.raises(FloatingPointError, match=MESSAGE):
            (za // zb).numpy()
        # Only the elements read count.
        np.testing.assert_array_equal((za // zb)[:300], _trunc_div_ref(a[:300], b[:300]))


@pytest.mark.parametrize("dtype", ints + uints)
def test_divide_by_zero_ignore(dtype):
    a, b = _arrays(dtype)
    za, zb = jix.compact(a), jix.compact(b)
    with jix.errstate(divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal((za // zb).numpy(), _trunc_div_ref(a, b))


@pytest.mark.parametrize("dtype", ints)
def test_min_floor_divide_minus_one_wraps(dtype):
    info = np.iinfo(dtype)
    a = np.array([info.min, 7, info.max], dtype=dtype)
    b = np.array([-1, -1, -1], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = (jix.compact(a) // jix.compact(b)).numpy()
    np.testing.assert_array_equal(result, np.array([info.min, -7, -info.max], dtype=dtype))


def test_compact_and_write_array_report(tmp_path):
    q = jix.compact(np.array([6, 7], dtype=np.int32)) // jix.compact(np.array([2, 0], dtype=np.int32))
    with pytest.warns(RuntimeWarning, match=MESSAGE):
        np.testing.assert_array_equal(jix.compact(q).numpy(), [3, 0])
    with pytest.warns(RuntimeWarning, match=MESSAGE):
        jix.write_array(q, tmp_path / "q.jix")
    np.testing.assert_array_equal(jix.read_array(tmp_path / "q.jix").numpy(), [3, 0])

    with jix.errstate(divide="raise"):
        with pytest.raises(FloatingPointError, match=MESSAGE):
            jix.compact(q)
        with pytest.raises(FloatingPointError, match=MESSAGE):
            jix.write_array(q, tmp_path / "q2.jix")


def test_modes_apply_at_read_time():
    # Operations are lazy: the modes in effect when the array is read apply.
    za = jix.compact(np.array([6, 7], dtype=np.int32))
    zb = jix.compact(np.array([2, 0], dtype=np.int32))
    with jix.errstate(divide="raise"):
        q = za // zb
    with pytest.warns(RuntimeWarning, match=MESSAGE):
        q.numpy()
    with jix.errstate(divide="raise"), pytest.raises(FloatingPointError):
        q.numpy()


def test_seterr_geterr():
    assert jix.geterr() == {"divide": "warn"}
    old = jix.seterr(divide="raise")
    try:
        assert old == {"divide": "warn"}
        assert jix.geterr() == {"divide": "raise"}
        assert jix.seterr(all="ignore") == {"divide": "raise"}
        assert jix.geterr() == {"divide": "ignore"}
        # An explicit kind takes precedence over `all`.
        jix.seterr(all="raise", divide="warn")
        assert jix.geterr() == {"divide": "warn"}
    finally:
        jix.seterr(**old)
    assert jix.geterr() == {"divide": "warn"}

    for bad in ["call", "print", "log", "Raise", ""]:
        with pytest.raises(ValueError, match="invalid error mode"):
            jix.seterr(divide=bad)
        with pytest.raises(ValueError, match="invalid error mode"):
            jix.errstate(all=bad)
    assert jix.geterr() == {"divide": "warn"}


def test_errstate_restores():
    with jix.errstate(divide="raise"):
        assert jix.geterr() == {"divide": "raise"}
        with jix.errstate(all="ignore"):
            assert jix.geterr() == {"divide": "ignore"}
        assert jix.geterr() == {"divide": "raise"}
    assert jix.geterr() == {"divide": "warn"}

    with pytest.raises(KeyError), jix.errstate(divide="ignore"):
        raise KeyError
    assert jix.geterr() == {"divide": "warn"}


def test_modes_are_per_thread():
    seen = []
    with jix.errstate(divide="raise"):
        thread = threading.Thread(target=lambda: seen.append(jix.geterr()))
        thread.start()
        thread.join()
    assert seen == [{"divide": "warn"}]


@pytest.mark.parametrize("dtype", ints + uints)
@given(st.data())
def test_divide_by_zero_random(dtype, data: DataObject):
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
    expected = _trunc_div_ref(np_a, np_b)
    with jix.errstate(divide="ignore"):
        np.testing.assert_array_equal((za // zb).numpy(), expected)
    with jix.errstate(divide="raise"):
        if (np_b == 0).any():
            with pytest.raises(FloatingPointError):
                (za // zb).numpy()
        else:
            np.testing.assert_array_equal((za // zb).numpy(), expected)
