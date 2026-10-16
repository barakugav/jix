"""
Arithmetic error handling: integer division by zero under jix.errstate.
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

MESSAGE = "divide by zero encountered"
DEFAULT_MODES = {"divide": "raise", "over": "warn", "under": "warn", "invalid": "raise"}


def _modes(**kinds):
    return {**DEFAULT_MODES, **kinds}


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
def test_divide_by_zero_warn(dtype):
    a, b = _arrays(dtype)
    za, zb = jix.compact(a), jix.compact(b)
    with jix.errstate(divide="warn"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = (za // zb).numpy()
    np.testing.assert_array_equal(result, _trunc_div_ref(a, b))
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
    np.testing.assert_array_equal((za // zb)[:300], _trunc_div_ref(a[:300], b[:300]))
    with jix.errstate(divide="raise"), pytest.raises(FloatingPointError, match=MESSAGE):
        (za // zb).numpy()


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
