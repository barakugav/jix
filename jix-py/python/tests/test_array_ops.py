"""
Tests for compact, plain, item, bool, read_array/write_array and the binary operator protocol.
"""

import operator
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jix

# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


def test_compact_produces_equal_array():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    za = jix.compact(arr)
    copied = jix.compact(za)
    assert copied.shape == za.shape
    assert copied.dtype == za.dtype
    np.testing.assert_array_equal(copied.numpy(), arr)


def test_compact_is_independent():
    arr = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(arr)
    copied = jix.compact(za)
    np.testing.assert_array_equal(copied.numpy(), za.numpy())


# ---------------------------------------------------------------------------
# plain
# ---------------------------------------------------------------------------


def test_plain_materializes_to_equal_array():
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    za = jix.compact(arr)
    # Materializing returns an equal array.
    plain = za.plain()
    assert plain.shape == za.shape
    assert plain.dtype == za.dtype
    np.testing.assert_array_equal(plain.numpy(), arr)
    # It also materializes a lazy view.
    np.testing.assert_array_equal((za + 1).plain().numpy(), arr + 1)


# ---------------------------------------------------------------------------
# item
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, expected_type",
    [
        (np.array([[1, 2], [3, 4]], dtype=np.int32), int),
        (np.array([1.5, 2.25], dtype=np.float64), float),
        (np.array([1 + 2j, 3 - 1j], dtype=np.complex64), complex),
    ],
)
def test_item_of_full_reduction(data, expected_type):
    result = jix.sum(jix.compact(data)).item()
    assert type(result) is expected_type
    assert result == data.sum().item()


def test_item_bool():
    assert jix.all(jix.compact(np.array([True, False]))).item() is False


@pytest.mark.parametrize("shape", [(1,), (2, 3)])
def test_item_requires_zero_dim(shape):
    with pytest.raises(ValueError, match="zero-dimensional"):
        jix.compact(np.zeros(shape)).item()


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(), (1,), (3,), (2, 3), (0,)])
def test_bool_always_raises(shape):
    za = jix.compact(np.ones(shape, dtype=np.int32))
    with pytest.raises(ValueError, match="truth value"):
        bool(za)


def test_bool_of_comparison_raises():
    a = jix.compact(np.array([1, 2, 3]))
    b = jix.compact(np.array([1, 2, 4]))
    with pytest.raises(ValueError, match="truth value"):
        if a == b:
            pass
    assert not (a == b).all().item()
    assert (a == b).any().item()


# ---------------------------------------------------------------------------
# read_array / write_array
# ---------------------------------------------------------------------------


def test_write_and_read_array_roundtrip():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path)
        assert loaded.shape == za.shape
        assert loaded.dtype == za.dtype
        np.testing.assert_array_equal(loaded.numpy(), arr)


def test_write_and_read_integer_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path)
        np.testing.assert_array_equal(loaded.numpy(), arr)


def test_write_and_read_bool_array():
    arr = np.array([True, False, True, True], dtype=bool)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path)
        np.testing.assert_array_equal(loaded.numpy(), arr)


def test_non_canonical_bool_bytes_are_true():
    # A NumPy bool array can hold any byte, any non-zero one is True
    arr = np.array([0, 1, 2, 255], dtype=np.uint8).view(bool)
    a = jix.asarray(arr)
    expected = np.array([False, True, True, True])
    np.testing.assert_array_equal((~a).numpy(), ~expected)
    np.testing.assert_array_equal((a == True).numpy(), expected)
    assert a.sum().item() == 3


def test_read_array_mmap():
    arr = np.arange(50, dtype=np.float64)
    za = jix.compact(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jix"
        jix.write_array(za, path)
        loaded = jix.read_array(path, mmap=True)
        np.testing.assert_array_equal(loaded.numpy(), arr)


def test_binary_ops_return_not_implemented_for_operands_that_opt_out():
    a = jix.compact(np.array([1, 2]))
    names = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "and", "or", "xor", "lshift", "rshift"]
    # Each reflected op returns its own name, so the result shows Python fell back to it
    reflected = {f"__r{name}__": lambda self, other, name=name: f"r{name}" for name in names}
    OptsOut = type("OptsOut", (), {"__array_ufunc__": None, "__gt__": lambda self, other: "gt", **reflected})

    for name in names:
        op = getattr(operator, f"{name}_" if name in ("and", "or") else name)
        assert op(a, OptsOut()) == f"r{name}"
    assert (a < OptsOut()) == "gt"
    assert (a == OptsOut()) is False
    # The reflected ops of jix opt out too, OptsOut has no __sub__
    with pytest.raises(TypeError):
        OptsOut() - a
    np.testing.assert_array_equal((a + [1, 1]).numpy(), [2, 3])
    np.testing.assert_array_equal((a + np.array([1, 1])).numpy(), [2, 3])
