"""Tests that malformed arguments and misbehaving file objects raise instead of panicking.

Every case here used to abort or unwind out of Rust: a `PanicException`, a bare `SystemError`,
or - for the writer that over-reports - a panic inside a destructor that killed the process.
"""

import io

import numpy as np
import pytest

import jix


def _array():
    return jix.compact(np.arange(24, dtype=np.int32).reshape(2, 3, 4))


# ---------------------------------------------------------------------------
# Compression level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [128, 1000, 2**31 - 1, -129, -1000, -(2**31)])
def test_compression_level_out_of_range(level):
    with pytest.raises(RuntimeError, match="compression level"):
        jix.compact(np.arange(8, dtype=np.int32), params={"compression_level": level})


@pytest.mark.parametrize("level", [-5, 0, 1, 3, 19, 22, 127])
def test_compression_level_in_range(level):
    # Anything that fits an i8 is handed to the codec, negative "fast" levels included.
    a = np.arange(64, dtype=np.int32)
    arr = jix.compact(a, params={"compression_level": level})
    np.testing.assert_array_equal(arr.numpy(), a)


# ---------------------------------------------------------------------------
# numpy dtypes that do not fit jix's dtype representation
# ---------------------------------------------------------------------------


def test_dtype_itemsize_too_large():
    # itemsize 80000 does not fit the u16 itemsize field.
    dtype = np.dtype([("a", "i4", (20000,))])
    with pytest.raises(ValueError, match="itemsize"):
        jix.compact(np.zeros(2, dtype=dtype))


def test_cast_dtype_itemsize_too_large():
    dtype = np.dtype([("a", "i4", (20000,))])
    with pytest.raises(ValueError, match="itemsize"):
        jix.cast(_array(), dtype)


def test_cast_none():
    with pytest.raises(TypeError, match="None"):
        jix.cast(_array(), None)


# ---------------------------------------------------------------------------
# dims and shapes take an int, or a tuple or list of ints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [1, (0, 1), [0, 1], np.int64(1), (np.int32(0), 1)])
def test_dims_accept_int_tuple_list(dim):
    a = jix.compact(np.ones((2, 3)))
    np_dim = tuple(dim) if isinstance(dim, list) else dim  # numpy's axis rejects lists
    np.testing.assert_array_equal(jix.sum(a, dim=dim).numpy(), np.sum(np.ones((2, 3)), axis=np_dim))


@pytest.mark.parametrize("dim", [np.array([1]), range(2), "1"])
def test_dims_reject_other_sequences(dim):
    with pytest.raises(TypeError, match="tuple or list of ints"):
        jix.sum(jix.compact(np.ones((2, 3))), dim=dim)


def test_max_with_array_second_arg_is_not_read_as_dims():
    a = jix.compact(np.ones((2, 3)))
    with pytest.raises(TypeError, match="tuple or list of ints"):
        jix.max(a, jix.asarray(np.array([1])))


def test_shape_rejects_ndarray():
    with pytest.raises(TypeError, match="tuple or list of ints"):
        jix.reshape(jix.compact(np.ones((2, 3))), np.array([3, 2]))


# ---------------------------------------------------------------------------
# numpy layouts jix cannot view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [jix.asarray, jix.compact])
def test_negative_strides_raise_value_error(fn):
    with pytest.raises(ValueError, match="Negative strides"):
        fn(np.arange(4)[::-1])


# ---------------------------------------------------------------------------
# File-like objects that lie about how much they read or wrote
# ---------------------------------------------------------------------------


class _OverReader:
    """A reader whose `read()` returns more bytes than were asked for."""

    def __init__(self, data):
        self._buf = io.BytesIO(data)

    def read(self, size=-1):
        return self._buf.read(size) + b"\x00" * 100_000

    def seek(self, offset, whence=0):
        return self._buf.seek(offset, whence)

    def tell(self):
        return self._buf.tell()


class _OverWriter:
    """A writer whose `write()` reports more bytes written than it was given."""

    def __init__(self):
        self._buf = io.BytesIO()

    def write(self, data):
        return self._buf.write(data) + 100

    def seek(self, offset, whence=0):
        return self._buf.seek(offset, whence)

    def tell(self):
        return self._buf.tell()


def _serialized():
    buf = io.BytesIO()
    jix.write_array(_array(), buf)
    return buf.getvalue()


def test_reader_returning_too_many_bytes():
    with pytest.raises(RuntimeError, match="read"):
        jix.read_array(_OverReader(_serialized()))


def test_writer_reporting_too_many_bytes():
    with pytest.raises(RuntimeError, match="write"):
        jix.write_array(_array(), _OverWriter())
