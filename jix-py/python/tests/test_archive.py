"""Tests for read_array and write_array with both file paths and file-like objects."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import jix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_array(values, dtype=np.float32):
    return jix.compact(np.array(values, dtype=dtype))


# ---------------------------------------------------------------------------
# write_array — file path
# ---------------------------------------------------------------------------


def test_write_array_path_creates_file():
    a = make_array([1.0, 2.0, 3.0, 4.0])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.jix"
        jix.write_array(a, p)
        assert p.exists()
        assert p.stat().st_size > 0


def test_write_array_path_string():
    a = make_array([1.0, 2.0, 3.0])
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "out.jix")
        jix.write_array(a, p)
        assert Path(p).exists()


def test_write_array_path_fails_if_exists():
    a = make_array([1.0, 2.0])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.jix"
        jix.write_array(a, p)
        with pytest.raises(Exception):
            jix.write_array(a, p)  # must not exist by default


def test_write_array_path_append():
    a = make_array([1.0, 2.0, 3.0])
    b = make_array([10, 20, 30], dtype=np.int32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "packed.jix"
        jix.write_array(a, p)
        offset = p.stat().st_size
        jix.write_array(b, p, append=True)

        a2 = jix.read_array(p)
        b2 = jix.read_array(p, offset=offset)
        np.testing.assert_array_equal(a2.numpy(), a.numpy())
        np.testing.assert_array_equal(b2.numpy(), b.numpy())


# ---------------------------------------------------------------------------
# write_array — file-like object
# ---------------------------------------------------------------------------


def test_write_array_bytesio():
    a = make_array([[1.0, 2.0], [3.0, 4.0]])
    buf = io.BytesIO()
    jix.write_array(a, buf)
    assert buf.tell() > 0


def test_write_array_open_file(tmp_path):
    a = make_array([5, 6, 7, 8], dtype=np.int64)
    p = tmp_path / "out.jix"
    with open(p, "wb") as f:
        jix.write_array(a, f)
    assert p.stat().st_size > 0


def test_write_array_bytesio_roundtrip():
    a = make_array([1.0, 2.0, 3.0, 4.0])
    buf = io.BytesIO()
    jix.write_array(a, buf)
    buf.seek(0)
    a2 = jix.read_array(buf)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_write_array_bad_object():
    a = make_array([1.0])
    with pytest.raises(TypeError):
        jix.write_array(a, 42)


# ---------------------------------------------------------------------------
# read_array — file path
# ---------------------------------------------------------------------------


def test_read_array_path(tmp_path):
    a = make_array([1.0, 2.0, 3.0, 4.0])
    p = tmp_path / "data.jix"
    jix.write_array(a, p)
    a2 = jix.read_array(p)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_read_array_path_string(tmp_path):
    a = make_array([10, 20, 30], dtype=np.int32)
    p = tmp_path / "data.jix"
    jix.write_array(a, p)
    a2 = jix.read_array(str(p))
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_read_array_path_mmap(tmp_path):
    a = make_array([[1.0, 2.0], [3.0, 4.0]])
    p = tmp_path / "data.jix"
    jix.write_array(a, p)
    a2 = jix.read_array(p, mmap=True)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_read_array_path_packed(tmp_path):
    a = make_array([1.0, 2.0, 3.0])
    b = make_array([10, 20, 30], dtype=np.int32)
    p = tmp_path / "packed.jix"
    jix.write_array(a, p)
    offset = p.stat().st_size
    jix.write_array(b, p, append=True)
    total = p.stat().st_size

    a2 = jix.read_array(p, offset=0, len=offset)
    b2 = jix.read_array(p, offset=offset, len=total - offset)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())
    np.testing.assert_array_equal(b2.numpy(), b.numpy())


def test_read_array_path_bad_offset(tmp_path):
    a = make_array([1.0])
    p = tmp_path / "data.jix"
    jix.write_array(a, p)
    with pytest.raises(Exception):
        jix.read_array(p, offset=10**9)


# ---------------------------------------------------------------------------
# read_array — file-like object
# ---------------------------------------------------------------------------


def test_read_array_bytesio():
    a = make_array([1.0, 2.0, 3.0, 4.0])
    buf = io.BytesIO()
    jix.write_array(a, buf)
    buf.seek(0)
    a2 = jix.read_array(buf)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_read_array_open_file(tmp_path):
    a = make_array([5, 6, 7, 8], dtype=np.int64)
    p = tmp_path / "data.jix"
    jix.write_array(a, p)
    with open(p, "rb") as f:
        a2 = jix.read_array(f)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_read_array_bytesio_2d():
    a = make_array([[1.0, 2.0], [3.0, 4.0]])
    buf = io.BytesIO()
    jix.write_array(a, buf)
    buf.seek(0)
    a2 = jix.read_array(buf)
    assert a2.shape == (2, 2)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_read_array_reader_at_offset():
    """Reader positioned partway into a packed buffer reads the correct array."""
    a = make_array([1.0, 2.0, 3.0])
    b = make_array([10, 20, 30], dtype=np.int32)
    buf = io.BytesIO()
    jix.write_array(a, buf)
    offset = buf.tell()
    jix.write_array(b, buf)

    buf.seek(offset)
    b2 = jix.read_array(buf)
    np.testing.assert_array_equal(b2.numpy(), b.numpy())


def test_read_array_reader_mmap_raises():
    buf = io.BytesIO()
    a = make_array([1.0])
    jix.write_array(a, buf)
    buf.seek(0)
    with pytest.raises(ValueError, match="mmap"):
        jix.read_array(buf, mmap=True)


def test_read_array_bad_object():
    with pytest.raises(TypeError):
        jix.read_array(42)


def test_read_array_object_missing_read():
    class NoRead:
        def seek(self, offset, whence=0):
            return 0

        def tell(self):
            return 0

    with pytest.raises(TypeError):
        jix.read_array(NoRead())


def test_read_array_object_missing_seek():
    class NoSeek:
        def read(self, n):
            return b""

        def tell(self):
            return 0

    with pytest.raises(TypeError):
        jix.read_array(NoSeek())


# ---------------------------------------------------------------------------
# Round-trips across dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int32, np.int64, np.uint8],
)
def test_roundtrip_bytesio_dtype(dtype):
    arr = np.arange(12, dtype=dtype).reshape(3, 4)
    a = jix.compact(arr)
    buf = io.BytesIO()
    jix.write_array(a, buf)
    buf.seek(0)
    a2 = jix.read_array(buf)
    np.testing.assert_array_equal(a2.numpy(), arr)


# ---------------------------------------------------------------------------
# write_array + read_array interop: path ↔ reader, writer ↔ path
# ---------------------------------------------------------------------------


def test_write_path_read_reader(tmp_path):
    a = make_array([1.0, 2.0, 3.0])
    p = tmp_path / "data.jix"
    jix.write_array(a, p)
    with open(p, "rb") as f:
        a2 = jix.read_array(f)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())


def test_write_writer_read_path(tmp_path):
    a = make_array([7, 8, 9], dtype=np.int32)
    p = tmp_path / "data.jix"
    with open(p, "wb") as f:
        jix.write_array(a, f)
    a2 = jix.read_array(p)
    np.testing.assert_array_equal(a2.numpy(), a.numpy())
