"""
Tests for the `out=` argument of `jix.Array.numpy()`.

`numpy(out=...)` decodes straight into a caller-owned NumPy array instead of
allocating a fresh one. Internally it always takes the *push* mode of the core
`to_ndarray_buf`, describing the destination as a `StridedBuf` over the NumPy
array's own memory and strides - so the destination does not have to be
C-contiguous. These tests cover the layouts that must work (C, Fortran, gapped
and transposed views), that nothing outside the destination is written, and the
layouts that must be rejected (reversed, broadcast, read-only).
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests_util import shape_strategy, sub_range_strategy

import jix

# Hosts are filled with this before a read, so any byte the read should not have
# touched is still recognizable afterwards. The source data starts at 1.
UNTOUCHED = 0

LAYOUTS = ["c", "f", "gapped", "transposed"]


def make_dst(shape, dtype, layout):
    """
    Build a destination of `shape` in the given layout.

    Returns `(host, dst)`: `dst` is what gets passed as `out=`, and `host` is the
    buffer it views, pre-filled with `UNTOUCHED` so the caller can check that only
    `dst`'s own elements were written.
    """
    shape = tuple(shape)
    if layout == "c":
        host = np.full(shape, UNTOUCHED, dtype=dtype)
        return host, host
    if layout == "f":
        host = np.full(shape, UNTOUCHED, dtype=dtype, order="F")
        return host, host
    if layout == "gapped":
        # Take every other element along the last axis out of a wider host, so the
        # destination has a gap between consecutive elements in its fastest axis.
        host = np.full(shape[:-1] + (shape[-1] * 2 + 1,), UNTOUCHED, dtype=dtype)
        return host, host[..., 1::2][..., : shape[-1]]
    if layout == "transposed":
        host = np.full(shape[::-1], UNTOUCHED, dtype=dtype)
        return host, host.T
    raise AssertionError(f"unknown layout {layout}")


def assert_only_dst_written(host, dst, expected, layout):
    """The destination holds `expected`, and nothing else in its host buffer was written."""
    np.testing.assert_array_equal(dst, expected)
    # Rebuild the host from scratch, writing `expected` through the same kind of view.
    # Anything the read touched outside the destination shows up as a mismatch here.
    ref_host, ref_dst = make_dst(dst.shape, host.dtype, layout)
    ref_dst[...] = expected
    np.testing.assert_array_equal(host, ref_host)


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_out_is_returned_as_is():
    np_a = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    dst = np.empty((3, 4), dtype=np.int32)
    assert a.numpy(out=dst) is dst
    np.testing.assert_array_equal(dst, np_a)


def test_out_is_keyword_only():
    a = jix.compact(np.arange(1, 5, dtype=np.int32))
    with pytest.raises(TypeError):
        a.numpy(None, np.empty((4,), dtype=np.int32))


def test_out_matches_the_allocating_path():
    np_a = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)
    a = jix.compact(np_a)
    dst = np.empty((2, 3, 4), dtype=np.float64)
    a.numpy(out=dst)
    np.testing.assert_array_equal(dst, a.numpy())


def test_out_can_be_reused_across_reads():
    np_a = np.arange(1, 17, dtype=np.int32).reshape(4, 4)
    a = jix.compact(np_a)
    dst = np.empty((2, 4), dtype=np.int32)
    for row in range(0, 4, 2):
        a.numpy(slice(row, row + 2), out=dst)
        np.testing.assert_array_equal(dst, np_a[row : row + 2])


# ---------------------------------------------------------------------------
# Destination layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [np.int32, np.float64, np.uint8])
def test_out_layouts(layout, dtype):
    shape = (5, 7)
    np_a = np.arange(1, 36, dtype=dtype).reshape(shape)
    a = jix.compact(np_a)
    host, dst = make_dst(shape, dtype, layout)
    assert a.numpy(out=dst) is dst
    assert_only_dst_written(host, dst, np_a, layout)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_out_layouts_over_a_split_read(layout):
    # A small read-size budget makes the core split the request into many pieces, each
    # written into the destination at the destination's own strides.
    shape = (128, 96)
    np_a = np.arange(1, 128 * 96 + 1, dtype=np.float32).reshape(shape)
    a = jix.compact(np_a, params={"block_shape": (16, 16), "read_size": (1024, 4096)})
    host, dst = make_dst(shape, np.float32, layout)
    a.numpy(out=dst)
    assert_only_dst_written(host, dst, np_a, layout)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_out_sub_region(layout):
    np_a = np.arange(1, 61, dtype=np.int64).reshape(6, 10)
    a = jix.compact(np_a)
    expected = np_a[1:5, 2:9]
    host, dst = make_dst(expected.shape, np.int64, layout)
    a.numpy((slice(1, 5), slice(2, 9)), out=dst)
    assert_only_dst_written(host, dst, expected, layout)


def test_out_into_a_slice_of_a_bigger_array():
    np_a = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
    a = jix.compact(np_a)
    big = np.zeros((3, 10), dtype=np.int32)
    a.numpy(out=big[:, 3:7])
    np.testing.assert_array_equal(big[:, 3:7], np_a)
    assert (big[:, :3] == 0).all()
    assert (big[:, 7:] == 0).all()


# ---------------------------------------------------------------------------
# Dropped axes - an integer index item removes its axis from the destination,
# but the core still wants one stride per array dimension.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_out_with_dropped_leading_axis(layout):
    np_a = np.arange(1, 25, dtype=np.int32).reshape(2, 3, 4)
    a = jix.compact(np_a)
    expected = np_a[1]
    host, dst = make_dst(expected.shape, np.int32, layout)
    a.numpy(1, out=dst)
    assert_only_dst_written(host, dst, expected, layout)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_out_with_dropped_middle_axis(layout):
    np_a = np.arange(1, 25, dtype=np.int32).reshape(2, 3, 4)
    a = jix.compact(np_a)
    expected = np_a[:, 2]
    host, dst = make_dst(expected.shape, np.int32, layout)
    a.numpy((slice(None), 2), out=dst)
    assert_only_dst_written(host, dst, expected, layout)


def test_out_with_all_axes_dropped():
    np_a = np.arange(1, 7, dtype=np.int32).reshape(2, 3)
    a = jix.compact(np_a)
    dst = np.empty((), dtype=np.int32)
    a.numpy((1, 2), out=dst)
    assert dst[()] == np_a[1, 2]


# ---------------------------------------------------------------------------
# Lazy pipelines push into `out` too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_out_from_a_lazy_op(layout):
    np_a = np.arange(1, 21, dtype=np.float32).reshape(4, 5)
    a = jix.compact(np_a)
    host, dst = make_dst((4, 5), np.float32, layout)
    (a * 2 + 1).numpy(out=dst)
    assert_only_dst_written(host, dst, np_a * 2 + 1, layout)


def test_out_from_a_reduction():
    np_a = np.arange(1, 21, dtype=np.float64).reshape(4, 5)
    a = jix.compact(np_a)
    dst = np.empty((5,), dtype=np.float64)
    a.sum(axis=0).numpy(out=dst)
    np.testing.assert_allclose(dst, np_a.sum(axis=0))


# ---------------------------------------------------------------------------
# Empty selections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,index", [((0, 4), None), ((3, 4), slice(1, 1))])
def test_out_empty(shape, index):
    np_a = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    a = jix.compact(np_a)
    expected = np_a if index is None else np_a[index]
    dst = np.empty(expected.shape, dtype=np.int32)
    assert a.numpy(index, out=dst) is dst
    assert dst.shape == expected.shape


# ---------------------------------------------------------------------------
# Rejected destinations
# ---------------------------------------------------------------------------


def test_out_rejects_wrong_dtype():
    a = jix.compact(np.arange(1, 13, dtype=np.int32).reshape(3, 4))
    with pytest.raises(ValueError, match="dtype"):
        a.numpy(out=np.empty((3, 4), dtype=np.float32))


@pytest.mark.parametrize("shape", [(4, 3), (3, 5), (3,), (3, 4, 1)])
def test_out_rejects_wrong_shape(shape):
    a = jix.compact(np.arange(1, 13, dtype=np.int32).reshape(3, 4))
    with pytest.raises(ValueError, match="shape"):
        a.numpy(out=np.empty(shape, dtype=np.int32))


def test_out_shape_must_match_the_index_not_the_array():
    a = jix.compact(np.arange(1, 13, dtype=np.int32).reshape(3, 4))
    with pytest.raises(ValueError, match="shape"):
        a.numpy(0, out=np.empty((3, 4), dtype=np.int32))


def test_out_rejects_read_only():
    a = jix.compact(np.arange(1, 13, dtype=np.int32).reshape(3, 4))
    dst = np.empty((3, 4), dtype=np.int32)
    dst.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        a.numpy(out=dst)


@pytest.mark.parametrize("reverse", [np.s_[::-1, :], np.s_[:, ::-1]])
def test_out_rejects_reversed_views(reverse):
    a = jix.compact(np.arange(1, 13, dtype=np.int32).reshape(3, 4))
    with pytest.raises(ValueError, match="stride"):
        a.numpy(out=np.empty((3, 4), dtype=np.int32)[reverse])


def test_out_rejects_broadcast_views():
    a = jix.compact(np.arange(1, 13, dtype=np.int32).reshape(3, 4))
    # `broadcast_to` hands back a read-only view; make it writeable so the zero stride,
    # not the read-only flag, is what the check has to catch.
    dst = np.lib.stride_tricks.as_strided(np.zeros(4, dtype=np.int32), shape=(3, 4), strides=(0, 4))
    assert dst.flags.writeable
    with pytest.raises(ValueError, match="stride"):
        a.numpy(out=dst)


def test_out_rejects_non_arrays():
    a = jix.compact(np.arange(1, 13, dtype=np.int32).reshape(3, 4))
    with pytest.raises(TypeError):
        a.numpy(out=[[0] * 4] * 3)


# ---------------------------------------------------------------------------
# Property test: whatever the shape, sub-range and destination layout, `out=`
# must produce exactly what the allocating path produces.
# ---------------------------------------------------------------------------


@given(
    shape=shape_strategy(),
    layout=st.sampled_from(LAYOUTS),
    data=st.data(),
)
@settings(max_examples=50, deadline=None)
def test_out_matches_allocating_path_property(shape, layout, data):
    shape = tuple(shape)
    np_a = np.arange(1, int(np.prod(shape)) + 1, dtype=np.int32).reshape(shape)
    a = jix.compact(np_a)

    if 0 in shape:
        index = None
    else:
        index = data.draw(sub_range_strategy(shape), label="index")
    expected = a.numpy(index)

    host, dst = make_dst(expected.shape, np.int32, layout)
    assert a.numpy(index, out=dst) is dst
    assert_only_dst_written(host, dst, expected, layout)
