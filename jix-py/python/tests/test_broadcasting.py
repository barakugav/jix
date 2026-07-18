"""
Tests for numpy-compatible broadcasting in binary ops.

Broadcasting is applied before dispatch: jix expands any 1-sized dimension and
prepends missing leading dimensions, exactly as NumPy does.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject

import jix

# ---------------------------------------------------------------------------
# Strategy: pairs of shapes that are numpy-broadcastable
# ---------------------------------------------------------------------------


@st.composite
def broadcastable_shapes_pair(draw, max_ndim: int = 6, max_dim: int = 8):
    """
    Generate two shapes that are broadcastable per numpy rules.

    Approach: build per-dimension choices aligned from the right. For each
    position the two dims are either equal, or one of them is 1 (which numpy
    broadcasts). One array may have fewer dimensions (the missing leading
    dims are implicitly 1).
    """
    ndim_a = draw(st.integers(1, max_ndim))
    ndim_b = draw(st.integers(1, max_ndim))
    ndim = max(ndim_a, ndim_b)

    dims_a = []
    dims_b = []

    for pos in range(ndim):
        # Position 0 is the leftmost; align from the right
        a_has = pos >= ndim - ndim_a
        b_has = pos >= ndim - ndim_b

        if a_has and b_has:
            choice = draw(st.integers(0, 2))
            d = draw(st.integers(1, max_dim))
            if choice == 0:
                dims_a.append(d)
                dims_b.append(d)
            elif choice == 1:
                dims_a.append(1)
                dims_b.append(d)
            else:
                dims_a.append(d)
                dims_b.append(1)
        elif a_has:
            dims_a.append(draw(st.integers(1, max_dim)))
        else:
            dims_b.append(draw(st.integers(1, max_dim)))

    return tuple(dims_a), tuple(dims_b)


@st.composite
def broadcast_int32_arrays(draw):
    """(np_a, za, np_b, zb) with broadcastable shapes and int32 dtype."""
    shape_a, shape_b = draw(broadcastable_shapes_pair())
    np_a = draw(
        st.builds(
            lambda s: np.arange(1, np.prod(s) + 1, dtype=np.int32).reshape(s),
            st.just(shape_a),
        )
    )
    np_b = draw(
        st.builds(
            lambda s: np.arange(1, np.prod(s) + 1, dtype=np.int32).reshape(s),
            st.just(shape_b),
        )
    )
    za = jix.compact(np_a)
    zb = jix.compact(np_b)
    return (np_a, za), (np_b, zb)


# ---------------------------------------------------------------------------
# Deterministic shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        # 1-D broadcasting
        ((5,), (5,)),
        ((1,), (5,)),
        ((5,), (1,)),
        # 2-D broadcasting
        ((3, 4), (3, 4)),
        ((3, 1), (3, 4)),
        ((1, 4), (3, 4)),
        ((3, 4), (3, 1)),
        ((3, 4), (1, 4)),
        ((1, 1), (3, 4)),
        # Different ndim
        ((3,), (2, 3)),
        ((4,), (3, 4)),
        ((1, 4), (3, 4)),
        ((3, 4), (4,)),
        # Classic 2-D outer-product pattern
        ((4, 1), (1, 3)),  # -> (4, 3)
        # 3-D
        ((3, 1, 4), (3, 5, 1)),  # -> (3, 5, 4)
        ((1, 3, 1), (2, 1, 4)),  # -> (2, 3, 4)
        ((1, 5, 1), (3, 1, 4)),  # NumPy docs classic
        # numpy docs image example
        ((256, 256, 3), (3,)),
        # The classic NumPy docs example
        ((8, 1, 6, 1), (7, 1, 5)),  # -> (8, 7, 6, 5)
    ],
)
def test_broadcast_add_shapes(shape_a, shape_b):
    """Element-wise add result has the numpy-broadcast shape and correct values."""
    np_a = np.arange(1, np.prod(shape_a) + 1, dtype=np.int32).reshape(shape_a)
    np_b = np.arange(1, np.prod(shape_b) + 1, dtype=np.int32).reshape(shape_b)
    za = jix.compact(np_a)
    zb = jix.compact(np_b)

    result = jix.add(za, zb)
    expected = np_a + np_b

    assert result.shape == expected.shape, f"shape: {result.shape} != {expected.shape}"
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((3,), (4,)),
        ((3, 4), (5, 4)),
        ((3, 4), (3, 5)),
        ((2, 3), (4, 5)),
    ],
)
def test_broadcast_incompatible_raises(shape_a, shape_b):
    """Incompatible shapes raise an error."""
    np_a = np.ones(shape_a, dtype=np.int32)
    np_b = np.ones(shape_b, dtype=np.int32)
    za = jix.compact(np_a)
    zb = jix.compact(np_b)

    with pytest.raises(Exception):
        _ = jix.add(za, zb).numpy()


# A couple of representative broadcastable shape pairs, reused from
# test_broadcast_add_shapes, for the concrete (non-property) tests below.
_BROADCAST_CONCRETE_SHAPE_PAIRS = [
    ((3, 1), (3, 4)),  # 2-D broadcast along one axis
    ((4, 1), (1, 3)),  # classic outer-product pattern -> (4, 3)
    ((8, 1, 6, 1), (7, 1, 5)),  # NumPy docs classic multi-dim example -> (8, 7, 6, 5)
]


# ---------------------------------------------------------------------------
# Property-based broadcasting tests
# ---------------------------------------------------------------------------


@given(st.data())
def test_broadcast_add_property(data: DataObject):
    """Broadcasting add matches numpy for arbitrary broadcastable int32 arrays."""
    (np_a, za), (np_b, zb) = data.draw(broadcast_int32_arrays(), label="arrays")

    result = jix.add(za, zb)
    expected = np_a + np_b

    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.numpy(), expected)


@given(st.data())
def test_broadcast_equal_property(data: DataObject):
    """Broadcasting equal matches numpy for arbitrary broadcastable int32 arrays."""
    (np_a, za), (np_b, zb) = data.draw(broadcast_int32_arrays(), label="arrays")

    result = jix.equal(za, zb)
    expected = np_a == np_b

    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.numpy(), expected)


# Concrete broadcasting tests across a handful of ops and dtypes, all sharing the same
# loop body over _BROADCAST_CONCRETE_SHAPE_PAIRS - collapsed into one parametrized test.
@pytest.mark.parametrize(
    "jix_op, np_op, dtype",
    [
        (jix.multiply, lambda a, b: a * b, np.int32),
        (jix.subtract, lambda a, b: a - b, np.int32),
        (jix.greater, lambda a, b: a > b, np.int32),
        (jix.maximum, np.maximum, np.int32),
        (jix.add, lambda a, b: a + b, np.float64),
    ],
)
def test_broadcast_concrete(jix_op, np_op, dtype):
    """Broadcasting matches numpy for a few representative broadcastable shapes."""
    for shape_a, shape_b in _BROADCAST_CONCRETE_SHAPE_PAIRS:
        np_a = np.arange(1, np.prod(shape_a) + 1, dtype=dtype).reshape(shape_a)
        np_b = np.arange(1, np.prod(shape_b) + 1, dtype=dtype).reshape(shape_b)
        za = jix.compact(np_a)
        zb = jix.compact(np_b)

        result = jix_op(za, zb)
        expected = np_op(np_a, np_b)

        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.numpy(), expected)


# ---------------------------------------------------------------------------
# Broadcasting with Python scalars
# ---------------------------------------------------------------------------


def test_broadcast_python_int_scalar():
    """Python int scalar operand broadcasts to the array shape."""
    np_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    za = jix.compact(np_a)

    result = jix.add(za, 10)
    np.testing.assert_array_equal(result.numpy(), np_a + 10)


def test_broadcast_python_float_scalar():
    """Python float scalar operand broadcasts to the array shape."""
    np_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    za = jix.compact(np_a)

    result = jix.add(za, 0.5)
    np.testing.assert_array_equal(result.numpy(), np_a + 0.5)


def test_broadcast_numpy_scalar():
    """numpy scalar operand broadcasts to the array shape."""
    np_a = np.array([1, 2, 3], dtype=np.int32)
    za = jix.compact(np_a)

    result = jix.add(za, np.int32(100))
    np.testing.assert_array_equal(result.numpy(), np_a + 100)


# ---------------------------------------------------------------------------
# Sub-range indexing after broadcasting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_a, shape_b, idx",
    [
        ((3, 1), (1, 4), (slice(1, 3), slice(2, 4))),
        ((1, 5), (3, 5), (slice(0, 2), slice(1, 4))),
        ((4,), (2, 4), (slice(0, 1), slice(1, 3))),
    ],
)
def test_broadcast_sub_index(shape_a, shape_b, idx):
    """Slicing a broadcast result gives the same values as numpy."""
    np_a = np.arange(1, np.prod(shape_a) + 1, dtype=np.int32).reshape(shape_a)
    np_b = np.arange(1, np.prod(shape_b) + 1, dtype=np.int32).reshape(shape_b)
    za = jix.compact(np_a)
    zb = jix.compact(np_b)

    result = jix.add(za, zb)
    expected = np_a + np_b

    np.testing.assert_array_equal(result[idx], expected[idx])
