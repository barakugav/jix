"""
Index handling for `Array.numpy()` and `Array.__getitem__`.

`.numpy()`, `array[...]` and `jix.slice()` all share `parse_basic_index`, but they consume
its result differently: `slice()` turns the dropped-axis flags into a `RemoveAxis`, while
`.numpy()` indexes them per axis to build the output shape and the destination strides.
`test_slice.py` covers the first consumer against `__getitem__`; this file pins all three
against NumPy itself, so a parser change cannot break them in lockstep and go unnoticed.
"""

import numpy as np
import pytest

import jix


def arange(shape):
    return np.arange(1, int(np.prod(shape)) + 1, dtype=np.int32).reshape(shape)


# (shape, index) pairs, covering every index form against a few ranks. Ellipsis cases are
# spread over ranks on purpose: an ellipsis expands to a different number of axes each
# time, which is exactly what a per-item (rather than per-axis) parser gets wrong.
CASES = [
    # rank 1
    ((5,), 0),
    ((5,), -1),
    ((5,), slice(1, 4)),
    ((5,), Ellipsis),
    ((5,), (Ellipsis,)),
    ((5,), (Ellipsis, 2)),
    ((5,), (2, Ellipsis)),
    # rank 2
    ((3, 4), 0),
    ((3, 4), -1),
    ((3, 4), (1, 2)),
    ((3, 4), (slice(None), 2)),
    ((3, 4), (1, slice(1, 3))),
    ((3, 4), Ellipsis),
    ((3, 4), (Ellipsis, 2)),
    ((3, 4), (0, Ellipsis)),
    ((3, 4), (Ellipsis, 0, 2)),
    ((3, 4), (0, Ellipsis, 2)),
    ((3, 4), (Ellipsis, slice(1, 3))),
    # rank 3 - an ellipsis here fills two axes at once
    ((2, 3, 4), 1),
    ((2, 3, 4), (1, 2)),
    ((2, 3, 4), (1, 2, 3)),
    ((2, 3, 4), (slice(None), 1)),
    ((2, 3, 4), (slice(None), slice(None), 2)),
    ((2, 3, 4), Ellipsis),
    ((2, 3, 4), (Ellipsis, 1)),
    ((2, 3, 4), (Ellipsis, -1)),
    ((2, 3, 4), (0, Ellipsis)),
    ((2, 3, 4), (0, Ellipsis, 1)),
    ((2, 3, 4), (Ellipsis, 1, 2)),
    ((2, 3, 4), (1, Ellipsis, 2)),
    ((2, 3, 4), (slice(0, 1), Ellipsis)),
    ((2, 3, 4), (Ellipsis, slice(1, 3))),
    # rank 4 - three-axis ellipsis fill
    ((2, 3, 4, 5), (Ellipsis, 2)),
    ((2, 3, 4, 5), (1, Ellipsis)),
    ((2, 3, 4, 5), (1, Ellipsis, 2)),
    ((2, 3, 4, 5), (1, 2, Ellipsis)),
    # a size-1 axis makes a mis-targeted axis removal succeed instead of erroring, so the
    # damage shows up only as a wrong shape
    ((2, 1, 3, 4), (Ellipsis, 2)),
    ((2, 1, 3, 4), (Ellipsis, 1, 2)),
    ((1, 1, 4), (Ellipsis, 2)),
]


def case_id(case):
    shape, index = case
    return f"{'x'.join(map(str, shape))}-{index!r}".replace(" ", "")


@pytest.mark.parametrize("shape,index", CASES, ids=[case_id(c) for c in CASES])
def test_getitem_matches_numpy(shape, index):
    np_a = arange(shape)
    a = jix.compact(np_a)
    expected = np_a[index]

    for label, got in [
        ("numpy(index)", a.numpy(index)),
        ("array[index]", a[index]),
        ("jix.slice()", jix.slice(a, index).numpy()),
    ]:
        assert got.shape == expected.shape, f"{label}: shape {got.shape} != {expected.shape}"
        np.testing.assert_array_equal(got, expected, err_msg=label)


@pytest.mark.parametrize("shape,index", CASES, ids=[case_id(c) for c in CASES])
def test_getitem_into_out_matches_numpy(shape, index):
    # The `out=` path rebuilds one stride per array dimension from the destination's own
    # strides, re-inserting the axes the index dropped - so it reads the same flags from a
    # different direction.
    np_a = arange(shape)
    a = jix.compact(np_a)
    expected = np_a[index]

    dst = np.empty(expected.shape, dtype=np.int32)
    assert a.numpy(index, out=dst) is dst
    np.testing.assert_array_equal(dst, expected)


def test_numpy_no_index_reads_the_whole_array():
    np_a = arange((2, 3, 4))
    a = jix.compact(np_a)
    np.testing.assert_array_equal(a.numpy(), np_a)
    np.testing.assert_array_equal(a.numpy(None), np_a)


def test_omitted_trailing_axes_are_full_slices():
    np_a = arange((2, 3, 4))
    a = jix.compact(np_a)
    np.testing.assert_array_equal(a.numpy((1,)), np_a[1])
    np.testing.assert_array_equal(a.numpy((1, 2)), np_a[1, 2])


def test_slice_without_an_integer_index_adds_no_remove_axis():
    # `drop_axes` carries one flag per axis, so "nothing dropped" is all-false rather than
    # empty - checking emptiness instead wrapped every slice in a no-op `RemoveAxis`.
    a = jix.compact(arange((2, 3, 4)))
    assert "RemoveAxis" not in str(jix.slice(a, (slice(0, 2), slice(0, 2))))
    assert "RemoveAxis" in str(jix.slice(a, (0, slice(0, 2))))


def test_out_with_a_reversed_length_one_axis_is_accepted():
    # An axis of extent 1 is never stepped, so NumPy handing back a negative stride for it
    # is harmless and must not be rejected.
    np_a = arange((1, 4))
    a = jix.compact(np_a)
    dst = np.empty((1, 4), dtype=np.int32)[::-1]
    assert dst.strides[0] < 0
    a.numpy(out=dst)
    np.testing.assert_array_equal(dst, np_a)
