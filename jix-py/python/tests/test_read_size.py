"""Tests for read_size kwarg accepting scalar int or (min, max) tuple."""

import numpy as np
import pytest

import jix


def test_read_size_accepts_scalar():
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    arr = jix.compact(a, params={"read_size": 4096})
    np.testing.assert_array_equal(arr.numpy(), a)


@pytest.mark.parametrize("seq", [(4096, 65536), [4096, 65536]])
def test_read_size_accepts_sequence(seq):
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    arr = jix.compact(a, params={"read_size": seq})
    np.testing.assert_array_equal(arr.numpy(), a)


@pytest.mark.parametrize(
    "bad,exc",
    [
        # Wrong arity is a value error (the elements parse, the length is wrong).
        ((1, 2, 3), ValueError),
        ((), ValueError),
        ([], ValueError),
        # Wrong element type / domain fails extraction and surfaces as a type error.
        (4096.0, TypeError),
        (-1, TypeError),
    ],
)
def test_read_size_rejects_bad_input(bad, exc):
    # Only a non-negative int or a 2-element sequence of non-negative ints is accepted;
    # anything else must be rejected rather than silently coerced.
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    with pytest.raises(exc):
        jix.compact(a, params={"read_size": bad})
