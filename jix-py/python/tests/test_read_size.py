"""Tests for read_size kwarg accepting scalar int or (min, max) tuple."""

import io

import numpy as np
import pytest

import jix


def test_read_size_accepts_scalar():
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    arr = jix.compact(a, params={"read_size": 4096})
    np.testing.assert_array_equal(arr.numpy(), a)


def test_read_size_accepts_tuple():
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    arr = jix.compact(a, params={"read_size": (4096, 65536)})
    np.testing.assert_array_equal(arr.numpy(), a)


@pytest.mark.parametrize(
    "bad,exc",
    [
        # Wrong arity is a value error (the elements parse, the length is wrong).
        ((1, 2, 3), ValueError),
        ((), ValueError),
        # Wrong type / domain fails extraction and surfaces as a type error. Only tuples count
        # as sequences, so lists are rejected too.
        ([4096, 65536], TypeError),
        ([], TypeError),
        (4096.0, TypeError),
        (-1, TypeError),
    ],
)
def test_read_size_rejects_bad_input(bad, exc):
    # Only a non-negative int or a 2-element tuple of non-negative ints is accepted;
    # anything else must be rejected rather than silently coerced.
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    with pytest.raises(exc):
        jix.compact(a, params={"read_size": bad})


def test_read_size_on_asarray_and_read_array(tmp_path):
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    np.testing.assert_array_equal(jix.asarray(a, params={"read_size": 16}).numpy(), a)

    path = tmp_path / "a.jix"
    jix.write_array(jix.compact(a), path)
    params = {"read_size": 16, "block_shape_fixed_dims": [True, False]}
    np.testing.assert_array_equal(jix.read_array(path, params=params).numpy(), a)


@pytest.mark.parametrize(
    "fn,params",
    [
        (jix.asarray, {"codec": "zstd"}),
        (jix.asarray, {"block_size": 1024}),
        (lambda a, params: jix.write_array(a, io.BytesIO(), params=params), {"read_size": 16}),
        (lambda a, params: jix.read_array(io.BytesIO(), params=params), {"codec": "zstd"}),
    ],
)
def test_unused_params_are_rejected(fn, params):
    a = np.arange(64, dtype=np.int32).reshape(8, 8)
    with pytest.raises(TypeError, match="Unexpected array params"):
        fn(a, params=params)
