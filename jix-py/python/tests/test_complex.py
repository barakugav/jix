"""Tests for `jix.real` / `jix.imag` and the matching `Array.real` / `Array.imag` getters."""

import numpy as np
import pytest

import jix


def test_real_complex64():
    a = jix.compact(np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex64))
    result = jix.real(a)
    assert result.dtype == np.float32
    assert result.shape == (3,)
    np.testing.assert_array_equal(result.numpy(), [1.0, 3.0, -5.0])


def test_real_complex128():
    a = jix.compact(np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex128))
    result = jix.real(a)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.numpy(), [1.0, 3.0, -5.0])


def test_imag_complex64():
    a = jix.compact(np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex64))
    result = jix.imag(a)
    assert result.dtype == np.float32
    assert result.shape == (3,)
    np.testing.assert_array_equal(result.numpy(), [2.0, -4.0, 6.0])


def test_imag_complex128():
    a = jix.compact(np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex128))
    result = jix.imag(a)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.numpy(), [2.0, -4.0, 6.0])


def test_real_preserves_shape_2d():
    np_a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    a = jix.compact(np_a)
    result = jix.real(a)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), np_a.real)


def test_imag_preserves_shape_2d():
    np_a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    a = jix.compact(np_a)
    result = jix.imag(a)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), np_a.imag)


def test_array_real_method_getter():
    a = jix.compact(np.array([1 + 2j, 3 - 4j], dtype=np.complex128))
    np.testing.assert_array_equal(a.real.numpy(), [1.0, 3.0])


def test_array_imag_method_getter():
    a = jix.compact(np.array([1 + 2j, 3 - 4j], dtype=np.complex128))
    np.testing.assert_array_equal(a.imag.numpy(), [2.0, -4.0])


def test_array_real_is_property_not_method():
    """real / imag are properties: accessed without parentheses (like np.ndarray.real)."""
    a = jix.compact(np.array([1 + 2j], dtype=np.complex64))
    # Property access yields a jix.Array directly.
    assert isinstance(a.real, jix.Array)
    assert isinstance(a.imag, jix.Array)


def test_real_accepts_python_list_of_complex():
    """Relaxed input: anything asarray() accepts."""
    result = jix.real([1 + 2j, 3 - 4j])
    np.testing.assert_array_equal(result.numpy(), [1.0, 3.0])


def test_real_rejects_float_input():
    """`real` dispatches only on complex dtypes; float inputs have no safe target."""
    a = jix.compact(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(Exception):
        _ = jix.real(a).numpy()


def test_imag_rejects_int_input():
    """`imag` dispatches only on complex dtypes; int inputs have no safe target."""
    a = jix.compact(np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(Exception):
        _ = jix.imag(a).numpy()
