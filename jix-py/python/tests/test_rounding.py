"""
Tests for element-wise rounding ops. Mirrors the test block in
jix/src/ops/elementwise/rounding.rs.
"""

import numpy as np
import pytest
from tests_util import check_op1_concrete

import jix


def test_floor_concrete():
    # Edge inputs: exact ints, +/- fractions, .5 tie; float32 + float64 paths;
    # a non-default block shape to cross block boundaries.
    values = [[-2.5, -0.5, 0.0], [0.5, 2.5, 3.9]]
    check_op1_concrete(jix.floor, np.floor, [(np.float32, values, [1, 2]), (np.float64, values, [1, 2])])


def test_ceil_concrete():
    # Same edge inputs as floor: exact ints, +/- fractions, .5 tie.
    values = [[-2.5, -0.5, 0.0], [0.5, 2.5, 3.9]]
    check_op1_concrete(jix.ceil, np.ceil, [(np.float32, values), (np.float64, values)])


def _round_half_away_from_zero(a: np.ndarray) -> np.ndarray:
    """Reference for jix.round: ties round away from zero (Rust f32/f64::round semantics)."""
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def test_round_concrete():
    # numpy.round uses banker's rounding (half-to-even); jix.round rounds half away from zero.
    # .5 ties round away from zero: 0.5 -> 1.0, -0.5 -> -1.0, 2.5 -> 3.0, -2.5 -> -3.0.
    values = [[-2.5, -0.5, 0.0], [0.5, 2.5, 1.4]]
    check_op1_concrete(jix.round, _round_half_away_from_zero, [(np.float32, values), (np.float64, values)])


@pytest.mark.parametrize("op", [jix.floor, jix.ceil, jix.round])
def test_rounding_returns_ints_as_is(op):
    for dtype in [np.int8, np.int64, np.uint8, np.uint64]:
        za = jix.compact(np.array([0, 5], dtype=dtype))
        assert op(za) is za
    assert op(7).numpy() == 7
    for dtype in [np.bool_, np.complex64]:
        with pytest.raises(TypeError):
            op(jix.compact(np.zeros(2, dtype=dtype)))
