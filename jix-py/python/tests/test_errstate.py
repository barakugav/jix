"""
Arithmetic error handling under jix.errstate: wrapping integer arithmetic. Mirrors the tests in
jix/src/ops/op1.rs and jix/src/ops/op2.rs.
"""

import numpy as np
import pytest
from tests_util import ints, uints

import jix


@pytest.mark.parametrize("dtype", ints + uints)
def test_arithmetic_wraps_like_numpy(dtype):
    info = np.iinfo(dtype)
    edges = np.array([info.min, info.min + 1, 0, 1, info.max // 2 + 1, info.max], dtype=dtype)
    a = np.resize(np.repeat(edges, len(edges)), 1000)
    b = np.resize(np.tile(edges, len(edges)), 1000)
    za, zb = jix.compact(a), jix.compact(b)
    with np.errstate(all="raise"), jix.errstate(all="raise"):
        np.testing.assert_array_equal((za + zb).numpy(), a + b)
        np.testing.assert_array_equal((za - zb).numpy(), a - b)
        np.testing.assert_array_equal((za * zb).numpy(), a * b)
        np.testing.assert_array_equal(jix.square(za).numpy(), np.square(a))
        np.testing.assert_array_equal(jix.absolute(za).numpy(), np.absolute(a))
        if np.issubdtype(dtype, np.signedinteger):
            np.testing.assert_array_equal(jix.negative(za).numpy(), np.negative(a))
