"""
Tests for element-wise trigonometric ops. Mirrors the test block in
jix/src/ops/elementwise/trig.rs.
"""

import numpy as np
from tests_util import check_op1_concrete

import jix


def test_sin_concrete():
    # 0, +/-pi/2, pi, and a couple of interior op_safe_strategy values.
    values = [0.0, np.pi / 2, np.pi, -np.pi / 2, 1.0, -1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.sin, np.sin, [(dtype, values)], rtol=rtol)


def test_cos_concrete():
    values = [0.0, np.pi / 2, np.pi, -np.pi / 2, 1.0, -1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.cos, np.cos, [(dtype, values)], rtol=rtol)


def test_tan_concrete():
    # Avoid drawing exactly at the +/-pi/2 asymptote; op_safe_strategy never hits it either.
    values = [0.0, np.pi / 4, -np.pi / 4, 1.0, -1.0, 10.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.tan, np.tan, [(dtype, values)], rtol=rtol)


def test_asin_concrete():
    # Domain [-1, 1]: both endpoints plus interior points.
    # f16 excluded: asin is in [-pi/2, pi/2], not representable precisely in f16.
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.asin, np.arcsin, [(dtype, values)], rtol=rtol)


def test_acos_concrete():
    # Domain [-1, 1]: both endpoints plus interior points.
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.acos, np.arccos, [(dtype, values)], rtol=rtol)


def test_atan_concrete():
    values = [0.0, 1.0, -1.0, 100.0, -100.0]
    for dtype, rtol in ((np.float32, 1e-5), (np.float64, 1e-12)):
        check_op1_concrete(jix.atan, np.arctan, [(dtype, values)], rtol=rtol)
