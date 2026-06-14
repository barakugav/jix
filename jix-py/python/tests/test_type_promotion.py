"""
Dtype-promotion tests for unary and binary ops.

Promotion rules (`jix-py/src/ops/common/dtype_promote.rs`):
- Dispatch table is iterated left-to-right; the first impl where every operand
  passes ``CastKind::Safe`` rules is selected, and operands are cast to that impl.
- Safe same-rank cast: ok iff ``src_precision <= dst_precision``.
- ``Bool -> *`` always safe.
- ``UInt -> Int``: requires ``src_precision.higher() <= dst_precision``
  (so u8 fits in i16 but not i8).
- ``UInt/Int -> Float/Complex``: ``dst_precision == P8`` is a hard short-circuit
  (any int/uint precision can land in f64/c128); otherwise the same
  ``higher() <= dst_precision`` rule.
- Float -> Complex: same-rank precision rule.
- Anything narrowing (Int -> UInt, Float -> Int, Complex -> Float, ...) is not safe.

Scalars carry a precision when the user passed a typed numpy scalar
(``np.int64(5)``) and ``None`` when they passed an untyped Python value
(``5``, ``5.0``, ``True``). An ``None`` precision matches any same-rank impl.

This file exists in part to catch a regression where ``DtypeScalarKind::I64``
was tagged ``Precision::P4`` in ``operand.rs`` instead of ``P8``: an
``np.int64`` scalar combined with a smaller signed-int array dispatched to
``i32`` and silently truncated the scalar.
"""

import numpy as np
import pytest

import jix


# ---------------------------------------------------------------------------
# Bug regression: np.int64 scalar must be tagged P8.
#
# Before the fix, an `np.int64` scalar was tagged Precision::P4, so combining
# it with any small signed-int array dispatched to the i32 impl. The scalar's
# 64-bit value was then truncated to fit i32, producing wrong values silently.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arr_dtype", [np.int8, np.int16, np.int32, np.int64])
def test_int64_scalar_promotes_array_to_int64(arr_dtype):
    a = jix.compact([1, 2, 3], dtype=arr_dtype)
    result = a + np.int64(5)
    assert result.dtype == np.int64, f"int64 scalar + {arr_dtype.__name__} array: expected int64, got {result.dtype}"


def test_int64_scalar_value_not_truncated():
    """The whole reason the precision tag matters: the scalar value must survive.

    If int64 scalars were tagged P4, dispatch would pick i32 and the scalar
    would lose its upper 32 bits before the add ever happened.
    """
    a = jix.compact([0, 0, 0], dtype=np.int8)
    big = 2**40 + 7  # value outside i32 range
    result = a + np.int64(big)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(
        result.numpy(),
        np.array([big, big, big], dtype=np.int64),
    )


@pytest.mark.parametrize("arr_dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_uint64_scalar_promotes_array_to_uint64(arr_dtype):
    """Symmetric check for the analogous uint64 path."""
    a = jix.compact([1, 2, 3], dtype=arr_dtype)
    result = a + np.uint64(5)
    assert result.dtype == np.uint64, f"uint64 scalar + {arr_dtype.__name__} array: expected uint64, got {result.dtype}"


def test_uint64_scalar_value_not_truncated():
    a = jix.compact([0, 0, 0], dtype=np.uint8)
    big = 2**40 + 7  # value outside u32 range
    result = a + np.uint64(big)
    assert result.dtype == np.uint64
    np.testing.assert_array_equal(
        result.numpy(),
        np.array([big, big, big], dtype=np.uint64),
    )


# ---------------------------------------------------------------------------
# Unary ops: result dtype preserves input dtype when the dtype is in dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.complex64, np.complex128],
)
def test_unary_negative_preserves_dtype(dtype):
    a = jix.compact(np.array([1, 2, 3], dtype=dtype))
    result = jix.negative(a)
    assert result.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unary_sqrt_preserves_float_dtype(dtype):
    a = jix.compact(np.array([1.0, 4.0, 9.0], dtype=dtype))
    result = jix.sqrt(a)
    assert result.dtype == np.dtype(dtype)


# ---------------------------------------------------------------------------
# Unary ops: auto-cast when input dtype is not directly dispatched.
#
# `negative` dispatch table: [i8, i16, i32, i64, f16, f32, f64, c64, c128].
# uint inputs must auto-cast to the smallest signed type whose Safe rule
# accepts them (UInt P_n needs Int P_{>n}).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_dtype, expected_dtype",
    [
        (np.uint8, np.int16),  # UInt P1 -> Int P2
        (np.uint16, np.int32),  # UInt P2 -> Int P4
        (np.uint32, np.int64),  # UInt P4 -> Int P8
    ],
)
def test_unary_uint_auto_casts_to_signed(src_dtype, expected_dtype):
    a = jix.compact(np.array([1, 2, 3], dtype=src_dtype))
    result = jix.negative(a)
    assert result.dtype == np.dtype(expected_dtype)


def test_unary_uint64_falls_back_to_float64():
    """u64 has no safe signed-int target (UInt P8 has no higher precision),
    but the `dst_precision == P8` short-circuit lets UInt P8 land in f64."""
    a = jix.compact(np.array([1, 2, 3], dtype=np.uint64))
    result = jix.negative(a)
    assert result.dtype == np.float64


def test_unary_bool_auto_casts_to_i8():
    """bool casts to anything safely; first signed-int impl is i8."""
    a = jix.compact(np.array([True, False, True], dtype=np.bool_))
    result = jix.negative(a)
    assert result.dtype == np.int8


# ---------------------------------------------------------------------------
# Binary array-array: identity (same dtype in -> same dtype out).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_binary_same_dtype_preserves_dtype(dtype):
    a = jix.compact(np.array([1, 2, 3], dtype=dtype))
    b = jix.compact(np.array([4, 5, 6], dtype=dtype))
    assert jix.add(a, b).dtype == np.dtype(dtype)
    assert jix.multiply(a, b).dtype == np.dtype(dtype)


# ---------------------------------------------------------------------------
# Binary array-array: mixed dtype.
#
# Each row is (a_dtype, b_dtype, expected_result_dtype). Cases hand-derived
# from the Safe-cast rules above.
# ---------------------------------------------------------------------------


_MIXED_ARRAY_CASES = [
    # Same-rank uint widening.
    (np.uint8, np.uint16, np.uint16),
    (np.uint8, np.uint32, np.uint32),
    (np.uint8, np.uint64, np.uint64),
    (np.uint16, np.uint64, np.uint64),
    # Same-rank int widening.
    (np.int8, np.int16, np.int16),
    (np.int8, np.int32, np.int32),
    (np.int8, np.int64, np.int64),
    (np.int16, np.int64, np.int64),
    # UInt -> Int requires `src_precision.higher() <= dst_precision`.
    (np.uint8, np.int16, np.int16),  # P1.higher() = P2 <= P2
    (np.uint8, np.int32, np.int32),
    (np.uint16, np.int32, np.int32),  # P2.higher() = P4 <= P4
    (np.uint16, np.int64, np.int64),
    (np.uint32, np.int64, np.int64),  # P4.higher() = P8 <= P8
    # UInt P8 has no higher precision, but Float P8 short-circuit lets it
    # land in f64 (lossy for the top bit, but allowed by the rules).
    (np.uint64, np.int64, np.float64),
    # Int -> Float: dst P8 short-circuits.
    (np.int8, np.float16, np.float16),  # P1.higher() = P2 <= P2 also works
    (np.int8, np.float32, np.float32),
    (np.int16, np.float32, np.float32),  # P2.higher() = P4 <= P4
    (np.int32, np.float32, np.float64),  # P4.higher() = P8, P8 <= P4 false -> try f64
    (np.int64, np.float32, np.float64),  # only f64 (P8 short-circuit)
    (np.int32, np.float64, np.float64),
    (np.int64, np.float64, np.float64),
    # Same-rank float widening.
    (np.float16, np.float32, np.float32),
    (np.float32, np.float64, np.float64),
    # Float -> Complex same-rank precision rule.
    (np.float32, np.complex64, np.complex64),
    (np.float32, np.complex128, np.complex128),
    (np.float64, np.complex64, np.complex128),  # Float P8 -> Complex P4 fails -> c128
    (np.float64, np.complex128, np.complex128),
    # Complex widening.
    (np.complex64, np.complex128, np.complex128),
    # Int -> Complex: small ints land in c64 (higher() <= P4); large ints need c128.
    (np.int8, np.complex64, np.complex64),
    (np.int16, np.complex64, np.complex64),
    (np.int32, np.complex64, np.complex128),  # needs c128 via P8 short-circuit
    (np.int64, np.complex64, np.complex128),
    # Bool casts to anything; first impl that fits both is selected.
    (np.bool_, np.bool_, np.uint8),  # no bool impl, first impl that fits both bools is u8
    (np.bool_, np.uint8, np.uint8),
    (np.bool_, np.int32, np.int32),
    (np.bool_, np.float32, np.float32),
    (np.bool_, np.complex64, np.complex64),
]


@pytest.mark.parametrize("a_dtype, b_dtype, expected", _MIXED_ARRAY_CASES)
def test_binary_array_array_promotion(a_dtype, b_dtype, expected):
    a = jix.compact(np.array([1, 0, 1], dtype=a_dtype))
    b = jix.compact(np.array([1, 1, 0], dtype=b_dtype))
    result = jix.add(a, b)
    assert result.dtype == np.dtype(expected), (
        f"add({a_dtype.__name__}, {b_dtype.__name__}): got {result.dtype}, expected {np.dtype(expected)}"
    )


@pytest.mark.parametrize("a_dtype, b_dtype, expected", _MIXED_ARRAY_CASES)
def test_binary_array_array_promotion_is_commutative(a_dtype, b_dtype, expected):
    """Argument order shouldn't change the result dtype (within the cases above)."""
    a = jix.compact(np.array([1, 0, 1], dtype=a_dtype))
    b = jix.compact(np.array([1, 1, 0], dtype=b_dtype))
    assert jix.add(a, b).dtype == jix.add(b, a).dtype


# ---------------------------------------------------------------------------
# Binary array-scalar: typed numpy scalars (precision is fixed).
# ---------------------------------------------------------------------------


_TYPED_SCALAR_CASES = [
    # int8 array combined with each typed integer scalar.
    (np.int8, np.int8(1), np.int8),
    (np.int8, np.int16(1), np.int16),
    (np.int8, np.int32(1), np.int32),
    (np.int8, np.int64(1), np.int64),  # << the regression case
    # int32 array.
    (np.int32, np.int8(1), np.int32),  # scalar promotes up to array
    (np.int32, np.int32(1), np.int32),
    (np.int32, np.int64(1), np.int64),  # << the regression case
    # int64 array stays int64 no matter what int scalar.
    (np.int64, np.int8(1), np.int64),
    (np.int64, np.int64(1), np.int64),
    # uint variants.
    (np.uint8, np.uint8(1), np.uint8),
    (np.uint8, np.uint64(1), np.uint64),
    (np.uint32, np.uint64(1), np.uint64),
    # Mixing typed uint with int array follows the UInt -> Int rule.
    (np.int16, np.uint8(1), np.int16),  # u8 fits in i16 safely
    (np.int8, np.uint8(1), np.int16),  # neither u8 nor i8 holds both -> i16
    # Typed float scalars.
    (np.float32, np.float32(1.5), np.float32),
    (np.float32, np.float64(1.5), np.float64),
    (np.int32, np.float32(1.5), np.float64),  # int P4 -> f32 fails -> f64
    (np.int8, np.float32(1.5), np.float32),  # int P1.higher() = P2 <= P4 -> f32
    (np.int64, np.float64(1.5), np.float64),
    # Typed complex scalars.
    (np.float32, np.complex64(1 + 2j), np.complex64),
    (np.float64, np.complex64(1 + 2j), np.complex128),
    (np.complex64, np.complex128(1 + 2j), np.complex128),
]


@pytest.mark.parametrize("arr_dtype, scalar, expected", _TYPED_SCALAR_CASES)
def test_binary_array_typed_scalar_promotion(arr_dtype, scalar, expected):
    a = jix.compact(np.array([1, 2, 3], dtype=arr_dtype))
    result = jix.add(a, scalar)
    assert result.dtype == np.dtype(expected), (
        f"add({arr_dtype.__name__}, {type(scalar).__name__}): got {result.dtype}, expected {np.dtype(expected)}"
    )


@pytest.mark.parametrize("arr_dtype, scalar, expected", _TYPED_SCALAR_CASES)
def test_binary_typed_scalar_array_promotion(arr_dtype, scalar, expected):
    """Scalar-first order via jix.add free-function to bypass numpy operator promotion."""
    a = jix.compact(np.array([1, 2, 3], dtype=arr_dtype))
    result = jix.add(scalar, a)
    assert result.dtype == np.dtype(expected)


# ---------------------------------------------------------------------------
# Binary array-scalar: untyped Python scalars (precision == None).
#
# An untyped scalar's rank fixes its lane (bool / Int / Float / Complex)
# but it matches every same-rank impl, so dispatch effectively picks the
# array's impl (or first impl that fits when ranks differ).
# ---------------------------------------------------------------------------


_PY_SCALAR_CASES = [
    # Python bool: rank Bool, casts to anything.
    (np.int32, True, np.int32),
    (np.float64, True, np.float64),
    # Python int: rank Int, no precision.
    (np.int8, 5, np.int8),
    (np.int32, 5, np.int32),
    (np.int64, 5, np.int64),
    # Python int with uint array: Int rank > UInt rank -> can't land in UInt
    # impls (src_rank > dst_rank fails Safe). First int impl that fits the
    # array's u8 input is i16 (u8 -> i16 needs P2 <= P2: ok).
    (np.uint8, 5, np.int16),
    # Python int with float array: untyped scalar's precision is None, so the
    # rule short-circuits via `src_precision.is_none()` -- the scalar matches
    # whichever float impl the array already fits into.
    (np.float32, 5, np.float32),
    (np.float64, 5, np.float64),
    # Python float: rank Float.
    (np.float32, 1.5, np.float32),
    (np.float64, 1.5, np.float64),
    (np.int32, 1.5, np.float64),  # int P4 -> f32 fails (P8 <= P4 false), -> f64
    # Python complex.
    (np.complex64, 1 + 2j, np.complex64),
    (np.complex128, 1 + 2j, np.complex128),
    (np.float32, 1 + 2j, np.complex64),  # f32 -> c64 same precision ok
]


@pytest.mark.parametrize("arr_dtype, scalar, expected", _PY_SCALAR_CASES)
def test_binary_array_python_scalar_promotion(arr_dtype, scalar, expected):
    a = jix.compact(np.array([1, 2, 3], dtype=arr_dtype))
    result = jix.add(a, scalar)
    assert result.dtype == np.dtype(expected), (
        f"add({arr_dtype.__name__}, py {type(scalar).__name__}={scalar!r}): "
        f"got {result.dtype}, expected {np.dtype(expected)}"
    )


# ---------------------------------------------------------------------------
# Values, not just dtypes, must be preserved after promotion.
# ---------------------------------------------------------------------------


def test_promotion_preserves_values_int8_plus_int64():
    a_np = np.array([1, -2, 3], dtype=np.int8)
    b_np = np.array([10, 20, 30], dtype=np.int64)
    result = jix.add(jix.compact(a_np), jix.compact(b_np))
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result.numpy(), a_np.astype(np.int64) + b_np)


def test_promotion_preserves_values_uint8_plus_int16():
    a_np = np.array([255, 100, 0], dtype=np.uint8)
    b_np = np.array([-1, 200, 32767], dtype=np.int16)
    result = jix.add(jix.compact(a_np), jix.compact(b_np))
    assert result.dtype == np.int16
    np.testing.assert_array_equal(result.numpy(), a_np.astype(np.int16) + b_np)


def test_promotion_preserves_values_int32_plus_float32():
    """int32 + float32 promotes to f64; the int32 value must survive the cast."""
    a_np = np.array([1_000_000, 2_000_000, 3_000_000], dtype=np.int32)
    b_np = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    result = jix.add(jix.compact(a_np), jix.compact(b_np))
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.numpy(), a_np.astype(np.float64) + b_np.astype(np.float64))


# ---------------------------------------------------------------------------
# Promotion + broadcasting compose correctly.
# ---------------------------------------------------------------------------


def test_promotion_with_broadcasting():
    a_np = np.array([1, 2, 3], dtype=np.int8).reshape(3, 1)
    b_np = np.array([10, 20, 30, 40], dtype=np.int64).reshape(1, 4)
    result = jix.add(jix.compact(a_np), jix.compact(b_np))
    assert result.dtype == np.int64
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result.numpy(), a_np.astype(np.int64) + b_np)


# ---------------------------------------------------------------------------
# Per-op promotion: the chosen impl depends on each op's dispatch table.
# ---------------------------------------------------------------------------


def test_divide_promotes_integers_to_float():
    """divide's dispatch is [f16, f32, f64, c64, c128] -- no int impls.
    Integer arrays must auto-cast into a float impl (f64 via P8 short-circuit)."""
    a = jix.compact(np.array([10, 20, 30], dtype=np.int32))
    b = jix.compact(np.array([2, 4, 5], dtype=np.int32))
    result = jix.divide(a, b)
    assert result.dtype == np.float64


def test_floor_divide_rejects_floats():
    """floor_divide's dispatch is integers only; float inputs have no safe target."""
    a = jix.compact(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = jix.compact(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(Exception):
        _ = jix.floor_divide(a, b).numpy()


def test_power_promotes_small_int_to_float32():
    """power's dispatch is [f32, f64]. int8 lands in f32 via the higher() rule
    (P1.higher() = P2 <= P4)."""
    a = jix.compact(np.array([2, 3, 4], dtype=np.int8))
    b = jix.compact(np.array([2, 2, 2], dtype=np.int8))
    result = jix.power(a, b)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.numpy(), np.array([4.0, 9.0, 16.0], dtype=np.float32))


def test_power_promotes_int64_to_float64():
    """int64 -> f32 fails (no higher precision); the P8 short-circuit lands it in f64."""
    a = jix.compact(np.array([2, 3, 4], dtype=np.int64))
    b = jix.compact(np.array([2, 2, 2], dtype=np.int64))
    result = jix.power(a, b)
    assert result.dtype == np.float64


def test_power_rejects_complex():
    """power has no complex impl; Complex -> Float is never safe."""
    a = jix.compact(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
    b = jix.compact(np.array([1 + 0j, 2 + 0j], dtype=np.complex64))
    with pytest.raises(Exception):
        _ = jix.power(a, b).numpy()
