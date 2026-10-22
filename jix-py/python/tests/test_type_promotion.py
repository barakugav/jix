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
(``5``, ``5.0``, ``True``). An ``None`` precision matches any same-rank impl, and
an untyped Python int also matches unsigned impls (it has no signedness). Its
value must fit the dtype dispatch picks, otherwise ``OverflowError`` is raised.
If every operand is untyped they all default to 64-bit; otherwise, if no float
or complex operand is typed, the untyped Python floats/complexes default to
32-bit (``i8 + 1.5`` is f32).

This file exists in part to catch a regression where ``ScalarKind::I64``
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
# `negative` dispatch table: uN (negated as iN), i8, i16, i32, i64, f16, f32,
# f64, c64, c128. bool auto-casts to the first one its Safe rule accepts.
# ---------------------------------------------------------------------------


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
    # Python int with uint array: an untyped int has no signedness, so it fills
    # the unsigned impl of the array's own dtype.
    (np.uint8, 5, np.uint8),
    (np.uint16, 5, np.uint16),
    (np.uint32, 5, np.uint32),
    (np.uint64, 5, np.uint64),
    # Python int with bool array: bool -> u8 is safe and the int fills u8 too.
    (np.bool_, 5, np.uint8),
    # Python int with float array: untyped scalar's precision is None, so the
    # rule short-circuits via `src_precision.is_none()` -- the scalar matches
    # whichever float impl the array already fits into.
    (np.float32, 5, np.float32),
    (np.float64, 5, np.float64),
    # Python float: rank Float. With a typed float operand it stays untyped and
    # follows that operand's precision.
    (np.float16, 1.5, np.float16),
    (np.float32, 1.5, np.float32),
    (np.float64, 1.5, np.float64),
    # Without a typed float/complex operand it defaults to 32-bit (f32), not the
    # narrowest float that fits the array.
    (np.bool_, 1.5, np.float32),
    (np.uint8, 1.5, np.float32),
    (np.int8, 1.5, np.float32),
    (np.int16, 1.5, np.float32),
    (np.int32, 1.5, np.float64),  # int P4 -> f32 fails (P8 <= P4 false), -> f64
    # Python complex: same rules, the 32-bit default is Complex<f32>.
    (np.complex64, 1 + 2j, np.complex64),
    (np.complex128, 1 + 2j, np.complex128),
    (np.float16, 1 + 2j, np.complex64),  # f16 -> c64 ok
    (np.float32, 1 + 2j, np.complex64),  # f32 -> c64 same precision ok
    (np.float64, 1 + 2j, np.complex128),
    (np.int8, 1 + 2j, np.complex64),
    (np.int32, 1 + 2j, np.complex128),
]


@pytest.mark.parametrize("arr_dtype, scalar, expected", _PY_SCALAR_CASES)
def test_binary_array_python_scalar_promotion(arr_dtype, scalar, expected):
    a = jix.compact(np.array([1, 2, 3], dtype=arr_dtype))
    result = jix.add(a, scalar)
    assert result.dtype == np.dtype(expected), (
        f"add({arr_dtype.__name__}, py {type(scalar).__name__}={scalar!r}): "
        f"got {result.dtype}, expected {np.dtype(expected)}"
    )


def test_python_float_defaults_to_f32_next_to_typed_int_scalar():
    """A typed numpy int scalar is not a float operand, so the Python float still defaults to f32."""
    result = jix.add(np.int8(1), 1.5)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.numpy(), np.float32(2.5))


def test_python_scalars_only_default_to_64_bit():
    """With only untyped operands the 64-bit default applies first, so the f32 default never does."""
    assert jix.add(1, 1.5).dtype == np.float64
    assert jix.add(1, 1j).dtype == np.complex128
    assert jix.sqrt(2.0).dtype == np.float64


@pytest.mark.parametrize("b", [True, np.bool_(True)])
def test_bool_scalar_has_fixed_dtype(b):
    """A bool scalar (Python or numpy) has a single width, so it is typed like a bool array and
    never triggers the all-untyped 64-bit default: True + 1 behaves like bool_arr + 1."""
    result = jix.add(b, 1)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result.numpy(), 2)
    assert jix.add(b, 1.5).dtype == np.float32
    assert jix.add(b, True).dtype == np.uint8


def test_python_float_default_keeps_f32_precision():
    """In f16, 1.0001 rounds to 1.0 and 1e10 overflows to inf. f32 holds both apart."""
    a = jix.compact(np.array([1, 0], dtype=np.int8))
    np.testing.assert_array_equal(jix.equal(a, 1.0001).numpy(), [False, False])
    np.testing.assert_array_equal(jix.add(a, 1e10).numpy(), np.array([1e10, 1e10], dtype=np.float32))


# ---------------------------------------------------------------------------
# An untyped Python int adopts the integer dtype dispatch picked for it, so its
# value must fit that dtype. Otherwise an OverflowError is raised rather than
# silently wrapping the scalar (like numpy). No special cases for comparisons
# or clamp: they raise too.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, arr_dtype, scalars",
    [
        (jix.add, np.int8, (1000,)),
        (jix.add, np.int8, (-1000,)),
        (jix.add, np.int16, (70000,)),
        (jix.add, np.int32, (2**40,)),
        (jix.add, np.int8, (2**63,)),
        (jix.add, np.int64, (2**63,)),
        (jix.add, np.bool_, (1000,)),
        (jix.subtract, np.int8, (1000,)),
        (jix.multiply, np.int16, (-70000,)),
        (jix.equal, np.int8, (1000,)),
        (jix.less, np.int32, (2**40,)),
        (jix.maximum, np.int8, (1000,)),
        (jix.bitwise_or, np.int8, (0x100,)),
        (jix.clamp, np.int8, (-1000, 1000)),
    ],
)
def test_python_int_out_of_bounds_raises(op, arr_dtype, scalars):
    a = jix.compact(np.array([0, 1, 1], dtype=arr_dtype))
    with pytest.raises(OverflowError, match="out of bounds"):
        op(a, *scalars)


@pytest.mark.parametrize(
    "arr_dtype, scalar",
    [
        (np.int8, 127),
        (np.int8, -128),
        (np.int16, -32768),
        (np.int32, 2**31 - 1),
        (np.int64, 2**63 - 1),
        (np.int64, -(2**63)),
    ],
)
def test_python_int_at_bounds_is_kept(arr_dtype, scalar):
    a_np = np.array([0, 0], dtype=arr_dtype)
    result = jix.add(jix.compact(a_np), scalar)
    assert result.dtype == np.dtype(arr_dtype)
    np.testing.assert_array_equal(result.numpy(), a_np + scalar)


def test_python_int_keeps_uint64_exact():
    """u64 + Python int stays u64. It used to promote to f64 and lose precision above 2**53."""
    a_np = np.array([2**53, 2**64 - 2], dtype=np.uint64)
    result = jix.add(jix.compact(a_np), 1)
    assert result.dtype == np.uint64
    np.testing.assert_array_equal(result.numpy(), a_np + np.uint64(1))


@pytest.mark.parametrize(
    "op, arr_dtype, scalar, expected",
    [
        (jix.bitwise_and, np.uint64, 1, [0, 1, 0]),
        (jix.bitwise_and, np.uint8, 0x0F, [0, 1, 2]),
        (jix.bitwise_left_shift, np.uint8, 1, [0, 2, 4]),
        (jix.bitwise_rotate_left, np.uint8, 1, [0, 2, 4]),
        (jix.maximum, np.uint16, 1, [1, 1, 2]),
        (jix.equal, np.uint32, 1, [False, True, False]),
    ],
)
def test_python_int_fills_unsigned_impl(op, arr_dtype, scalar, expected):
    a = jix.compact(np.array([0, 1, 2], dtype=arr_dtype))
    result = op(a, scalar)
    if result.dtype != np.bool_:
        assert result.dtype == np.dtype(arr_dtype)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize(
    "op, scalar",
    [(jix.add, -1), (jix.add, 256), (jix.subtract, -1), (jix.equal, 300), (jix.less, -1)],
)
def test_python_int_out_of_unsigned_bounds_raises(op, scalar):
    """The int lands in the u8 impl, so a value outside [0, 255] raises - for comparisons too."""
    a = jix.compact(np.array([0, 1, 2], dtype=np.uint8))
    with pytest.raises(OverflowError, match="out of bounds"):
        op(a, scalar)


def test_typed_scalar_out_of_bounds_promotes_instead_of_raising():
    """The bounds check applies only to untyped Python ints. A typed numpy scalar carries its own
    dtype, which promotion respects."""
    a = jix.compact(np.array([0, 1], dtype=np.int8))
    result = jix.add(a, np.int16(1000))
    assert result.dtype == np.int16
    np.testing.assert_array_equal(result.numpy(), [1000, 1001])


@pytest.mark.parametrize("arr_dtype", [np.uint8, np.int64, np.uint64, np.float16, np.float64])
@pytest.mark.parametrize("scalar", [2**63, 2**64, -(2**63) - 1, 10**400])
def test_python_int_outside_int64_raises(arr_dtype, scalar):
    """numpy raises too, but only with integer arrays, and below 2**64 not with uint64 ones."""
    a = jix.compact(np.array([0, 1], dtype=arr_dtype))
    with pytest.raises(OverflowError, match="out of bounds"):
        jix.add(a, scalar)
    with pytest.raises(OverflowError, match="out of bounds"):
        jix.add(scalar, 1)


def test_python_int_into_float_is_not_bounds_checked():
    """Only integer targets are checked: a Python int cast to a float rounds or overflows to inf."""
    a = jix.compact(np.array([1.0, 2.0], dtype=np.float16))
    np.testing.assert_array_equal(jix.add(a, 70000).numpy(), [np.inf, np.inf])


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


def test_integer_only_inputs_skip_f16():
    """Integer-only inputs to a float-only op default to at least f32, like untyped Python floats:
    f16 is used only when some operand is already a float."""
    u8_np = np.array([1, 2, 200], dtype=np.uint8)
    u8 = jix.compact(u8_np)
    i8 = jix.compact(np.array([1, 2, 3], dtype=np.int8))
    assert jix.divide(u8, 2).dtype == np.float32
    assert jix.divide(u8, 2.0).dtype == np.float32
    assert jix.divide(i8, i8).dtype == np.float32
    assert jix.sqrt(u8).dtype == np.float32
    np.testing.assert_array_equal(jix.sqrt(u8).numpy(), np.sqrt(u8_np.astype(np.float32)))
    assert jix.exp(jix.compact(np.array([True, False]))).dtype == np.float32
    assert jix.sqrt(np.int8(4)).dtype == np.float32
    assert jix.sqrt(True).dtype == np.float32
    # Wider ints still go to the float that holds them.
    assert jix.sqrt(jix.compact(np.array([4], dtype=np.int32))).dtype == np.float64


def test_float_operand_keeps_f16():
    """With an f16 operand present, integer operands still promote into f16."""
    i8 = jix.compact(np.array([1, 2], dtype=np.int8))
    f16 = jix.compact(np.array([1.0, 2.0], dtype=np.float16))
    assert jix.add(i8, f16).dtype == np.float16
    assert jix.divide(i8, np.float16(2)).dtype == np.float16


def test_divide_floor_keeps_float_dtypes():
    """divide_floor's dispatch has the floats: float inputs keep their dtype."""
    a = jix.compact(np.array([7.5, -7.5], dtype=np.float32))
    b = jix.compact(np.array([2.0, 2.0], dtype=np.float32))
    result = jix.divide_floor(a, b)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.numpy(), [3.0, -4.0])


# ---------------------------------------------------------------------------
# power dispatch table (base, exponent) pairs:
#   integers: (u8,u8) (i8,u8) (u16,u16) (i16,u16) (u32,u32) (i32,u32) (u64,u32) (i64,u32)
#             (i8,i8) (i16,i16) (i32,i32) (i64,i64)
#   floats:   (f16,i32) (f16,f16) (f32,i32) (f32,f32) (f64,i32) (f64,f64)
# The widest unsigned exponent slot is u32, and no signed slot holds a u64, so a
# u64 exponent, or a u64 base with a signed exponent, promotes to float.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_dtype, exp_dtype, expected_dtype",
    [
        (np.uint8, np.uint8, np.uint8),  # direct
        (np.int8, np.uint8, np.int8),  # direct (signed base, unsigned exp)
        (np.int32, np.uint32, np.int32),  # direct
        (np.uint64, np.uint32, np.uint64),  # direct, widest unsigned base/exp
        (np.uint8, np.uint16, np.uint16),  # u8 base widens to the u16 exp impl
    ],
)
def test_power_integer_dispatch_keeps_integer(base_dtype, exp_dtype, expected_dtype):
    """Integer base with an unsigned exponent dispatches to an integer impl - no float promotion."""
    a = jix.compact(np.array([2, 3, 4], dtype=base_dtype))
    b = jix.compact(np.array([3, 2, 1], dtype=exp_dtype))
    result = jix.power(a, b)
    assert result.dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), np.array([8, 9, 4], dtype=expected_dtype))


@pytest.mark.parametrize(
    "base_dtype,exp_dtype,expected_dtype",
    [
        (np.int8, np.int8, np.int8),  # direct
        (np.int32, np.int32, np.int32),  # direct
        (np.int64, np.int64, np.int64),  # direct
        (np.uint8, np.int8, np.int16),  # both widen to the first signed slot, as numpy
        (np.uint32, np.int32, np.int64),
        (np.uint64, np.int64, np.float64),  # no signed integer holds u64
    ],
)
def test_power_signed_exponent_keeps_integer(base_dtype, exp_dtype, expected_dtype):
    """An integer base with a signed exponent dispatches to an integer impl, as numpy."""
    a = jix.compact(np.array([2, 3, 4], dtype=base_dtype))
    b = jix.compact(np.array([3, 2, 1], dtype=exp_dtype))
    result = jix.power(a, b)
    assert result.dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(result.numpy(), np.array([8, 9, 4], dtype=expected_dtype))


def test_power_promotes_uint64_exponent_to_float64():
    """A 64-bit exponent has no slot (widest is u32 = 32-bit), so even u8 ** u64 promotes to f64."""
    a = jix.compact(np.array([2, 3, 4], dtype=np.uint8))
    b = jix.compact(np.array([2, 2, 2], dtype=np.uint64))
    result = jix.power(a, b)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.numpy(), np.array([4.0, 9.0, 16.0], dtype=np.float64))


def test_power_python_int_exponent_keeps_integer_base():
    """A plain Python int has no signedness, so it fills the unsigned exponent slot:
    power(int32_array, 3) stays int32, like a typed signed or unsigned exponent."""
    a = jix.compact(np.array([2, 3, 4], dtype=np.int32))
    assert jix.power(a, 3).dtype == np.int32
    np.testing.assert_array_equal(jix.power(a, 3).numpy(), [8, 27, 64])
    assert jix.power(a, np.uint32(3)).dtype == np.int32
    assert jix.power(a, np.int32(3)).dtype == np.int32


def test_power_python_int_base_keeps_unsigned_exponent_dtype():
    """A Python int base fills the base slot of the exponent's own dtype: 2 ** u8 -> u8."""
    b = jix.compact(np.array([0, 1, 7], dtype=np.uint8))
    result = jix.power(2, b)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result.numpy(), [1, 2, 128])


def test_power_f16_base_keeps_f16():
    """f16 has its own pairs (computed in f32), so an f16 base with a Python scalar exponent or a
    small int exponent stays f16. A Python float base follows a typed f16 exponent."""
    a_np = np.array([1.0, 2.0, 4.0], dtype=np.float16)
    a = jix.compact(a_np)
    for exponent in (3, 1.5, np.int8(3), jix.compact(np.array([1, 2, 3], dtype=np.uint8))):
        result = jix.power(a, exponent)
        assert result.dtype == np.float16
    np.testing.assert_array_equal(jix.power(a, 3).numpy(), a_np**3)
    np.testing.assert_array_equal(jix.power(a, 1.5).numpy(), a_np ** np.float16(1.5))
    assert jix.power(2.0, a).dtype == np.float16


def test_power_negative_python_int_exponent_raises_for_integer_base():
    """A negative Python int does not fit the unsigned exponent slot of an integer base."""
    a = jix.compact(np.array([2, 3, 4], dtype=np.int32))
    with pytest.raises(OverflowError, match="out of bounds"):
        jix.power(a, -1)


def test_power_scalar_exponent_keeps_float_base_narrow():
    """A float base prefers the signed (f32, i32) slot, so a scalar exponent must not widen it."""
    a = jix.compact(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    assert jix.power(a, 3).dtype == np.float32
    assert jix.power(a, np.int32(3)).dtype == np.float32


def test_power_rejects_complex():
    """power has no complex impl; Complex -> Float is never safe."""
    a = jix.compact(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
    b = jix.compact(np.array([1 + 0j, 2 + 0j], dtype=np.complex64))
    with pytest.raises(Exception):
        _ = jix.power(a, b).numpy()


# ---------------------------------------------------------------------------
# Scalar operands are built directly at the dispatch dtype.
#
# Dispatch casts every operand to the dtype of the impl it picked. For an array
# that means wrapping it in a lazy `Cast` view, but a scalar has nothing to wrap:
# the value is cast up front and the `Scalar` storage is created at the target
# dtype. A `Cast<Scalar>` in the storage chain means that stopped happening.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arr_dtype", [np.int8, np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("scalar", [2, 2.0, True, np.float32(2.0), np.int8(2), np.float16(2.0)])
def test_scalar_operand_has_no_cast_view(arr_dtype, scalar):
    a = jix.compact(np.array([1, 2, 3], dtype=arr_dtype))
    storage = repr(a * scalar)
    assert "Scalar" in storage, storage
    # `Cast<Compact>` is fine - the array operand really does need a lazy cast view.
    assert "Cast<Scalar" not in storage, storage


def test_scalar_operand_has_no_cast_view_for_cmp_and_bitwise():
    a = jix.compact(np.array([1, 2, 3], dtype=np.float32))
    assert "Cast<Scalar" not in repr(a > 2.0)
    b = jix.compact(np.array([1, 2, 3], dtype=np.int32))
    assert "Cast<Scalar" not in repr(b | 1)


def test_scalar_operand_complex_has_no_cast_view():
    a = jix.compact(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
    storage = repr(a * 2.0)
    assert "Scalar" in storage, storage
    assert "Cast<Scalar" not in storage, storage


def test_python_float_scalar_rounds_to_dispatch_dtype():
    """`0.1` is not representable in f32: the scalar must round exactly the way numpy's
    float32(0.1) does, whether it is cast before or after the Scalar storage is built."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = jix.compact(arr)
    np.testing.assert_array_equal((a * 0.1).numpy(), arr * np.float32(0.1))


def test_float16_scalar_keeps_its_value():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    a = jix.compact(arr)
    np.testing.assert_array_equal((a * np.float16(0.1)).numpy(), arr * np.float16(0.1))


def test_bool_scalar_keeps_its_value():
    arr = np.array([1, 2, 3], dtype=np.int32)
    a = jix.compact(arr)
    np.testing.assert_array_equal((a * True).numpy(), arr * np.int32(1))
    np.testing.assert_array_equal((a * False).numpy(), arr * np.int32(0))


def test_complex_scalar_keeps_its_value():
    arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    a = jix.compact(arr)
    np.testing.assert_array_equal((a * (0.1 + 0.2j)).numpy(), arr * np.complex64(0.1 + 0.2j))


# ---------------------------------------------------------------------------
# where / concatenate / stack: inputs of different dtypes are cast to a common
# dtype, picked with the same rules as the dispatch tables. Inputs that already
# share a dtype, which may be a struct dtype, are used as they are.
# ---------------------------------------------------------------------------

_ALL_DTYPES = [
    np.bool_,
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


@pytest.mark.parametrize("a_dtype", _ALL_DTYPES)
@pytest.mark.parametrize("b_dtype", _ALL_DTYPES)
def test_where_concatenate_stack_promote_arrays_like_numpy(a_dtype, b_dtype):
    """For two arrays the common dtype is numpy's, for every pair of dtypes."""
    a_np = np.array([0, 1], dtype=a_dtype)
    b_np = np.array([1, 0], dtype=b_dtype)
    cond_np = np.array([True, False])
    a, b, cond = jix.compact(a_np), jix.compact(b_np), jix.compact(cond_np)
    expected = np.result_type(a_dtype, b_dtype)
    for result, ref in [
        (jix.where(cond, a, b), np.where(cond_np, a_np, b_np)),
        (jix.concatenate([a, b]), np.concatenate([a_np, b_np])),
        (jix.stack([a, b]), np.stack([a_np, b_np])),
    ]:
        assert result.dtype == expected, (a_dtype, b_dtype)
        np.testing.assert_array_equal(result.numpy(), ref)


def test_concatenate_stack_promote_many_arrays():
    """The common dtype is picked for all inputs at once, not pairwise: u8, i8 and f16 all fit f16,
    as numpy, although u8 and i8 alone promote to i16, which then needs f32 with f16."""
    arrays = [np.array([1], dtype=np.uint8), np.array([-1], dtype=np.int8), np.array([0.5], dtype=np.float16)]
    zs = [jix.compact(a) for a in arrays]
    assert jix.concatenate(zs).dtype == np.float16
    np.testing.assert_array_equal(jix.concatenate(zs).numpy(), np.concatenate(arrays))
    assert jix.stack(zs).dtype == np.float16
    np.testing.assert_array_equal(jix.stack(zs).numpy(), np.stack(arrays))


@pytest.mark.parametrize(
    "arr_dtype, scalar, expected",
    [
        (np.uint8, 0, np.uint8),
        (np.int8, -1, np.int8),
        (np.uint64, 1, np.uint64),
        (np.float16, 1.5, np.float16),
        (np.float32, 1.5, np.float32),
        (np.int8, 1.5, np.float32),  # no f16 input, so not f16 (numpy: f64)
        (np.int32, 1.5, np.float64),
        (np.bool_, True, np.bool_),
        (np.bool_, 1, np.uint8),  # bool + int -> u8 (numpy: i64)
        (np.int8, 1j, np.complex64),
        (np.float64, 1j, np.complex128),
    ],
)
def test_where_python_scalar_takes_array_dtype(arr_dtype, scalar, expected):
    """A Python scalar follows the same rules as in the element-wise ops, in either position."""
    a_np = np.array([0, 1], dtype=arr_dtype)
    cond_np = np.array([True, False])
    a, cond = jix.compact(a_np), jix.compact(cond_np)
    for args, np_args in [((a, scalar), (a_np, scalar)), ((scalar, a), (scalar, a_np))]:
        result = jix.where(cond, *args)
        assert result.dtype == np.dtype(expected)
        np.testing.assert_array_equal(result.numpy(), np.where(cond_np, *np_args).astype(expected))


@pytest.mark.parametrize("x, y, expected", [(0, 1, np.int64), (0, 1.5, np.float64), (True, False, np.bool_)])
def test_where_python_scalars_only(x, y, expected):
    """With only untyped operands, the 64-bit default applies."""
    cond_np = np.array([True, False])
    result = jix.where(jix.compact(cond_np), x, y)
    assert result.dtype == np.dtype(expected)
    np.testing.assert_array_equal(result.numpy(), np.where(cond_np, x, y).astype(expected))


@pytest.mark.parametrize("arr_dtype, scalar", [(np.uint8, 300), (np.uint8, -1), (np.int8, -1000)])
def test_where_python_int_out_of_bounds_raises(arr_dtype, scalar):
    a = jix.compact(np.array([0, 1], dtype=arr_dtype))
    cond = jix.compact(np.array([True, False]))
    with pytest.raises(OverflowError, match="out of bounds"):
        jix.where(cond, a, scalar)


_STRUCT_DTYPE = np.dtype([("x", np.int32), ("y", np.float64)])


def test_where_concatenate_stack_keep_shared_struct_dtype():
    """Inputs that already share a dtype skip promotion, so struct dtypes keep working."""
    a_np = np.array([(1, 1.5), (2, 2.5)], dtype=_STRUCT_DTYPE)
    b_np = np.array([(3, 3.5), (4, 4.5)], dtype=_STRUCT_DTYPE)
    cond_np = np.array([True, False])
    a, b, cond = jix.asarray(a_np), jix.asarray(b_np), jix.compact(cond_np)
    for result, ref in [
        (jix.where(cond, a, b), np.where(cond_np, a_np, b_np)),
        (jix.concatenate([a, b]), np.concatenate([a_np, b_np])),
        (jix.stack([a, b]), np.stack([a_np, b_np])),
    ]:
        assert result.dtype == _STRUCT_DTYPE
        np.testing.assert_array_equal(result.numpy(), ref)


def test_where_concatenate_stack_struct_and_scalar_dtype_raise():
    """A struct dtype has no common dtype with a scalar one."""
    s = jix.asarray(np.array([(1, 1.5), (2, 2.5)], dtype=_STRUCT_DTYPE))
    i = jix.compact(np.array([1, 2], dtype=np.int32))
    cond = jix.compact(np.array([True, False]))
    for fn in [lambda: jix.where(cond, s, i), lambda: jix.concatenate([s, i]), lambda: jix.stack([s, i])]:
        with pytest.raises(TypeError, match="does not support operands with dtypes"):
            fn()
