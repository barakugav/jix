use std::fmt::Debug;
use std::ops::Range;

use crate::dtype::Dtyped;
use crate::storage::block::BlockSize;
use crate::storage::Compact;
use crate::util::AlignedBytes;
use crate::{ArrayParams, DimDyn, Ty};

// ---------------------------------------------------------------------------
// arr_params - shared test helper (previously duplicated in every test module)
// ---------------------------------------------------------------------------

pub(crate) fn arr_params(block_shape: &[usize]) -> ArrayParams {
    ArrayParams {
        block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
        ..ArrayParams::default()
    }
}

// ---------------------------------------------------------------------------
// gen_data_bytes_from_slice - convert a typed slice to aligned bytes
// ---------------------------------------------------------------------------

/// Convert a slice of typed values to aligned bytes.
/// Used by codec filter roundtrip tests that receive values from proptest strategies.
pub(crate) fn gen_data_bytes_from_slice<T: Dtyped>(items: &[T]) -> AlignedBytes {
    let bytes = unsafe { crate::util::cast_slice::<T, u8>(items) };
    AlignedBytes::from_slice(T::DTYPE.alignment().as_usize(), bytes)
}

// ---------------------------------------------------------------------------
// ScalarStrategy - proptest strategies for scalar types
// ---------------------------------------------------------------------------

use proptest::strategy::BoxedStrategy;

/// Proptest strategies for scalar types used across test modules.
///
/// - `any_strategy()`: full domain (used for codec roundtrip tests).
/// - `op_safe_strategy()`: bounded range that avoids overflow when values are
///   combined arithmetically (used for op2, cast tests). For integer types,
///   values are small and positive so that e.g. `a = a_extra + b` still fits
///   in the type. Floats default to `any_strategy()`.
pub(crate) trait ScalarStrategy:
    Dtyped + Default + core::fmt::Debug + Clone + 'static
{
    fn any_strategy() -> BoxedStrategy<Self>;
    #[allow(unused)]
    fn logical_op_strategy() -> BoxedStrategy<Self> {
        Self::any_strategy()
            .prop_union(Just(Self::default()).boxed())
            .boxed()
    }
    fn maybe_non_finite_strategy() -> BoxedStrategy<Self> {
        Self::any_strategy()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        Self::any_strategy()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self>;
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        Self::op_safe_strategy()
    }
    /// Values in the closed interval [-1, 1]. Used for ops whose domain is restricted to that
    /// range (e.g. `asin`, `acos`). Overridden for `f32` and `f64`; falls back to
    /// `op_safe_strategy` for all other types (where the method is not expected to be called).
    fn unit_strategy() -> BoxedStrategy<Self> {
        Self::op_safe_strategy()
    }
    /// Values in `[0, bit_width)`. Used as shift amounts so that `a << b` / `a >> b` never
    /// panic in debug mode. Overridden for all integer types; falls back to `any_strategy`
    /// for non-integer types (not expected to be called for those).
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        Self::any_strategy()
    }
    /// Small set of 3 distinct values used for equality/comparison tests. Ensures that
    /// ~33 % of generated pairs are equal, so both the `true` and `false` branches of
    /// `equal`/`not_equal` are exercised. For float types the set includes `NaN` to cover
    /// the `NaN != NaN` (IEEE 754) edge case. Defaults to `any_strategy` for types where
    /// equal pairs happen naturally (e.g. `bool` has only 2 values -> 50 % hit rate).
    fn comparable_strategy() -> BoxedStrategy<Self> {
        Self::any_strategy()
    }
}

use proptest::prelude::*;

impl ScalarStrategy for u8 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u8>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (0u8..=4).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        (1u8..=4).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0u8..8).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0u8), Just(1u8), Just(2u8)].boxed()
    }
}
impl ScalarStrategy for u16 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u16>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        // three_arrays test does (a*b)*c where a=a_extra+b+c <= 3r.
        // max (3r)*r*r = 3r^3 <= u16::MAX=65535 -> r <= 27.
        (0u16..=27).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        (1u16..=27).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0u16..16).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0u16), Just(1u16), Just(2u16)].boxed()
    }
}
impl ScalarStrategy for u32 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u32>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (0u32..=30).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        (1u32..=30).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0u32..32).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0u32), Just(1u32), Just(2u32)].boxed()
    }
}
impl ScalarStrategy for u64 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u64>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (0u64..=30).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        (1u64..=30).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0u64..64).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0u64), Just(1u64), Just(2u64)].boxed()
    }
}
impl ScalarStrategy for i8 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i8>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (-4i8..=4).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        prop_oneof![(-4i8..=-1).boxed(), (1i8..=4).boxed()].boxed()
    }
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        (0i8..=4).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0i8..8).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0i8), Just(1i8), Just(2i8)].boxed()
    }
}
impl ScalarStrategy for i16 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i16>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        // three_arrays test does (a*b)*c where a=a_extra+b+c <= 3r.
        // max (3r)*r*r = 3r^3 <= i16::MAX=32767 -> r <= 22.
        (-22i16..=22).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        prop_oneof![(-22i16..=-1).boxed(), (1i16..=22).boxed()].boxed()
    }
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        (0i16..=22).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0i16..16).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0i16), Just(1i16), Just(2i16)].boxed()
    }
}
impl ScalarStrategy for i32 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i32>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (-100i32..=100).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        prop_oneof![(-100i32..=-1).boxed(), (1i32..=100).boxed()].boxed()
    }
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        (0i32..=100).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0i32..32).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0i32), Just(1i32), Just(2i32)].boxed()
    }
}
impl ScalarStrategy for i64 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i64>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (-100i64..=100).boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        prop_oneof![(-100i64..=-1).boxed(), (1i64..=100).boxed()].boxed()
    }
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        (0i64..=100).boxed()
    }
    fn shift_safe_strategy() -> BoxedStrategy<Self> {
        (0i64..64).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0i64), Just(1i64), Just(2i64)].boxed()
    }
}

#[cfg(feature = "half")]
impl ScalarStrategy for crate::scalar::f16 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<f32>().prop_map(crate::scalar::f16::from_f32).boxed()
    }

    fn maybe_non_finite_strategy() -> BoxedStrategy<Self> {
        f32::maybe_non_finite_strategy()
            .prop_map(crate::scalar::f16::from_f32)
            .boxed()
    }

    fn op_safe_strategy() -> BoxedStrategy<Self> {
        f32::op_safe_strategy()
            .prop_map(crate::scalar::f16::from_f32)
            .boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        f32::op_safe_non_zero_strategy()
            .prop_map(crate::scalar::f16::from_f32)
            .boxed()
    }
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        f32::op_safe_non_negative_strategy()
            .prop_map(crate::scalar::f16::from_f32)
            .boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        f32::comparable_strategy()
            .prop_map(crate::scalar::f16::from_f32)
            .boxed()
    }
}
impl ScalarStrategy for f32 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<f32>().boxed()
    }

    fn maybe_non_finite_strategy() -> BoxedStrategy<Self> {
        prop::strategy::Union::new_weighted(vec![
            (7, any::<f32>().boxed()),
            (
                1,
                prop_oneof![Just(f32::INFINITY), Just(f32::NEG_INFINITY), Just(f32::NAN)].boxed(),
            ),
        ])
        .boxed()
    }

    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (0..=(2 * 100 * 100))
            .prop_map(|x| ((x - 100 * 100) as f32) / 100.0)
            .boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        Self::op_safe_strategy()
            .prop_filter("zero", |&x| x != 0.0)
            .boxed()
    }
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        (0..=(100 * 100)).prop_map(|x| (x as f32) / 100.0).boxed()
    }
    fn unit_strategy() -> BoxedStrategy<Self> {
        (-100i8..=100).prop_map(|x| x as f32 / 100.0).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0.0f32), Just(1.0f32), Just(2.4f32), Just(f32::NAN)].boxed()
    }
}
impl ScalarStrategy for f64 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<f64>().boxed()
    }

    fn maybe_non_finite_strategy() -> BoxedStrategy<Self> {
        prop::strategy::Union::new_weighted(vec![
            (7, any::<f64>().boxed()),
            (
                1,
                prop_oneof![Just(f64::INFINITY), Just(f64::NEG_INFINITY), Just(f64::NAN)].boxed(),
            ),
        ])
        .boxed()
    }

    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (0..=(2 * 100 * 100))
            .prop_map(|x| ((x - 100 * 100) as f64) / 100.0)
            .boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        Self::op_safe_strategy()
            .prop_filter("zero", |&x| x != 0.0)
            .boxed()
    }
    fn op_safe_non_negative_strategy() -> BoxedStrategy<Self> {
        (0..=(100 * 100)).prop_map(|x| (x as f64) / 100.0).boxed()
    }

    fn unit_strategy() -> BoxedStrategy<Self> {
        (-100i8..=100).prop_map(|x| x as f64 / 100.0).boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![Just(0.0f64), Just(1.0f64), Just(2.4f64), Just(f64::NAN)].boxed()
    }
}
impl ScalarStrategy for bool {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<bool>().boxed()
    }

    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        Just(true).boxed()
    }
}

#[cfg(feature = "num-complex")]
impl ScalarStrategy for crate::scalar::Complex<f32> {
    fn any_strategy() -> BoxedStrategy<Self> {
        (any::<f32>(), any::<f32>())
            .prop_map(|(re, im)| crate::scalar::Complex { re, im })
            .boxed()
    }

    fn maybe_non_finite_strategy() -> BoxedStrategy<Self> {
        (
            f32::maybe_non_finite_strategy(),
            f32::maybe_non_finite_strategy(),
        )
            .prop_map(|(re, im)| crate::scalar::Complex { re, im })
            .boxed()
    }

    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (f32::op_safe_strategy(), f32::op_safe_strategy())
            .prop_map(|(re, im)| crate::scalar::Complex { re, im })
            .boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        (
            f32::op_safe_non_zero_strategy(),
            f32::op_safe_non_zero_strategy(),
        )
            .prop_map(|(re, im)| crate::scalar::Complex { re, im })
            .boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        prop_oneof![
            Just(crate::scalar::Complex {
                re: 0.0f32,
                im: 0.0
            }),
            Just(crate::scalar::Complex {
                re: 1.0f32,
                im: 0.0
            }),
            Just(crate::scalar::Complex {
                re: 0.0f32,
                im: 1.0
            }),
            Just(crate::scalar::Complex {
                re: f32::NAN,
                im: 0.0
            }),
            Just(crate::scalar::Complex {
                re: 0.0,
                im: f32::NAN
            }),
        ]
        .boxed()
    }
}

#[cfg(feature = "num-complex")]
impl ScalarStrategy for crate::scalar::Complex<f64> {
    fn any_strategy() -> BoxedStrategy<Self> {
        <crate::scalar::Complex<f32>>::any_strategy()
            .prop_map(crate::scalar::Cast::<Self>::cast)
            .boxed()
    }

    fn maybe_non_finite_strategy() -> BoxedStrategy<Self> {
        <crate::scalar::Complex<f32>>::maybe_non_finite_strategy()
            .prop_map(crate::scalar::Cast::<Self>::cast)
            .boxed()
    }

    fn op_safe_strategy() -> BoxedStrategy<Self> {
        <crate::scalar::Complex<f32>>::op_safe_strategy()
            .prop_map(crate::scalar::Cast::<Self>::cast)
            .boxed()
    }
    fn op_safe_non_zero_strategy() -> BoxedStrategy<Self> {
        <crate::scalar::Complex<f32>>::op_safe_non_zero_strategy()
            .prop_map(crate::scalar::Cast::<Self>::cast)
            .boxed()
    }
    fn comparable_strategy() -> BoxedStrategy<Self> {
        <crate::scalar::Complex<f32>>::comparable_strategy()
            .prop_map(crate::scalar::Cast::<Self>::cast)
            .boxed()
    }
}

pub(crate) fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    if cfg!(not(miri)) {
        prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=100, 1)),
            (2, proptest::collection::vec(100..=1000, 1)),
            // 2D
            (8, proptest::collection::vec(1..=20, 2)),
            (2, proptest::collection::vec(20..=50, 2)),
            // 3D
            (5, proptest::collection::vec(1..=16, 3)),
            // 4D
            (5, proptest::collection::vec(1..=10, 4)),
            // Many dims
            (3, proptest::collection::vec(1..=4, 1..=8)),
            // Zero-length dims
            (1, proptest::collection::vec(0..=3, 0..=8)),
        ])
    } else {
        prop::strategy::Union::new_weighted(vec![
            // 1D
            (8, proptest::collection::vec(1usize..=100, 1)),
            // 2D
            (8, proptest::collection::vec(1..=16, 2)),
            (2, proptest::collection::vec(10..=25, 2)),
            // 3D
            (5, proptest::collection::vec(1..=5, 3)),
            // 4D
            (5, proptest::collection::vec(1..=5, 4)),
            // Many dims
            (3, proptest::collection::vec(1..=4, 1..=8)),
            // Zero-length dims
            (1, proptest::collection::vec(0..=3, 0..=8)),
        ])
    }
}

// pub(crate) fn ndarray_strategy<T>() -> impl Strategy<Value = ndarray::ArrayD<T>>
// where
//     T: ScalarStrategy + Debug,
// {
//     ndarray_strategy_generic(shape_strategy(), T::any_strategy())
// }

pub(crate) fn ndarray_strategy_generic<T>(
    shape: impl Strategy<Value = Vec<usize>>,
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = ndarray::ArrayD<T>>
where
    T: Debug,
{
    shape
        .prop_flat_map(move |shape| {
            let total_len: usize = shape.iter().product();
            let elements = prop::collection::vec(element.clone(), total_len);
            (Just(shape), elements)
        })
        .prop_map(|(shape, data)| {
            ndarray::ArrayD::<T>::from_shape_vec(shape.as_slice(), data).unwrap()
        })
}

pub(crate) fn block_shape_strategy(
    shape: &[usize],
) -> impl Strategy<Value = Vec<BlockSize>> + use<> {
    // Per-dim block size is clamped to [1, max(1, s)] to satisfy the invariant
    // enforced by `ArrayParams::tune`: `1 <= b <= s.max(1)`.
    shape
        .iter()
        .map(|&s| 1u32..=4u32.min(s.max(1) as u32))
        .collect::<Vec<_>>()
}

pub(crate) fn carray_strategy_any<T>(
) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::Array<Compact<Ty<T>, DimDyn>>)>
where
    T: ScalarStrategy + Debug,
{
    carray_strategy_from_shape(shape_strategy(), T::any_strategy())
}

pub(crate) fn carray_strategy_from_shape<T>(
    shape: impl Strategy<Value = Vec<usize>>,
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::Array<Compact<Ty<T>, DimDyn>>)>
where
    T: ScalarStrategy + Debug,
{
    carray_strategy_from_data(ndarray_strategy_generic::<T>(shape, element))
}

pub(crate) fn carray_strategy_from_data<T>(
    data: impl Strategy<Value = ndarray::ArrayD<T>>,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::Array<Compact<Ty<T>, DimDyn>>)>
where
    T: ScalarStrategy + Debug,
{
    data.prop_flat_map(|arr| {
        let block_shape = block_shape_strategy(arr.shape());
        let block_size = Just(None)
            .boxed()
            .prop_union((1u64..100).prop_map(Some).boxed());
        let read_size = Just(None).boxed().prop_union(
            (1u64..600, 1u64..600)
                .prop_map(|(a, b)| Some((a.min(b), a.max(b))))
                .boxed(),
        );

        (Just(arr), block_shape, block_size, read_size)
    })
    .prop_map(|(arr, block_shape, block_size, read_size)| {
        let block_shape = block_shape;
        let mut params = ArrayParams::default();
        params.block_shape(&block_shape);
        if let Some(block_size) = block_size {
            params.block_size(block_size);
        }
        if let Some(read_size) = read_size {
            params.read_size(read_size);
        }
        let compact = crate::Array::compact_ndarray_with(&arr, params).unwrap();
        (arr, compact)
    })
}

// pub(crate) fn carrays2_strategy<T>() -> impl Strategy<
//     Value = (
//         (ndarray::ArrayD<T>, crate::Array<Compact>),
//         (ndarray::ArrayD<T>, crate::Array<Compact>),
//     ),
// >
// where
//     T: ScalarStrategy + Debug,
// {
//     carrays2_strategy_generic(shape_strategy(), T::any_strategy())
// }

pub(crate) fn carrays2_strategy_generic<T>(
    shape: impl Strategy<Value = Vec<usize>>,
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<
    Value = (
        (ndarray::ArrayD<T>, crate::Array<Compact<Ty<T>, DimDyn>>),
        (ndarray::ArrayD<T>, crate::Array<Compact<Ty<T>, DimDyn>>),
    ),
>
where
    T: ScalarStrategy + Debug,
{
    shape.prop_flat_map(move |shape| {
        (
            carray_strategy_from_shape(Just(shape.clone()), element.clone()),
            carray_strategy_from_shape(Just(shape.clone()), element.clone()),
        )
    })
}

/// Generates a random sub-range for an array of the given shape.
/// Each dimension independently gets `start..end` with `0 <= start <= end <= shape[i]`.
pub(crate) fn sub_range_strategy(shape: &[u64]) -> BoxedStrategy<Vec<Range<u64>>> {
    shape.iter().fold(Just(vec![]).boxed(), |acc, &len| {
        acc.prop_flat_map(move |ranges| {
            (0u64..=len, 0u64..=len).prop_map(move |(a, b)| {
                let mut v = ranges.clone();
                v.push(a.min(b)..a.max(b));
                v
            })
        })
        .boxed()
    })
}

/// Asserts that `actual` contains the same values as `expected`, comparing each
/// element bit-for-bit. Checks the full array first, then 16 random sub-ranges
/// via `to_ndarray_sub`.
///
/// Use [`assert_array_matches_approx`] for results of order-dependent
/// floating-point arithmetic (e.g. reductions), where exact equality is too
/// strict.
pub(crate) fn assert_array_matches<S, T, D>(
    actual: &crate::Array<S>,
    expected: &ndarray::Array<T, D>,
) where
    S: crate::ArrayStorage,
    T: crate::dtype::Dtyped + std::fmt::Debug + Clone + PartialEq,
    D: ndarray::Dimension,
{
    assert_array_matches_with(actual, expected, |a, b| a == b);
}

/// Like [`assert_array_matches`], but compares elements with [`ApproxEq`] under
/// the given relative and absolute tolerances instead of bit-for-bit equality.
///
/// This is the right check for results of order-dependent floating-point
/// arithmetic: a reduction reads its input block-by-block, so the accumulator
/// reassociates and may land a few ULP away from a reference computed by a
/// straight sequential fold.
///
/// [`ApproxEq`]: crate::scalar::ApproxEq
pub(crate) fn assert_array_matches_approx<S, T, D>(
    actual: &crate::Array<S>,
    expected: &ndarray::Array<T, D>,
    rtol: <T as crate::scalar::ApproxEq>::RelativeTolerance,
    atol: <T as crate::scalar::ApproxEq>::AbsoluteTolerance,
) where
    S: crate::ArrayStorage,
    T: crate::dtype::Dtyped + std::fmt::Debug + Clone + crate::scalar::ApproxEq,
    D: ndarray::Dimension,
{
    assert_array_matches_with(actual, expected, move |a, b| a.approx_eq(b, &rtol, &atol));
}

/// Shared implementation of [`assert_array_matches`] and
/// [`assert_array_matches_approx`]: checks storage invariants, then compares the
/// full read and 16 random sub-range reads against `expected` using the
/// element-wise comparator `eq`.
fn assert_array_matches_with<S, T, D>(
    actual: &crate::Array<S>,
    expected: &ndarray::Array<T, D>,
    eq: impl Fn(&T, &T) -> bool,
) where
    S: crate::ArrayStorage,
    T: crate::dtype::Dtyped + std::fmt::Debug + Clone,
    D: ndarray::Dimension,
{
    use proptest::test_runner::{Config, TestCaseError, TestRunner};

    // take the opportunity to check some invariants
    let spec = actual.storage.spec();
    let ndim = actual.shape().len();
    assert_eq!(spec.block_shape().len(), ndim);
    assert!(spec
        .block_shape()
        .iter()
        .zip(actual.shape())
        .all(|(&b, &s)| (0..=s.max(1)).contains(&(b as u64))));
    assert_eq!(spec.block_shape_tag().len(), ndim);
    assert!(spec.block_size() > 0);
    assert!(spec.read_size().min > 0);

    let expected = expected.view().into_dyn();
    let actual = actual.as_ref().into_typed::<T>().unwrap();
    let full = actual.to_ndarray().unwrap().into_dyn();
    if let Err(msg) = elementwise_eq(&full, &expected, &eq) {
        unreachable!("full array mismatch: {msg}");
    }

    let ctx = actual.read_ctx();
    let shape = actual.shape();
    let mut runner = TestRunner::new(Config {
        cases: 16,
        failure_persistence: None,
        ..Config::default()
    });
    runner
        .run(&sub_range_strategy(&shape), |ranges| {
            let actual_sub = actual
                .to_ndarray_sub(&ranges, &ctx)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let ranges_usize: Vec<Range<usize>> = ranges
                .iter()
                .map(|r| r.start as usize..r.end as usize)
                .collect();
            let expected_sub = ndarray_slice(&expected, &ranges_usize);
            elementwise_eq(&actual_sub.into_dyn(), &expected_sub, &eq)
                .map_err(TestCaseError::fail)?;
            Ok(())
        })
        .unwrap();
}

/// Compares two dynamic-dimension arrays element-wise (in logical order) with
/// `eq`. Returns `Err` describing the first shape or value mismatch found.
fn elementwise_eq<A, B, T>(
    actual: &ndarray::ArrayBase<A, ndarray::IxDyn>,
    expected: &ndarray::ArrayBase<B, ndarray::IxDyn>,
    eq: &impl Fn(&T, &T) -> bool,
) -> Result<(), String>
where
    A: ndarray::Data<Elem = T>,
    B: ndarray::Data<Elem = T>,
    T: std::fmt::Debug,
{
    if actual.shape() != expected.shape() {
        return Err(format!(
            "shape mismatch: actual {:?} != expected {:?}",
            actual.shape(),
            expected.shape()
        ));
    }
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        if !eq(a, b) {
            return Err(format!(
                "value mismatch at flat index {i}: actual {a:?} != expected {b:?}"
            ));
        }
    }
    Ok(())
}

fn ndarray_slice<S, D>(
    array: &ndarray::ArrayBase<S, D>,
    index: &[Range<usize>],
) -> ndarray::ArrayD<S::Elem>
where
    S: ndarray::Data,
    S::Elem: Clone,
    D: ndarray::Dimension,
{
    let mut view = array.view();
    for (dim, range) in index.iter().enumerate() {
        view.slice_axis_inplace(
            ndarray::Axis(dim),
            ndarray::Slice::from(range.start as isize..range.end as isize),
        );
    }
    view.to_owned().into_dimensionality().unwrap()
}
