use std::fmt::Debug;
use std::ops::Range;

use crate::dtype::Dtyped;
use crate::params::ArrayParams;
use crate::storage::block::BlockSize;
use crate::storage::Compact;
use crate::util::AlignedBytes;

// ---------------------------------------------------------------------------
// arr_params — shared test helper (previously duplicated in every test module)
// ---------------------------------------------------------------------------

pub(crate) fn arr_params(block_shape: &[usize]) -> ArrayParams {
    ArrayParams {
        block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
        ..ArrayParams::default()
    }
}

// ---------------------------------------------------------------------------
// gen_data_bytes_from_slice — convert a typed slice to aligned bytes
// ---------------------------------------------------------------------------

/// Convert a slice of typed values to aligned bytes.
/// Used by codec filter roundtrip tests that receive values from proptest strategies.
pub(crate) fn gen_data_bytes_from_slice<T: Dtyped>(items: &[T]) -> AlignedBytes {
    let bytes = unsafe { crate::util::cast_slice::<T, u8>(items) };
    AlignedBytes::from_slice(T::DTYPE.alignment().as_usize(), bytes)
}

// ---------------------------------------------------------------------------
// ScalarStrategy — proptest strategies for scalar types
// ---------------------------------------------------------------------------

use proptest::strategy::BoxedStrategy;

/// Proptest strategies for scalar types used across test modules.
///
/// - `any_strategy()`: full domain (used for codec roundtrip tests).
/// - `op_safe_strategy()`: bounded range that avoids overflow when values are
///   combined arithmetically (used for op2, astype tests). For integer types,
///   values are small and positive so that e.g. `a = a_extra + b` still fits
///   in the type. Floats default to `any_strategy()`.
pub(crate) trait ScalarStrategy: Dtyped + core::fmt::Debug + Clone + 'static {
    fn any_strategy() -> BoxedStrategy<Self>;
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        Self::any_strategy()
    }
    /// Values in the closed interval [-1, 1]. Used for ops whose domain is restricted to that
    /// range (e.g. `asin`, `acos`). Overridden for `f32` and `f64`; falls back to
    /// `op_safe_strategy` for all other types (where the method is not expected to be called).
    fn unit_strategy() -> BoxedStrategy<Self> {
        Self::op_safe_strategy()
    }
}

use proptest::prelude::*;

impl ScalarStrategy for u8 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u8>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u8..=4).boxed()
    }
}
impl ScalarStrategy for u16 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u16>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        // three_arrays test does (a*b)*c where a=a_extra+b+c ≤ 3r.
        // max (3r)·r·r = 3r³ ≤ u16::MAX=65535 → r ≤ 27.
        (1u16..=27).boxed()
    }
}
impl ScalarStrategy for u32 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u32>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u32..=30).boxed()
    }
}
impl ScalarStrategy for u64 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<u64>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u64..=30).boxed()
    }
}
impl ScalarStrategy for i8 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i8>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1i8..=4).boxed()
    }
}
impl ScalarStrategy for i16 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i16>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        // three_arrays test does (a*b)*c where a=a_extra+b+c ≤ 3r.
        // max (3r)·r·r = 3r³ ≤ i16::MAX=32767 → r ≤ 22.
        (1i16..=22).boxed()
    }
}
impl ScalarStrategy for i32 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i32>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1i32..=100).boxed()
    }
}
impl ScalarStrategy for i64 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<i64>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1i64..=100).boxed()
    }
}
impl ScalarStrategy for f32 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<f32>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u8..=100).prop_map(|x| x as f32).boxed()
    }
    fn unit_strategy() -> BoxedStrategy<Self> {
        (-100i8..=100).prop_map(|x| x as f32 / 100.0).boxed()
    }
}
impl ScalarStrategy for f64 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<f64>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u8..=100).prop_map(|x| x as f64).boxed()
    }
    fn unit_strategy() -> BoxedStrategy<Self> {
        (-100i8..=100).prop_map(|x| x as f64 / 100.0).boxed()
    }
}
impl ScalarStrategy for bool {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<bool>().boxed()
    }
}

#[cfg(feature = "half")]
impl ScalarStrategy for crate::dtype::f16 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<f32>().prop_map(crate::dtype::f16::from_f32).boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u8..=15)
            .prop_map(|x| crate::dtype::f16::from_f32(x as f32))
            .boxed()
    }
}

impl ScalarStrategy for crate::dtype::Complex<f32> {
    fn any_strategy() -> BoxedStrategy<Self> {
        (any::<f32>(), any::<f32>())
            .prop_map(|(re, im)| crate::dtype::Complex { re, im })
            .boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u8..=15)
            .prop_map(|x| crate::dtype::Complex {
                re: x as f32,
                im: 0.0,
            })
            .boxed()
    }
}

impl ScalarStrategy for crate::dtype::Complex<f64> {
    fn any_strategy() -> BoxedStrategy<Self> {
        (any::<f64>(), any::<f64>())
            .prop_map(|(re, im)| crate::dtype::Complex { re, im })
            .boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u8..=15)
            .prop_map(|x| crate::dtype::Complex {
                re: x as f64,
                im: 0.0,
            })
            .boxed()
    }
}

pub(crate) fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
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
}

pub(crate) fn ndarray_strategy<T>() -> impl Strategy<Value = ndarray::ArrayD<T>>
where
    T: ScalarStrategy + Debug,
{
    ndarray_strategy_generic(shape_strategy(), T::any_strategy())
}

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

pub(crate) fn block_shape_strategy(ndim: usize) -> impl Strategy<Value = Vec<BlockSize>> {
    prop::collection::vec(1u32..=4, ndim)
}

pub(crate) fn compact_array_strategy<T>(
) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::Array<Compact>)>
where
    T: ScalarStrategy + Debug,
{
    compact_array_strategy_generic(T::any_strategy())
}

pub(crate) fn compact_array_strategy_generic<T>(
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::Array<Compact>)>
where
    T: ScalarStrategy + Debug,
{
    ndarray_strategy_generic::<T>(shape_strategy(), element)
        .prop_flat_map(|arr| {
            let block_shape = block_shape_strategy(arr.ndim());
            (Just(arr), block_shape)
        })
        .prop_map(|(arr, block_shape)| {
            let block_shape = block_shape;
            let mut params = ArrayParams::default();
            params.block_shape(&block_shape);
            let compact = crate::Array::compact_array_with(&arr, params).unwrap();
            (arr, compact)
        })
}

pub(crate) fn compact_arrays2_strategy<T>() -> impl Strategy<
    Value = (
        (ndarray::ArrayD<T>, crate::Array<Compact>),
        (ndarray::ArrayD<T>, crate::Array<Compact>),
    ),
>
where
    T: ScalarStrategy + Debug,
{
    let element = T::any_strategy();
    shape_strategy()
        .prop_flat_map(move |shape| {
            let total_len: usize = shape.iter().product();
            let elements1 = prop::collection::vec(element.clone(), total_len);
            let elements2 = prop::collection::vec(element.clone(), total_len);
            let block_shape1 = block_shape_strategy(shape.len());
            let block_shape2 = block_shape_strategy(shape.len());
            (
                Just(shape),
                elements1,
                elements2,
                block_shape1,
                block_shape2,
            )
        })
        .prop_map(|(shape, data1, data2, block_shape1, block_shape2)| {
            let arr1 = ndarray::ArrayD::<T>::from_shape_vec(shape.as_slice(), data1).unwrap();
            let arr2 = ndarray::ArrayD::<T>::from_shape_vec(shape.as_slice(), data2).unwrap();
            let compact1 = crate::Array::compact_array_with(
                &arr1,
                ArrayParams::default().block_shape(&block_shape1).clone(),
            )
            .unwrap();
            let compact2 = crate::Array::compact_array_with(
                &arr2,
                ArrayParams::default().block_shape(&block_shape2).clone(),
            )
            .unwrap();
            ((arr1, compact1), (arr2, compact2))
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

/// Asserts that `actual` contains the same values as `expected`.
/// Checks the full array first, then 16 random sub-ranges via `to_ndarray_sub`.
pub(crate) fn assert_array_matches<S, T>(actual: &crate::Array<S>, expected: &ndarray::ArrayD<T>)
where
    S: crate::storage::ArrayStorage,
    T: crate::dtype::Dtyped + std::fmt::Debug + Clone + PartialEq,
{
    use proptest::prelude::*;
    use proptest::test_runner::{Config, TestCaseError, TestRunner};

    let full = actual.to_ndarray::<T>().unwrap();
    assert_eq!(&full, expected);

    let ctx = actual.read_ctx();
    let shape: Vec<u64> = actual.shape().to_vec();
    let mut runner = TestRunner::new(Config {
        cases: 16,
        ..Config::default()
    });
    runner
        .run(&sub_range_strategy(&shape), |ranges| {
            let actual_sub = actual
                .to_ndarray_sub::<T>(&ranges, &ctx)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let ranges_usize: Vec<Range<usize>> = ranges
                .iter()
                .map(|r| r.start as usize..r.end as usize)
                .collect();
            let expected_sub = ndarray_slice(expected, &ranges_usize);
            prop_assert_eq!(actual_sub, expected_sub);
            Ok(())
        })
        .unwrap();
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
    view.to_owned()
        .into_dimensionality::<ndarray::IxDyn>()
        .unwrap()
}
