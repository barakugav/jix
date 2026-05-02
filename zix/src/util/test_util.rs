use std::fmt::Debug;

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
}
impl ScalarStrategy for f64 {
    fn any_strategy() -> BoxedStrategy<Self> {
        any::<f64>().boxed()
    }
    fn op_safe_strategy() -> BoxedStrategy<Self> {
        (1u8..=100).prop_map(|x| x as f64).boxed()
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

pub(crate) fn ndarray_strategy<T>() -> impl Strategy<Value = ndarray::ArrayD<T>>
where
    T: ScalarStrategy + Debug,
{
    ndarray_strategy_generic(
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
        ]),
        T::any_strategy(),
    )
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

pub(crate) fn compact_array_strategy<T>(
) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::Array<Compact>)>
where
    T: ScalarStrategy + Debug,
{
    ndarray_strategy::<T>()
        .prop_flat_map(|arr| {
            let block_shape = prop::collection::vec(1u32..=4, arr.ndim());
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
