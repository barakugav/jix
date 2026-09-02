use std::fmt::Debug;
use std::ops::Range;

use crate::dtype::Dtyped;
use crate::storage::block::BlockSize;
use crate::storage::Compact;
use crate::util::AlignedBytes;
use crate::{Array, ArrayParams, ArrayStorage, DimDyn, Ty, TypeDyn};

// ---------------------------------------------------------------------------
// arr_params - shared test helper
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

/// A random array of a random shape. See [`array_strategy_from_data`].
pub(crate) fn array_strategy_any<T>() -> impl Strategy<Value = (ndarray::ArrayD<T>, TestArray<T>)>
where
    T: ScalarStrategy + Debug,
{
    array_strategy_from_shape(shape_strategy(), T::any_strategy())
}

/// A random array of the given shape. See [`array_strategy_from_data`].
pub(crate) fn array_strategy_from_shape<T>(
    shape: impl Strategy<Value = Vec<usize>>,
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, TestArray<T>)>
where
    T: ScalarStrategy + Debug,
{
    array_strategy_from_data(ndarray_strategy_generic::<T>(shape, element))
}

pub(crate) type TestArray<T> = Array<crate::ops::IntoType<crate::storage::ArrayStorageAny, Ty<T>>>;

#[derive(Debug, Clone, Copy)]
enum TestArrayKind {
    Compact,
    Plain,
}

/// Wraps `data` in an array whose storage backend and memory layout are drawn at random.
///
/// The backend is an even split between [`Compact`] and [`Plain`]. Both are built from the same
/// hand-laid byte buffer, so `Compact` covers the strided ingest path while `Plain` - the only
/// backend that lends its own bytes out of `read_data` rather than a packed, aligned copy - pushes
/// those strides through every read downstream.
///
/// The `ndarray` half of the pair stays row-major so it remains a clean reference to compare
/// against, but it does repeat along any axis the layout gave a zero stride.
pub(crate) fn array_strategy_from_data<T>(
    data: impl Strategy<Value = ndarray::ArrayD<T>>,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, TestArray<T>)>
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
        let kind = prop_oneof![Just(TestArrayKind::Compact), Just(TestArrayKind::Plain)];
        let layout = layout_strategy(arr.ndim(), T::DTYPE.alignment().as_usize());

        (Just(arr), block_shape, block_size, read_size, kind, layout)
    })
    .prop_map(|(arr, block_shape, block_size, read_size, kind, layout)| {
        let mut params = ArrayParams::default();
        params.block_shape(&block_shape);
        if let Some(block_size) = block_size {
            params.block_size(block_size);
        }
        if let Some(read_size) = read_size {
            params.read_size(read_size);
        }
        let arr = collapse_zero_stride_axes(arr, &layout.zero_stride_axes);
        let array = build_test_array(&arr, &layout, kind, params);
        (arr, array)
    })
}

/// Two independently generated arrays of the same shape. Each draws its own backend and layout.
#[allow(clippy::type_complexity)]
pub(crate) fn arrays2_strategy_generic<T>(
    shape: impl Strategy<Value = Vec<usize>>,
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<
    Value = (
        (ndarray::ArrayD<T>, TestArray<T>),
        (ndarray::ArrayD<T>, TestArray<T>),
    ),
>
where
    T: ScalarStrategy + Debug,
{
    shape.prop_flat_map(move |shape| {
        (
            array_strategy_from_shape(Just(shape.clone()), element.clone()),
            array_strategy_from_shape(Just(shape.clone()), element.clone()),
        )
    })
}

/// Always [`Compact`] storage, row-major. Only for tests that need real compact storage -
/// `as_compact()` and the archive's compact write path. Prefer [`array_strategy_any`] otherwise.
pub(crate) fn carray_strategy_any<T>(
) -> impl Strategy<Value = (ndarray::ArrayD<T>, Array<Compact<Ty<T>, DimDyn>>)>
where
    T: ScalarStrategy + Debug,
{
    carray_strategy_from_shape(shape_strategy(), T::any_strategy())
}

pub(crate) fn carray_strategy_from_shape<T>(
    shape: impl Strategy<Value = Vec<usize>>,
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, Array<Compact<Ty<T>, DimDyn>>)>
where
    T: ScalarStrategy + Debug,
{
    carray_strategy_from_data(ndarray_strategy_generic::<T>(shape, element))
}

pub(crate) fn carray_strategy_from_data<T>(
    data: impl Strategy<Value = ndarray::ArrayD<T>>,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, Array<Compact<Ty<T>, DimDyn>>)>
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
        let mut params = ArrayParams::default();
        params.block_shape(&block_shape);
        if let Some(block_size) = block_size {
            params.block_size(block_size);
        }
        if let Some(read_size) = read_size {
            params.read_size(read_size);
        }
        let compact = Array::compact_ndarray_with(&arr, params).unwrap();
        (arr, compact)
    })
}

#[derive(Debug, Clone)]
struct TestArrayLayout {
    /// Memory order of the axes, outermost (largest stride) first. `[0, 1, .., n-1]` is row-major.
    axis_order: Vec<usize>,
    /// Extra bytes inserted before each axis' stride, indexed like `axis_order`. All-zero is packed.
    padding_strides: Vec<usize>,
    /// Axes given a stride of 0, so every index along them aliases one element. Indexed by axis.
    zero_stride_axes: Vec<bool>,
    /// Bytes the base pointer is pushed past its natural alignment. `0` leaves it aligned.
    ptr_offset: usize,
}

impl TestArrayLayout {
    fn compute_strides(&self, shape: &[usize], itemsize: usize) -> Vec<usize> {
        let mut strides = vec![0usize; shape.len()];
        let mut min_stride = itemsize;
        for (k, &dim) in self.axis_order.iter().enumerate().rev() {
            // A zero-stride axis spans no bytes, so it neither takes padding nor grows the extent.
            if self.zero_stride_axes[dim] {
                continue;
            }
            strides[dim] = min_stride + self.padding_strides[k];
            min_stride = strides[dim] * shape[dim].max(1);
        }
        strides
    }
}

/// Repeats `arr` along each zero-stride axis, so its values match the aliasing the buffer will
/// have. Only the source side of a read is ever given zero strides, never a write destination.
fn collapse_zero_stride_axes<T: Clone>(
    mut arr: ndarray::ArrayD<T>,
    zero_stride_axes: &[bool],
) -> ndarray::ArrayD<T> {
    let shape = arr.shape().to_vec();
    for (dim, &zero) in zero_stride_axes.iter().enumerate() {
        if zero && shape[dim] > 1 {
            arr = arr
                .index_axis(ndarray::Axis(dim), 0)
                .insert_axis(ndarray::Axis(dim))
                .broadcast(shape.as_slice())
                .unwrap()
                .to_owned();
        }
    }
    arr
}

/// Draws a memory layout: 75 % row-major, 12.5 % a transpose, 12.5 % a transpose with padding and
/// possibly zero strides. Independently, 80 % keep the base pointer and every stride aligned to the
/// dtype and the rest break one or both.
///
/// Only a padded layout can carry strides that are not multiples of the alignment, so drawing
/// unaligned strides forces padding on and lifts the padded share above 12.5 %. Types with
/// alignment 1 have nothing to misalign and always come out aligned.
fn layout_strategy(ndim: usize, align: usize) -> BoxedStrategy<TestArrayLayout> {
    #[derive(Debug, Clone, Copy)]
    enum Contiguity {
        Contiguous,
        PermuteAxes,
        Strided,
    }
    let contiguity = prop::strategy::Union::new_weighted(vec![
        (6, Just(Contiguity::Contiguous)),
        (1, Just(Contiguity::PermuteAxes)),
        (1, Just(Contiguity::Strided)),
    ]);
    // (pointer aligned, strides aligned)
    let alignment = prop::strategy::Union::new_weighted(vec![
        (24, Just((true, true))),
        (2, Just((true, false))),
        (2, Just((false, true))),
        (2, Just((false, false))),
    ]);

    (contiguity, alignment)
        .prop_flat_map(move |(contiguity, (ptr_aligned, strides_aligned))| {
            let padded =
                matches!(contiguity, Contiguity::Strided) || (!strides_aligned && align > 1);

            let axis_order: BoxedStrategy<Vec<usize>> =
                if matches!(contiguity, Contiguity::Contiguous) {
                    Just((0..ndim).collect::<Vec<_>>()).boxed()
                } else {
                    Just((0..ndim).collect::<Vec<_>>()).prop_shuffle().boxed()
                };

            let padding_strides = if !padded || ndim == 0 {
                Just(vec![0usize; ndim]).boxed()
            } else if strides_aligned {
                prop::collection::vec((0usize..=2).prop_map(move |p| p * align), ndim).boxed()
            } else {
                (prop::collection::vec(0usize..=(2 * align), ndim), 0..ndim)
                    .prop_map(move |(mut pads, k)| {
                        if pads[k].is_multiple_of(align) {
                            pads[k] += 1;
                        }
                        pads
                    })
                    .boxed()
            };

            let zero_stride_axes: BoxedStrategy<Vec<bool>> =
                if matches!(contiguity, Contiguity::Strided) && ndim > 0 {
                    prop::collection::vec(
                        prop::strategy::Union::new_weighted(vec![
                            (3, Just(false)),
                            (1, Just(true)),
                        ]),
                        ndim,
                    )
                    .boxed()
                } else {
                    Just(vec![false; ndim]).boxed()
                };

            let ptr_offset: BoxedStrategy<usize> = if ptr_aligned || align == 1 {
                Just(0usize).boxed()
            } else {
                (1..align).boxed()
            };

            (axis_order, padding_strides, zero_stride_axes, ptr_offset)
        })
        .prop_map(
            |(axis_order, padding_strides, zero_stride_axes, ptr_offset)| TestArrayLayout {
                axis_order,
                padding_strides,
                zero_stride_axes,
                ptr_offset,
            },
        )
        .boxed()
}

fn build_strided_bytes<T: Dtyped>(
    arr: &ndarray::ArrayD<T>,
    layout: &TestArrayLayout,
) -> (AlignedBytes, usize, Vec<usize>) {
    let itemsize = T::DTYPE.itemsize() as usize;
    let align = T::DTYPE.alignment().as_usize();
    let strides = layout.compute_strides(arr.shape(), itemsize);
    let span = crate::util::strided_span_bytes(arr.shape(), &strides, itemsize);

    // A zero-element array still needs somewhere to point, and a zero-length allocation would
    // leave the storage holding a dangling pointer.
    let len = (span + layout.ptr_offset).max(1);
    let mut buf = AlignedBytes::with_capacity_exact(align, len);
    buf.resize(len, 0);

    for (idx, value) in arr.indexed_iter() {
        let offset =
            layout.ptr_offset + (0..arr.ndim()).map(|d| idx[d] * strides[d]).sum::<usize>();
        // Byte-wise, because `offset` is under no obligation to be a multiple of the alignment.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(value).cast::<u8>(),
                buf.as_mut_ptr().add(offset),
                itemsize,
            );
        }
    }

    (buf, layout.ptr_offset, strides)
}

/// Builds a [`TestArray`] holding `arr`'s values, in the given storage backend and memory layout.
fn build_test_array<T: Dtyped>(
    arr: &ndarray::ArrayD<T>,
    layout: &TestArrayLayout,
    kind: TestArrayKind,
    params: ArrayParams,
) -> TestArray<T> {
    let (buf, offset, strides) = build_strided_bytes(arr, layout);
    let shape: Vec<u64> = arr.shape().iter().map(|&s| s as u64).collect();
    let data = unsafe { buf.as_ptr().add(offset) };

    let array = match kind {
        // `Compact` re-blocks and compresses, so the layout shapes only the one read that copies
        // the data in - which is the ingest path worth exercising. `buf` outlives the call.
        TestArrayKind::Compact => unsafe {
            Array::<Compact<TypeDyn, DimDyn>>::compact_nd_ptr(
                data,
                &shape,
                &strides,
                T::DTYPE,
                params,
            )
        }
        .unwrap()
        .into_any(),
        // `Plain` keeps the buffer as its storage, so every read downstream sees these strides and
        // this alignment. `Plain::new` takes the allocation by value, so the buffer travels along.
        TestArrayKind::Plain => {
            let storage = unsafe {
                crate::storage::Plain::<AlignedBytes, TypeDyn, DimDyn>::new(
                    buf,
                    data,
                    &shape,
                    &strides,
                    T::DTYPE,
                    params,
                )
            }
            .unwrap();
            Array::from_storage(storage).into_any()
        }
    };
    array.into_typed::<T>().unwrap()
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
pub(crate) fn assert_array_matches<S, T, D>(actual: &Array<S>, expected: &ndarray::Array<T, D>)
where
    S: ArrayStorage,
    T: Dtyped + std::fmt::Debug + Clone + PartialEq,
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
    actual: &Array<S>,
    expected: &ndarray::Array<T, D>,
    rtol: <T as crate::scalar::ApproxEq>::RelativeTolerance,
    atol: <T as crate::scalar::ApproxEq>::AbsoluteTolerance,
) where
    S: ArrayStorage,
    T: Dtyped + std::fmt::Debug + Clone + crate::scalar::ApproxEq,
    D: ndarray::Dimension,
{
    assert_array_matches_with(actual, expected, move |a, b| a.approx_eq(b, &rtol, &atol));
}

/// Shared implementation of [`assert_array_matches`] and
/// [`assert_array_matches_approx`]: checks storage invariants, then compares the
/// full read and 16 random sub-range reads against `expected` using the
/// element-wise comparator `eq`.
fn assert_array_matches_with<S, T, D>(
    actual: &Array<S>,
    expected: &ndarray::Array<T, D>,
    eq: impl Fn(&T, &T) -> bool,
) where
    S: ArrayStorage,
    T: Dtyped + std::fmt::Debug + Clone,
    D: ndarray::Dimension,
{
    // Erase the storage to `&dyn ArrayStorage` and the dimension to `IxDyn` so the comparison
    // body (`assert_array_matches_with_dyn`) is monomorphized once instead of once per
    // (storage type x ndim).
    let storage: &dyn ArrayStorage = &actual.storage;
    let actual = Array::from_storage(storage);
    let expected = expected.view().into_dyn();
    assert_array_matches_dyn::<T>(&actual, expected, &eq);
}

/// Storage- and dimension-agnostic core of [`assert_array_matches_with`]: checks storage
/// invariants, then compares the full read and 16 random sub-range reads against `expected`.
fn assert_array_matches_dyn<T>(
    actual: &Array<&dyn ArrayStorage>,
    expected: ndarray::ArrayViewD<'_, T>,
    eq: &impl Fn(&T, &T) -> bool,
) where
    T: Dtyped + std::fmt::Debug + Clone,
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
    assert_eq!(spec.block_shape_fixed_dims().len(), ndim);
    assert!(spec.block_size() > 0);
    assert!(spec.read_size().min > 0);

    let actual = actual.view().into_typed::<T>().unwrap();
    let full = actual.to_ndarray().unwrap().into_dyn();
    if let Err(msg) = elementwise_eq(&full, &expected, eq) {
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
        .run(&sub_range_strategy(shape), |ranges| {
            let actual_sub = actual
                .to_ndarray_sub(&ranges, &ctx)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let ranges_usize: Vec<Range<usize>> = ranges
                .iter()
                .map(|r| r.start as usize..r.end as usize)
                .collect();
            let expected_sub = ndarray_slice(&expected, &ranges_usize);
            elementwise_eq(&actual_sub.into_dyn(), &expected_sub, eq)
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    /// Classifies a drawn layout into the buckets [`layout_strategy`] documents.
    fn classify(
        layout: &TestArrayLayout,
        shape: &[usize],
        itemsize: usize,
        align: usize,
    ) -> (bool, bool, bool, bool) {
        let strides = layout.compute_strides(shape, itemsize);
        let packed = layout.padding_strides.iter().all(|&p| p == 0);
        let row_major = packed && layout.axis_order == (0..shape.len()).collect::<Vec<_>>();
        let ptr_aligned = layout.ptr_offset.is_multiple_of(align);
        let strides_aligned = strides.iter().all(|&s| s.is_multiple_of(align));
        (row_major, packed, ptr_aligned, strides_aligned)
    }

    /// The generated layouts must actually span the interesting cases - a green test suite proves
    /// nothing if the dice always land on "row-major and aligned".
    #[test]
    fn layout_strategy_covers_the_documented_mix() {
        const N: usize = 4000;
        let shape = [3usize, 4, 5];
        let (itemsize, align) = (4usize, 4usize);

        let strategy = layout_strategy(shape.len(), align);
        let mut runner = TestRunner::deterministic();
        let (mut row_major, mut padded, mut unaligned_ptr, mut unaligned_strides) = (0, 0, 0, 0);
        let (mut transposed, mut zero_stride) = (0, 0);
        for _ in 0..N {
            let layout = strategy.new_tree(&mut runner).unwrap().current();
            let (is_row_major, is_packed, ptr_aligned, strides_aligned) =
                classify(&layout, &shape, itemsize, align);
            row_major += usize::from(is_row_major);
            transposed += usize::from(is_packed && !is_row_major);
            padded += usize::from(!is_packed);
            unaligned_ptr += usize::from(!ptr_aligned);
            unaligned_strides += usize::from(!strides_aligned);
            zero_stride += usize::from(layout.zero_stride_axes.iter().any(|&z| z));
        }
        let pct = |n: usize| 100.0 * n as f64 / N as f64;
        eprintln!(
            "row-major {:.1}%  transposed {:.1}%  padded {:.1}%  unaligned-ptr {:.1}%  unaligned-strides {:.1}%  zero-stride {:.1}%",
            pct(row_major), pct(transposed), pct(padded), pct(unaligned_ptr), pct(unaligned_strides), pct(zero_stride),
        );

        // Wide bands: the point is that every case is reachable and none dominates, not that the
        // sampler hits an exact ratio.
        assert!(
            (55.0..75.0).contains(&pct(row_major)),
            "row-major {:.1}%",
            pct(row_major)
        );
        assert!(
            (5.0..20.0).contains(&pct(transposed)),
            "transposed {:.1}%",
            pct(transposed)
        );
        assert!(
            (15.0..35.0).contains(&pct(padded)),
            "padded {:.1}%",
            pct(padded)
        );
        assert!(
            (5.0..20.0).contains(&pct(unaligned_ptr)),
            "unaligned ptr {:.1}%",
            pct(unaligned_ptr)
        );
        assert!(
            (5.0..20.0).contains(&pct(unaligned_strides)),
            "unaligned strides {:.1}%",
            pct(unaligned_strides)
        );
        assert!(
            (3.0..20.0).contains(&pct(zero_stride)),
            "zero stride {:.1}%",
            pct(zero_stride)
        );
    }

    /// A dtype with alignment 1 has nothing to misalign, so every layout must come out aligned.
    #[test]
    fn layout_strategy_leaves_byte_dtypes_aligned() {
        let strategy = layout_strategy(2, 1);
        let mut runner = TestRunner::deterministic();
        for _ in 0..500 {
            let layout = strategy.new_tree(&mut runner).unwrap().current();
            assert_eq!(layout.ptr_offset, 0);
        }
    }

    /// Walks a storage tree down to its first leaf and returns that leaf's name.
    fn leaf_storage_name(storage: &dyn ArrayStorage) -> String {
        let info = storage.info();
        match info.dependencies() {
            [] => info.name().to_string(),
            [dep, ..] => leaf_storage_name(*dep),
        }
    }

    /// The whole point of the `array_strategy_*` family is that tests see both backends. `Plain` is
    /// the one that lends its own strided, possibly unaligned bytes straight out of `read_data`, so
    /// if it stopped being generated the extra coverage would vanish silently.
    #[test]
    fn array_strategy_mixes_both_storage_backends() {
        const N: usize = 200;
        let strategy = array_strategy_any::<i32>();
        let mut runner = TestRunner::deterministic();
        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        for _ in 0..N {
            let (_nd, za) = strategy.new_tree(&mut runner).unwrap().current();
            *counts.entry(leaf_storage_name(&za.storage)).or_default() += 1;
        }
        eprintln!("storage backends over {N} draws: {counts:?}");
        assert!(
            counts.get("Compact").copied().unwrap_or(0) > N / 5,
            "{counts:?}"
        );
        assert!(
            counts.get("Plain").copied().unwrap_or(0) > N / 5,
            "{counts:?}"
        );
    }

    /// Every layout must survive a round-trip through both storage backends: the values that come
    /// back are the ones that went in, whatever the strides and alignment were.
    #[test]
    fn build_test_array_roundtrips_every_layout() {
        let arr =
            ndarray::ArrayD::<i32>::from_shape_vec(ndarray::IxDyn(&[2, 3, 4]), (0..24).collect())
                .unwrap();
        let align = i32::DTYPE.alignment().as_usize();

        // name, axis order, padding, zero-stride axes, ptr offset, strides expected unaligned
        let cases = [
            ("row-major", [0, 1, 2], [0, 0, 0], [false; 3], 0, false),
            ("transposed", [2, 0, 1], [0, 0, 0], [false; 3], 0, false),
            ("padded", [1, 2, 0], [4, 8, 4], [false; 3], 0, false),
            ("unaligned ptr", [0, 1, 2], [0, 0, 0], [false; 3], 1, false),
            (
                "unaligned strides",
                [0, 1, 2],
                [1, 3, 1],
                [false; 3],
                0,
                true,
            ),
            (
                "unaligned ptr and strides",
                [2, 1, 0],
                [1, 2, 3],
                [false; 3],
                3,
                true,
            ),
            (
                "one zero stride",
                [0, 1, 2],
                [0, 0, 0],
                [true, false, false],
                0,
                false,
            ),
            (
                "zero stride, padded, unaligned",
                [1, 0, 2],
                [1, 2, 3],
                [false, true, false],
                3,
                true,
            ),
            (
                "every axis zero stride",
                [0, 1, 2],
                [0, 0, 0],
                [true; 3],
                0,
                false,
            ),
        ];

        for (name, order, pads, zero_stride_axes, ptr_offset, expect_unaligned) in cases {
            let layout = TestArrayLayout {
                axis_order: order.to_vec(),
                padding_strides: pads.to_vec(),
                zero_stride_axes: zero_stride_axes.to_vec(),
                ptr_offset,
            };
            let strides = layout.compute_strides(arr.shape(), i32::DTYPE.itemsize() as usize);
            assert_eq!(
                !strides.iter().all(|s| s.is_multiple_of(align)),
                expect_unaligned,
                "{name}: strides {strides:?} do not match the intended alignment",
            );
            assert_eq!(
                strides.iter().filter(|&&s| s == 0).count(),
                zero_stride_axes.iter().filter(|&&z| z).count(),
                "{name}: strides {strides:?} do not match the intended zero-stride axes",
            );

            // The reference has to repeat wherever the buffer aliases.
            let expected = collapse_zero_stride_axes(arr.clone(), &zero_stride_axes);
            for kind in [TestArrayKind::Compact, TestArrayKind::Plain] {
                eprintln!("case {name} / {kind:?}: strides {strides:?}, offset {ptr_offset}");
                let za = build_test_array(&expected, &layout, kind, ArrayParams::default());
                assert_array_matches(&za, &expected);
            }
        }
    }
}
