use crate::array::ArrayParams;
use crate::dtype::Dtyped;
use crate::storage::block::BlockSize;
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
    AlignedBytes::from_slice(T::DTYPE.alignment() as usize, bytes)
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
