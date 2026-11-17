//! EXPERIMENT: `core::ops` operators between an array and a scalar (`Array<S> + 1.5`), on stable
//! Rust, running on [`Op1`] alone and without broadcasting the scalar to an array.
//!
//! Every binary op already has a named kernel type implementing [`Op2Kernel`] (e.g. the
//! `AddKernel` behind [`Add`](crate::ops::Add)). [`ScalarRhs`] fixes its right operand, turning
//! it into an [`Op1Kernel`], so `Array<S> + x` reuses the op's scalar and SIMD kernels as-is.
//!
//! A closure passed to [`Map`](crate::ops::Map) would also run on [`Op1`], but its type is
//! anonymous and cannot be written in `Add::Output` on stable Rust (short of boxing it, which
//! costs a virtual call per element). The kernel here is a named type, so the result type is
//! nameable, statically dispatched and inlined.

use super::arith::AddKernel;
use crate::ops::elementwise::op1::{Op1, Op1Kernel};
use crate::ops::elementwise::op2::Op2Kernel;
use crate::ops::prelude::*;

/// [`Op1Kernel`] applying the [`Op2Kernel`] `K` with a fixed right operand.
pub(crate) struct ScalarRhs<K, T> {
    kernel: K,
    rhs: T,
}
impl<K, T1, T2> Op1Kernel<T1> for ScalarRhs<K, T2>
where
    K: Op2Kernel<T1, T2>,
    T2: Copy,
{
    type Output = K::Output;

    #[inline(always)]
    fn apply(&self, x: T1, status: &mut ArithmeticStatus) -> Self::Output {
        self.kernel.apply(x, self.rhs, status)
    }

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        xs: [T1; N],
        status: &mut ArithmeticStatus,
        simd: S,
    ) -> [Self::Output; N] {
        self.kernel.apply_bulk(xs, [self.rhs; N], status, simd)
    }
}

/// Element-wise addition of an array and a scalar (`a + x`).
///
/// Same per-element semantics as [`Add`](crate::ops::Add), see the
/// [`jix::scalar::Add`](crate::scalar::Add) scalar trait.
///
/// The result is a lazy view.
///
/// This struct is the bare storage implementation, the operation is also available as the `+`
/// operator between an array and a scalar of its element type.
///
/// # Examples
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
/// assert_eq!((a + 0.5).to_ndarray()?, array![1.5f32, 2.5, 3.5]);
///
/// let b = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// assert_eq!((b + 10).to_ndarray()?, array![11, 12, 13]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct AddScalar<S, T>(Op1<S, ScalarRhs<AddKernel, T>>);
impl<S, T> AddScalar<S, T>
where
    S: ArrayStorageTyped<Item = T>,
    T: Dtyped + crate::scalar::Add<T, Output: Dtyped>,
{
    /// Constructs an [`AddScalar`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, rhs: T) -> Result<Self> {
        let kernel = ScalarRhs {
            kernel: AddKernel,
            rhs,
        };
        Ok(Self(Op1::new(array, kernel)?))
    }

    /// Constructs an array with [`AddScalar`] storage. See the storage struct docs for semantics
    /// and examples.
    pub fn new_array(array: Array<S>, rhs: T) -> Result<Array<Self>> {
        Self::new(array.into_storage(), rhs).map(Array::from_storage)
    }
}
impl<S, T> crate::storage::ArrayStorageImpl for AddScalar<S, T> {}
impl<S, T> ArrayStorage for AddScalar<S, T>
where
    S: ArrayStorageTyped<Item = T>,
    T: Dtyped + crate::scalar::Add<T, Output: Dtyped>,
{
    type ElementType = Ty<<T as crate::scalar::Add<T>>::Output>;
    crate::storage::impl_array_storage_forward!('a, U, <S, T>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("AddScalar", [&self.0.array])
    }

    crate::ops::impl_element_type_change_default!();
}

impl<S, T> core::ops::Add<T> for Array<S>
where
    S: ArrayStorageTyped<Item = T>,
    T: Dtyped + crate::scalar::Add<T, Output: Dtyped>,
{
    type Output = Array<AddScalar<S, T>>;

    /// Applies the [`AddScalar`] operation, see the op struct docs for details.
    #[track_caller]
    fn add(self, rhs: T) -> Self::Output {
        AddScalar::new_array(self, rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::ops::elementwise::op1::tests::check_op1;
    use crate::util::{assert_array_matches, ScalarStrategy};
    use crate::Array;

    #[test]
    fn add_scalar() {
        check_op1::<f32>(f32::op_safe_strategy(), |nd, za| {
            assert_array_matches(&(za + 2.5), &nd.mapv(|x| x + 2.5));
        });
        check_op1::<f64>(f64::op_safe_strategy(), |nd, za| {
            assert_array_matches(&(za + -0.25), &nd.mapv(|x| x + -0.25));
        });
        check_op1::<i32>(i32::any_strategy(), |nd, za| {
            assert_array_matches(&(za + 7), &nd.mapv(|x| x.wrapping_add(7)));
        });
        check_op1::<u8>(u8::any_strategy(), |nd, za| {
            assert_array_matches(&(za + 200), &nd.mapv(|x| x.wrapping_add(200)));
        });
    }

    #[test]
    fn add_scalar_composes_with_other_ops() -> crate::error::Result<()> {
        let a = Array::compact_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
        let b = Array::compact_ndarray(&array![[10.0f32, 20.0], [30.0, 40.0]])?;

        // Scalar add feeding array-array ops, and another scalar add.
        let result = -(a.view() + 1.0) * b.view() + 0.5;
        assert_eq!(
            result.to_ndarray()?,
            array![[-19.5f32, -59.5], [-119.5, -199.5]]
        );
        // Under a shape op and a reduction: [[2, 4], [3, 5]] summed over dim 0.
        assert_eq!(
            (a.transpose() + 1.0).sum(0usize).to_ndarray()?,
            array![5.0f32, 9.0]
        );
        Ok(())
    }
}
