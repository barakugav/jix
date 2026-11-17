//! EXPERIMENT: `core::ops` operators between an array and a scalar, built on [`Map`] and a
//! closure, without a dedicated op storage (like [`Add`](crate::ops::Add)).
//!
//! `Array<S> + 1.5f32` must name its result type in `Add::Output`, but the closure
//! `move |x| x + rhs` has an anonymous type, and stable Rust has no way to write it there. So the
//! result type depends on the toolchain:
//!
//! - **nightly, `--cfg jix_nightly`** (`#![feature(type_alias_impl_trait)]`): the closure type
//!   gets a name, `AddF32Fn = impl Fn(f32) -> f32`, and the result is
//!   `Array<Map<S, AddF32Fn>>`. The lambda is part of the static type, statically dispatched and
//!   inlined into the read loop like any other [`Array::map`].
//! - **stable**: the closure is boxed, `Array<Map<S, Box<dyn Fn(f32) -> f32 + Send + Sync>>>`.
//!   That names the type, but the lambda is erased from it and called through a vtable once per
//!   element, so the pipeline cannot be inlined or vectorized across it.
//!
//! The nightly alternative of writing the closure inline, `type Output = Array<Map<S, impl
//! Fn(f32) -> f32>>` (`impl_trait_in_assoc_type`), compiles but is worse: that opaque type
//! captures the impl's generic `S`, so for a view (`S = &Compact`) the borrow checker assumes
//! dropping the result may use the borrow, and the source array stays borrowed until the result
//! is dropped instead of until its last use. A free-standing alias captures nothing.
//!
//! Build and test the nightly variant with:
//! `RUSTFLAGS="--cfg jix_nightly" cargo +nightly test --all-features scalar_core_ops`

use crate::ops::Map;
use crate::storage::ArrayStorageTyped;
use crate::Array;

/// The closure behind `Array<S> + f32`: `move |x: f32| x + rhs`.
#[cfg(jix_nightly)]
pub type AddF32Fn = impl Fn(f32) -> f32 + Copy + Send + Sync;

#[cfg(jix_nightly)]
#[define_opaque(AddF32Fn)]
fn add_f32_fn(rhs: f32) -> AddF32Fn {
    move |x| x + rhs
}

#[cfg(jix_nightly)]
impl<S> core::ops::Add<f32> for Array<S>
where
    S: ArrayStorageTyped<Item = f32>,
{
    type Output = Array<Map<S, AddF32Fn>>;

    /// Adds `rhs` to every element, as a lazy [`Map`].
    #[track_caller]
    fn add(self, rhs: f32) -> Self::Output {
        self.map(add_f32_fn(rhs))
    }
}

#[cfg(not(jix_nightly))]
type BoxedAddFn = Box<dyn Fn(f32) -> f32 + Send + Sync>;

#[cfg(not(jix_nightly))]
impl<S> core::ops::Add<f32> for Array<S>
where
    S: ArrayStorageTyped<Item = f32>,
{
    type Output = Array<Map<S, BoxedAddFn>>;

    /// Adds `rhs` to every element, as a lazy [`Map`].
    #[track_caller]
    fn add(self, rhs: f32) -> Self::Output {
        self.map::<f32, BoxedAddFn>(Box::new(move |x| x + rhs))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::ops::Map;
    use crate::storage::ArrayStorageTyped;
    use crate::util::{assert_array_matches, ScalarStrategy};
    use crate::Array;

    #[test]
    fn add_f32_scalar() {
        crate::ops::elementwise::op1::tests::check_op1::<f32>(f32::any_strategy(), |nd, za| {
            let result = za + 2.5f32;
            let expected = nd.mapv(|x| x + 2.5);
            assert_array_matches(&result, &expected);
        });
    }

    #[test]
    fn add_f32_scalar_composes_with_other_ops() -> crate::error::Result<()> {
        let a = Array::compact_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
        let b = Array::compact_ndarray(&array![[10.0f32, 20.0], [30.0, 40.0]])?;

        // Scalar add feeding array-array ops, a reduction and another scalar add.
        let result = -(a.view() + 1.0) * b.view() + 0.5;
        assert_eq!(
            result.to_ndarray()?,
            array![[-19.5f32, -59.5], [-119.5, -199.5]]
        );
        // `result` still holds a borrow of `a`, but is not used again, so `a` can be moved. This
        // fails with an `impl Fn` written inline in `Add::Output`, see the module docs.
        // [[2, 4], [3, 5]] summed over dim 0.
        assert_eq!(
            (a.transpose() + 1.0).sum(0usize).to_ndarray()?,
            array![5.0f32, 9.0]
        );
        Ok(())
    }

    /// The result is a [`Map`] over the input storage, with no extra op storage around it.
    #[test]
    fn add_f32_scalar_is_a_map() -> crate::error::Result<()> {
        fn add_one<S>(a: Array<S>) -> Array<Map<S, impl Fn(f32) -> f32>>
        where
            S: ArrayStorageTyped<Item = f32>,
        {
            a + 1.0
        }
        let a = Array::compact_ndarray(&array![1.0f32, 2.0])?;
        assert_eq!(add_one(a).to_ndarray()?, array![2.0f32, 3.0]);
        Ok(())
    }
}
