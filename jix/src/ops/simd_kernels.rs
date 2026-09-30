//! SIMD bodies of element-wise kernels, over `fearless_simd` vectors.
//!
//! A kernel's `apply_bulk` passes its elements here, with the scalar kernel as the fallback. The
//! element types with a SIMD body are selected by [`TypeId`], which is resolved at compile time,
//! so the kernels keep their generic bounds and every other type runs the scalar kernel. Op1
//! kernels write their SIMD bodies in `define_op1!`'s `simd:` argument.
//!
//! Nodes of the pipeline pass `[T; N]` arrays to each other: a body loads its array into vectors
//! and stores the results into a new array, and LLVM keeps these in registers along a chain.

use std::any::TypeId;

use fearless_simd::{Simd, SimdBase};

/// `x` as a `U`, if `T` is `U`.
#[inline(always)]
pub(crate) fn checked_transmute<T: 'static, U: 'static>(x: T) -> Option<U> {
    // SAFETY: `T` and `U` are the same type.
    (TypeId::of::<T>() == TypeId::of::<U>())
        .then(|| unsafe { std::mem::transmute_copy::<T, U>(&x) })
}

/// Vector `c` of `xs`. `(c + 1) * V::LEN <= N`.
#[inline(always)]
fn load<S: Simd, V: SimdBase<S>, const N: usize>(simd: S, xs: &[V::Element; N], c: usize) -> V {
    debug_assert!((c + 1) * V::LEN <= N);
    // SAFETY: in bounds per the caller, and `V::Array` is `[V::Element; V::LEN]`.
    V::load_array_ref(simd, unsafe { &*xs.as_ptr().cast::<V::Array>().add(c) })
}

/// Store `v` as vector `c` of `xs`. `(c + 1) * V::LEN <= N`.
#[inline(always)]
fn store<S: Simd, V: SimdBase<S>, const N: usize>(v: V, xs: &mut [V::Element; N], c: usize) {
    debug_assert!((c + 1) * V::LEN <= N);
    // SAFETY: in bounds, and `V::Array` is `[V::Element; V::LEN]`.
    v.store_array(unsafe { &mut *xs.as_mut_ptr().add(c * V::LEN).cast() });
}

/// `f` on the vectors of `V` in `a` and `b`, if their elements are `V`'s and `N` a multiple of
/// `V::LEN`.
#[inline(always)]
fn try_map2<S, V, T1, T2, O, const N: usize>(
    simd: S,
    a: [T1; N],
    b: [T2; N],
    f: impl Fn(V, V) -> V,
) -> Option<[O; N]>
where
    S: Simd,
    V: SimdBase<S, Element: 'static>,
    T1: 'static,
    T2: 'static,
    O: 'static,
{
    let a =
        checked_transmute::<[T1; N], [V::Element; N]>(a).filter(|_| N.is_multiple_of(V::LEN))?;
    let b = checked_transmute::<[T2; N], [V::Element; N]>(b)?;
    let mut out = a;
    for c in 0..N / V::LEN {
        store(
            f(load::<S, V, N>(simd, &a, c), load(simd, &b, c)),
            &mut out,
            c,
        );
    }
    checked_transmute(out)
}

macro_rules! op2 {
    ($(#[$meta:meta])* $name:ident, $op:tt) => {
        $(#[$meta])*
        #[inline(always)]
        pub(crate) fn $name<S, T1, T2, O, const N: usize>(
            simd: S,
            a: [T1; N],
            b: [T2; N],
            scalar: impl Fn(T1, T2) -> O,
        ) -> [O; N]
        where
            S: Simd,
            T1: Copy + 'static,
            T2: Copy + 'static,
            O: 'static,
        {
            if let Some(r) = try_map2::<S, S::f32s, _, _, _, N>(simd, a, b, |a, b| a $op b) {
                return r;
            }
            if let Some(r) = try_map2::<S, S::f64s, _, _, _, N>(simd, a, b, |a, b| a $op b) {
                return r;
            }
            if let Some(r) = try_map2::<S, S::i32s, _, _, _, N>(simd, a, b, |a, b| a $op b) {
                return r;
            }
            crate::array_from_fn_inline(|i| scalar(a[i], b[i]))
        }
    };
}
op2!(
    /// `a + b`. SIMD for f32, f64 and i32 (wrapping).
    add, +
);
op2!(
    /// `a - b`. SIMD for f32, f64 and i32 (wrapping).
    sub, -
);
op2!(
    /// `a * b`. SIMD for f32, f64 and i32 (wrapping).
    mul, *
);

#[cfg(test)]
mod tests {
    use fearless_simd::{Level, Simd};

    const N: usize = 32;

    fn check<S: Simd>(simd: S) {
        fn inputs<T>(f: impl Fn(i32) -> T) -> [[T; N]; 2] {
            [
                std::array::from_fn(|i| f(i as i32 * 7 - 100)),
                std::array::from_fn(|i| f(i as i32 * -3 + 5)),
            ]
        }
        let [a, b] = inputs(|x| x as f32 * 0.37);
        let [c, d] = inputs(|x| x as f64 * 0.37);
        let [e, f] = inputs(|x| x.wrapping_mul(0x0123_4567));
        simd.vectorize(|| {
            macro_rules! check_op2 {
                ($op:ident, $f:expr, $wrapping:expr) => {
                    let ab: [f32; N] = std::array::from_fn(|i| $f(a[i], b[i]));
                    assert_eq!(super::$op(simd, a, b, $f), ab);
                    let cd: [f64; N] = std::array::from_fn(|i| $f(c[i], d[i]));
                    assert_eq!(super::$op(simd, c, d, $f), cd);
                    let ef: [i32; N] = std::array::from_fn(|i| $wrapping(e[i], f[i]));
                    assert_eq!(super::$op(simd, e, f, $wrapping), ef);
                };
            }
            check_op2!(add, |x, y| x + y, i32::wrapping_add);
            check_op2!(sub, |x, y| x - y, i32::wrapping_sub);
            check_op2!(mul, |x, y| x * y, i32::wrapping_mul);
            // Not a multiple of the vector length: scalar.
            assert_eq!(super::add(simd, [1.5f32], [2.0], |x, y| x + y), [3.5]);
        });
    }

    #[test]
    fn all_levels() {
        let level = Level::new();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(s) = level.as_sse2() {
                check(s);
            }
            if let Some(s) = level.as_sse4_2() {
                check(s);
            }
            if let Some(s) = level.as_avx2() {
                check(s);
            }
            if let Some(s) = level.as_avx512() {
                check(s);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(s) = level.as_neon() {
            check(s);
        }
    }
}
