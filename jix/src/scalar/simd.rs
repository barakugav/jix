//! Building blocks of the SIMD `*_bulk` functions of the scalar traits.

use crate::util::{array_from_fn_inline, ArrayExt};
use fearless_simd::{Simd, SimdBase, SimdElement};

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum SimdLevel {
    Sse2,
    Sse4_2,
    Avx2,
    Avx512,
    Neon,
    /// The others (fallback, WASM).
    Other,
}

#[inline(always)]
pub(crate) fn simd_level<S: Simd>(simd: S) -> SimdLevel {
    match simd.level() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Sse2(_) => SimdLevel::Sse2,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Sse4_2(_) => SimdLevel::Sse4_2,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Avx2(_) => SimdLevel::Avx2,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Avx512(_) => SimdLevel::Avx512,
        #[cfg(target_arch = "aarch64")]
        fearless_simd::Level::Neon(_) => SimdLevel::Neon,
        _ => SimdLevel::Other,
    }
}

/// Like [`fearless_simd::dispatch!`], but with [`Level::baseline`](fearless_simd::Level::baseline)
macro_rules! dispatch_baseline {
    ($simd:ident => $op:expr) => {{
        match const { fearless_simd::Level::baseline() } {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fearless_simd::Level::Sse2($simd) => {
                fearless_simd::Simd::vectorize($simd, #[inline(always)] || $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fearless_simd::Level::Sse4_2($simd) => {
                fearless_simd::Simd::vectorize($simd, #[inline(always)] || $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fearless_simd::Level::Avx2($simd) => {
                fearless_simd::Simd::vectorize($simd, #[inline(always)] || $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            fearless_simd::Level::Avx512($simd) => {
                fearless_simd::Simd::vectorize($simd, #[inline(always)] || $op)
            }
            #[cfg(target_arch = "aarch64")]
            fearless_simd::Level::Neon($simd) => {
                fearless_simd::Simd::vectorize($simd, #[inline(always)] || $op)
            }
            #[allow(unreachable_patterns)]
            level => fearless_simd::dispatch!(level, $simd => $op),
        }
    }};
}
pub(crate) use dispatch_baseline;

pub(crate) trait SimdElementExt: SimdElement {
    type V<S: Simd>: SimdBase<S, Element = Self>;

    fn mask8s_from_mask<S: Simd>(
        mask: impl Fn(usize) -> <Self::V<S> as SimdBase<S>>::Mask,
    ) -> S::mask8s;
}

macro_rules! impl_simd_element_ext {
    (
        $t:ident,
        V = $v:ident,
        mask8s_from_mask = |$mask:ident| $narrow:expr
    ) => {
        impl SimdElementExt for $t {
            type V<S: Simd> = S::$v;

            #[inline(always)]
            fn mask8s_from_mask<S: Simd>(
                $mask: impl Fn(usize) -> <Self::V<S> as SimdBase<S>>::Mask,
            ) -> S::mask8s {
                #[allow(unused_imports)]
                use fearless_simd::MaskNarrow;
                $narrow
            }
        }
    };
}
impl_simd_element_ext!(i8, V = i8s, mask8s_from_mask = |m| m(0));
impl_simd_element_ext!(u8, V = u8s, mask8s_from_mask = |m| m(0));
impl_simd_element_ext!(i16, V = i16s, mask8s_from_mask = |m| m(0).narrow(m(1)));
impl_simd_element_ext!(u16, V = u16s, mask8s_from_mask = |m| m(0).narrow(m(1)));
impl_simd_element_ext!(
    i32,
    V = i32s,
    mask8s_from_mask = |m| m(0).narrow(m(1)).narrow(m(2).narrow(m(3)))
);
impl_simd_element_ext!(
    u32,
    V = u32s,
    mask8s_from_mask = |m| m(0).narrow(m(1)).narrow(m(2).narrow(m(3)))
);
impl_simd_element_ext!(
    f32,
    V = f32s,
    mask8s_from_mask = |m| m(0).narrow(m(1)).narrow(m(2).narrow(m(3)))
);
impl_simd_element_ext!(
    i64,
    V = i64s,
    mask8s_from_mask = |m| {
        let lo = m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
        lo.narrow(m(4).narrow(m(5)).narrow(m(6).narrow(m(7))))
    }
);
impl_simd_element_ext!(
    u64,
    V = u64s,
    mask8s_from_mask = |m| {
        let lo = m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
        lo.narrow(m(4).narrow(m(5)).narrow(m(6).narrow(m(7))))
    }
);
impl_simd_element_ext!(
    f64,
    V = f64s,
    mask8s_from_mask = |m| {
        let lo = m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
        lo.narrow(m(4).narrow(m(5)).narrow(m(6).narrow(m(7))))
    }
);

#[inline(always)]
pub(crate) fn map_vectors<T, U, const N: usize, S>(
    xs: [T; N],
    f_scalar: impl FnMut(T) -> U,
    mut f_simd: impl FnMut(T::V<S>) -> U::V<S>,
    simd: S,
) -> [U; N]
where
    T: SimdElementExt,
    U: SimdElementExt,
    S: Simd,
{
    map_vectors_different_width::<T, U, N, 1, 1, S>(
        xs,
        f_scalar,
        #[inline(always)]
        |[x]| [f_simd(x)],
        simd,
    )
}

#[inline(always)]
pub(crate) fn map_vectors_different_width<
    T,
    U,
    const N: usize,
    const KT: usize,
    const KU: usize,
    S,
>(
    xs: [T; N],
    f_scalar: impl FnMut(T) -> U,
    mut f_simd: impl FnMut([T::V<S>; KT]) -> [U::V<S>; KU],
    simd: S,
) -> [U; N]
where
    T: SimdElementExt,
    U: SimdElementExt,
    S: Simd,
{
    let in_lanes = <T::V<S> as SimdBase<S>>::LEN;
    let out_lanes = <U::V<S> as SimdBase<S>>::LEN;
    const { assert!(KT * <T::V<S> as SimdBase<S>>::LEN == KU * <U::V<S> as SimdBase<S>>::LEN) };
    let chunk = KT * in_lanes;
    if !N.is_multiple_of(chunk) {
        return xs.map_inline(f_scalar);
    }
    let mut ys = [U::default(); N];
    for_each_unrolled(
        N / chunk,
        #[inline(always)]
        |c| {
            let x = array_from_fn_inline(|k| {
                <T::V<S> as SimdBase<S>>::from_slice(
                    simd,
                    &xs[c * chunk + k * in_lanes..][..in_lanes],
                )
            });
            let y = f_simd(x);
            for k in 0..KU {
                y[k].store_slice(&mut ys[c * chunk + k * out_lanes..][..out_lanes]);
            }
        },
    );
    ys
}

#[inline(always)]
pub(crate) fn map_vectors2<T1, T2, U, const N: usize, S>(
    xs: [T1; N],
    ys: [T2; N],
    mut f_scalar: impl FnMut(T1, T2) -> U,
    mut f_simd: impl FnMut(T1::V<S>, T2::V<S>) -> U::V<S>,
    simd: S,
) -> [U; N]
where
    T1: SimdElementExt,
    T2: SimdElementExt,
    U: SimdElementExt,
    S: Simd,
{
    let lanes = const {
        let lanes = <T1::V<S> as SimdBase<S>>::LEN;
        assert!(lanes == <T2::V<S> as SimdBase<S>>::LEN && lanes == <U::V<S> as SimdBase<S>>::LEN);
        lanes
    };
    if !N.is_multiple_of(lanes) {
        return array_from_fn_inline(
            #[inline(always)]
            move |i| f_scalar(xs[i], ys[i]),
        );
    }
    let mut zs = [U::default(); N];
    for_each_unrolled(
        N / lanes,
        #[inline(always)]
        |c| {
            let x = <T1::V<S> as SimdBase<S>>::from_slice(simd, &xs[c * lanes..][..lanes]);
            let y = <T2::V<S> as SimdBase<S>>::from_slice(simd, &ys[c * lanes..][..lanes]);
            f_simd(x, y).store_slice(&mut zs[c * lanes..][..lanes]);
        },
    );
    zs
}

#[inline(always)]
pub(crate) fn map_vectors_to_bools<T, const N: usize, S>(
    xs: [T; N],
    f_scalar: impl Fn(T) -> bool,
    f_simd: impl Fn(T::V<S>) -> <T::V<S> as SimdBase<S>>::Mask,
    simd: S,
) -> [bool; N]
where
    T: SimdElementExt,
    S: Simd,
{
    let lanes = <T::V<S> as SimdBase<S>>::LEN;
    map_masks_to_bools::<T, N, S>(
        #[inline(always)]
        |i| f_scalar(xs[i]),
        #[inline(always)]
        |i| {
            let x = <T::V<S> as SimdBase<S>>::from_slice(simd, &xs[i..][..lanes]);
            f_simd(x)
        },
        simd,
    )
}

#[inline(always)]
pub(crate) fn map_vectors2_to_bools<T, const N: usize, S>(
    xs: [T; N],
    ys: [T; N],
    f_scalar: impl Fn(T, T) -> bool,
    f_simd: impl Fn(T::V<S>, T::V<S>) -> <T::V<S> as SimdBase<S>>::Mask,
    simd: S,
) -> [bool; N]
where
    T: SimdElementExt,
    S: Simd,
{
    let lanes = <T::V<S> as SimdBase<S>>::LEN;
    map_masks_to_bools::<T, N, S>(
        #[inline(always)]
        |i| f_scalar(xs[i], ys[i]),
        #[inline(always)]
        |i| {
            let x = <T::V<S> as SimdBase<S>>::from_slice(simd, &xs[i..][..lanes]);
            let y = <T::V<S> as SimdBase<S>>::from_slice(simd, &ys[i..][..lanes]);
            f_simd(x, y)
        },
        simd,
    )
}

#[inline(always)]
fn map_masks_to_bools<T, const N: usize, S>(
    f_scalar: impl Fn(usize) -> bool,
    f_mask: impl Fn(usize) -> <T::V<S> as SimdBase<S>>::Mask,
    simd: S,
) -> [bool; N]
where
    T: SimdElementExt,
    S: Simd,
{
    use fearless_simd::SimdMask;
    let lanes_t = <T::V<S> as SimdBase<S>>::LEN;
    let lanes_u8 = <S::u8s as SimdBase<S>>::LEN;
    // static analysis shows the simd path is only beneficial on AVX2
    if !N.is_multiple_of(lanes_u8) || simd_level(simd) != SimdLevel::Avx2 {
        return array_from_fn_inline(f_scalar);
    }
    let mut ys = [0i8; N];
    for_each_unrolled(
        N / lanes_u8,
        #[inline(always)]
        |c| {
            let m = T::mask8s_from_mask::<S>(
                #[inline(always)]
                |k| f_mask(c * lanes_u8 + k * lanes_t),
            );
            // `-1` / `0` bytes to `1` / `0`.
            let y = m.to_vector() & 1;
            y.store_slice(&mut ys[c * lanes_u8..][..lanes_u8]);
        },
    );
    ys.map_inline(
        #[inline(always)]
        |y| y != 0,
    )
}

#[inline(always)]
fn for_each_unrolled(n: usize, mut f: impl FnMut(usize)) {
    #[cfg(debug_assertions)]
    for i in 0..n {
        f(i);
    }
    #[cfg(not(debug_assertions))]
    {
        macro_rules! calls {
            ($($i:literal)*) => {$(
                if $i < n {
                    f($i);
                }
            )*};
        }
        calls!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31);
        for i in 32..n {
            f(i);
        }
    }
}
