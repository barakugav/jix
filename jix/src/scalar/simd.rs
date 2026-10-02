//! Building blocks of the SIMD `*_bulk` functions of the scalar traits.

use crate::util::{array_from_fn_inline, ArrayExt};
use fearless_simd::{Simd, SimdBase, SimdElement};

/// The fearless_simd levels a SIMD body can be limited to.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Each target constructs only the variants of its arch.
pub(crate) enum Level {
    Sse2,
    Sse4_2,
    Avx2,
    Avx512,
    Neon,
    /// The others (fallback, WASM).
    Other,
}

/// The level of `simd`. A constant for each `S`, so a check of it folds away.
#[inline(always)]
pub(crate) fn level<S: Simd>(simd: S) -> Level {
    match simd.level() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Sse2(_) => Level::Sse2,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Sse4_2(_) => Level::Sse4_2,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Avx2(_) => Level::Avx2,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fearless_simd::Level::Avx512(_) => Level::Avx512,
        #[cfg(target_arch = "aarch64")]
        fearless_simd::Level::Neon(_) => Level::Neon,
        _ => Level::Other,
    }
}

/// An element type with a native SIMD vector type.
pub(crate) trait SimdLane: SimdElement {
    type V<S: Simd>: SimdBase<S, Element = Self>;
}
macro_rules! impl_simd_lane {
    ($($t:ident => $v:ident),*) => {$(
        impl SimdLane for $t {
            type V<S: Simd> = S::$v;
        }
    )*};
}
impl_simd_lane!(
    i8 => i8s, i16 => i16s, i32 => i32s, i64 => i64s,
    u8 => u8s, u16 => u16s, u32 => u32s, u64 => u64s,
    f32 => f32s, f64 => f64s
);

/// Map `xs` by `f`, which takes `KA` vectors of `A` and gives `KB` vectors of `B`, the same
/// elements. If `N` is not a multiple of the elements `f` takes, map by `scalar`.
#[inline(always)]
pub(crate) fn map_vectors<S, A, B, const KA: usize, const KB: usize, const N: usize>(
    simd: S,
    xs: [A; N],
    scalar: impl Fn(A) -> B,
    f: impl Fn([A::V<S>; KA]) -> [B::V<S>; KB],
) -> [B; N]
where
    S: Simd,
    A: SimdLane,
    B: SimdLane,
{
    let la = <A::V<S> as SimdBase<S>>::LEN;
    let lb = <B::V<S> as SimdBase<S>>::LEN;
    const { assert!(KA * <A::V<S> as SimdBase<S>>::LEN == KB * <B::V<S> as SimdBase<S>>::LEN) };
    let chunk = KA * la;
    if !N.is_multiple_of(chunk) {
        return xs.map_inline(scalar);
    }
    let mut ys = [B::default(); N];
    for c in 0..N / chunk {
        let x = array_from_fn_inline(|k| {
            <A::V<S> as SimdBase<S>>::from_slice(simd, &xs[c * chunk + k * la..][..la])
        });
        let y = f(x);
        for k in 0..KB {
            y[k].store_slice(&mut ys[c * chunk + k * lb..][..lb]);
        }
    }
    ys
}

/// Map the pairs of `xs` and `ys` by `f`, on vectors of as many lanes of `A`, `B` and `C`. If `N`
/// is not a multiple of the lanes, map by `scalar`.
#[inline(always)]
pub(crate) fn map_vectors2<S, A, B, C, const N: usize>(
    simd: S,
    xs: [A; N],
    ys: [B; N],
    scalar: impl Fn(A, B) -> C,
    f: impl Fn(A::V<S>, B::V<S>) -> C::V<S>,
) -> [C; N]
where
    S: Simd,
    A: SimdLane,
    B: SimdLane,
    C: SimdLane,
{
    let lanes = const {
        let lanes = <A::V<S> as SimdBase<S>>::LEN;
        assert!(lanes == <B::V<S> as SimdBase<S>>::LEN && lanes == <C::V<S> as SimdBase<S>>::LEN);
        lanes
    };
    if !N.is_multiple_of(lanes) {
        return array_from_fn_inline(
            #[inline(always)]
            |i| scalar(xs[i], ys[i]),
        );
    }
    let mut zs = [C::default(); N];
    for c in 0..N / lanes {
        let x = <A::V<S> as SimdBase<S>>::from_slice(simd, &xs[c * lanes..][..lanes]);
        let y = <B::V<S> as SimdBase<S>>::from_slice(simd, &ys[c * lanes..][..lanes]);
        f(x, y).store_slice(&mut zs[c * lanes..][..lanes]);
    }
    zs
}
