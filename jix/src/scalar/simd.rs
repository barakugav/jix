//! Building blocks of the SIMD `*_bulk` functions of the scalar traits.

use crate::util::{array_from_fn_inline, ArrayExt};
use fearless_simd::{Simd, SimdBase, SimdElement};

/// The fearless_simd levels a SIMD body can be limited to.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Each target constructs only the variants of its arch.
pub(crate) enum SimdLevel {
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

/// An element type with a native SIMD vector type.
pub(crate) trait SimdLane: SimdElement {
    type V<S: Simd>: SimdBase<S, Element = Self>;

    /// The byte mask of the `size_of::<Self>()` masks `mask(0..)` of consecutive vectors of
    /// `Self`, narrowed in order.
    fn narrow_masks<S: Simd>(
        mask: impl Fn(usize) -> <Self::V<S> as SimdBase<S>>::Mask,
    ) -> S::mask8s;
}
macro_rules! impl_simd_lane {
    ($($t:ident => $v:ident, |$mask:ident| $narrow:expr;)*) => {$(
        impl SimdLane for $t {
            type V<S: Simd> = S::$v;

            #[inline(always)]
            fn narrow_masks<S: Simd>(
                $mask: impl Fn(usize) -> <Self::V<S> as SimdBase<S>>::Mask,
            ) -> S::mask8s {
                #[allow(unused_imports)]
                use fearless_simd::MaskNarrow;
                $narrow
            }
        }
    )*};
}
impl_simd_lane!(
    i8 => i8s, |m| m(0);
    u8 => u8s, |m| m(0);
    i16 => i16s, |m| m(0).narrow(m(1));
    u16 => u16s, |m| m(0).narrow(m(1));
    i32 => i32s, |m| m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
    u32 => u32s, |m| m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
    f32 => f32s, |m| m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
    i64 => i64s, |m| {
        let lo = m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
        lo.narrow(m(4).narrow(m(5)).narrow(m(6).narrow(m(7))))
    };
    u64 => u64s, |m| {
        let lo = m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
        lo.narrow(m(4).narrow(m(5)).narrow(m(6).narrow(m(7))))
    };
    f64 => f64s, |m| {
        let lo = m(0).narrow(m(1)).narrow(m(2).narrow(m(3)));
        lo.narrow(m(4).narrow(m(5)).narrow(m(6).narrow(m(7))))
    };
);

/// The `bool`s of the masks `mask(i)` of the vectors of `A` at the elements `i` (of `N`): a vector
/// of bytes at a time, of `size_of::<A>()` masks narrowed to bytes. `scalar(i)` per element if `N`
/// is not a multiple of the lanes of a vector of bytes, or not on AVX2.
///
/// AVX2 only (static analysis): there LLVM's own vectorization of the scalar comparisons is
/// slower, by up to 1.9x (comparisons, casts to `bool`) and 6.7x (`is_finite` / `is_infinite` of
/// `f64`). On SSE2 / SSE4.2 the narrowing (`packssdw` / `packsswb`) is mostly what LLVM emits
/// anyway; on AVX-512 LLVM stores a compare's `k` mask directly (`vmovdqu8 {k}`), where the
/// narrowing of fearless_simd's masks goes through general registers (0.5-0.9x); on NEON LLVM
/// splits the narrowing of `vmovn` halves into lane moves (0.1-0.6x).
#[inline(always)]
pub(crate) fn masks_to_bools<S, A, const N: usize>(
    simd: S,
    scalar: impl Fn(usize) -> bool,
    mask: impl Fn(usize) -> <A::V<S> as SimdBase<S>>::Mask,
) -> [bool; N]
where
    S: Simd,
    A: SimdLane,
{
    use fearless_simd::SimdMask;
    let la = <A::V<S> as SimdBase<S>>::LEN;
    let bytes = <S::u8s as SimdBase<S>>::LEN;
    if !N.is_multiple_of(bytes) || simd_level(simd) != SimdLevel::Avx2 {
        return array_from_fn_inline(scalar);
    }
    let mut ys = [0i8; N];
    for_each_unrolled(
        N / bytes,
        #[inline(always)]
        |c| {
            let m = A::narrow_masks::<S>(
                #[inline(always)]
                |k| mask(c * bytes + k * la),
            );
            // `-1` / `0` bytes to `1` / `0`.
            let y = m.to_vector() & 1;
            y.store_slice(&mut ys[c * bytes..][..bytes]);
        },
    );
    ys.map_inline(
        #[inline(always)]
        |y| y != 0,
    )
}

/// [`masks_to_bools`] of `f` of the vectors of `xs`.
#[inline(always)]
pub(crate) fn map_vectors_to_bools<S, A, const N: usize>(
    simd: S,
    xs: [A; N],
    scalar: impl Fn(A) -> bool,
    f: impl Fn(A::V<S>) -> <A::V<S> as SimdBase<S>>::Mask,
) -> [bool; N]
where
    S: Simd,
    A: SimdLane,
{
    let la = <A::V<S> as SimdBase<S>>::LEN;
    masks_to_bools::<S, A, N>(
        simd,
        #[inline(always)]
        |i| scalar(xs[i]),
        #[inline(always)]
        |i| f(<A::V<S> as SimdBase<S>>::from_slice(simd, &xs[i..][..la])),
    )
}

/// [`masks_to_bools`] of `f` of the pairs of vectors of `xs` and `ys`.
#[inline(always)]
pub(crate) fn map_vectors2_to_bools<S, A, const N: usize>(
    simd: S,
    xs: [A; N],
    ys: [A; N],
    scalar: impl Fn(A, A) -> bool,
    f: impl Fn(A::V<S>, A::V<S>) -> <A::V<S> as SimdBase<S>>::Mask,
) -> [bool; N]
where
    S: Simd,
    A: SimdLane,
{
    let la = <A::V<S> as SimdBase<S>>::LEN;
    masks_to_bools::<S, A, N>(
        simd,
        #[inline(always)]
        |i| scalar(xs[i], ys[i]),
        #[inline(always)]
        |i| {
            let x = <A::V<S> as SimdBase<S>>::from_slice(simd, &xs[i..][..la]);
            let y = <A::V<S> as SimdBase<S>>::from_slice(simd, &ys[i..][..la]);
            f(x, y)
        },
    )
}

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
    for_each_unrolled(
        N / chunk,
        #[inline(always)]
        |c| {
            let x = array_from_fn_inline(|k| {
                <A::V<S> as SimdBase<S>>::from_slice(simd, &xs[c * chunk + k * la..][..la])
            });
            let y = f(x);
            for k in 0..KB {
                y[k].store_slice(&mut ys[c * chunk + k * lb..][..lb]);
            }
        },
    );
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
    for_each_unrolled(
        N / lanes,
        #[inline(always)]
        |c| {
            let x = <A::V<S> as SimdBase<S>>::from_slice(simd, &xs[c * lanes..][..lanes]);
            let y = <B::V<S> as SimdBase<S>>::from_slice(simd, &ys[c * lanes..][..lanes]);
            f(x, y).store_slice(&mut zs[c * lanes..][..lanes]);
        },
    );
    zs
}

/// `f(0)`, .., `f(n - 1)`, as separate calls in the source for `n <= 32` (`n` a constant after
/// inlining, so the guards fold away). The vectors of a `*_bulk` live in local arrays, which stay
/// in registers only if LLVM fully unrolls the loop over them, and it does not always: e.g. an
/// `i64` multiply of fearless_simd 1.1 (`pmuludq` steps) on SSE2 / SSE4.2 left a loop over the
/// arrays on the stack. Debug builds loop: unrolled, the code of every caller is multiplied (and
/// the test crate does not compile in 12 GB).
#[inline(always)]
pub(crate) fn for_each_unrolled(n: usize, mut f: impl FnMut(usize)) {
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
