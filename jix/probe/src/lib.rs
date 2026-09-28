//! Asm probe for the byte-shuffle filter kernels (decode main loop only).
//!
//! The kernels are `#[path]`-included straight from the `jix` sources, so this crate compiles the
//! exact code `jix` ships, without `jix`'s C dependencies (zstd), for any rustup target.
//!
//! `jix` reaches `decode_simd` through `fearless_simd::dispatch!`, which runs it inside a
//! `#[target_feature]` function of the detected level. Here the level is chosen statically instead:
//! `analyze.py` compiles the whole crate with that level's target features, and `simd()` returns
//! the matching token, so each `probe_*` function contains the same code as that dispatch arm.
//!
//! Driven by `analyze.py`, see `README.md`.

#[path = "../../src/codec/filter/byte_shuffle/kernels.rs"]
pub mod byte_shuffle;

/// The fearless_simd token of the level enabled at compile time.
#[inline(always)]
#[allow(unreachable_code)]
fn simd() -> impl fearless_simd::Simd {
    // SAFETY: the probe is only compiled, never run.
    #[cfg(target_feature = "avx512vbmi")]
    return unsafe { fearless_simd::Avx512::assume_supported() };
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512vbmi")))]
    return unsafe { fearless_simd::Avx2::assume_supported() };
    #[cfg(all(target_feature = "sse4.2", not(target_feature = "avx2")))]
    return unsafe { fearless_simd::Sse4_2::assume_supported() };
    #[cfg(all(target_feature = "sse2", not(target_feature = "sse4.2")))]
    return unsafe { fearless_simd::Sse2::assume_supported() };
    #[cfg(target_arch = "aarch64")]
    return unsafe { fearless_simd::Neon::assume_supported() };
}

macro_rules! probe {
    ($($name:ident => $itemsize:literal;)*) => {
        $(
            #[unsafe(no_mangle)]
            pub fn $name(src: &[u8], dst: &mut [u8]) -> usize {
                byte_shuffle::decode_simd::<_, $itemsize>(simd(), src, dst)
            }
        )*
    };
}

probe! {
    probe_byte_shuffle_decode_2 => 2;
    probe_byte_shuffle_decode_4 => 4;
    probe_byte_shuffle_decode_8 => 8;
    probe_byte_shuffle_decode_16 => 16;
}
