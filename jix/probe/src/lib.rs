//! Asm probe for the byte-shuffle filter kernels.
//!
//! The kernels are `#[path]`-included straight from the `jix` sources, so this crate compiles the
//! exact code `jix` ships, without `jix`'s C dependencies (zstd), for any rustup target. Every
//! kernel instantiation `jix` uses is reached from an exported `probe_*` function, so it is
//! codegened in this crate and visible to `cargo asm`.
//!
//! Driven by `analyze.py`, see `README.md`.

#[path = "../../src/codec/filter/byte_shuffle/kernels.rs"]
pub mod byte_shuffle;

macro_rules! probe {
    ($($name:ident => $f:path;)*) => {
        $(
            #[unsafe(no_mangle)]
            pub fn $name(src: &[u8], dst: &mut [u8]) {
                $f(src, dst)
            }
        )*
    };
}

probe! {
    probe_byte_shuffle_encode_2 => byte_shuffle::encode_impl::<2, 64>;
    probe_byte_shuffle_encode_4 => byte_shuffle::encode_impl::<4, 32>;
    probe_byte_shuffle_encode_8 => byte_shuffle::encode_impl::<8, 16>;
    probe_byte_shuffle_encode_16 => byte_shuffle::encode_impl::<16, 8>;
    probe_byte_shuffle_decode_2 => byte_shuffle::decode_impl::<2, 64>;
    probe_byte_shuffle_decode_4 => byte_shuffle::decode_impl::<4, 32>;
    probe_byte_shuffle_decode_8 => byte_shuffle::decode_impl::<8, 16>;
    probe_byte_shuffle_decode_16 => byte_shuffle::decode_impl::<16, 8>;
}

/// Runtime-itemsize fallback (odd itemsizes, struct dtypes, and the `<LANES` tail).
#[unsafe(no_mangle)]
pub fn probe_byte_shuffle_encode_generic(src: &[u8], dst: &mut [u8], itemsize: usize) {
    byte_shuffle::encode_impl_generic(src, dst, itemsize, 0)
}

/// Runtime-itemsize fallback (odd itemsizes, struct dtypes, and the `<LANES` tail).
#[unsafe(no_mangle)]
pub fn probe_byte_shuffle_decode_generic(src: &[u8], dst: &mut [u8], itemsize: usize) {
    byte_shuffle::decode_impl_generic(src, dst, itemsize, 0)
}
