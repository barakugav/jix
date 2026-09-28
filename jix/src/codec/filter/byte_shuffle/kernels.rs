//! Byte-shuffle kernels.
//!
//! This file is intentionally self-contained (only `core`/`std`, plus the optional
//! `multiversion` attribute) so that `jix/probe` can `#[path]`-include it and compile it for any
//! target without pulling in the C dependencies of `jix` (zstd). Keep it that way.

/// Byte-shuffle `src` into `dst` for elements of `itemsize` bytes.
pub fn encode(src: &[u8], dst: &mut [u8], itemsize: usize) {
    assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(itemsize));
    match itemsize {
        1 => dst.copy_from_slice(src), // identity permutation
        2 => encode_impl::<2, 64>(src, dst),
        4 => encode_impl::<4, 32>(src, dst),
        8 => encode_impl::<8, 16>(src, dst),
        16 => encode_impl::<16, 8>(src, dst),
        _ => encode_impl_generic(src, dst, itemsize, 0),
    }
}

/// Inverse of [`encode`].
pub fn decode(src: &[u8], dst: &mut [u8], itemsize: usize) {
    assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(itemsize));
    match itemsize {
        1 => dst.copy_from_slice(src), // identity permutation
        2 => decode_impl::<2, 64>(src, dst),
        4 => decode_impl::<4, 32>(src, dst),
        8 => decode_impl::<8, 16>(src, dst),
        16 => decode_impl::<16, 8>(src, dst),
        _ => decode_impl_generic(src, dst, itemsize, 0),
    }
}

#[inline(never)]
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
pub fn encode_impl<const ITEMSIZE: usize, const LANES: usize>(src: &[u8], dst: &mut [u8]) {
    let nitems = src.len() / ITEMSIZE;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut i = 0;
    let body_limit = nitems - nitems % LANES;
    while i < body_limit {
        let elms = unsafe {
            src_ptr
                .cast::<[u8; ITEMSIZE]>()
                .add(i)
                .cast::<[[u8; ITEMSIZE]; LANES]>()
                .read()
        };
        #[allow(clippy::needless_range_loop)]
        for b in 0..ITEMSIZE {
            let byte_elms = std::array::from_fn(|j| elms[j][b]);
            unsafe {
                dst_ptr
                    .add(b * nitems + i)
                    .cast::<[u8; LANES]>()
                    .write(byte_elms);
            }
        }
        i += LANES;
    }
    // Tail of the remaining <LANES items
    encode_impl_generic(src, dst, ITEMSIZE, i);
}

#[inline(never)]
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
pub fn encode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));
    let nitems = src.len() / itemsize;
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    for b in 0..itemsize {
        for i in start..nitems {
            unsafe {
                let elm = src.add(i * itemsize + b).read();
                dst.add(b * nitems + i).write(elm);
            }
        }
    }
}

#[inline(never)]
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
pub fn decode_impl<const ITEMSIZE: usize, const LANES: usize>(src: &[u8], dst: &mut [u8]) {
    let nitems = src.len() / ITEMSIZE;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut i = 0;
    let body_limit = nitems - nitems % LANES;
    while i < body_limit {
        let mut elms = [[std::mem::MaybeUninit::<u8>::uninit(); ITEMSIZE]; LANES];
        #[allow(clippy::needless_range_loop)]
        for b in 0..ITEMSIZE {
            let byte_elms = unsafe { src_ptr.add(b * nitems + i).cast::<[u8; LANES]>().read() };
            for j in 0..LANES {
                elms[j][b].write(byte_elms[j]);
            }
        }
        // SAFETY: every one of the ITEMSIZE * LANES entries was written above.
        let elms = unsafe { std::mem::transmute_copy::<_, [[u8; ITEMSIZE]; LANES]>(&elms) };
        unsafe {
            dst_ptr
                .cast::<[u8; ITEMSIZE]>()
                .add(i)
                .cast::<[[u8; ITEMSIZE]; LANES]>()
                .write(elms);
        }
        i += LANES;
    }
    // Tail of the remaining <LANES items
    decode_impl_generic(src, dst, ITEMSIZE, i);
}

#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
#[inline(never)]
pub fn decode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));
    let nitems = src.len() / itemsize;
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    for i in start..nitems {
        for b in 0..itemsize {
            unsafe {
                let elm = src.add(b * nitems + i).read();
                dst.add(i * itemsize + b).write(elm);
            }
        }
    }
}
