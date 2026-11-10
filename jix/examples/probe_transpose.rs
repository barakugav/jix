//! Scratch experiment: how to copy row bands of a big C-order matrix into its transpose.
//!
//! Models a transposed read: bands of `R` decoded source rows are gathered in a staging buffer
//! `T[R][N]` and written to `D[j][i] = S[i][j]`. Variants differ in the band size and in how the
//! band is copied out. Not part of the library - it picks the strategy for `NdCopier::copy_blocked`.

#![allow(clippy::all)] // scratch probe

use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const N: usize = 8192;

fn best(mut f: impl FnMut()) -> f64 {
    f();
    (0..3)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::MAX, f64::min)
}

/// dst-ordered direct copy of one band tile: rows `[i0, i0+r)`, cols `[j0, j0+c)`.
#[inline(always)]
unsafe fn tile_direct(
    t: *const i32,
    d: *mut i32,
    band0: usize,
    i0: usize,
    r: usize,
    j0: usize,
    c: usize,
) {
    for j in j0..j0 + c {
        for i in i0..i0 + r {
            unsafe { *d.add(j * N + band0 + i) = *t.add(i * N + j) };
        }
    }
}

/// Gather the tile into a compact buffer, transpose it there, then write out whole runs.
#[inline(always)]
unsafe fn tile_3step(
    t: *const i32,
    d: *mut i32,
    band0: usize,
    i0: usize,
    r: usize,
    j0: usize,
    c: usize,
    a: &mut [i32],
    b: &mut [i32],
    nt: bool,
) {
    unsafe {
        // 1. gather: r runs of c elements
        for i in 0..r {
            std::ptr::copy_nonoverlapping(t.add((i0 + i) * N + j0), a.as_mut_ptr().add(i * c), c);
        }
        // 2. transpose a[r][c] -> b[c][r], both compact
        for j in 0..c {
            for i in 0..r {
                *b.get_unchecked_mut(j * r + i) = *a.get_unchecked(i * c + j);
            }
        }
        // 3. scatter: c runs of r elements
        for j in 0..c {
            let dst = d.add((j0 + j) * N + band0 + i0);
            let src = b.as_ptr().add(j * r);
            if nt {
                copy_run_nt(src.cast(), dst.cast(), r * 4);
            } else {
                std::ptr::copy_nonoverlapping(src, dst, r);
            }
        }
    }
}

/// Copy `len` bytes with non-temporal stores for the 64-byte-aligned middle of the destination.
#[inline(always)]
unsafe fn copy_run_nt(src: *const u8, dst: *mut u8, len: usize) {
    unsafe {
        let head = dst.align_offset(64).min(len);
        std::ptr::copy_nonoverlapping(src, dst, head);
        let mut off = head;
        #[cfg(target_arch = "x86_64")]
        while off + 64 <= len {
            let v0 = _mm_loadu_si128(src.add(off).cast());
            let v1 = _mm_loadu_si128(src.add(off + 16).cast());
            let v2 = _mm_loadu_si128(src.add(off + 32).cast());
            let v3 = _mm_loadu_si128(src.add(off + 48).cast());
            _mm_stream_si128(dst.add(off).cast(), v0);
            _mm_stream_si128(dst.add(off + 16).cast(), v1);
            _mm_stream_si128(dst.add(off + 32).cast(), v2);
            _mm_stream_si128(dst.add(off + 48).cast(), v3);
            off += 64;
        }
        std::ptr::copy_nonoverlapping(src.add(off), dst.add(off), len - off);
    }
}

fn main() {
    let s: Vec<i32> = (0..N * N).map(|x| x as i32).collect();
    let mut d = vec![0i32; N * N];
    let mb = (N * N * 4) as f64 / 1e6;

    let check = |d: &[i32]| {
        for &(i, j) in &[(0, 0), (1, 7), (N - 1, 3), (5, N - 1), (4097, 2049)] {
            assert_eq!(d[j * N + i], s[i * N + j]);
        }
    };

    // Reference: plain copy through the same staging band (no transpose).
    {
        let band = 32;
        let mut t = vec![0i32; band * N];
        let time = best(|| {
            for b0 in (0..N).step_by(band) {
                t.copy_from_slice(&s[b0 * N..(b0 + band) * N]);
                d[b0 * N..(b0 + band) * N].copy_from_slice(&t);
            }
        });
        println!(
            "{:<44} {:7.1} ms {:7.0} MB/s",
            "straight copy via 1MB stage",
            time * 1e3,
            mb / time
        );
    }

    // The library's own nd copy on the same band shapes.
    for (band, shift) in [
        (16usize, 0usize),
        (32, 0),
        (256, 0),
        (256, 28),
        (256, 1000),
        (256, 12),
    ] {
        // `shift` moves the staging base relative to the page, in elements.
        let mut t_alloc = vec![0i32; band * N + 1024];
        let t = &mut t_alloc[shift..shift + band * N];
        let dtype = <i32 as jix::dtype::Dtyped>::DTYPE;
        let time = best(|| {
            for b0 in (0..N).step_by(band) {
                t.copy_from_slice(&s[b0 * N..(b0 + band) * N]);
                // region dims: [cols (dst rows), band rows]
                let src =
                    unsafe { std::slice::from_raw_parts(t.as_ptr().cast::<u8>(), band * N * 4) };
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        d.as_mut_ptr().add(b0).cast::<u8>(),
                        (N * N - b0) * 4,
                    )
                };
                unsafe {
                    jix::__private::bench_util::nd_copy(
                        src,
                        dst,
                        &[N, band],
                        &[4, N * 4],
                        &[N * 4, 4],
                        &dtype,
                    )
                };
            }
        });
        check(&d);
        let name = format!(
            "band={band:<4} jix nd_copy shift={shift} t={:p} d={:p}",
            t.as_ptr(),
            d.as_ptr()
        );
        println!("{name:<44} {:7.1} ms {:7.0} MB/s", time * 1e3, mb / time);
    }

    // (band rows, tile rows r, tile cols c, mode)
    let variants: &[(usize, usize, usize, &str)] = &[
        (16, 16, N, "direct"),
        (32, 32, N, "direct"),
        (64, 64, N, "direct"),
        (256, 256, 256, "direct"),
        (256, 64, 64, "direct"),
        (256, 16, 256, "direct"),
        (256, 8, 1024, "direct"),
        (256, 64, 64, "3step"),
        (256, 32, 128, "3step"),
        (256, 128, 32, "3step"),
        (256, 256, 16, "3step"),
        (256, 256, 64, "3step"),
        (256, 256, 256, "3step"),
        (256, 64, 64, "3step-nt"),
        (256, 128, 32, "3step-nt"),
        (256, 256, 16, "3step-nt"),
        (256, 256, 64, "3step-nt"),
        (1024, 256, 16, "3step-nt"),
        (1024, 512, 8, "3step-nt"),
        (1024, 1024, 16, "3step-nt"),
        (32, 32, 128, "3step"),
        (32, 32, 128, "3step-nt"),
        (64, 64, 64, "3step-nt"),
    ];
    for &(band, r, c, mode) in variants {
        let mut t = vec![0i32; band * N];
        let mut a = vec![0i32; r * c];
        let mut b = vec![0i32; r * c];
        let time = best(|| {
            for b0 in (0..N).step_by(band) {
                // stage: the decoded band
                t.copy_from_slice(&s[b0 * N..(b0 + band) * N]);
                let (tp, dp) = (t.as_ptr(), d.as_mut_ptr());
                for i0 in (0..band).step_by(r) {
                    for j0 in (0..N).step_by(c) {
                        unsafe {
                            match mode {
                                "direct" => tile_direct(tp, dp, b0, i0, r, j0, c),
                                "3step" => {
                                    tile_3step(tp, dp, b0, i0, r, j0, c, &mut a, &mut b, false)
                                }
                                _ => tile_3step(tp, dp, b0, i0, r, j0, c, &mut a, &mut b, true),
                            }
                        }
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                _mm_sfence()
            };
        });
        check(&d);
        let name = format!("band={band:<4} tile={r}x{c} {mode}");
        println!("{name:<44} {:7.1} ms {:7.0} MB/s", time * 1e3, mb / time);
    }
}
