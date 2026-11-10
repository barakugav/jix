//! Scratch probe: raw memory bandwidth of this machine, to calibrate what a big read can reach.
//!
//! Times copying a cache-resident source (like a decoded block) into a large destination, with
//! regular and with non-temporal stores, plus plain memcpy / memset over the whole buffer.

#![allow(clippy::all)] // scratch probe

use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn best(mut f: impl FnMut()) -> f64 {
    f();
    (0..7)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::MAX, f64::min)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_nt(src: *const u8, dst: *mut u8, len: usize) {
    let mut i = 0;
    while i + 32 <= len {
        unsafe {
            let v = _mm256_loadu_si256(src.add(i).cast());
            _mm256_stream_si256(dst.add(i).cast(), v);
        }
        i += 32;
    }
}

fn main() {
    let total: usize = 256 << 20;
    let mut dst = vec![0u8; total + 64];
    let off = dst.as_ptr().align_offset(64);
    let dst_ptr = unsafe { dst.as_mut_ptr().add(off) };
    let src_big = vec![1u8; total];
    let mb = total as f64 / 1e6;

    let t = best(|| unsafe { std::ptr::copy_nonoverlapping(src_big.as_ptr(), dst_ptr, total) });
    println!(
        "memcpy  big->big          {:7.1} ms  {:8.0} MB/s",
        t * 1e3,
        mb / t
    );
    let t = best(|| unsafe { std::ptr::write_bytes(dst_ptr, 7, total) });
    println!(
        "memset  big               {:7.1} ms  {:8.0} MB/s",
        t * 1e3,
        mb / t
    );

    for block in [32 << 10, 256 << 10, 1 << 20] {
        let src = vec![3u8; block];
        let t = best(|| {
            for i in 0..total / block {
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr.add(i * block), block)
                };
            }
        });
        println!(
            "memcpy  {:4}K block->big   {:7.1} ms  {:8.0} MB/s",
            block >> 10,
            t * 1e3,
            mb / t
        );
        #[cfg(target_arch = "x86_64")]
        {
            let t = best(|| {
                for i in 0..total / block {
                    unsafe { copy_nt(src.as_ptr(), dst_ptr.add(i * block), block) };
                }
                unsafe { _mm_sfence() };
            });
            println!(
                "nt copy {:4}K block->big   {:7.1} ms  {:8.0} MB/s",
                block >> 10,
                t * 1e3,
                mb / t
            );
        }
        // Same source into a destination that stays in cache: the no-DRAM reference.
        let mut small = vec![0u8; block];
        let t = best(|| {
            for _ in 0..total / block {
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), small.as_mut_ptr(), block) };
                std::hint::black_box(&mut small);
            }
        });
        println!(
            "memcpy  {:4}K block->cache {:7.1} ms  {:8.0} MB/s",
            block >> 10,
            t * 1e3,
            mb / t
        );
    }
    std::hint::black_box(&dst);
}
