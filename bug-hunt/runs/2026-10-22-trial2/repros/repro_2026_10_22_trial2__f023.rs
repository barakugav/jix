// F023: Repeat computes `out_strides[k] * repeats` (repeat.rs:181) even when output dim k has
// length 1, whose out stride is never stepped and so may be anything. When the product exceeds
// usize::MAX, a debug build (the default `cargo build` / `cargo test` profile for downstream crates
// too) panics with "attempt to multiply with overflow" on valid input.
//
// These tests assert the CORRECT result, so they fail while the bug exists.
use jix::storage::StridedBuf;
use jix::{Array, ReadContext};
use ndarray::array;

/// Huge stride on the length-1 out dim, small repeat count.
#[test]
fn repeat_len1_out_dim_huge_stride() {
    let a = Array::compact_ndarray(&array![[1u8, 2, 3, 4]]).unwrap();
    let r = a.repeat(4, 0); // shape [4, 4]
    let mut dst = [0u8; 4];
    let ctx = ReadContext::default();
    {
        // SAFETY: dim 0 has length 1 so its stride is never stepped; the span is 4 bytes, all of
        // `dst`, and no two indices map to the same byte.
        let mut out = unsafe {
            StridedBuf::from_raw_parts_mut(dst.as_mut_ptr(), &[1, 4], &[1usize << 62, 1], 1)
        };
        r.to_ndarray_buf(&[1..2, 0..4], &ctx, Some(&mut out))
            .unwrap();
    }
    assert_eq!(dst, [1, 2, 3, 4]);
}

/// Ordinary out stride (a 16 KiB row pitch, e.g. one row of a wider buffer) with a large lazy
/// repeat count: 2^14 * 2^50 = 2^64 overflows.
#[test]
fn repeat_len1_out_dim_row_pitch_large_repeats() {
    let a = Array::compact_ndarray(&array![[1u8, 2, 3, 4]]).unwrap();
    let r = a.repeat(1u64 << 50, 0); // shape [2^50, 4], lazy
    let mut backing = vec![0u8; 2 << 14];
    let ctx = ReadContext::default();
    {
        // SAFETY: shape [1, 4] at strides [16384, 1] touches bytes 0..4 of `backing`.
        let mut out = unsafe {
            StridedBuf::from_raw_parts_mut(backing.as_mut_ptr(), &[1, 4], &[1usize << 14, 1], 1)
        };
        r.to_ndarray_buf(&[5..6, 0..4], &ctx, Some(&mut out))
            .unwrap();
    }
    assert_eq!(&backing[..4], &[1, 2, 3, 4]);
}
