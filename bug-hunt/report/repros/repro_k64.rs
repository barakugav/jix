//! K64: `Array::to_ndarray_buf` in pull mode (`out = None`) routes every region above
//! `read_size.min` of a storage without PLAIN_READ (Broadcast, element-wise pipelines, ...) to the
//! block-splitting slow path, which allocates the WHOLE region and pushes every piece into it.
//! The storage itself answers the same pull request with an O(1) 0-stride view.
//! Ported to main 1169965 (#106); file renamed from repro_2026_10_23_batch3__f024.rs.
//!
//! The tests assert the correct (cheap) behavior, so they fail while the bug exists.
//!
//! WARNING: `pull_of_huge_broadcast` is `#[ignore]`d because it ABORTS THE PROCESS
//! ("memory allocation of 1099511627776 bytes failed", SIGABRT). Run it on its own with
//! `--test repro_k64 -- --ignored --exact pull_of_huge_broadcast`.
//!
//! Ported: `Array::to_ndarray_buf(index, ctx, None)` is now `Array::to_buf_strided(index, ctx)`.
//! `ArrayStorage::read_data` can no longer be called from outside the crate (`OutBuf` has no
//! public constructor), so each `storage().read_data(index, ctx, None)` control is replaced by
//! `to_buf_strided` on the same view of an array whose `read_size` floor covers the whole region
//! (`direct_params`): that region counts as small, so `to_buf_strided` skips the splitting path
//! and returns the storage's own pull buffer unchanged.
#![allow(clippy::single_range_in_vec_init)]

use jix::{Array, ArrayParams, ReadContext};
use ndarray::array;

/// Params whose `read_size` floor (4 TiB) covers every region read here, so `to_buf_strided`
/// forwards the read to the storage's own pull without splitting it.
fn direct_params() -> ArrayParams {
    ArrayParams::new().read_size((1 << 42, 1 << 42))
}

#[test]
fn pull_of_large_broadcast_keeps_zero_strides() {
    let n = 4096u64;
    let b = Array::compact_ndarray(&array![[1.5f32]])
        .unwrap()
        .broadcast(&[n, n]);
    let ctx = ReadContext::default();
    let index = [0..n, 0..n];

    let b_direct = Array::compact_ndarray_with(&array![[1.5f32]], direct_params())
        .unwrap()
        .broadcast(&[n, n]);
    let direct = b_direct.to_buf_strided(&index, &ctx).unwrap();
    println!(
        "broadcast storage().read_data pull strides: {:?}",
        direct.strides()
    );
    assert_eq!(direct.strides(), &[0, 0]);
    drop(direct);

    let t = std::time::Instant::now();
    let via_array = b.to_buf_strided(&index, &ctx).unwrap();
    println!(
        "broadcast Array::to_buf_strided pull strides: {:?}, {:?}",
        via_array.strides(),
        t.elapsed()
    );
    // The value is right either way - only the cost differs.
    let v = unsafe { via_array.data_ptr().cast::<f32>().read_unaligned() };
    assert_eq!(v, 1.5);
    assert_eq!(
        via_array.strides(),
        &[0, 0],
        "pull materialized a 64 MiB region"
    );
}

#[test]
fn pull_of_large_neg_broadcast_keeps_zero_strides() {
    let n = 4096u64;
    let b = Array::compact_ndarray(&array![[1.5f32]])
        .unwrap()
        .broadcast(&[n, n]);
    let neg = -b;
    let ctx = ReadContext::default();
    let index = [0..n, 0..n];

    let neg_direct = -Array::compact_ndarray_with(&array![[1.5f32]], direct_params())
        .unwrap()
        .broadcast(&[n, n]);
    let direct = neg_direct.to_buf_strided(&index, &ctx).unwrap();
    println!(
        "neg storage().read_data pull strides: {:?}",
        direct.strides()
    );
    assert_eq!(direct.strides(), &[0, 0]);
    drop(direct);

    let t = std::time::Instant::now();
    let via_array = neg.to_buf_strided(&index, &ctx).unwrap();
    println!(
        "neg Array::to_buf_strided pull strides: {:?}, {:?}",
        via_array.strides(),
        t.elapsed()
    );
    let v = unsafe { via_array.data_ptr().cast::<f32>().read_unaligned() };
    assert_eq!(v, -1.5);
    assert_eq!(
        via_array.strides(),
        &[0, 0],
        "pull ran the kernel 2^24 times"
    );
}

/// ABORTS THE PROCESS while the bug exists: 2^38 f32 = 1 TiB pooled allocation.
#[test]
#[ignore]
fn pull_of_huge_broadcast() {
    let (n, m) = (1u64 << 20, 1u64 << 18);
    let b = Array::compact_ndarray(&array![[1.5f32]])
        .unwrap()
        .broadcast(&[n, m]);
    let ctx = ReadContext::default();
    let index = [0..n, 0..m];
    let b_direct = Array::compact_ndarray_with(&array![[1.5f32]], direct_params())
        .unwrap()
        .broadcast(&[n, m]);
    let direct = b_direct.to_buf_strided(&index, &ctx).unwrap();
    println!("storage().read_data pull strides: {:?}", direct.strides());
    drop(direct);
    let r = b
        .to_buf_strided(&index, &ctx)
        .map(|buf| buf.strides().to_vec());
    assert_eq!(r.unwrap(), vec![0, 0]);
}

/// The same value array built two ways: a `Scalar` storage (doc-hidden `__private`, what jix-py
/// builds for Python scalars) keeps PLAIN_READ and gets the 0-stride view through
/// `to_ndarray_buf`; the equivalent public broadcast loses it.
#[test]
fn scalar_and_broadcast_pull_agree() {
    let n = 4096u64;
    let ctx = ReadContext::default();
    let index = [0..n, 0..n];
    let s = Array::from_storage(
        jix::__private::Scalar::new(1.5f32, &[n, n], jix::ArrayParams::default()).unwrap(),
    );
    let s_strides = s.to_buf_strided(&index, &ctx).unwrap().strides().to_vec();
    let b = Array::compact_ndarray(&array![[1.5f32]])
        .unwrap()
        .broadcast(&[n, n]);
    let b_strides = b.to_buf_strided(&index, &ctx).unwrap().strides().to_vec();
    println!("plain_scalar pull strides: {s_strides:?}, broadcast pull strides: {b_strides:?}");
    assert_eq!(b_strides, s_strides);
}

/// A zero-copy re-strided view (PermuteDims over Plain) is turned into a full copy as well.
#[test]
fn pull_of_permuted_plain_stays_a_view() {
    let n = 2048usize;
    let nd = ndarray::Array2::from_shape_fn((n, n), |(i, j)| (i * n + j) as f32);
    let p = Array::plain_ndarray_view(nd.view())
        .unwrap()
        .permute_dims(&[1, 0]);
    let ctx = ReadContext::default();
    let index = [0..n as u64, 0..n as u64];
    let p_direct = Array::plain_ndarray_view_with(nd.view(), direct_params())
        .unwrap()
        .permute_dims(&[1, 0]);
    let direct = p_direct.to_buf_strided(&index, &ctx).unwrap();
    let direct_ptr = direct.data_ptr();
    println!(
        "permuted plain storage().read_data pull strides: {:?}",
        direct.strides()
    );
    drop(direct);
    let via_array = p.to_buf_strided(&index, &ctx).unwrap();
    println!(
        "permuted plain Array::to_buf_strided pull strides: {:?}, same memory as source: {}",
        via_array.strides(),
        via_array.data_ptr() == direct_ptr
    );
    assert_eq!(
        via_array.data_ptr(),
        direct_ptr,
        "pull copied a 16 MiB view"
    );
}
