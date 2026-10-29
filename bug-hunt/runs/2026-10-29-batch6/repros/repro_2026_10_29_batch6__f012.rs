//! Repro for F012 (run 2026-10-29-batch6).
//!
//! 32-bit only: an empty read region whose other extents multiply past usize::MAX passes
//! `check_region_size` / `check_get_buffer_size` (both bound the TRUE element product, which is 0),
//! and the usize stride / size math that follows overflows. A debug i686 build panics with
//! 'attempt to multiply with overflow'; release wraps silently.
//!
//! Each test asserts the correct behavior: `Ok` (nothing to read) or a clean
//! `Err(InvalidShapeOperation)` - the answer `to_ndarray_sub` already gives for the same region
//! since the K81 fix. A panic fails the test.
//!
//! x86_64: `cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_29_batch6__f012`
//! i686:   `bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_29_batch6__f012`
use jix::error::ErrorKind;
use jix::Array;
use ndarray::array;

const BIG: u64 = 1 << 20; // BIG * BIG = 2^40 > u32::MAX

fn assert_ok_or_clean_err<T>(r: Result<T, jix::Error>) {
    if let Err(e) = r {
        assert_eq!(e.kind(), ErrorKind::InvalidShapeOperation, "{e:?}");
    }
}

// to_ndarray_slice -> default_strides(read_shape) (util/mod.rs:141) multiplies 2^20 * 2^20.
#[test]
fn to_ndarray_slice_empty_region_huge_dims() {
    let b = Array::compact_ndarray(&array![[[7u8]]])
        .unwrap()
        .broadcast(&[1, BIG, BIG]);
    let ctx = b.read_ctx();
    let mut buf: [u8; 0] = [];
    assert_ok_or_clean_err(b.to_ndarray_slice(&[0..0, 0..BIG, 0..BIG], &mut buf, &ctx));
}

// Compact::read_data: `out_shape.iter().product::<usize>()` (compact.rs:295) before nitems == 0.
#[test]
fn to_ndarray_buf_pull_compact_empty_region_huge_dims() {
    let a = Array::compact_fn((BIG, BIG, 0u64), |_| 0u8).unwrap();
    let ctx = a.read_ctx();
    assert_ok_or_clean_err(a.to_ndarray_buf(&[0..BIG, 0..BIG, 0..0], &ctx, None).map(|_| ()));
}

// Broadcast pull -> materialize_out_buf (buf.rs:386) multiplies the same extents.
#[test]
fn to_ndarray_buf_pull_broadcast_empty_region_huge_dims() {
    let b = Array::compact_ndarray(&array![[[7u8]]])
        .unwrap()
        .broadcast(&[BIG, BIG, 1]);
    let ctx = b.read_ctx();
    assert_ok_or_clean_err(b.to_ndarray_buf(&[0..BIG, 0..BIG, 0..0], &ctx, None).map(|_| ()));
}

// Control: to_ndarray_sub (K81 fix, array.rs:775-783) on the same kind of region.
#[test]
fn to_ndarray_sub_empty_region_huge_dims_control() {
    let b = Array::compact_ndarray(&array![[[7u8]]])
        .unwrap()
        .broadcast(&[1, BIG, BIG]);
    let ctx = b.read_ctx();
    let r = b.to_ndarray_sub::<ndarray::Ix3>(&[0..0, 0..BIG, 0..BIG], &ctx);
    println!("to_ndarray_sub: {:?}", r.as_ref().map(|a| a.shape().to_vec()));
    assert_ok_or_clean_err(r);
}

// Same region through an element-wise op (map) in pull mode.
#[test]
fn to_ndarray_buf_pull_map_empty_region_huge_dims() {
    let m = Array::compact_ndarray(&array![[[7u8]]])
        .unwrap()
        .broadcast(&[BIG, BIG, 1])
        .map(|x: u8| x.wrapping_add(1));
    let ctx = m.read_ctx();
    assert_ok_or_clean_err(m.to_ndarray_buf(&[0..BIG, 0..BIG, 0..0], &ctx, None).map(|_| ()));
}
