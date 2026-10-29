//! F034: spans / products over an EMPTY region's extents do not stop at the zero extent, so the
//! never-stepped strides / extents of the other dims overflow (debug panic; release wraps and
//! happens to return the right empty result).
//!
//! Asserts the CORRECT behavior, so every test fails while the bug exists. The last test only
//! fails on a 32-bit target (`bug-hunt/tools/cargo-i686 test ...`); on 64-bit it passes.

use std::ptr::NonNull;

use jix::dtype::Dtyped;
use jix::storage::StridedBuf;
use jix::Array;
use ndarray::{Array3, Ix3};

/// `StridedBuf::from_raw_parts_mut`'s documented contract holds for an empty region with any
/// strides (no byte is reachable), yet the span computation overflows on the non-empty dims.
#[test]
fn from_raw_parts_empty_region_with_large_strides() {
    let data = NonNull::<u8>::dangling().as_ptr();
    let buf = unsafe {
        StridedBuf::from_raw_parts_mut(data, &[3, 3, 0], &[usize::MAX / 2, usize::MAX / 2, 8], 8)
    };
    assert_eq!(buf.strides().len(), 3);
}

/// A Plain view of a zero-element buffer whose non-empty dims have large strides (what NumPy's
/// `as_strided(np.zeros(1, np.uint32), (3, 3, 0), (2**63 - 1,) * 2 + (4,))` hands to jix-py;
/// u32 rather than f64 so the i686 run is not stopped first by the known f64 alignment gap, K8).
/// Plain::new's contract ("valid to read for all index combinations") is vacuously met: there are
/// none.
fn empty_plain_with_large_strides(
    backing: &[u32],
) -> Array<jix::storage::Plain<&'static (), jix::Ty<u32>>> {
    unsafe {
        Array::plain_ndarray_ptr(
            backing.as_ptr().cast::<u8>(),
            [3u64, 3, 0].as_slice(),
            &[usize::MAX / 2, usize::MAX / 2, 4],
            u32::DTYPE,
            Default::default(),
        )
        .unwrap()
    }
}

/// Plain::read_data computes `strided_span_bytes` for every request, even an empty one.
#[test]
fn plain_empty_read_with_large_strides() {
    let backing = [0u32];
    let e = empty_plain_with_large_strides(&backing);
    let ctx = e.read_ctx();
    let buf = e.to_ndarray_buf(&[0..3, 0..3, 0..0], &ctx, None).unwrap();
    assert_eq!(buf.strides().len(), 3);
}

/// Concatenate reads the zero-length middle member with an empty sub-index, which lands in
/// Plain::read_data -> strided_span_bytes (the Python trigger).
#[test]
fn concatenate_with_empty_strided_member() {
    let backing = [0u32];
    let ones = Array3::<u32>::ones((3, 3, 2));
    let y1 = Array::compact_ndarray(&ones).unwrap();
    let y2 = Array::compact_ndarray(&ones).unwrap();
    let e = empty_plain_with_large_strides(&backing);
    let c = jix::ops::concatenate((y1, e, y2), 2);
    let got = c.to_ndarray::<Ix3>().unwrap();
    assert_eq!(got, Array3::<u32>::ones((3, 3, 4)));
}

/// 32-bit only: an empty region whose non-zero extents multiply past usize::MAX. The array shape
/// is valid (its zero-padded product, 2^40, fits u64) and check_region_size accepts the region
/// (its byte size is 0), but materialize_out_buf's `usize` product of the extents overflows.
#[test]
fn empty_region_large_extents_pull() {
    let one = Array::compact_ndarray(&ndarray::array![[[7u8]]]).unwrap();
    let b = one.broadcast(&[1 << 20, 1 << 20, 0]);
    let full = [0..(1u64 << 20), 0..(1u64 << 20), 0..0];
    let ctx = b.read_ctx();
    let buf = b.to_ndarray_buf(&full, &ctx, None).unwrap();
    assert_eq!(buf.strides().len(), 3);
}
