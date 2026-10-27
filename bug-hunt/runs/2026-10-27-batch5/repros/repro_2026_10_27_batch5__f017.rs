// Repro for bug-hunt run 2026-10-27-batch5, finding F017.
//
// An empty array (a 0-length dim) whose other dims multiply to nearly 2^64 is accepted by
// compact_fn_with / write_to (check_shape_overflow: product with zeros as 1, times itemsize, fits
// u64), but the archive reader rejects the file jix just wrote: it runs try_shape_product over the
// PADDED shape (dims rounded up to a multiple of the block dim), which counts the 0 dim as 1 for
// its overflow guard, so the padding pushes the guard past u64::MAX although nitems is 0.
//
// The tests assert the CORRECT behavior (the archive round-trips), so they fail while the bug
// exists.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};

fn round_trip(shape: [u64; 3], params: ArrayParams) {
    let a = Array::compact_fn_with(shape, params, |_| 0u8).expect("valid empty array");
    assert_eq!(a.shape(), &shape);
    // The same array round-trips through an in-memory compact.
    assert_eq!(a.view().compact().expect("in-memory compact").shape(), &shape);

    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).expect("write_to accepts the array");
    let bytes = buf.into_inner();
    let len = bytes.len() as u64;
    let b = Array::read_from_reader(
        Cursor::new(bytes),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .expect("an empty array written by jix must read back");
    assert_eq!(b.shape(), &shape);
}

/// 3 * 0x5555_5555_5555_5555 == u64::MAX; block dim 2 pads the 3 to 4.
#[test]
fn empty_array_explicit_block_shape_round_trips() {
    let mut p = ArrayParams::new();
    p.block_shape(&[1, 2, 1]);
    round_trip([0, 3, 0x5555_5555_5555_5555], p);
}

/// (2^32 - 1) * (2^32 + 1) == u64::MAX; the auto block shape pads the last dim.
#[test]
fn empty_array_auto_block_shape_round_trips() {
    round_trip([0, (1 << 32) - 1, (1 << 32) + 1], ArrayParams::new());
}
