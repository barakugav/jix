// F041 repro: an archive with one out-of-bounds block location, read with
// ArchiveValidation::Minimal. BlockTable documents "offset + len <= block_data.len()" for every
// block as an invariant, but read_content only checks it under Strict, and BlockTable::new only
// debug_asserts it. So the same file loads in a release build (intact blocks readable) but panics
// at LOAD in a debug build (assert at storage/block.rs:129).
//
// The test asserts the build-independent behavior: Minimal load succeeds and the intact blocks
// read back correctly. It FAILS in debug builds while the bug exists, passes with --release.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::{ArrayD, IxDyn};

/// Writes a u32 [64] array with block_shape [16] (4 blocks) and corrupts the len of block 3.
fn archive_with_bad_last_location() -> (Vec<u8>, ArrayD<u32>) {
    let data = ArrayD::from_shape_fn(IxDyn(&[64]), |i| i[0] as u32);
    let mut p = ArrayParams::new();
    p.block_shape(&[16]);
    let a = Array::compact_ndarray_with(&data, p).unwrap();
    let mut c = Cursor::new(Vec::new());
    a.write_to(&mut c).unwrap();
    let mut bytes = c.into_inner();

    // TOC = [Section; 2] = {offset: i64, size: u64} x 2. The first is blocks_loc (4 * 12 bytes),
    // and it starts right after the TOC (offset relative to the archive start = 0 here).
    let toc = (0..bytes.len() - 32)
        .find(|&p| {
            let off = i64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
            let size = u64::from_le_bytes(bytes[p + 8..p + 16].try_into().unwrap());
            size == 48 && off == (p + 32) as i64
        })
        .expect("TOC not found");
    // BlockLocation = {offset: u64, len: u32}, 12 bytes. Block 3's len := u32::MAX.
    let loc3 = toc + 32 + 3 * 12;
    bytes[loc3 + 8..loc3 + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    (bytes, data)
}

#[test]
fn strict_rejects_bad_location() {
    let (bytes, _) = archive_with_bad_last_location();
    let len = bytes.len() as u64;
    let r = Array::read_from_reader(
        Cursor::new(bytes),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    );
    assert!(r.is_err(), "Strict must reject the archive");
}

#[test]
fn minimal_load_does_not_depend_on_build_profile() {
    let (bytes, data) = archive_with_bad_last_location();
    let len = bytes.len() as u64;
    // Debug build: panics HERE, in BlockTable::new's debug_assert (storage/block.rs:129).
    let a = Array::read_from_reader(
        Cursor::new(bytes),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Minimal,
    )
    .expect("Minimal load should succeed (it skips per-block location checks)");
    let a = a.into_typed::<u32>().unwrap();
    let ctx = a.read_ctx();
    // Blocks 0..3 are intact.
    let got = a.to_ndarray_sub::<IxDyn>(&[0..48], &ctx).unwrap();
    assert_eq!(got.as_slice().unwrap(), &data.as_slice().unwrap()[..48]);
}
