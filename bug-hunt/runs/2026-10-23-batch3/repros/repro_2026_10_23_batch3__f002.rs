// Repro for F002 (run 2026-10-23-batch3).
//
// Reading an archive with `ArrayParams` that carry `block_shape_fixed_dims` with a `false`
// entry makes `ArrayParams::tune` re-scale the FILE's block shape. The spec block shape that
// `read_data` uses then no longer matches the geometry of the stored blocks.
//
// Each test asserts the CORRECT behavior (the read either rejects the params up front, or returns
// the original data), so every test FAILS while the bug exists.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::{Array1, Array2, Ix1, Ix2};

fn archive_2d(src: &Array2<i32>, block_shape: &[u32]) -> Vec<u8> {
    let mut p = ArrayParams::new();
    p.block_shape(block_shape);
    let a = Array::compact_ndarray_with(src, p).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    buf.into_inner()
}

fn read(
    bytes: Vec<u8>,
    params: ArrayParams,
) -> Result<Array<jix::storage::Compact<jix::TypeDyn>>, jix::Error> {
    Array::read_from_reader(Cursor::new(bytes), None, params, ArchiveValidation::Strict)
}

fn src_64x64() -> Array2<i32> {
    Array2::from_shape_fn((64, 64), |(i, j)| (i * 64 + j) as i32)
}

/// Same block volume after the re-tune: [8, 32] on disk becomes [4, 64] in the spec. Every read
/// succeeds but scatters each decoded block with the wrong shape.
#[test]
fn same_volume_drift_returns_original_values() {
    let src = src_64x64();
    let bytes = archive_2d(&src, &[8, 32]);
    let mut p = ArrayParams::new();
    p.block_shape_fixed_dims(&[false, false]).block_size(1024);
    let Ok(a) = read(bytes, p) else {
        return; // rejecting the params up front is an acceptable fix
    };
    let got = a.into_typed::<i32>().unwrap().to_ndarray::<Ix2>().unwrap();
    let bad = got.iter().zip(src.iter()).filter(|(g, s)| g != s).count();
    println!("mismatches: {bad} / {}", src.len());
    println!("got[1, 0..4] = {:?}  expected {:?}", &got.row(1).to_vec()[..4], &src.row(1).to_vec()[..4]);
    assert_eq!(got, src);
}

/// Different block volume after the re-tune (1-D, [16] -> L1-sized): every read fails with
/// InvalidBufferSize although the file is valid.
#[test]
fn volume_drift_1d_reads_fail() {
    let src = Array1::from_iter(0..100i32);
    let mut p = ArrayParams::new();
    p.block_shape(&[16]);
    let a = Array::compact_ndarray_with(&src, p).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    let mut p = ArrayParams::new();
    p.block_shape_fixed_dims(&[false]);
    let Ok(a) = read(buf.into_inner(), p) else {
        return;
    };
    let got = a.into_typed::<i32>().unwrap().to_ndarray::<Ix1>();
    println!("1-D read: {:?}", got.as_ref().map(|g| g.len()));
    assert_eq!(got.unwrap(), src);
}

/// The same params dict reused on two machines with different L1 sizes (simulated by two
/// different block_size values): a partially fixed block shape re-tunes to another volume.
#[test]
fn param_reuse_with_different_block_size_target() {
    let src = src_64x64();
    // "Machine A": L1-derived target of 1 KiB -> block [8, 32].
    let mut pa = ArrayParams::new();
    pa.block_shape(&[8, 1]).block_shape_fixed_dims(&[true, false]).block_size(1024);
    let a = Array::compact_ndarray_with(&src, pa).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    // "Machine B": same params, L1-derived target of 2 KiB.
    let mut pb = ArrayParams::new();
    pb.block_shape(&[8, 1]).block_shape_fixed_dims(&[true, false]).block_size(2048);
    let Ok(b) = read(buf.into_inner(), pb) else {
        return;
    };
    let got = b.into_typed::<i32>().unwrap().to_ndarray::<Ix2>();
    println!("param reuse read: {:?}", got.as_ref().map(|g| g.dim()).map_err(|e| e.to_string()));
    assert_eq!(got.unwrap(), src);
}

/// Same volume, but the spec grid has MORE blocks than the file: shape [64, 300], file block
/// [32, 5] (120 blocks), spec block [1, 160] (128 blocks). Reading the last rows indexes past
/// the block-location table and panics.
#[test]
fn drift_with_larger_grid_panics() {
    let src = Array2::from_shape_fn((64, 300), |(i, j)| (i * 300 + j) as i32);
    let bytes = archive_2d(&src, &[32, 5]);
    let mut p = ArrayParams::new();
    p.block_shape_fixed_dims(&[false, false]).block_size(1024);
    let Ok(a) = read(bytes, p) else {
        return;
    };
    let got = a.into_typed::<i32>().unwrap().to_ndarray::<Ix2>().unwrap();
    assert_eq!(got, src);
}

/// write_to of a drifted array copies the blocks verbatim but writes the drifted block shape into
/// the header. The new file passes Strict validation and reads back permuted with DEFAULT params.
#[test]
fn write_back_after_drift_persists_corrupt_file() {
    let src = src_64x64();
    let bytes = archive_2d(&src, &[8, 32]);
    let mut p = ArrayParams::new();
    p.block_shape_fixed_dims(&[false, false]).block_size(1024);
    let Ok(a) = read(bytes, p) else {
        return;
    };
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    let b = read(buf.into_inner(), ArrayParams::default())
        .expect("re-written archive passes Strict validation");
    let got = b.into_typed::<i32>().unwrap().to_ndarray::<Ix2>().unwrap();
    let bad = got.iter().zip(src.iter()).filter(|(g, s)| g != s).count();
    println!("re-written file, default params: mismatches {bad} / {}", src.len());
    assert_eq!(got, src);
}
