// Verifier repro for F005: the mmap reader bounds archive sections by the caller's `len`, never by
// the mapping. A file shorter than `offset + len` (truncated download / interrupted copy, or a
// wrong `len` from an external index) opens successfully, and reading a block whose compressed
// bytes lie past the end of the mapping makes zstd read memory outside the mapping.
//
//   cargo test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_21_pilot__f005 -- --nocapture --test-threads 1 <test name>
//
// - `open_truncated_mmap_should_fail`: the heap reader errors, the mmap reader returns Ok -> the
//   assertion fails (no crash; demonstrates the missing check).
// - `read_truncated_mmap_last_block`: reads the last element -> the test process dies with
//   SIGSEGV (out-of-bounds read ~14 MB past a 64 KiB mapping).
use std::fs::OpenOptions;
use std::path::Path;

use jix::{ArchiveValidation, Array, ArrayParams, ReadContext};
use ndarray::{Array1, Ix1};

const N: usize = 2_000_000;
const KEEP: u64 = 64 * 1024;

fn lcg_f64(n: usize, mut s: u64) -> Array1<f64> {
    Array1::from_iter((0..n).map(|_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }))
}

/// Writes a ~14 MB archive, truncates the file to 64 KiB, returns the original length.
fn write_and_truncate(path: &Path) -> u64 {
    Array::compact_ndarray(&lcg_f64(N, 1))
        .unwrap()
        .write_to_file(path)
        .unwrap();
    let full = std::fs::metadata(path).unwrap().len();
    OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(KEEP)
        .unwrap();
    eprintln!(
        "archive length {full}, file truncated to {}",
        std::fs::metadata(path).unwrap().len()
    );
    full
}

#[test]
fn open_truncated_mmap_should_fail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.jix");
    let full = write_and_truncate(&path);

    let heap = Array::read_from_file_section(
        &path,
        0,
        full,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    );
    eprintln!("heap reader: {:?}", heap.as_ref().map(|a| a.shape().to_vec()));
    assert!(heap.is_err());

    // Safety (as documented): the file is not modified while mapped.
    let mm = unsafe {
        Array::read_from_file_mmap(
            &path,
            0,
            full,
            ArrayParams::default(),
            ArchiveValidation::Strict,
        )
    };
    eprintln!("mmap reader: {:?}", mm.as_ref().map(|a| a.shape().to_vec()));
    assert!(
        mm.is_err(),
        "mmap reader accepted an archive whose sections extend {} bytes past the end of the file",
        full - KEEP
    );
}

#[test]
fn read_truncated_mmap_last_block() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.jix");
    let full = write_and_truncate(&path);

    // Safety (as documented): the file is not modified while mapped.
    let a = unsafe {
        Array::read_from_file_mmap(
            &path,
            0,
            full,
            ArrayParams::default(),
            ArchiveValidation::Strict,
        )
    }
    .unwrap()
    .into_typed::<f64>()
    .unwrap();
    eprintln!("opened mmap array, shape {:?}; reading the last element", a.shape());
    let n = N as u64;
    let r = a.to_ndarray_sub::<Ix1>(&[n - 1..n], &ReadContext::default());
    eprintln!("read returned: {r:?}");
}
