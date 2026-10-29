//! F023: after `Array::read_from_reader`, the reader is not left at the end of the archive (the
//! core reads the block_data section before the earlier blocks_loc section and never seeks back),
//! so arrays written back-to-back with `write_to(&mut w)` cannot be read back with successive
//! `read_from_reader(&mut r)`. `write_to` does leave the stream at the archive end
//! (`jix/src/archive/block.rs:244-248`). The tests assert the symmetric (correct) behavior, so
//! they fail while the bug exists. The Python binding adds a second cause (an internal 8 KiB
//! BufReader that reads ahead and is dropped, `jix-py/src/archive.rs:142`); see
//! bug-hunt/runs/2026-10-29-batch6/repros/F023.py.

use std::io::{Cursor, Seek};

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::array;

fn packed() -> (Vec<u8>, Vec<u64>) {
    let a = Array::compact_ndarray(&array![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
    let b = Array::compact_ndarray(&array![0.0f64, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut w = Cursor::new(Vec::new());
    a.write_to(&mut w).unwrap();
    let end_a = w.stream_position().unwrap();
    b.write_to(&mut w).unwrap();
    let end_b = w.stream_position().unwrap();
    let bytes = w.into_inner();
    assert_eq!(end_b, bytes.len() as u64, "write_to leaves the stream at the archive end");
    (bytes, vec![end_a, end_b])
}

#[test]
fn reader_left_at_end_of_archive() {
    let (bytes, ends) = packed();
    let mut r = Cursor::new(bytes);
    let a = Array::read_from_reader(&mut r, None, ArrayParams::default(), ArchiveValidation::default())
        .unwrap();
    assert_eq!(a.shape(), &[10]);
    let pos = r.stream_position().unwrap();
    println!("after read_from_reader: position = {pos}, end of first archive = {}", ends[0]);
    assert_eq!(pos, ends[0]);
}

#[test]
fn successive_reads_of_packed_stream() {
    let (bytes, _) = packed();
    let mut r = Cursor::new(bytes);
    let a = Array::read_from_reader(&mut r, None, ArrayParams::default(), ArchiveValidation::default())
        .unwrap()
        .into_typed::<i32>()
        .unwrap();
    assert_eq!(a.to_ndarray().unwrap(), array![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_dyn());
    let b = Array::read_from_reader(&mut r, None, ArrayParams::default(), ArchiveValidation::default());
    println!("second read_from_reader: {:?}", b.as_ref().map(|b| b.shape().to_vec()));
    let b = b.unwrap().into_typed::<f64>().unwrap();
    assert_eq!(b.to_ndarray().unwrap(), array![0.0f64, 1.0, 2.0, 3.0, 4.0].into_dyn());
}
