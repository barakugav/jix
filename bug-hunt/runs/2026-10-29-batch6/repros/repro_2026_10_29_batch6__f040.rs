//! F040: an array read from a .jix file does not carry the file's filters in its spec, so
//! `compact()` and writes of lazy views over it re-encode with the default byte-shuffle instead of
//! the file's bit-shuffle. The same array built in memory keeps bit-shuffle.
//!
//! Asserts the CORRECT behavior (codec settings inherited from the source), so it fails while the
//! bug exists.

use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams, ArrayStorage, Filter};

#[derive(Debug, PartialEq)]
enum F {
    Byte,
    Bit,
}

fn read_varint(b: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let byte = b[*pos];
        *pos += 1;
        v |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}

/// Returns the filters listed in the archive's BlockTableHeader (field 5 of the third
/// length-delimited message after the "jix1" magic).
fn filters_of(bytes: &[u8]) -> Vec<F> {
    assert_eq!(&bytes[..4], b"jix1");
    let mut pos = 4;
    let mut msgs = Vec::new();
    for _ in 0..3 {
        let len = read_varint(bytes, &mut pos) as usize;
        msgs.push(&bytes[pos..pos + len]);
        pos += len;
    }
    let header = msgs[2];
    let mut p = 0;
    let mut out = Vec::new();
    while p < header.len() {
        let tag = read_varint(header, &mut p);
        let (field, wire) = (tag >> 3, tag & 7);
        match wire {
            0 => {
                read_varint(header, &mut p);
            }
            2 => {
                let len = read_varint(header, &mut p) as usize;
                let body = &header[p..p + len];
                p += len;
                if field == 5 {
                    // Filter { oneof kind { Empty byte_shuffle = 1; Empty bit_shuffle = 2; } }
                    out.push(match body[0] >> 3 {
                        1 => F::Byte,
                        2 => F::Bit,
                        k => panic!("unknown filter kind {k}"),
                    });
                }
            }
            w => panic!("unexpected wire type {w}"),
        }
    }
    out
}

fn write<S: ArrayStorage>(a: &Array<S>) -> Vec<u8> {
    let mut c = Cursor::new(Vec::new());
    a.write_to(&mut c).unwrap();
    c.into_inner()
}

#[test]
fn file_read_array_keeps_file_filters_on_reencode() {
    let x = ndarray::Array2::from_shape_fn((64, 64), |(i, j)| (i * 64 + j) as u16);
    let mut params = ArrayParams::new();
    params.filters(&[Filter::BitShuffle]).unwrap();
    let c = Array::compact_ndarray_with(&x, params).unwrap();

    let file = write(&c);
    assert_eq!(filters_of(&file), vec![F::Bit], "file written from c");

    // In-memory source: codec settings are inherited (control).
    let c_compact = c.compact().unwrap();
    assert_eq!(filters_of(&write(&c_compact)), vec![F::Bit], "c.compact()");
    let c_view = c.view().map(|v: u16| v.wrapping_add(1));
    assert_eq!(filters_of(&write(&c_view)), vec![F::Bit], "c + 1");

    // Same array read back from the file.
    let len = file.len() as u64;
    let r = Array::read_from_reader(
        Cursor::new(file),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .unwrap();
    // The compressed-block copy fast path keeps the file's filters.
    assert_eq!(filters_of(&write(&r)), vec![F::Bit], "copy of r");

    let r_compact = r.compact().unwrap();
    let r_view = r.into_typed::<u16>().unwrap().map(|v: u16| v.wrapping_add(1));
    let got_compact = filters_of(&write(&r_compact));
    let got_view = filters_of(&write(&r_view));
    println!("r.compact() filters: {got_compact:?}");
    println!("r + 1       filters: {got_view:?}");
    // Values are still right; only the encoding differs.
    assert_eq!(
        r_view.to_ndarray::<ndarray::Ix2>().unwrap(),
        x.mapv(|v| v.wrapping_add(1))
    );
    assert_eq!(got_compact, vec![F::Bit], "r.compact() lost the file's bit-shuffle");
    assert_eq!(got_view, vec![F::Bit], "write of r + 1 lost the file's bit-shuffle");
}
