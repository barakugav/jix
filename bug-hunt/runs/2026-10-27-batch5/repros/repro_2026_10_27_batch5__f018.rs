// Repro for F018 (bug-hunt run 2026-10-27-batch5): u64 archive header fields (ArrayHeader
// block_shape dims, BlockTableHeader.block_size) are truncated to u32 with `as` BEFORE they are
// validated, so a value >= 2^32 is accepted as its low 32 bits.
//
// Each test asserts the CORRECT behavior (the crafted archive is rejected with an error), so it
// fails while the bug exists.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};

fn read_varint(b: &[u8], pos: &mut usize) -> u64 {
    let (mut v, mut shift) = (0u64, 0);
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

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Splits an archive into magic, the three length-prefixed messages and the rest (TOC + body).
/// The TOC section offsets are relative to the archive start, so when a message changes length
/// they are shifted by the size difference.
fn rewrite(bytes: &[u8], edit: impl Fn(usize, Vec<u8>) -> Vec<u8>) -> Vec<u8> {
    let mut pos = 4;
    let mut out = bytes[..4].to_vec();
    for i in 0..3 {
        let len = read_varint(bytes, &mut pos) as usize;
        let msg = edit(i, bytes[pos..pos + len].to_vec());
        pos += len;
        write_varint(&mut out, msg.len() as u64);
        out.extend_from_slice(&msg);
    }
    let delta = out.len() as i64 - pos as i64;
    for _ in 0..2 {
        let off = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        out.extend_from_slice(&(off + delta).to_le_bytes());
        out.extend_from_slice(&bytes[pos + 8..pos + 16]);
        pos += 16;
    }
    out.extend_from_slice(&bytes[pos..]);
    out
}

/// Replaces the varint field `field` of a protobuf message (re-encoding every other field as is).
fn set_varint_field(msg: &[u8], field: u64, value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    let mut found = false;
    while pos < msg.len() {
        let start = pos;
        let tag = read_varint(msg, &mut pos);
        match tag & 7 {
            0 => {
                read_varint(msg, &mut pos);
            }
            2 => {
                let len = read_varint(msg, &mut pos) as usize;
                pos += len;
            }
            w => panic!("unexpected wire type {w}"),
        }
        if tag >> 3 == field {
            assert_eq!(tag & 7, 0);
            write_varint(&mut out, tag);
            write_varint(&mut out, value);
            found = true;
        } else {
            out.extend_from_slice(&msg[start..pos]);
        }
    }
    assert!(found, "field {field} not present");
    out
}

/// ArrayHeader { shape = [4], block_shape = [block_dim] } (both packed repeated uint64).
fn array_header(block_dim: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for (field, v) in [(1u64, 4u64), (2, block_dim)] {
        let mut body = Vec::new();
        write_varint(&mut body, v);
        write_varint(&mut out, (field << 3) | 2);
        write_varint(&mut out, body.len() as u64);
        out.extend_from_slice(&body);
    }
    out
}

/// A valid archive of a [4] u8 array with block shape [2].
fn valid_archive() -> Vec<u8> {
    let mut p = ArrayParams::new();
    p.block_shape(&[2]);
    let a = Array::compact_fn_with([4u64], p, |i| i as u8).unwrap();
    let mut c = Cursor::new(Vec::new());
    a.write_to(&mut c).unwrap();
    c.into_inner()
}

fn read(bytes: Vec<u8>) -> Result<Array<jix::storage::Compact<jix::TypeDyn>>, jix::Error> {
    let len = bytes.len() as u64;
    Array::read_from_reader(
        Cursor::new(bytes),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
}

const OVER_U32: u64 = (1 << 32) + 2;

#[test]
fn sanity_unmodified_rewrite_reads_back() {
    let bytes = rewrite(&valid_archive(), |i, m| if i == 1 { array_header(2) } else { m });
    let a = read(bytes).unwrap().into_typed::<u8>().unwrap();
    assert_eq!(a.to_ndarray().unwrap(), ndarray::array![0u8, 1, 2, 3]);
}

/// Prints what a wrongly accepted archive loaded as, and whether writing it back reproduces the
/// crafted bytes (it does not: the header is re-emitted with the truncated values).
fn report_accepted(a: Array<jix::storage::Compact<jix::TypeDyn>>, crafted: &[u8]) {
    let a = a.into_typed::<u8>().unwrap();
    eprintln!(
        "accepted: shape {:?}, data {:?}",
        a.shape(),
        a.to_ndarray::<ndarray::Ix1>().unwrap()
    );
    let mut c = Cursor::new(Vec::new());
    a.write_to(&mut c).unwrap();
    let rewritten = c.into_inner();
    eprintln!(
        "write_to of the loaded array: equals crafted input: {}, equals the original valid \
         archive (block [2]): {}",
        rewritten == crafted,
        rewritten == valid_archive()
    );
}

/// ArrayHeader.block_shape = [2^32 + 2] for shape [4]: the block dim exceeds the dim length and
/// does not fit u32, so the archive is invalid.
#[test]
fn array_header_block_dim_over_u32_is_rejected() {
    let bytes = rewrite(&valid_archive(), |i, m| if i == 1 { array_header(OVER_U32) } else { m });
    match read(bytes.clone()) {
        Err(e) => eprintln!("rejected: {e}"),
        Ok(a) => {
            report_accepted(a, &bytes);
            panic!("ArrayHeader block_shape [2^32 + 2] for shape [4] accepted");
        }
    }
}

/// BlockTableHeader.block_size = 2^32 + 2 with nitems = 4: nitems is not a multiple of the
/// declared block size, so the archive is invalid.
#[test]
fn block_table_block_size_over_u32_is_rejected() {
    let bytes = rewrite(&valid_archive(), |i, m| {
        if i == 2 { set_varint_field(&m, 3, OVER_U32) } else { m }
    });
    match read(bytes.clone()) {
        Err(e) => eprintln!("rejected: {e}"),
        Ok(a) => {
            report_accepted(a, &bytes);
            panic!("BlockTableHeader block_size 2^32 + 2 (nitems 4) accepted");
        }
    }
}
