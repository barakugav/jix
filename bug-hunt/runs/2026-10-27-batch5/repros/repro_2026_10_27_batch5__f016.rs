// Repro for bug-hunt run 2026-10-27-batch5, finding F016.
//
// ArrayBlockTableStorageBase::read_from computes `s.next_multiple_of(b)` on untrusted header dims
// before any check that dims are <= i64::MAX. A header dim > i64::MAX (not a multiple of the
// block dim) overflows: debug builds panic ("attempt to add with overflow"), release builds wrap.
// The tests assert the CORRECT behavior (Err(InvalidArchive)), so they fail while the bug exists.
use std::io::Cursor;

use jix::error::ErrorKind;
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
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// ArrayHeader { shape = 1 (packed u64), block_shape = 2 (packed u64) }.
fn array_header(shape: &[u64], block_shape: &[u64]) -> Vec<u8> {
    let mut msg = Vec::new();
    for (field, vals) in [(1u64, shape), (2, block_shape)] {
        let mut body = Vec::new();
        vals.iter().for_each(|&v| write_varint(&mut body, v));
        write_varint(&mut msg, (field << 3) | 2);
        write_varint(&mut msg, body.len() as u64);
        msg.extend_from_slice(&body);
    }
    msg
}

/// Write `a`, then replace its ArrayHeader and shift the TOC (offsets relative to the archive
/// start) by the change in header length. Layout: magic | FileMetadata | ArrayHeader |
/// BlockTableHeader | TOC [{offset: i64, size: u64}; 2] | rest.
fn with_header<S: jix::ArrayStorage>(a: &Array<S>, shape: &[u64], block_shape: &[u64]) -> Vec<u8> {
    let mut c = Cursor::new(Vec::new());
    a.write_to(&mut c).unwrap();
    let bytes = c.into_inner();
    let mut pos = 4;
    let mut msgs = Vec::new();
    for _ in 0..3 {
        let len = read_varint(&bytes, &mut pos) as usize;
        msgs.push(bytes[pos..pos + len].to_vec());
        pos += len;
    }
    msgs[1] = array_header(shape, block_shape);
    let mut out = bytes[..4].to_vec();
    for m in &msgs {
        write_varint(&mut out, m.len() as u64);
        out.extend_from_slice(m);
    }
    let delta = out.len() as i64 - pos as i64;
    for i in 0..2 {
        let at = pos + 16 * i;
        let off = i64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        out.extend_from_slice(&(off + delta).to_le_bytes());
        out.extend_from_slice(&bytes[at + 8..at + 16]);
    }
    out.extend_from_slice(&bytes[pos + 32..]);
    out
}

fn read_kind(bytes: Vec<u8>) -> Result<Vec<u64>, ErrorKind> {
    let len = bytes.len() as u64;
    Array::read_from_reader(
        Cursor::new(bytes),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .map(|a| a.shape().to_vec())
    .map_err(|e| {
        eprintln!("error: {e:?}");
        e.kind()
    })
}

/// Shape [u64::MAX], block [2]. Debug: panics in next_multiple_of before the block table is read.
#[test]
fn header_dim_u64_max_is_invalid_archive() {
    let a = Array::compact_fn([2u64], |_| 7u8).unwrap();
    let bytes = with_header(&a, &[u64::MAX], &[2]);
    assert_eq!(read_kind(bytes), Err(ErrorKind::InvalidArchive));
}

/// Shape [u64::MAX, 2], block [2, 1] over a valid empty block table (nitems 0, block_size 2).
/// Debug: same panic. Release: the padded-nitems product and the nblocks product both wrap to 0,
/// every archive check passes, and only into_spec's check_shape_overflow rejects it, as
/// InvalidShapeOperation.
#[test]
fn header_dim_u64_max_passes_archive_checks_in_release() {
    let mut p = ArrayParams::new();
    p.block_shape(&[1, 2]);
    let a = Array::compact_fn_with([0u64, 2], p, |_| 0u8).unwrap();
    let bytes = with_header(&a, &[u64::MAX, 2], &[2, 1]);
    assert_eq!(read_kind(bytes), Err(ErrorKind::InvalidArchive));
}

/// Sanity: the header rewriting keeps a valid archive valid.
#[test]
fn rewrite_sanity() {
    let a = Array::compact_fn([2u64], |_| 7u8).unwrap();
    let bytes = with_header(&a, &[2], &[2]);
    assert_eq!(read_kind(bytes), Ok(vec![2]));
}
