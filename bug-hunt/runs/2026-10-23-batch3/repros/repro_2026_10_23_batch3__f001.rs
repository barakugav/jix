// Verifier repro for bug-hunt run 2026-10-23-batch3, finding F001.
//
// Struct layout validation (jix/src/dtype.rs is_aligned_struct / is_packed_struct /
// from_fields::determine_itemsize_and_alignment) sums field ends in unchecked u16. A field list
// whose running end passes 65535 wraps (release) and validates against a tiny itemsize, so a
// struct dtype whose fields lie up to 64 KiB past the element is accepted. SubDtype trusts the
// field offset and reads past the buffer.
//
// Every test asserts the CORRECT behavior and fails while the bug exists:
// - debug build: each test panics with "attempt to add with overflow" / "attempt to multiply with
//   overflow" inside the validators (instead of returning Err).
// - release build (`--release`): the validators return Ok. WARNING: in release,
//   `crafted_archive_field_read` and `safe_api_relabel_field_read` read 64 KiB past a 4-byte
//   buffer and may CRASH THE PROCESS (SIGSEGV); run them one at a time by name.
//
// cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_23_batch3__f001 -- --nocapture
use std::io::Cursor;
use std::ops::Range;

use jix::dtype::{Alignment, Dtype, Dtyped};
use jix::error::ErrorKind;
use jix::ops::IntoType;
use jix::storage::{ArraySpec, ArrayStorageInfo, StridedBuf};
use jix::{ArchiveValidation, Array, ArrayParams, ArrayStorage, ElementType, ReadContext, TypeDyn};
use ndarray::{array, Ix1};

// ---- tiny protobuf encoder / walker (archive surgery) -------------------------------------------

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}
fn get_varint(buf: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let b = buf[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}
fn put_key(out: &mut Vec<u8>, field: u32, wire: u8) {
    put_varint(out, ((field as u64) << 3) | wire as u64);
}
fn put_varint_field(out: &mut Vec<u8>, field: u32, v: u64) {
    put_key(out, field, 0);
    put_varint(out, v);
}
fn put_bytes_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    put_key(out, field, 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// jix.v1.Dtype: little-endian U8 scalar with the given inner shape.
fn u8_dtype(shape: &[u32]) -> Vec<u8> {
    let mut scalar = Vec::new();
    put_varint_field(&mut scalar, 1, 5); // SCALAR_KIND_U8
    put_varint_field(&mut scalar, 2, 1); // ENDIANNESS_LITTLE
    let mut d = Vec::new();
    put_bytes_field(&mut d, 1, &scalar);
    let mut packed = Vec::new();
    for &s in shape {
        put_varint(&mut packed, s as u64);
    }
    if !shape.is_empty() {
        put_bytes_field(&mut d, 3, &packed);
    }
    put_varint_field(&mut d, 4, shape.iter().product::<u32>() as u64);
    put_varint_field(&mut d, 5, 1);
    d
}

/// jix.v1.Dtype struct with (name, offset, dtype bytes) fields.
fn struct_dtype(fields: &[(&str, u32, Vec<u8>)], itemsize: u32, alignment: u32) -> Vec<u8> {
    let mut st = Vec::new();
    for (name, offset, dt) in fields {
        let mut f = Vec::new();
        put_bytes_field(&mut f, 1, name.as_bytes());
        put_varint_field(&mut f, 2, *offset as u64);
        put_bytes_field(&mut f, 3, dt);
        put_bytes_field(&mut st, 1, &f);
    }
    let mut d = Vec::new();
    put_bytes_field(&mut d, 2, &st);
    put_varint_field(&mut d, 4, itemsize as u64);
    put_varint_field(&mut d, 5, alignment as u64);
    d
}

/// Replace the dtype in a valid archive's BlockTableHeader and shift the TOC offsets.
fn replace_dtype(archive: &[u8], new_dtype: &[u8]) -> Vec<u8> {
    let mut pos = 4; // magic
    let l = get_varint(archive, &mut pos) as usize; // FileMetadata
    pos += l;
    let l = get_varint(archive, &mut pos) as usize; // ArrayHeader
    pos += l;
    let hdr_len_pos = pos;
    let l = get_varint(archive, &mut pos) as usize; // BlockTableHeader
    let hdr = &archive[pos..pos + l];
    let after_hdr = pos + l;

    let mut new_hdr = Vec::new();
    put_bytes_field(&mut new_hdr, 1, new_dtype);
    let mut p = 0;
    while p < hdr.len() {
        let start = p;
        let key = get_varint(hdr, &mut p);
        let (field, wire) = (key >> 3, key & 7);
        match wire {
            0 => {
                get_varint(hdr, &mut p);
            }
            2 => {
                let n = get_varint(hdr, &mut p) as usize;
                p += n;
            }
            _ => panic!("unexpected wire type {wire}"),
        }
        if field != 1 {
            new_hdr.extend_from_slice(&hdr[start..p]);
        }
    }
    let mut out = archive[..hdr_len_pos].to_vec();
    put_varint(&mut out, new_hdr.len() as u64);
    out.extend_from_slice(&new_hdr);
    let delta = out.len() as i64 - after_hdr as i64;

    // TOC: [Section { offset: i64, size: u64 }; 2]
    let mut toc = archive[after_hdr..after_hdr + 32].to_vec();
    for s in 0..2 {
        let o = i64::from_le_bytes(toc[s * 16..s * 16 + 8].try_into().unwrap()) + delta;
        toc[s * 16..s * 16 + 8].copy_from_slice(&o.to_le_bytes());
    }
    out.extend_from_slice(&toc);
    out.extend_from_slice(&archive[after_hdr + 32..]);
    out
}

fn base_archive() -> Vec<u8> {
    let a = Array::compact_ndarray(&array![1u8, 2, 3, 4]).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    buf.into_inner()
}

fn read(bytes: Vec<u8>) -> Result<Array<jix::storage::Compact<TypeDyn>>, jix::Error> {
    Array::read_from_reader(
        Cursor::new(bytes),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
}

/// Struct dtype {a: u8[65535] @0, b: u8[2] @65535} with itemsize 1, alignment 1.
/// Aligned check: end = 65535 + 2 = 65537 -> wraps to 1 == itemsize.
fn crafted() -> Vec<u8> {
    let d = struct_dtype(
        &[("a", 0, u8_dtype(&[65535])), ("b", 65535, u8_dtype(&[2]))],
        1,
        1,
    );
    replace_dtype(&base_archive(), &d)
}

fn describe(d: &Dtype) -> String {
    format!(
        "itemsize={} align={} aligned={} fields={:?}",
        d.itemsize(),
        d.alignment(),
        d.is_aligned(),
        d.fields().map(|f| f
            .iter()
            .map(|(n, o, d)| (n.to_string(), *o, d.itemsize()))
            .collect::<Vec<_>>())
    )
}

fn u8_shaped(n: u16) -> Dtype {
    let mut d = u8::DTYPE;
    d.set_shape(&[n]).unwrap();
    d
}

/// Control: the archive surgery itself is sound - a legit {a: u8 @0} struct reads back.
#[test]
fn control_surgery_roundtrip() {
    let d = struct_dtype(&[("a", 0, u8_dtype(&[]))], 1, 1);
    let arr = read(replace_dtype(&base_archive(), &d)).unwrap();
    let got = arr.dtype_sub_field::<u8>("a").to_ndarray::<Ix1>().unwrap();
    assert_eq!(got, array![1u8, 2, 3, 4]);
}

/// A crafted archive with a struct dtype whose field b lies at offset 65535 of a 1-byte element
/// must be rejected with InvalidArchive.
#[test]
fn crafted_archive_rejected() {
    match read(crafted()) {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidArchive, "{e}"),
        Ok(arr) => panic!(
            "crafted archive accepted, dtype violates C1: {}",
            describe(arr.dtype())
        ),
    }
}

/// Same archive, then read field b. In release this reads 64 KiB past the 4-byte decoded buffer
/// (SubDtype push path, sub_dtype.rs:149) - may SIGSEGV.
#[test]
fn crafted_archive_field_read() {
    if let Ok(arr) = read(crafted()) {
        eprintln!("ACCEPTED: {}", describe(arr.dtype()));
        let b = arr.dtype_sub_field::<[u8; 2]>("b").to_ndarray::<Ix1>();
        panic!("crafted archive accepted; field b (bytes 65535.. of a 4-byte buffer) = {b:?}");
    }
}

/// Public safe constructors: both must reject the layout with InvalidArgument.
#[test]
fn from_fields_rejects_field_past_itemsize() {
    let r = Dtype::from_fields(vec![
        ("a".to_string(), 0, u8_shaped(65535)),
        ("b".to_string(), 65535, u8_shaped(2)),
    ]);
    match r {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{e}"),
        Ok(d) => panic!("from_fields accepted: {}", describe(&d)),
    }
}

#[test]
fn new_struct_rejects_field_past_itemsize() {
    let r = Dtype::new_struct(
        vec![
            ("a".to_string(), 0, u8_shaped(65535)),
            ("b".to_string(), 65535, u8_shaped(2)),
        ],
        &[],
        1,
        Alignment::new(1).unwrap(),
    );
    match r {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{e}"),
        Ok(d) => panic!("new_struct accepted: {}", describe(&d)),
    }
}

/// Ceil-only wrap: {a: u16 @0, b: u8[65533] @2} needs a 65536-byte aligned struct; the final
/// ceil_to_multiple(65535, 2) wraps to 0.
#[test]
fn from_fields_rejects_aligned_itemsize_overflow() {
    let r = Dtype::from_fields(vec![
        ("a".to_string(), 0, u16::DTYPE),
        ("b".to_string(), 2, u8_shaped(65533)),
    ]);
    match r {
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{e}"),
        Ok(d) => panic!("from_fields accepted: {}", describe(&d)),
    }
}

// ---- 100% safe code, no crafted file -----------------------------------------------------------

/// A safe user storage that forwards everything to an inner storage but reports a different
/// (jix-validated) dtype of the same itemsize. Its buffers are exactly what its dtype promises
/// (one byte per element), so it does not lie about anything except through the Dtype that
/// Dtype::from_fields certified.
struct Relabel<S> {
    inner: S,
    et: TypeDyn,
}
impl<S: ArrayStorage> ArrayStorage for Relabel<S> {
    type ElementType = TypeDyn;

    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>, jix::Error> {
        self.inner.read_data(index, context, out)
    }
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
    fn dtype(&self) -> &Dtype {
        self.et.dtype()
    }
    fn spec(&self) -> ArraySpec<'_> {
        self.inner.spec()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        self.inner.info()
    }
    type ElementTypeChange<NewET: ElementType> = IntoType<Self, NewET>;
    fn element_type_change<NewET: ElementType>(
        self,
    ) -> Result<IntoType<Self, NewET>, jix::Error> {
        IntoType::new(self)
    }
}

/// Safe code only: from_fields hands out a 1-byte struct dtype with field b at offset 65535;
/// dtype_sub_field("b") then reads 64 KiB past a 4-byte Vec. Correct: from_fields returns Err.
#[test]
fn safe_api_relabel_field_read() {
    let d = match Dtype::from_fields(vec![
        ("a".to_string(), 0, u8_shaped(65535)),
        ("b".to_string(), 65535, u8_shaped(2)),
    ]) {
        Err(_) => return, // correct behavior
        Ok(d) => d,
    };
    eprintln!("from_fields ACCEPTED: {}", describe(&d));
    assert_eq!(d.itemsize(), 1);
    let inner = Array::plain_ndarray(array![1u8, 2, 3, 4]).unwrap().into_storage();
    let arr = Array::from_storage(Relabel {
        inner,
        et: TypeDyn::from_dtype(d).unwrap(),
    });
    let b = arr.dtype_sub_field::<[u8; 2]>("b").to_ndarray::<Ix1>();
    panic!("safe code read field b (bytes 65535.. of a 4-byte Vec) = {b:?}");
}
