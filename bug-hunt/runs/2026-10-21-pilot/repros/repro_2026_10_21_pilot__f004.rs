// Verifier repro for F004: `Array::read_from_reader` (safe, public) passes UNINITIALIZED
// `&mut [u8]` buffers to the caller's `Read` impl (util/mod.rs:571-583 value_from_io,
// archive/common.rs:204-215 read_slice, archive/block.rs:452-458 Owned::read_section).
// `Read` is a safe trait, so a safe impl may read `buf` before filling it; std's `Read::read`
// docs: "Calling `read` with an uninitialized `buf` ... is not safe, and can lead to undefined
// behavior". Each test asserts the correct behavior (the buffer jix hands over was initialized by
// jix), so it fails while the bug is present.
//
// Native:  cargo test --manifest-path jix/Cargo.toml --all-features \
//              --test repro_2026_10_21_pilot__f004 -- --nocapture --test-threads=1
// Miri:    python jix/run_miri.py --all-features --test repro_2026_10_21_pilot__f004 -- miri_
use std::io::{Cursor, Read, Seek, SeekFrom};

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::Array1;

/// A purely safe `Read + Seek` wrapper. Once armed it inspects every buffer it is handed
/// BEFORE filling it (typed `u8` loads of `buf`), and records a copy.
struct Snoop {
    inner: Cursor<Vec<u8>>,
    /// arm on the first `seek` call: with `Take<R>` in between, the first seek that reaches the
    /// user's reader is the one in `ArchiveReader::read_section_into`, i.e. the first armed read
    /// is the block-data read of `Owned::read_section` (archive/block.rs:452-458).
    arm_on_seek: bool,
    armed: bool,
    seen: Vec<Vec<u8>>,
}
impl Snoop {
    fn new(bytes: Vec<u8>, arm_on_seek: bool) -> Self {
        Self {
            inner: Cursor::new(bytes),
            arm_on_seek,
            armed: !arm_on_seek,
            seen: Vec::new(),
        }
    }
}
impl Read for Snoop {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.armed {
            let mut copy = Vec::with_capacity(buf.len());
            for &b in buf.iter() {
                copy.push(b); // safe code reading the caller's buffer
            }
            self.seen.push(copy);
        }
        self.inner.read(buf)
    }
}
impl Seek for Snoop {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        if self.arm_on_seek {
            self.armed = true;
        }
        self.inner.seek(pos)
    }
}

fn archive_bytes(data: Vec<u8>) -> Vec<u8> {
    let a = Array::compact_ndarray(&Array1::from(data)).unwrap();
    let mut cur = Cursor::new(Vec::new());
    a.write_to(&mut cur).unwrap();
    cur.into_inner()
}

fn read_back(r: &mut Snoop) {
    Array::read_from_reader(
        r,
        None,
        ArrayParams::default(),
        ArchiveValidation::default(),
    )
    .unwrap();
}

/// Miri: the first armed read is `Owned::read_section`'s block-data buffer
/// (`Vec::<MaybeUninit<u8>>::with_capacity` + `set_len` + `cast_slice_mut`). Miri reports UB at
/// the `for &b in buf.iter()` load in `Snoop::read`.
#[test]
fn miri_owned_read_section_hands_uninit_buf_to_reader() {
    let bytes = archive_bytes(vec![1u8, 2, 3, 4]);
    let mut r = Snoop::new(bytes, true);
    read_back(&mut r);
    assert!(!r.seen.is_empty());
}

/// Native: stale bytes of a freed allocation reach safe code. Pass 1 learns the size of the
/// block-data section; pass 2 frees a `vec![0xA5; len]` of that size right before the read, and
/// the buffer jix then hands to `Snoop::read` contains those 0xA5 bytes.
#[test]
fn native_stale_heap_bytes_reach_safe_reader() {
    let mut s: u64 = 0x1234_5678;
    let data: Vec<u8> = (0..600)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect();
    let bytes = archive_bytes(data);

    let mut r = Snoop::new(bytes.clone(), true);
    read_back(&mut r);
    let len = r.seen[0].len();
    println!("block-data section buffer handed to Read::read: {len} bytes");

    let mut r = Snoop::new(bytes, true);
    {
        let junk = vec![0xA5u8; len];
        drop(std::hint::black_box(junk));
    }
    read_back(&mut r);
    let first = &r.seen[0];
    let n_a5 = first.iter().filter(|&&b| b == 0xA5).count();
    println!(
        "buffer contents seen by the safe reader before it wrote anything: {n_a5}/{} bytes are \
         0xA5 (left behind by a freed Vec); first 32: {:02x?}",
        first.len(),
        &first[..first.len().min(32)]
    );
    assert_eq!(
        n_a5, 0,
        "jix handed an uninitialized buffer (stale heap contents) to a safe Read impl"
    );
}
