// Repro for bug-hunt finding F011 (run 2026-10-22-trial2).
//
// A small (~8 KB) archive that passes ArchiveValidation::Strict declares one block of
// 2^28 - 1 elements of a 128-byte dtype, i.e. 34_359_738_240 bytes. The first read of ANY element
// asks the allocator for a full block of scratch (storage/compact.rs:395-397) and, when that
// allocation fails, RawAlignedBytes calls `handle_alloc_error`, which ABORTS THE PROCESS instead of
// returning an Err.
//
// THIS TEST ABORTS THE PROCESS while the bug exists ("memory allocation of 34359738240 bytes
// failed").
//
// To make the outcome independent of the RAM / overcommit settings of the machine running it,
// the test binary installs a global allocator that refuses (returns null for) any single request
// above ALLOC_CAP, like a machine with less memory than the archive claims. Without the cap the
// result is the same on any machine where the kernel refuses a 34 GB allocation (e.g. 15 GB RAM,
// no swap, vm.overcommit_memory=0). The header fields can go up to 2^32 - 1 elements x 65535 bytes
// (~2.8e14 bytes), which exceeds the 47-bit x86_64 user address space, so a crafted file can make
// the allocation fail everywhere.
//
// Correct behavior (asserted): either the load is rejected, or the read returns an Err.
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

use jix::{ArchiveValidation, Array, ArrayParams, Filter};
use ndarray::{Array1, Ix1};

/// Single allocations above this size fail (the allocator returns null).
const ALLOC_CAP: usize = 4 << 30;

struct CappedAlloc;
static LARGEST_REQUEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CappedAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST_REQUEST.fetch_max(layout.size(), Ordering::Relaxed);
        if layout.size() > ALLOC_CAP {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST_REQUEST.fetch_max(layout.size(), Ordering::Relaxed);
        if layout.size() > ALLOC_CAP {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST_REQUEST.fetch_max(new_size, Ordering::Relaxed);
        if new_size > ALLOC_CAP {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CappedAlloc = CappedAlloc;

const ITEM: usize = 128;
type Item = [u8; ITEM];

/// A valid archive of 2^21 x [u8; 128] (one block, no filters), then the four 4-byte varints
/// holding 2^21 (shape, block_shape, nitems, block_size) are patched to 2^28 - 1. The varint
/// lengths do not change, so every protobuf length prefix stays valid.
fn build() -> Vec<u8> {
    let n = 1usize << 21; // uleb128: 80 80 80 01
    let data = Array1::from_elem(n, [0u8; ITEM]);
    let mut params = ArrayParams::new();
    params.block_shape(&[n as u32]);
    params.filters(&[] as &[Filter]).unwrap();
    let a = Array::compact_ndarray_with(&data, params).unwrap();
    drop(data);
    let mut cur = Cursor::new(Vec::new());
    a.write_to(&mut cur).unwrap();
    let mut b = cur.into_inner();
    // Only patch the metadata, which precedes the first zstd frame.
    let frame = b
        .windows(4)
        .position(|w| w == [0x28, 0xB5, 0x2F, 0xFD])
        .unwrap();
    let mut patched = 0;
    for i in 0..frame - 3 {
        if b[i..i + 4] == [0x80, 0x80, 0x80, 0x01] {
            b[i..i + 4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0x7F]); // 2^28 - 1
            patched += 1;
        }
    }
    assert_eq!(patched, 4, "expected shape, block_shape, nitems, block_size");
    b
}

#[test]
fn small_archive_with_huge_block_read_errors_instead_of_aborting() {
    let bytes = build();
    let file_len = bytes.len();
    let a = match Array::read_from_reader(
        Cursor::new(bytes),
        Some(file_len as u64),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    ) {
        Err(e) => {
            // Rejecting the implausible block at load time is a correct outcome.
            eprintln!("load rejected: {e}");
            return;
        }
        Ok(a) => a.into_typed::<Item>().unwrap(),
    };
    eprintln!(
        "archive of {file_len} bytes loaded under Strict: shape {:?}, block bytes {}",
        a.shape(),
        a.shape()[0] as usize * ITEM
    );
    LARGEST_REQUEST.store(0, Ordering::Relaxed);
    let ctx = a.read_ctx();
    eprintln!("reading element 1..2 (allocations above {ALLOC_CAP} bytes fail)...");
    let r = a.to_ndarray_sub::<Ix1>(&[1..2], &ctx);
    eprintln!(
        "read returned {:?}; largest single allocation request: {} bytes",
        r.as_ref().map(|_| ()).map_err(|e| e.to_string()),
        LARGEST_REQUEST.load(Ordering::Relaxed)
    );
    assert!(r.is_err(), "the block cannot decode to 2^28-1 items");
}
