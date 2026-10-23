// Repro for F006 (run 2026-10-23-batch3): an archive message length (uleb128) is allocated and
// zero-filled (jix/src/archive/common.rs:161 -> :210 `tmp_buf.resize(len, 0)`) before it is
// compared with the bytes left in the archive.
//
// Every test asserts the CORRECT behavior (a tiny file whose first message claims a length
// past the end of the archive is rejected with an Err, without allocating that length), so each
// fails while the bug exists:
// - `msg_len_above_isize_max_is_an_error`: panics with 'capacity overflow'.
// - `msg_len_past_eof_does_not_allocate_it`: assertion on the peak allocation fails (256 MiB
//   zero-filled for a 9-byte file before the EOF error comes back).
// - `msg_len_2_44_is_an_error_not_an_abort`: the read ABORTS its process ('memory allocation of
//   17592186044416 bytes failed', SIGABRT). It is run in a child process (this same test binary,
//   filtered to the ignored `child_msg_len_2_44`) so the abort does not take the harness down.
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use jix::{Array, ArrayParams};

struct PeakAlloc;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(c, Ordering::SeqCst);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(c, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::SeqCst);
    }
}
#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;
/// Serializes the tests so one test's allocations (e.g. a panic backtrace) do not show up in the
/// peak measured by another.
static SERIAL: Mutex<()> = Mutex::new(());
fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn uleb(mut x: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let b = (x & 0x7f) as u8;
        x >>= 7;
        if x == 0 {
            out.push(b);
            return out;
        }
        out.push(b | 0x80);
    }
}

/// Writes `b"jix1" + uleb128(msg_len)` (the FileMetadata length prefix and nothing else).
fn tiny_file(dir: &Path, msg_len: u64) -> PathBuf {
    let mut bytes = b"jix1".to_vec();
    bytes.extend(uleb(msg_len));
    let path = dir.join(format!("len_{msg_len}.jix"));
    std::fs::write(&path, &bytes).unwrap();
    path
}

#[test]
fn msg_len_above_isize_max_is_an_error() {
    let _g = serial();
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_file(dir.path(), 1 << 63); // 14-byte file
    let r = Array::read_from_file(&path, ArrayParams::default());
    assert!(r.is_err(), "expected an Err for a message longer than the file");
}

#[test]
fn msg_len_past_eof_does_not_allocate_it() {
    let _g = serial();
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_file(dir.path(), 1 << 28); // 9-byte file claiming a 256 MiB message
    let base = CUR.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let r = Array::read_from_file(&path, ArrayParams::default());
    let peak = PEAK.load(Ordering::SeqCst) - base;
    println!("result: {:?}", r.as_ref().map(|_| ()));
    println!("peak heap growth while reading a 9-byte file: {peak} bytes");
    assert!(r.is_err());
    assert!(
        peak < (1 << 20),
        "reading a 9-byte file allocated {peak} bytes before rejecting it"
    );
}

/// Child half of `msg_len_2_44_is_an_error_not_an_abort`; aborts the process while the bug exists.
#[test]
#[ignore]
fn child_msg_len_2_44() {
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_file(dir.path(), 1 << 44); // 11-byte file claiming a 16 TiB message
    let r = Array::read_from_file(&path, ArrayParams::default());
    assert!(r.is_err());
}

#[test]
fn msg_len_2_44_is_an_error_not_an_abort() {
    let _g = serial();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "child_msg_len_2_44", "--nocapture"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!("child status: {:?}", out.status);
    println!("child stderr: {}", stderr.trim());
    assert!(
        out.status.success(),
        "reading an 11-byte file killed the process: {:?}, stderr: {}",
        out.status,
        stderr.trim()
    );
}
