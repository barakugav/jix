// Verifier repro for F009 (run 2026-10-21-pilot).
//
// `AlignedBytes` / `RawAlignedBytes` are pub(crate), so the public API cannot reach them. This
// test compiles the UNMODIFIED source files jix/src/util/aligned_vec/{mod.rs,raw.rs} into the test
// crate via `#[path]` (with a stand-in for `crate::util::cpu_cache::CACHE_LINE_SIZE`, which only
// `new_padded` / `with_capacity_padded` use) and calls the SAFE constructors with align = 0.
//
// The UB is caught by the standard library's debug precondition checks (ub_checks), which abort
// the process (non-unwinding panic). Each trigger therefore runs in a child process: the parent
// tests re-execute this test binary with `--ignored --exact <child>` and inspect its stderr/status.

#[allow(dead_code, unused_imports, clippy::all)]
#[path = "../src/util/aligned_vec/mod.rs"]
pub(crate) mod aligned_vec;

// The included files refer to `crate::util::cpu_cache::CACHE_LINE_SIZE` and
// `crate::util::aligned_vec::RuntimeAlign`; provide those paths.
mod util {
    pub(crate) use super::aligned_vec;
    pub mod cpu_cache {
        pub const CACHE_LINE_SIZE: usize = 128;
    }
}

use aligned_vec::AlignedBytes;

// ---- children (each one triggers the UB path; run only as a subprocess) ----

#[test]
#[ignore]
fn child_new_exact_0() {
    // Safe fn. RuntimeAlign::new(0) accepts 0 -> RawAlignedBytes::new_unchecked(0)
    // -> null_mut().wrapping_add(0) == null -> NonNull::new_unchecked(null).
    let v = AlignedBytes::new_exact(0);
    eprintln!("CHILD_SURVIVED new_exact(0): ptr={:p} align={}", v.as_ptr(), v.alignment());
}

#[test]
#[ignore]
fn child_with_capacity_exact_0_16() {
    // Safe fn. RuntimeAlign::new(0) accepts 0 -> with_capacity_unchecked(16, 0)
    // -> is_valid_alloc computes `0 - 1` -> Layout::from_size_align_unchecked(16, 0) + alloc.
    let v = AlignedBytes::with_capacity_exact(0, 16);
    eprintln!("CHILD_SURVIVED with_capacity_exact(0,16): ptr={:p} cap={}", v.as_ptr(), v.capacity());
}

#[test]
#[ignore]
fn child_from_slice_0() {
    // Safe fn: from_slice(0, ..) -> with_capacity_exact(0, 1) -> same as above.
    let v = AlignedBytes::from_slice(0, &[1u8]);
    eprintln!("CHILD_SURVIVED from_slice(0,[1]): {:?}", &v[..]);
}

// ---- control: a valid alignment works ----

#[test]
fn control_align_1_ok() {
    let v = AlignedBytes::new_exact(1);
    assert_eq!(v.alignment(), 1);
    let w = AlignedBytes::with_capacity_exact(1, 16);
    assert_eq!(w.capacity(), 16);
    let s = AlignedBytes::from_slice(1, &[1u8]);
    assert_eq!(&s[..], &[1u8]);
}

#[test]
fn control_align_3_panics() {
    // Non-zero non-power-of-two IS rejected with an ordinary panic.
    let r = std::panic::catch_unwind(|| AlignedBytes::new_exact(3));
    assert!(r.is_err());
}

// ---- parents ----

fn run_child(name: &str) -> (Option<i32>, String) {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args(["--ignored", "--exact", name, "--nocapture", "--test-threads=1"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    eprintln!("=== child {name}: status {:?}", out.status);
    for line in stderr.lines().chain(stdout.lines()) {
        if line.contains("precondition")
            || line.contains("CHILD_SURVIVED")
            || line.contains("panicked")
            || line.contains("requires")
            || line.contains("overflow")
            || line.contains("abort")
        {
            eprintln!("    {line}");
        }
    }
    (out.status.code(), stderr + &stdout)
}

#[test]
fn parent_new_exact_0() {
    let (code, text) = run_child("child_new_exact_0");
    assert!(!text.contains("CHILD_SURVIVED"), "child survived: {text}");
    assert!(code != Some(0), "child exited 0");
}

#[test]
fn parent_with_capacity_exact_0_16() {
    let (code, text) = run_child("child_with_capacity_exact_0_16");
    assert!(!text.contains("CHILD_SURVIVED"), "child survived: {text}");
    assert!(code != Some(0), "child exited 0");
}

#[test]
fn parent_from_slice_0() {
    let (code, text) = run_child("child_from_slice_0");
    assert!(!text.contains("CHILD_SURVIVED"), "child survived: {text}");
    assert!(code != Some(0), "child exited 0");
}
