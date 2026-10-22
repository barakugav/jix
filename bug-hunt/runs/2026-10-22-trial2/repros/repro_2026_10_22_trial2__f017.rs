//! F017 repro: the contiguous inner loop of the element-wise pipeline keeps 8 lanes (4 on x86)
//! however wide the element is, so a pipeline over wide elements puts several `[T; 8]` values on
//! the stack per node and overflows a 2 MiB thread stack.
//!
//! THE CRASHING STEP ABORTS THE PROCESS (stack overflow -> SIGABRT). To report it as an ordinary
//! test failure, each case runs in a child process: this test binary re-executed with
//! `--exact child_*` and `F017_CHILD=1`; the parent asserts that the child exited successfully.
//!
//! Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//!          --test repro_2026_10_22_trial2__f017 -- --nocapture
//! (add `--release` for the release-build numbers; `-- --ignored measure` prints the minimum
//! stack each path needs).
use std::process::Command;

use jix::Array;
use ndarray::{Array1, Ix1, s};

/// Element width in bytes: a valid `Dtyped` (`[T; N]` is accepted up to `u16::MAX` bytes).
const W: usize = 60000;
/// Rust's default stack for spawned threads (also what `cargo test` gives each test).
const DEFAULT_STACK: usize = 2 << 20;

fn stack_size() -> usize {
    std::env::var("F017_STACK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_STACK)
}

/// `map(id).map(id)` over 16 elements of `[u8; W]`, on a thread with `stack_size()` bytes of stack.
/// `contiguous` picks a packed input (contiguous inner loop, 8 lanes) or a step-2 view of the same
/// data (strided inner loop, lanes capped to 1 by `LanesInfo`).
fn map_twice_on_thread(contiguous: bool) {
    std::thread::Builder::new()
        .stack_size(stack_size())
        .spawn(move || {
            let data = Array1::from(vec![[1u8; W]; 32]);
            let view = if contiguous {
                data.slice(s![..16])
            } else {
                data.slice(s![..;2])
            };
            let a = Array::plain_ndarray_view(view).unwrap();
            let got = a
                .map(|x: [u8; W]| x)
                .map(|x: [u8; W]| x)
                .to_ndarray::<Ix1>()
                .unwrap();
            assert_eq!(got.len(), 16);
            assert!(got.iter().all(|x| x.iter().all(|&b| b == 1)));
        })
        .unwrap()
        .join()
        .unwrap();
}

fn is_child() -> bool {
    std::env::var_os("F017_CHILD").is_some()
}

// Child entry points: no-ops unless re-executed by `run_child`.
#[test]
fn child_contiguous() {
    if is_child() {
        map_twice_on_thread(true);
    }
}
#[test]
fn child_strided() {
    if is_child() {
        map_twice_on_thread(false);
    }
}

fn run_child(name: &str, stack: usize) -> std::process::Output {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env("F017_CHILD", "1")
        .env("F017_STACK", stack.to_string())
        .output()
        .unwrap()
}

fn tail(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(4)..].join("\n")
}

/// Fails while the bug exists: the child aborts with a stack overflow.
#[test]
fn f017_wide_elements_contiguous_map_on_default_thread_stack() {
    let out = run_child("child_contiguous", DEFAULT_STACK);
    println!("contiguous [u8; {W}] map.map on a 2 MiB thread: {}", out.status);
    assert!(
        out.status.success(),
        "child aborted: {}\n{}",
        out.status,
        tail(&out.stderr)
    );
}

/// Control: the same element width and op chain through the strided inner loop.
#[test]
fn f017_control_wide_elements_strided_map_on_default_thread_stack() {
    let out = run_child("child_strided", DEFAULT_STACK);
    println!("strided [u8; {W}] map.map on a 2 MiB thread: {}", out.status);
    assert!(
        out.status.success(),
        "child aborted: {}\n{}",
        out.status,
        tail(&out.stderr)
    );
}

/// Prints the smallest thread stack (to 16 KiB) each path needs.
#[test]
#[ignore]
fn measure() {
    for (name, label) in [("child_contiguous", "contiguous"), ("child_strided", "strided")] {
        let (mut lo, mut hi) = (64usize << 10, 64usize << 20);
        assert!(run_child(name, hi).status.success(), "{label} fails even at {hi}");
        while hi - lo > 16 << 10 {
            let mid = (lo + hi) / 2;
            if run_child(name, mid).status.success() {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        println!("{label:>10} [u8; {W}] map.map: min stack ~ {} KiB", hi >> 10);
    }
}
