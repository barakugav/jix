//! F007 repro: `reduce_unordered` over a wide `Dtyped` element (here `[u8; 65000]`) aborts the
//! process with a stack overflow on a thread with Rust's default 2 MiB spawned-thread stack.
//!
//! THE BUGGY CASES CRASH THE PROCESS (SIGABRT, "has overflowed its stack"), so every case runs
//! in a child process (this test binary re-executed with `--exact <child> --ignored`) and the
//! parent asserts the child exited 0. Each child runs its reduction on a thread spawned with an
//! explicit 2 MiB stack (Rust's default for spawned threads and cargo test's per-test thread).
//!
//! The one-cell fold (reduced dim innermost) keeps `LANES` copies of the item and of the state on
//! the stack: LANES = 16 on the strided path and >= 8 on the contiguous path, whatever the element
//! width. The multi-cell fold (kept dim innermost) handles one item at a time and is the control.
//!
//! - contiguous: a Compact array with `block_shape([16])` and the DEFAULT read size. The default
//!   read-size floor is the block size in bytes, so each tile holds 16 items -> 8 lanes.
//! - strided: a plain step-2 view with `read_size((1 MiB, 8 MiB))`, which is exactly the default
//!   `ArrayParams` picks on a CPU reporting a 16 MiB L2 (Apple M-series P-cluster:
//!   min = max(l1d / 2, l2 / 16), max = l2 / 2, params.rs:428-432). Tile >= 16 items -> 16 lanes.
//!   On this 1 MiB-L2 host the default floor is 64 KiB (one 65000-byte item per tile).
//!
//! Run: cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_24_batch4__f007 -- --nocapture
//! (add --release for the shipped-build behavior; `-- --ignored measure --nocapture` prints the
//! smallest stack each case needs)
use jix::{Array, ArrayParams};
use ndarray::{Array1, Array2, Ix0, Ix1, s};
use std::process::Command;

const W: usize = 65000;
type Wide = [u8; W];
const STACK_ENV: &str = "JIX_F007_STACK";

fn combine(a: Wide, b: Wide) -> Wide {
    let mut r = a;
    for i in 0..W {
        r[i] = a[i].wrapping_add(b[i]);
    }
    r
}

/// Runs `f` on a thread with the stack size from `JIX_F007_STACK` (default 2 MiB).
fn on_thread(f: impl FnOnce() + Send + 'static) {
    let stack = std::env::var(STACK_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2 << 20);
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

/// Strided one-cell fold: 32 items at stride 2 * W, all reduced into one cell. LANES = 16.
#[test]
#[ignore]
fn child_strided_one_cell() {
    on_thread(|| {
        let data = Array1::from(vec![[1u8; W]; 64]);
        let mut p = ArrayParams::new();
        p.read_size((1 << 20, 8 << 20)); // the automatic default on a 16 MiB-L2 CPU
        let a = Array::plain_ndarray_view_with(data.slice(s![..;2]), p).unwrap();
        let r = a.reduce_unordered(0, combine).to_ndarray::<Ix0>().unwrap();
        assert!(r[[]].iter().all(|&b| b == 32));
    });
}

/// Contiguous one-cell fold: 32 contiguous items in blocks of 16, all reduced into one cell.
/// LANES = 8.
#[test]
#[ignore]
fn child_contiguous_one_cell() {
    on_thread(|| {
        let data = Array1::from(vec![[1u8; W]; 32]);
        let mut p = ArrayParams::new();
        p.block_shape(&[16]);
        let a = Array::compact_ndarray_with(&data, p).unwrap();
        let r = a.reduce_unordered(0, combine).to_ndarray::<Ix0>().unwrap();
        assert!(r[[]].iter().all(|&b| b == 32));
    });
}

/// Control: same element type, block layout and item count, but the kept dim is innermost, so
/// the multi-cell fold (one item at a time) runs.
#[test]
#[ignore]
fn child_multi_cell_control() {
    on_thread(|| {
        let data = Array2::from_shape_vec((16, 2), vec![[1u8; W]; 32]).unwrap();
        let mut p = ArrayParams::new();
        p.block_shape(&[16, 2]);
        let a = Array::compact_ndarray_with(&data, p).unwrap();
        let r = a.reduce_unordered(0, combine).to_ndarray::<Ix1>().unwrap();
        assert!(r.iter().all(|c| c.iter().all(|&b| b == 16)));
    });
}

fn run_child(name: &str, stack: Option<usize>) -> (bool, String) {
    let exe = std::env::current_exe().unwrap();
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", name, "--nocapture", "--ignored", "--test-threads=1"]);
    if let Some(stack) = stack {
        cmd.env(STACK_ENV, stack.to_string());
    }
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let notes = stderr
        .lines()
        .filter(|l| l.contains("overflow") || l.contains("panicked") || l.contains("fatal"))
        .collect::<Vec<_>>()
        .join(" / ");
    (out.status.success(), format!("{name}: {} | {notes}", out.status))
}

fn check(name: &str) {
    let (ok, msg) = run_child(name, None);
    println!("{msg}");
    assert!(ok, "child crashed on a 2 MiB thread stack: {msg}");
}

#[test]
fn f007_control_multi_cell_wide_reduce_on_2mib_stack() {
    check("child_multi_cell_control");
}

#[test]
fn f007_strided_one_cell_wide_reduce_on_2mib_stack() {
    check("child_strided_one_cell");
}

#[test]
fn f007_contiguous_one_cell_wide_reduce_on_2mib_stack() {
    check("child_contiguous_one_cell");
}

/// Binary-searches (16 KiB steps) the smallest thread stack each case needs.
#[test]
#[ignore]
fn measure() {
    for name in [
        "child_multi_cell_control",
        "child_contiguous_one_cell",
        "child_strided_one_cell",
    ] {
        let (mut lo, mut hi) = (16usize << 10, 64usize << 20);
        if !run_child(name, Some(hi)).0 {
            println!("{name}: fails even with {} KiB", hi >> 10);
            continue;
        }
        while hi - lo > 16 << 10 {
            let mid = (lo + hi) / 2 / 4096 * 4096;
            if run_child(name, Some(mid)).0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        println!("{name}: min stack ~ {} KiB", hi >> 10);
    }
}
