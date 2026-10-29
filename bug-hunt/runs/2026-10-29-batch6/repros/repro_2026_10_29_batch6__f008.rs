// Repro for F008 (run 2026-10-29-batch6). CRASHES THE PROCESS while the bug exists: the test thread
// overflows its stack and Rust aborts with "thread '<unnamed>' has overflowed its stack" (SIGABRT).
//
// A type-erased op chain - the shape every jix-py op result has, Any(Add(IntoType(Any(Add(...))), ..)) -
// is read and dropped by recursing once per op level, with no depth limit. A chain built in a loop
// (`x = x + 1` repeated) therefore overflows an ordinary 8 MiB (Linux main-thread sized) stack.
//
// The tests assert the CORRECT behavior (the value, or simply finishing the drop), so they fail while
// the bug exists. Run them one at a time - an abort in one kills the whole test binary:
//   cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_29_batch6__f008 -- --nocapture deep_chain_read
//   cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_29_batch6__f008 -- --nocapture deep_chain_drop
use jix::{Array, ArrayAny};
use ndarray::{array, Ix1};

const STACK: usize = 8 << 20; // 8 MiB, the usual Linux main-thread stack (ulimit -s 8192)

/// `x = x + [1, 1]` repeated `n` times, type-erasing every intermediate like jix-py does.
fn build(n: usize) -> ArrayAny {
    let mut x: ArrayAny = Array::compact_ndarray(&array![1.0f64, 2.0]).unwrap().into_any();
    for _ in 0..n {
        let one = Array::compact_ndarray(&array![1.0f64, 1.0]).unwrap();
        x = (x.into_typed::<f64>().unwrap() + one).into_any();
    }
    x
}

#[test]
fn deep_chain_read() {
    let n = 10_000;
    let x = build(n);
    eprintln!("built a {n}-deep chain; reading it on an 8 MiB thread");
    let got = std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || {
            let r = x.into_typed::<f64>().unwrap().to_ndarray::<Ix1>();
            // Drop of a 10k chain fits easily; only the read is under test here.
            r
        })
        .unwrap()
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(got, array![1.0 + n as f64, 2.0 + n as f64]);
}

#[test]
fn deep_chain_drop() {
    let n = 100_000;
    let x = build(n);
    eprintln!("built a {n}-deep chain; dropping it on an 8 MiB thread");
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || drop(x))
        .unwrap()
        .join()
        .unwrap();
}
