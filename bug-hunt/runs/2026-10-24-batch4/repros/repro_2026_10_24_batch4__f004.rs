// Repro for bug-hunt run 2026-10-24-batch4, finding F004.
//
// Reshape's read-time dim-matching scan (jix/src/ops/shape_ops/reshape.rs:296-314) gives up for
// good at the first new dim it cannot match, so for any split or merge reshape (flatten,
// [N]->[a,b], [a,b]->[N], [6,4]->[2,3,4], ...) NO dim matches and `read_data` issues one 1-element
// inner read per output element (reshape.rs:347-381). On a Compact inner each of those reads
// decompresses a whole block.
//
// These tests assert the CORRECT behavior, so they fail while the bug exists:
// - `inner_read_count_*`: counts the inner reads deterministically. The inner storage is a Map
//   over a 1-element array broadcast to the full shape, so the map closure runs once per inner
//   `read_data` call (Broadcast reads 0..1 of its input on each call). A reshape that matches
//   the trailing (stride-1) dim, as the code's own comment at reshape.rs:195-201 describes,
//   needs at most one inner read per output row.
// - `compact_block_aligned_reshape_cost`: wall-clock ratio of reading a block-aligned reshape of a
//   Compact array against reading the array itself (generous 50x threshold; the bug gives ~1000x).
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//        --test repro_2026_10_24_batch4__f004 -- --nocapture --test-threads 1
use jix::{Array, ArrayParams};
use ndarray::{Array1, Array2, Ix1, Ix2, Ix3, array};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static CALLS_SPLIT: AtomicUsize = AtomicUsize::new(0);
static CALLS_FLATTEN: AtomicUsize = AtomicUsize::new(0);
static CALLS_3D: AtomicUsize = AtomicUsize::new(0);
static CALLS_BASE: AtomicUsize = AtomicUsize::new(0);

#[test]
fn inner_read_count_baseline() {
    // Sanity: reading the broadcast view directly calls the map closure once.
    let a = Array::compact_ndarray(&array![7.0f32]).unwrap();
    let v = a
        .map(|x: f32| {
            CALLS_BASE.fetch_add(1, Ordering::Relaxed);
            x
        })
        .broadcast(&[4096]);
    let got = v.to_ndarray::<Ix1>().unwrap();
    assert!(got.iter().all(|&x| x == 7.0));
    let calls = CALLS_BASE.load(Ordering::Relaxed);
    println!("broadcast [4096] read directly: {calls} inner read(s)");
    assert!(calls <= 4, "baseline: {calls} map calls");
}

#[test]
fn inner_read_count_split() {
    // [4096] -> [64, 64]: each output row is a contiguous run of 64 source elements.
    let a = Array::compact_ndarray(&array![7.0f32]).unwrap();
    let v = a
        .map(|x: f32| {
            CALLS_SPLIT.fetch_add(1, Ordering::Relaxed);
            x
        })
        .broadcast(&[4096])
        .reshape([64u64, 64]);
    let got = v.to_ndarray::<Ix2>().unwrap();
    assert!(got.iter().all(|&x| x == 7.0));
    let calls = CALLS_SPLIT.load(Ordering::Relaxed);
    println!("[4096] -> [64, 64]: {calls} inner reads (one per row would be 64)");
    assert!(calls <= 64, "[4096]->[64,64] issued {calls} inner reads, expected <= 64 (one per row)");
}

#[test]
fn inner_read_count_flatten() {
    // [64, 64] -> [4096] (flatten): one inner read per source row (64) suffices.
    let a = Array::compact_ndarray(&array![[7.0f32]]).unwrap();
    let v = a
        .map(|x: f32| {
            CALLS_FLATTEN.fetch_add(1, Ordering::Relaxed);
            x
        })
        .broadcast(&[64, 64])
        .reshape([4096u64]);
    let got = v.to_ndarray::<Ix1>().unwrap();
    assert!(got.iter().all(|&x| x == 7.0));
    let calls = CALLS_FLATTEN.load(Ordering::Relaxed);
    println!("[64, 64] -> [4096] (flatten): {calls} inner reads (one per source row would be 64)");
    assert!(calls <= 64, "flatten [64,64] issued {calls} inner reads, expected <= 64");
}

#[test]
fn inner_read_count_comment_example() {
    // The exact example of the comment at reshape.rs:195-201: [6, 4] -> [2, 3, 4], where the
    // comment says new dims 1 and 2 are matched, so one inner read per new dim-0 index (2).
    let a = Array::compact_ndarray(&array![[7.0f32]]).unwrap();
    let v = a
        .map(|x: f32| {
            CALLS_3D.fetch_add(1, Ordering::Relaxed);
            x
        })
        .broadcast(&[6, 4])
        .reshape([2u64, 3, 4]);
    let got = v.to_ndarray::<Ix3>().unwrap();
    assert!(got.iter().all(|&x| x == 7.0));
    let calls = CALLS_3D.load(Ordering::Relaxed);
    println!("[6, 4] -> [2, 3, 4]: {calls} inner reads (comment says dims 1, 2 match -> 2)");
    assert!(calls <= 6, "[6,4]->[2,3,4] issued {calls} inner reads, expected <= 6");
}

fn best_of<F: FnMut()>(n: usize, mut f: F) -> Duration {
    (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .min()
        .unwrap()
}

#[test]
fn compact_block_aligned_reshape_cost() {
    // [16384] f32 stored in blocks of 1024 -> reshape [16, 1024]: every new row is exactly one
    // stored block, so a correct read costs about one pass over the 16 blocks, like reading `a`.
    const N: usize = 16384;
    const B: u32 = 1024;
    let data = Array1::from_iter((0..N).map(|i| i as f32));
    let mut p = ArrayParams::new();
    p.block_shape(&[B]);
    let a = Array::compact_ndarray_with(&data, p).unwrap();

    let expect = Array2::from_shape_vec((N / B as usize, B as usize), data.to_vec()).unwrap();
    let r = a.view().reshape([(N / B as usize) as u64, B as u64]);
    assert_eq!(r.to_ndarray::<Ix2>().unwrap(), expect); // results are correct

    let base = best_of(5, || {
        a.view().to_ndarray::<Ix1>().unwrap();
    });
    let reshaped = best_of(2, || {
        r.to_ndarray::<Ix2>().unwrap();
    });
    let ratio = reshaped.as_secs_f64() / base.as_secs_f64();
    println!(
        "block-aligned [16384] (block 1024) -> [16, 1024]: base {base:?}, reshaped {reshaped:?} \
         ({ratio:.0}x)"
    );
    assert!(
        ratio < 50.0,
        "block-aligned reshape read is {ratio:.0}x slower than reading the array itself"
    );
}
