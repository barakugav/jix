// Repro for bug-hunt run 2026-10-24-batch4, finding F012.
//
// Stepped Slice (any dim with step > 1) over a non-plain inner takes the general path of
// `Slice::read_data` (jix/src/ops/shape_ops/slice.rs:166-206). It iterates over EVERY output
// position of every stepped dim (`iter_shape`, slice.rs:181-187) and issues one inner `read_data`
// per position (slice.rs:191-205). On a Compact inner each of those reads decompresses every block
// it touches (no block cache in `ReadContext`, jix/src/codec/mod.rs), so stepping the innermost dim
// costs about one whole-block decompression per output element.
//
// These tests assert the CORRECT behavior, so they fail while the bug exists:
// - `inner_read_count_*`: counts the inner reads deterministically. The inner storage is a Map
//   over a 1-element array broadcast to the full shape, so the map closure runs once per inner
//   `read_data` call (Broadcast reads 0..1 of its input on each call; baseline test = 1 call).
//   A stepped slice selects a subset of one bounding box, so a correct read needs O(1) inner reads
//   (or at most one per output row).
// - `compact_stepped_slice_cost`: wall-clock ratio of reading `a.slice(::2)` of a Compact array
//   against reading the whole array (generous 50x threshold; the bug gives ~1000x).
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//        --test repro_2026_10_24_batch4__f012 -- --nocapture --test-threads 1
use jix::ops::SliceItem;
use jix::{Array, ArrayParams};
use ndarray::{Array1, Ix1, Ix2, array};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static CALLS_BASE: AtomicUsize = AtomicUsize::new(0);
static CALLS_1D: AtomicUsize = AtomicUsize::new(0);
static CALLS_2D_COLS: AtomicUsize = AtomicUsize::new(0);
static CALLS_2D_BOTH: AtomicUsize = AtomicUsize::new(0);

fn step(s: i64) -> SliceItem {
    SliceItem::new(None, None, s)
}

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
fn inner_read_count_1d_step2() {
    // [4096][::2] -> 2048 elements, all inside one bounding box 0..4095.
    let a = Array::compact_ndarray(&array![7.0f32]).unwrap();
    let v = a
        .map(|x: f32| {
            CALLS_1D.fetch_add(1, Ordering::Relaxed);
            x
        })
        .broadcast(&[4096])
        .slice((step(2),));
    let got = v.to_ndarray::<Ix1>().unwrap();
    assert_eq!(got.len(), 2048);
    assert!(got.iter().all(|&x| x == 7.0));
    let calls = CALLS_1D.load(Ordering::Relaxed);
    println!("[4096][::2]: {calls} inner reads (one bounding-box read would be 1)");
    assert!(calls <= 4, "[4096][::2] issued {calls} inner reads, expected O(1)");
}

#[test]
fn inner_read_count_2d_cols() {
    // [64, 64][:, ::2]: one inner read per output column today.
    let a = Array::compact_ndarray(&array![[7.0f32]]).unwrap();
    let v = a
        .map(|x: f32| {
            CALLS_2D_COLS.fetch_add(1, Ordering::Relaxed);
            x
        })
        .broadcast(&[64, 64])
        .slice((step(1), step(2)));
    let got = v.to_ndarray::<Ix2>().unwrap();
    assert_eq!(got.dim(), (64, 32));
    assert!(got.iter().all(|&x| x == 7.0));
    let calls = CALLS_2D_COLS.load(Ordering::Relaxed);
    println!("[64, 64][:, ::2]: {calls} inner reads (one bounding-box read would be 1)");
    assert!(calls <= 4, "[64,64][:, ::2] issued {calls} inner reads, expected O(1)");
}

#[test]
fn inner_read_count_2d_both() {
    // [64, 64][::2, ::2]: one inner read per output ELEMENT today (32 * 32 = 1024).
    let a = Array::compact_ndarray(&array![[7.0f32]]).unwrap();
    let v = a
        .map(|x: f32| {
            CALLS_2D_BOTH.fetch_add(1, Ordering::Relaxed);
            x
        })
        .broadcast(&[64, 64])
        .slice((step(2), step(2)));
    let got = v.to_ndarray::<Ix2>().unwrap();
    assert_eq!(got.dim(), (32, 32));
    assert!(got.iter().all(|&x| x == 7.0));
    let calls = CALLS_2D_BOTH.load(Ordering::Relaxed);
    println!("[64, 64][::2, ::2]: {calls} inner reads (one per output row would be 32)");
    assert!(calls <= 32, "[64,64][::2, ::2] issued {calls} inner reads, expected <= 32");
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
fn compact_stepped_slice_cost() {
    // [32768] f32 stored in blocks of 4096 (8 blocks). `[::2]` selects half the elements of the
    // same 8 blocks, so a correct read costs about one pass over those blocks, like reading `a`.
    const N: usize = 32768;
    const B: u32 = 4096;
    let data = Array1::from_iter((0..N).map(|i| i as f32));
    let mut p = ArrayParams::new();
    p.block_shape(&[B]);
    let a = Array::compact_ndarray_with(&data, p).unwrap();

    let expect = Array1::from_iter((0..N).step_by(2).map(|i| i as f32));
    let s = a.view().slice((step(2),));
    assert_eq!(s.to_ndarray::<Ix1>().unwrap(), expect); // results are correct

    let base = best_of(5, || {
        a.view().to_ndarray::<Ix1>().unwrap();
    });
    let stepped = best_of(2, || {
        s.to_ndarray::<Ix1>().unwrap();
    });
    let ratio = stepped.as_secs_f64() / base.as_secs_f64();
    println!(
        "Compact [32768] f32 (block 4096) [::2]: full read {base:?}, stepped read {stepped:?} \
         ({ratio:.0}x)"
    );
    assert!(
        ratio < 50.0,
        "step-2 slice read is {ratio:.0}x slower than reading the whole array"
    );
}
