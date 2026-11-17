//! K185 (run 2026-11-16-scan-106-107): 32-bit only. #107 caps `ReadSize` at isize::MAX bytes
//! (params.rs:26-27) "so read tiles and compaction chunks fit in memory", but `scale_read_shape`
//! (util/mod.rs:503-509) grows a dim with a non-NONE `ScaleWeight` (Broadcast marks its broadcast
//! dims FULL) up to `2 * max_nitems`, and compaction floors `max_chunk` at one target block. So:
//! - a compaction chunk reaches up to 2 * read_size.max bytes, or 2 target blocks (2 GiB with
//!   1 GiB blocks, at the DEFAULT read size), and `chunk_buf.reserve` (array/compact.rs:262-264)
//!   panics with "capacity overflow" (util/aligned_vec/raw.rs:33);
//! - a reduction tile over a broadcast dim reaches 2 * read_size.max bytes, and the Broadcast's
//!   `check_out_buf` rejects it: `sum` returns InvalidShapeOperation although the output is a
//!   scalar and the (zero-copy) read would work with a tile capped at isize::MAX.
//!
//! Each test asserts the CORRECT behavior, so it fails while the bug exists: compaction must not
//! panic (an Ok with the right data or a clean Err both pass), and the sums must be complete.
//! Gated to 32-bit: on 64-bit the doubled tile still fits in memory.
//!
//! Run: bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//!        --test repro_k185 -- --nocapture --test-threads 1
#![cfg(target_pointer_width = "32")]

use jix::{Array, ArrayParams};
use ndarray::{array, Ix0};

fn compact_broadcast_ones(src: ArrayParams, n: u64, block: u32) {
    let a = Array::compact_ndarray_with(&array![1u8], src).unwrap();
    let b = a.broadcast(&[n]);
    let ctx = b.read_ctx();
    // Panics today with "capacity overflow" (a chunk of 2^31 bytes > isize::MAX).
    let res = b.compact_with(ArrayParams::new().block_shape(&[block]).unwrap(), &ctx);
    println!("compact_with -> {:?}", res.as_ref().map(|c| c.shape().to_vec()));
    if let Ok(c) = res {
        assert_eq!(c.shape(), &[n]);
        let mut tail = vec![0u8; 16];
        c.to_buf_slice(&[n - 16..n], &mut tail, &c.read_ctx()).unwrap();
        assert_eq!(tail, vec![1u8; 16]);
    }
}

/// A 1 GiB read size (below isize::MAX, so #107 does not clamp it): max_chunk = 1024 blocks of
/// 1 MiB, doubled along the broadcast dim to 2048 blocks = 2^31 bytes.
#[test]
fn compact_with_1_gib_read_size_does_not_panic() {
    compact_broadcast_ones(ArrayParams::new().read_size((1, 1 << 30)), 1 << 31, 1 << 20);
}

/// DEFAULT read size, 1 GiB target blocks (the BLOCK_SIZE_BYTES_MAX cap): max_chunk is floored at
/// 1 block, doubled along the broadcast dim to 2 blocks = 2^31 bytes.
#[test]
fn compact_with_default_read_size_1_gib_blocks_does_not_panic() {
    compact_broadcast_ones(ArrayParams::new(), 1 << 31, 1 << 30);
}

fn sum_broadcast_ones(read_size: (u64, u64), n: u64) {
    let a = Array::compact_ndarray_with(&array![1u8], ArrayParams::new().read_size(read_size))
        .unwrap();
    let res = a.broadcast(&[n]).sum(0).to_ndarray::<Ix0>();
    println!("sum over {n} ones with read_size {read_size:?}: {res:?}");
    assert_eq!(res.unwrap()[[]], n);
}

/// 1 GiB budget (not clamped): the tile grows to 2^31 bytes on the broadcast dim.
#[test]
fn sum_over_long_broadcast_1_gib_read_size() {
    sum_broadcast_ones((1, 1 << 30), (1 << 31) + 5);
}

/// The K153 budget (8 GiB, clamped by #107 to isize::MAX): the tile is 2 * (2^31 - 1) bytes.
#[test]
fn sum_over_long_broadcast_clamped_read_size() {
    sum_broadcast_ones((1, 1 << 33), (1 << 32) + 5);
}

/// Control (passes today): the same reduction with a 512 MiB budget. The doubled tile is 1 GiB,
/// which fits, so the sum over the same 2^31 + 5 broadcast elements completes. This shows the
/// failures above come from the 2 * max growth, not from the array or the reduction itself.
#[test]
#[ignore = "control; slow in debug (folds 2^31 elements)"]
fn control_sum_with_512_mib_read_size_completes() {
    sum_broadcast_ones((1, 1 << 29), (1 << 31) + 5);
}
