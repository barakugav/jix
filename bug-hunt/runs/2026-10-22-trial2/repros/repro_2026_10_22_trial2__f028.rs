// Verifier repro for F028 (run 2026-10-22-trial2).
//
// Claim: in the multi-cell fold path (`fold_multi_cell_impl`, jix/src/ops/reduction.rs), a cell's
// items reach `update_state` in the order of the `NdIterUnordered` outer walk, which is sorted by
// memory strides, not in ascending stream position. When two reduced dims are laid out in memory
// in the reverse of their logical order, a cell is visited at stream positions 0, 2, 1, 3.
// `ArgMax::update` / `ArgMin::update` assume ascending order ("a tie keeps the earlier one"), so a
// multi-dim argmax returns the later of two tied maxima.
//
// `ReductionOp` / `ArgMaxKernel` are `pub(crate)` and the public `argmax` is single-dim, so the
// argmax itself is not reachable from here. This test drives the SAME driver path through the
// public `reduce_unordered`, whose kernel maps `init_state` -> first item and
// `update_state(s, x)` -> `f(s, x)`, with `f` = the public scalar `jix::scalar::ArgMax::update`
// on (value, stream-position) packed into a u64. In the multi-cell path only `init_state` and
// `update_state` run (no `merge_states`), so this is exactly what `ArgMaxKernel` computes there.
//
// Both tests assert the CORRECT behavior and therefore fail while the bug exists.
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_22_trial2__f028 -- --nocapture

use std::sync::Mutex;

use jix::dtype::Dtyped;
use jix::{Array, ArrayParams, Ty};

const SHAPE: [usize; 3] = [2, 2, 3]; // (r0, r1, k): reduce r0 and r1, keep k
const ITEM: usize = 8; // u64
// Memory order: k innermost, then r0, then r1 outermost - the reverse of the logical r0, r1.
const STRIDES: [usize; 3] = [3 * ITEM, 6 * ITEM, ITEM];

fn offset(r0: usize, r1: usize, k: usize) -> usize {
    r0 * STRIDES[0] + r1 * STRIDES[1] + k * STRIDES[2]
}

/// Row-major position of `(r0, r1)` in the reduced sub-shape `[2, 2]`.
fn stream_pos(r0: usize, r1: usize) -> u64 {
    (r0 * SHAPE[1] + r1) as u64
}

fn build(backing: &[u64]) -> Array<jix::storage::Plain<&'static (), Ty<u64>>> {
    let shape = SHAPE.map(|s| s as u64);
    unsafe {
        // SAFETY: `backing` outlives the array, every index stays in bounds (max offset 88 + 8 <=
        // 12 * 8 bytes), and the base is 8-aligned.
        Array::plain_ndarray_ptr(
            backing.as_ptr().cast::<u8>(),
            &shape,
            &STRIDES,
            <u64 as Dtyped>::DTYPE,
            ArrayParams::new(),
        )
    }
    .unwrap()
}

fn pack(value: u32, idx: u64) -> u64 {
    ((value as u64) << 32) | idx
}

#[test]
fn multi_cell_fold_visits_cell_items_in_stream_order() {
    // Each element stores its own stream position; record the order `update_state` sees for
    // cell k = 0.
    let mut backing = vec![0u64; 12];
    for r0 in 0..2 {
        for r1 in 0..2 {
            for k in 0..3 {
                backing[offset(r0, r1, k) / ITEM] = pack(k as u32, stream_pos(r0, r1));
            }
        }
    }
    let seen = Mutex::new(Vec::<u64>::new());
    let arr = build(&backing);
    let _ = arr
        .reduce_unordered([0usize, 1].as_slice(), |acc: u64, x: u64| {
            if x >> 32 == 0 {
                let mut seen = seen.lock().unwrap();
                if seen.is_empty() {
                    seen.push(acc & 0xffff_ffff); // the seed (init_state)
                }
                seen.push(x & 0xffff_ffff);
            }
            acc
        })
        .to_ndarray::<ndarray::Ix1>()
        .unwrap();
    let seen = seen.into_inner().unwrap();
    eprintln!("stream positions folded into cell 0, in visit order: {seen:?}");
    // `ArgMaxKernel::update_state` relies on ascending order.
    assert_eq!(seen, vec![0, 1, 2, 3], "multi-cell fold visited a cell out of stream order");
}

#[test]
fn argmax_update_rule_keeps_first_tied_maximum_over_two_permuted_reduced_dims() {
    // Cell k: maxima tied at (r0=0, r1=1) [pos 1] and (r0=1, r1=0) [pos 2]; the first one, pos 1,
    // is the correct argmax (NumPy: np.argmax(x[:, :, k]) == 1).
    let mut backing = vec![0u64; 12];
    for r0 in 0..2 {
        for r1 in 0..2 {
            for k in 0..3 {
                let p = stream_pos(r0, r1);
                let v = if p == 1 || p == 2 { 100 + k as u32 } else { k as u32 };
                backing[offset(r0, r1, k) / ITEM] = pack(v, p);
            }
        }
    }
    let arr = build(&backing);
    // `ArgMaxKernel`: init_state = (idx, item); update_state = ArgMax::update(state, item, idx).
    let got = arr
        .reduce_unordered([0usize, 1].as_slice(), |acc: u64, x: u64| {
            let state = (acc & 0xffff_ffff, (acc >> 32) as u32);
            let (i, v) = <u32 as jix::scalar::ArgMax>::update(state, (x >> 32) as u32, x & 0xffff_ffff);
            pack(v, i)
        })
        .to_ndarray::<ndarray::Ix1>()
        .unwrap();
    let got_idx = got.mapv(|s| s & 0xffff_ffff);
    eprintln!("argmax stream positions per cell: {got_idx:?}");
    assert_eq!(got_idx, ndarray::array![1u64, 1, 1]);
}

#[test]
fn multi_cell_fold_visits_cell_items_in_stream_order_across_tiles_c_order() {
    // Second trigger, no permuted strides needed: a plain C-order Compact array [r0, r1, k] =
    // [4, 4, 2] with 2x2x2 blocks and a read window of 8 items, so each tile is [2, 2, 2]. Tiles
    // are swept row-major over the reduced dims, so tile (0, 0) holds stream positions
    // {0, 1, 4, 5} and tile (0, 1) holds {2, 3, 6, 7}: position 4 is folded before position 2.
    // The kept dim k is innermost in memory, so this is the multi-cell path, where tiles after
    // the first one continue the cell with `update_state` (not the index-aware `merge_states`).
    let (n0, n1, nk) = (4usize, 4usize, 2usize);
    let nd = ndarray::Array3::<u64>::from_shape_fn((n0, n1, nk), |(r0, r1, k)| {
        pack(k as u32, (r0 * n1 + r1) as u64)
    });
    let mut params = ArrayParams::new();
    params.block_shape(&[2, 2, 2]).read_size((64, 64)); // 8 u64 items
    let arr = Array::compact_ndarray_with(&nd, params).unwrap();

    let seen = Mutex::new(Vec::<u64>::new());
    let _ = arr
        .reduce_unordered([0usize, 1].as_slice(), |acc: u64, x: u64| {
            if x >> 32 == 0 {
                let mut seen = seen.lock().unwrap();
                if seen.is_empty() {
                    seen.push(acc & 0xffff_ffff);
                }
                seen.push(x & 0xffff_ffff);
            }
            acc
        })
        .to_ndarray::<ndarray::Ix1>()
        .unwrap();
    let seen = seen.into_inner().unwrap();
    eprintln!("C order, multi-tile: stream positions folded into cell 0, in visit order: {seen:?}");
    assert_eq!(
        seen,
        (0..16).collect::<Vec<u64>>(),
        "multi-cell fold visited a cell out of stream order"
    );
}
