// Bug-hunt repro (run 2026-10-27-batch5, finding F002).
// 32-bit: the reduction driver computes reduced-stream positions in usize
// (jix/src/ops/reduction.rs:486-497 and 588-593). On a 32-bit target a tile that starts at stream
// position k*2^32 gets base 0, so the fold re-initializes the cell (fold_one_cell: merge iff
// base_item_idx > 0, reduction.rs:818; multi-cell: init iff base_item_idx == 0, reduction.rs:655)
// and every item before it is lost. ArgMax indices are truncated mod 2^32.
//
// The tests assert the CORRECT result. They only make sense on a 32-bit target (on 64-bit they
// would pass after folding 2^32+ items), so the file is gated. Run:
//   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_27_batch5__f002 --release -- --nocapture --test-threads 1
// (release shows the silent wrong results; debug panics at reduction.rs:495 / :492 instead).
#![cfg(target_pointer_width = "32")]

use jix::Array;
use ndarray::{array, Ix0, Ix1};

const N: u64 = (1u64 << 32) + 16;

// One reduced dim longer than 2^32, one-cell fold path.
#[test]
fn sum_one_dim_longer_than_2pow32() {
    let s = Array::compact_ndarray(&array![1u8]).unwrap().broadcast(&[N]).sum(0);
    let got = s.to_ndarray::<Ix0>().unwrap()[()];
    assert_eq!(got, N);
}

// Mean divides the partial sum by the correct full count.
#[test]
fn mean_one_dim_longer_than_2pow32() {
    let m = Array::compact_ndarray(&array![1.0f32]).unwrap().broadcast(&[N]).mean(0);
    let got = m.to_ndarray::<Ix0>().unwrap()[()];
    assert_eq!(got, 1.0);
}

// Two reduced dims with more than 2^32 items in total (each dim small).
#[test]
fn sum_two_dims_more_than_2pow32_items() {
    let s = Array::compact_ndarray(&array![[1u8]])
        .unwrap()
        .broadcast(&[65537, 65536])
        .sum(&[0usize, 1][..]);
    let got = s.to_ndarray::<Ix0>().unwrap()[()];
    assert_eq!(got, 65537u64 * 65536);
}

// Multi-cell fold path: reduced dim outer, kept dim inner.
#[test]
fn sum_multi_cell_layout_longer_than_2pow32() {
    let s = Array::compact_ndarray(&array![[1u8, 1u8]]).unwrap().broadcast(&[N, 2]).sum(0);
    let got = s.to_ndarray::<Ix1>().unwrap();
    assert_eq!(got, array![N, N]);
}

// ArgMax: the only maximum sits past position 2^32; its index must not be truncated.
#[test]
fn argmax_index_past_2pow32() {
    let zeros = Array::compact_ndarray(&array![0u8]).unwrap().broadcast(&[N]);
    let one = Array::compact_ndarray(&array![1u8]).unwrap().broadcast(&[1]);
    let a = jix::ops::concatenate((zeros, one), 0);
    let got = a.argmax(0).to_ndarray::<Ix0>().unwrap()[()];
    assert_eq!(got, N);
}
