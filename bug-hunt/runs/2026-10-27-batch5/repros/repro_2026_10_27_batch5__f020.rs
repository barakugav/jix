// Repro for F020 (bug-hunt run 2026-10-27-batch5).
//
// The read-cost hint `element_cost` (f32) is never saturated, so a lazy pipeline whose estimated
// per-element cost passes f32::MAX (~2^128) gets cost +inf. Concatenate weights each input's cost
// by its share of the concat dim, and for a zero-length member that is 0 * inf = NaN. The next
// multi-input constructor (Concatenate, Stack, Op2, where, map_multiple) then panics in
// `max_cost_layout_order` (`partial_cmp(..).unwrap()` on NaN, params.rs:912).
//
// Debug builds fail earlier, at the `debug_assert!(element_cost.is_finite())` in
// `Array::from_storage` (array.rs:1662), on the op whose cost first becomes inf.
//
// These tests assert the CORRECT behavior (constructors return Ok, values are right), so they
// fail while the bug exists:
//   cargo test --manifest-path jix/Cargo.toml --all-features --release \
//       --test repro_2026_10_27_batch5__f020 -- --nocapture
//
// The release build shows the params.rs:912 panic; the debug build the array.rs:1662 one.

use jix::Array;
use jix::ArrayAny;
use jix::ops::Concatenate;
use ndarray::{Array1, array};

/// A zero-length f32 array whose estimated element cost is +inf: `x = x + x` doubles the cost
/// (c -> 2c + 1) each step, so ~125 steps pass f32::MAX.
fn empty_with_inf_cost_add_chain() -> ArrayAny {
    let mut x = Array::compact_ndarray(&Array1::<f32>::zeros(0))
        .unwrap()
        .into_any();
    for _ in 0..130 {
        let a = x.clone().into_typed::<f32>().unwrap();
        let b = x.into_typed::<f32>().unwrap();
        x = (a + b).into_any();
    }
    x
}

/// Same, reached through nested reductions over lazy broadcasts (the hunter's trigger): each
/// level multiplies the cost by the reduced length + 4.
fn empty_with_inf_cost_nested_sum() -> ArrayAny {
    // cost: 8 -> 8 * (2^62 + 4) ~ 2^65 -> ~2^125 -> ~2^185 = inf (u8 sums to u64, so the
    // later broadcasts stay at 2^60 to keep the byte size within u64).
    let base = Array::compact_ndarray(&array![[1u8]]).unwrap();
    let l1 = base.broadcast(&[1, 1 << 62]).sum(1);
    let l2 = l1.insert_dim(1).broadcast(&[1, 1 << 60]).sum(1);
    let l3 = l2.insert_dim(1).broadcast(&[1, 1 << 60]).sum(1);
    l3.into_any().slice((0..0,)).into_any()
}

fn check_concat_twice(empty: ArrayAny, other: ArrayAny) {
    // First join: the inf-cost member has weight 0, so the result's cost is 0 * inf = NaN.
    let c = Concatenate::new_array(vec![empty, other.clone()], 0).unwrap();
    assert_eq!(c.shape(), &[1]);
    // Second join combines the NaN cost: panics in max_cost_layout_order.
    let c2 = Concatenate::new_array(vec![c.into_any(), other], 0).unwrap();
    assert_eq!(c2.shape(), &[2]);
}

#[test]
fn concat_after_inf_cost_add_chain() {
    let other = Array::compact_ndarray(&array![5.0f32]).unwrap().into_any();
    let empty = empty_with_inf_cost_add_chain();
    assert_eq!(empty.shape(), &[0]);
    check_concat_twice(empty.clone(), other.clone());

    // An element-wise op on the NaN-cost concatenation panics the same way.
    let c = Concatenate::new_array(vec![empty, other.clone()], 0).unwrap();
    let sum = c.into_any().into_typed::<f32>().unwrap() + other.into_typed::<f32>().unwrap();
    assert_eq!(sum.to_ndarray().unwrap(), array![10.0f32]);
}

#[test]
fn concat_after_inf_cost_nested_sum() {
    let other = Array::compact_ndarray(&array![5u64]).unwrap().into_any();
    let empty = empty_with_inf_cost_nested_sum();
    assert_eq!(empty.shape(), &[0]);
    check_concat_twice(empty, other);
}
