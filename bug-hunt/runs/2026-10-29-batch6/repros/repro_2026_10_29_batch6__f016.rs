// Repro for F016 (run 2026-10-29-batch6): the Vec / slice `MapMultiple` scratch in
// jix/src/util/arr_sequence.rs calls `Vec::set_len(narrays * M)` over uninitialized `S::Item`
// slots (455 / 579) and then fills them with `tmp_buf[i] = item` (462 / 586), which derefs the
// Vec to a `&mut [S::Item]` covering the still-uninitialized slots.
//
// The values are correct in normal builds, so these tests pass natively. They FAIL under strict
// Miri (references must point to initialized values), which is how the defect is shown:
//
//   MIRIFLAGS="-Zmiri-recursive-validation" \
//     python jix/run_miri.py --all-features --test repro_2026_10_29_batch6__f016 -- vec_scratch
//
// Inputs are Plain arrays and the read pushes into a zeroed destination, so the crate's other
// uninitialized byte buffers (pool buffers, compact decode) are not involved and the first
// report is the scratch itself. They pass once the scratch is filled through
// `spare_capacity_mut()` (or a `Vec<MaybeUninit<T>>`) and `set_len` runs after the fill.
use jix::Array;
use jix::ops::map_multiple;
use ndarray::array;

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn vec_scratch_has_no_reference_to_uninit() {
    let a = Array::plain_ndarray(array![1i32, 2, 3]).unwrap();
    let b = Array::plain_ndarray(array![10i32, 20, 30]).unwrap();
    // `Vec<Array<_>>` sequence: arr_sequence.rs:450-467.
    let m = map_multiple(vec![a, b], |xs: &[i32]| xs[0] + xs[1]);
    let ctx = m.read_ctx();
    let mut dst = [0u8; 12];
    m.to_ndarray_slice(&[0..3], &mut dst, &ctx).unwrap();
    assert_eq!(decode_i32(&dst), vec![11, 22, 33]);
}

#[test]
fn slice_scratch_has_no_reference_to_uninit() {
    let arrays = vec![
        Array::plain_ndarray(array![1i32, 2, 3]).unwrap(),
        Array::plain_ndarray(array![10i32, 20, 30]).unwrap(),
    ];
    // `&[Array<_>]` sequence: arr_sequence.rs:574-591.
    let m = map_multiple(&arrays[..], |xs: &[i32]| xs[0] * xs[1]);
    let ctx = m.read_ctx();
    let mut dst = [0u8; 12];
    m.to_ndarray_slice(&[0..3], &mut dst, &ctx).unwrap();
    assert_eq!(decode_i32(&dst), vec![10, 40, 90]);
}
