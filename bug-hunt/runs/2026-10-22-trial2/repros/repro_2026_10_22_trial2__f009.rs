// F009: Roll::new reduces `shift` with `shift.rem_euclid(len as i64)` (roll.rs:54). For a dim
// longer than i64::MAX, `len as i64` is negative, so the modulus is wrong (wrong values), and for
// len == u64::MAX with shift == i64::MIN the remainder overflows and panics.
// These tests assert the CORRECT behavior, so they fail while the bug exists.
use jix::{Array, ReadContext};
use ndarray::{array, Ix1};

fn first<S: jix::ArrayStorage<ElementType = jix::Ty<u8>>>(a: &Array<S>) -> u8 {
    a.to_ndarray_sub::<Ix1>(&[0..1], &ReadContext::default()).unwrap()[0]
}

#[test]
fn roll_neg_shift_len_2pow63_plus_1() {
    // [0,1,2] tiled r times: len = 3r = 2^63 + 1 > i64::MAX; in[j] = j % 3.
    let r: u64 = 3074457345618258603;
    let t = Array::compact_ndarray(&array![0u8, 1, 2]).unwrap().tile(r, 0);
    assert_eq!(t.shape(), &[(1u64 << 63) + 1]);
    // roll(-1): out[0] = in[1] = 1.
    assert_eq!(first(&t.roll(-1, 0)), 1);
}

#[test]
fn roll_pos_shift_len_u64_max() {
    let t = Array::compact_ndarray(&array![0u8, 1, 2]).unwrap().tile(u64::MAX / 3, 0);
    assert_eq!(t.shape(), &[u64::MAX]);
    // roll(1): out[0] = in[L-1] = (u64::MAX - 1) % 3 = 2.
    assert_eq!(first(&t.roll(1, 0)), 2);
}

#[test]
fn roll_i64_min_len_u64_max_returns_ok() {
    let b = Array::compact_ndarray(&array![7u8]).unwrap().broadcast(&[u64::MAX]);
    let r = jix::ops::Roll::new_array(b, i64::MIN, 0);
    assert!(r.is_ok());
    assert_eq!(first(&r.unwrap()), 7);
}
