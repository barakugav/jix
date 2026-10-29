// Repro for F013 (run 2026-10-29-batch6): a Scalar or Plain operand overrides a compact operand's
// explicit block shape in combine_block_layout, so compact() of `a * scalar` / `a + plain` flips
// the user's column blocks [N, 1] to row blocks [1, N].
//
// Each test asserts the CORRECT behavior (the explicit layout survives, as it does for `a + a`),
// so it FAILS while the bug exists.
//
// The block layout of an array is observed through the public `ArrayParams::override_from_array`
// plus `ArrayParams`'s derived Debug output.
use jix::{Array, ArrayParams};
use ndarray::{Array2, Ix2};

const N: usize = 1024;

fn block_layout_of<S: jix::ArrayStorage>(a: &Array<S>) -> String {
    let mut p = ArrayParams::new();
    p.override_from_array(a);
    let s = format!("{p:?}");
    let start = s.find("block_shape:").unwrap();
    let end = s.find(", block_size").unwrap();
    s[start..end].to_string()
}

fn column_blocked() -> Array<jix::storage::Compact<jix::Ty<f32>>> {
    let data = Array2::<f32>::from_shape_fn((N, N), |(i, j)| (i * N + j) as f32);
    let mut params = ArrayParams::new();
    params.block_shape(&[N as u32, 1]); // column blocks, all dims fixed by default
    Array::compact_ndarray_with(&data, params).unwrap()
}

/// Control: an element-wise op between two compact operands keeps the explicit layout.
#[test]
fn control_compact_operand_keeps_explicit_block_shape() {
    let a = column_blocked();
    let src = block_layout_of(&a);
    println!("a:                  {src}");
    let aa = (a.view() + a.view()).compact().unwrap();
    let got = block_layout_of(&aa);
    println!("(a + a).compact():  {got}");
    assert!(got.starts_with("block_shape: Some([1024, 1])"), "got {got}");
}

/// Control: the same scalar expressed as a broadcast 0-d array keeps the explicit layout, because
/// Broadcast forwards the source's granularity (ones) instead of an L1-sized auto shape.
#[test]
fn control_broadcast_scalar_keeps_explicit_block_shape() {
    let a = column_blocked();
    let s0 = Array::compact_ndarray(&ndarray::arr0(2.0f32)).unwrap();
    let s = s0.insert_dim(0).insert_dim(0).broadcast(&[N as u64, N as u64]);
    println!("broadcast 0-d spec: {}", block_layout_of(&s));
    let lazy = a.view() * s;
    println!("a * bcast (lazy):   {}", block_layout_of(&lazy));
    let b = lazy.compact().unwrap();
    let got = block_layout_of(&b);
    println!("(a * bcast).compact(): {got}");
    assert!(got.starts_with("block_shape: Some([1024, 1])"), "got {got}");
}

/// Public Rust API: a Plain (ndarray) operand.
#[test]
fn plain_operand_keeps_explicit_block_shape() {
    let a = column_blocked();
    let p = Array::plain_ndarray(Array2::<f32>::ones((N, N))).unwrap();
    println!("plain spec:         {}", block_layout_of(&p));
    let lazy = a.view() + p;
    println!("a + plain (lazy):   {}", block_layout_of(&lazy));
    let b = lazy.compact().unwrap();
    let got = block_layout_of(&b);
    println!("(a + plain).compact(): {got}");
    // Data stays correct.
    let v = b.to_ndarray::<Ix2>().unwrap();
    assert_eq!(v[[3, 7]], (3 * N + 7) as f32 + 1.0);
    assert!(got.starts_with("block_shape: Some([1024, 1])"), "got {got}");
}

/// The Scalar storage Python builds for every Python scalar operand (`a * 2.0`).
#[test]
fn scalar_operand_keeps_explicit_block_shape() {
    let a = column_blocked();
    let s = Array::from_storage(
        jix::__private::Scalar::new(2.0f32, [N as u64, N as u64], ArrayParams::new()).unwrap(),
    );
    println!("scalar spec:        {}", block_layout_of(&s));
    let lazy = a.view() * s;
    println!("a * scalar (lazy):  {}", block_layout_of(&lazy));
    let b = lazy.compact().unwrap();
    let got = block_layout_of(&b);
    println!("(a * scalar).compact(): {got}");
    let v = b.to_ndarray::<Ix2>().unwrap();
    assert_eq!(v[[3, 7]], 2.0 * (3 * N + 7) as f32);
    assert!(got.starts_with("block_shape: Some([1024, 1])"), "got {got}");
}
