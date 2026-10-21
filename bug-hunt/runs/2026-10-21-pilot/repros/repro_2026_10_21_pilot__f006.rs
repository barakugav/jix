//! F006: ArrayParams::tune divides by zero when a fixed block dim is 0 and another dim is scaled.
use jix::{Array, ArrayParams};
use ndarray::{Array2, Array4};

fn describe<T>(r: &std::thread::Result<Result<T, jix::Error>>) -> String {
    match r {
        Ok(Ok(_)) => "Ok(..)".to_string(),
        Ok(Err(e)) => format!("Err({e})"),
        Err(p) => format!(
            "PANIC: {}",
            p.downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_default()
        ),
    }
}

/// Reference: all dims fixed -> scaling skipped -> clean Err from the validation in `tune`.
#[test]
fn all_fixed_zero_dim_is_clean_err() {
    let nd = Array2::<f32>::zeros((3, 5));
    let mut p = ArrayParams::new();
    p.block_shape(&[0, 4]);
    let r = std::panic::catch_unwind(|| Array::compact_ndarray_with(&nd, p).map(|_| ()));
    println!("[all fixed]   block_shape [0, 4], fixed [T, T] -> {}", describe(&r));
    assert!(matches!(r, Ok(Err(_))));
}

/// Bug: same block shape, but dim 1 is auto-scaled -> integer divide by zero before validation.
#[test]
fn fixed_zero_dim_with_scaled_dim_panics() {
    let nd = Array2::<f32>::zeros((3, 5));
    let mut p = ArrayParams::new();
    p.block_shape(&[0, 4]).block_shape_fixed_dims(&[true, false]);
    let r = std::panic::catch_unwind(|| Array::compact_ndarray_with(&nd, p).map(|_| ()));
    println!("[one scaled]  block_shape [0, 4], fixed [T, F] -> {}", describe(&r));
    assert!(matches!(r, Ok(Err(_))), "expected Err(InvalidArgument), got a panic");
}

/// Second report's trigger (zero in the last dim, scaled dim first).
#[test]
fn fixed_zero_last_dim_with_scaled_dim_panics() {
    let nd = Array2::<f32>::zeros((8, 4));
    let mut p = ArrayParams::new();
    p.block_shape(&[4, 0]).block_shape_fixed_dims(&[false, true]);
    let r = std::panic::catch_unwind(|| Array::compact_ndarray_with(&nd, p).map(|_| ()));
    println!("[one scaled]  block_shape [4, 0], fixed [F, T] -> {}", describe(&r));
    assert!(matches!(r, Ok(Err(_))), "expected Err(InvalidArgument), got a panic");
}

/// Same path through `compact_with` on an existing array (array.rs:1289).
#[test]
fn compact_with_fixed_zero_dim_panics() {
    let a = Array::compact_ndarray(&Array2::<f32>::zeros((3, 5))).unwrap();
    let ctx = a.read_ctx();
    let mut p = ArrayParams::new();
    p.block_shape(&[0, 4]).block_shape_fixed_dims(&[true, false]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        a.view().compact_with(p, &ctx).map(|_| ())
    }));
    println!("[compact_with] block_shape [0, 4], fixed [T, F] -> {}", describe(&r));
    assert!(matches!(r, Ok(Err(_))), "expected Err(InvalidArgument), got a panic");
}

/// Overflow variant: the fixed-dim volume product is unchecked (debug: multiply overflow panic).
#[test]
fn huge_fixed_dims_volume_overflow_panics() {
    let nd = Array4::<f32>::zeros((3, 3, 3, 5));
    let mut p = ArrayParams::new();
    p.block_shape(&[1 << 31, 1 << 31, 4, 1]).block_shape_fixed_dims(&[true, true, true, false]);
    let r = std::panic::catch_unwind(|| Array::compact_ndarray_with(&nd, p).map(|_| ()));
    println!(
        "[overflow]    block_shape [2^31, 2^31, 4, 1], fixed [T, T, T, F] -> {}",
        describe(&r)
    );
    assert!(matches!(r, Ok(Err(_))), "expected Err(InvalidArgument), got a panic");
}
