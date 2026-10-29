//! F036: f32 / f16 base ** i32 exponent rounds the exponent to f32 (`b as f32`), so an odd
//! exponent above 2^24 becomes even and a negative base gives a result with the wrong sign.
//! The f64 ** i32 path is exact (i32 -> f64 is exact) and serves as the reference.
//! Asserts the CORRECT behavior, so it fails while the bug exists.
use jix::Array;
use jix::error::{ArithmeticErrorKind, ArithmeticErrorMode};
use jix::scalar::f16;
use ndarray::{Ix1, array};

const N: i32 = 16_777_217; // 2^24 + 1, odd; `N as f32` == 16777216.0 (even)

fn ignore_errors(ctx: &mut jix::ReadContext) {
    for k in [ArithmeticErrorKind::Overflow, ArithmeticErrorKind::InvalidOperation] {
        ctx.set_arithmetic_error_mode(k, ArithmeticErrorMode::Ignore);
    }
}

#[test]
fn f64_pow_odd_large_i32_keeps_sign_control() {
    let base = Array::compact_ndarray(&array![-1.0f64, -0.5, -2.0]).unwrap();
    let exp = Array::compact_ndarray(&array![N, N, N]).unwrap();
    let mut ctx = base.read_ctx();
    ignore_errors(&mut ctx);
    let got = base.pow(exp).to_ndarray_sub::<Ix1>(&[0..3], &ctx).unwrap();
    println!("f64 ** i32 {N}: {got:?}");
    assert_eq!(got[0], -1.0);
    assert!(got[1] == 0.0 && got[1].is_sign_negative());
    assert_eq!(got[2], f64::NEG_INFINITY);
}

#[test]
fn f32_pow_odd_large_i32_keeps_sign() {
    let base = Array::compact_ndarray(&array![-1.0f32, -0.5, -2.0]).unwrap();
    let exp = Array::compact_ndarray(&array![N, N, N]).unwrap();
    let mut ctx = base.read_ctx();
    ignore_errors(&mut ctx);
    let got = base.pow(exp).to_ndarray_sub::<Ix1>(&[0..3], &ctx).unwrap();
    println!("f32 ** i32 {N}: {got:?}  (want [-1.0, -0.0, -inf])");
    assert_eq!(got[0], -1.0, "(-1f32) ** {N}");
    assert!(got[1] == 0.0 && got[1].is_sign_negative(), "(-0.5f32) ** {N} = {}", got[1]);
    assert_eq!(got[2], f32::NEG_INFINITY, "(-2f32) ** {N}");
}

#[test]
fn f16_pow_odd_large_i32_keeps_sign() {
    let base = Array::compact_ndarray(&array![
        f16::from_f32(-1.0),
        f16::from_f32(-0.5),
        f16::from_f32(-2.0)
    ])
    .unwrap();
    let exp = Array::compact_ndarray(&array![N, N, N]).unwrap();
    let mut ctx = base.read_ctx();
    ignore_errors(&mut ctx);
    let got = base.pow(exp).to_ndarray_sub::<Ix1>(&[0..3], &ctx).unwrap();
    println!("f16 ** i32 {N}: {got:?}  (want [-1.0, -0.0, -inf])");
    assert_eq!(got[0].to_f32(), -1.0, "(-1f16) ** {N}");
    assert!(
        got[1].to_f32() == 0.0 && got[1].is_sign_negative(),
        "(-0.5f16) ** {N} = {}",
        got[1]
    );
    assert_eq!(got[2].to_f32(), f32::NEG_INFINITY, "(-2f16) ** {N}");
}

#[test]
fn f32_pow_odd_large_negative_i32_keeps_sign() {
    // Negative odd exponent: (-1)^-(2^24+1) = -1, (-2)^-(2^24+1) = -0.0.
    let base = Array::compact_ndarray(&array![-1.0f32, -2.0]).unwrap();
    let exp = Array::compact_ndarray(&array![-N, -N]).unwrap();
    let mut ctx = base.read_ctx();
    ignore_errors(&mut ctx);
    let got = base.pow(exp).to_ndarray_sub::<Ix1>(&[0..2], &ctx).unwrap();
    println!("f32 ** i32 {}: {got:?}  (want [-1.0, -0.0])", -N);
    assert_eq!(got[0], -1.0, "(-1f32) ** {}", -N);
    assert!(got[1] == 0.0 && got[1].is_sign_negative(), "(-2f32) ** {} = {}", -N, got[1]);
}
