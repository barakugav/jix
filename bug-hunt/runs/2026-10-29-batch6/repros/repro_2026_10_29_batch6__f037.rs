//! F037: float Add / Sub / Mul / Square and the Sum / Product reductions never report
//! InvalidOperation or Overflow, while Div reports the same conditions.
//!
//! Each test asserts the behavior consistent with `Div` (and numpy's errstate): a NaN result from
//! non-NaN operands is InvalidOperation, an infinite result from finite operands is Overflow.
//! They fail while the bug exists.
use jix::error::{ArithmeticErrorKind, ArithmeticErrorMode, ErrorKind};
use jix::{Array, ReadContext};
use ndarray::{array, Ix0, Ix1};

fn strict(mut ctx: ReadContext) -> ReadContext {
    for k in [
        ArithmeticErrorKind::DivideByZero,
        ArithmeticErrorKind::Overflow,
        ArithmeticErrorKind::Underflow,
        ArithmeticErrorKind::InvalidOperation,
    ] {
        ctx.set_arithmetic_error_mode(k, ArithmeticErrorMode::Error);
    }
    ctx
}

fn inf1() -> Array<jix::storage::Compact<jix::Ty<f64>>> {
    Array::compact_ndarray(&array![f64::INFINITY]).unwrap()
}

/// Default modes (InvalidOperation = Error): `inf / inf` fails the read, `inf - inf` must too.
#[test]
fn default_modes_inf_minus_inf_reported_like_inf_div_inf() {
    let q = inf1() / inf1();
    let div = q.to_ndarray_sub::<Ix1>(&[0..1], &q.read_ctx());
    println!("default modes: inf / inf -> {div:?}");
    assert_eq!(div.as_ref().unwrap_err().kind(), ErrorKind::Arithmetic);

    let d = inf1() - inf1();
    let sub = d.to_ndarray_sub::<Ix1>(&[0..1], &d.read_ctx());
    println!("default modes: inf - inf -> {sub:?}");
    assert!(
        sub.as_ref().is_err_and(|e| e.kind() == ErrorKind::Arithmetic),
        "inf - inf gave NaN with no InvalidOperation (inf / inf fails): {sub:?}"
    );
}

#[test]
fn mul_zero_times_inf_reports_invalid() {
    let a = Array::compact_ndarray(&array![0.0f64]).unwrap();
    let m = a * inf1();
    let r = m.to_ndarray_sub::<Ix1>(&[0..1], &strict(m.read_ctx()));
    println!("all=Error: 0 * inf -> {r:?}");
    assert!(r.is_err(), "0 * inf gave NaN with no InvalidOperation: {r:?}");
}

#[test]
fn add_overflow_reports_overflow() {
    let a = Array::compact_ndarray(&array![1e308f64]).unwrap();
    let b = Array::compact_ndarray(&array![1e308f64]).unwrap();
    let s = a + b;
    let r = s.to_ndarray_sub::<Ix1>(&[0..1], &strict(s.read_ctx()));
    println!("all=Error: 1e308 + 1e308 -> {r:?}");
    assert!(r.is_err(), "1e308 + 1e308 gave inf with no Overflow: {r:?}");
}

#[test]
fn square_overflow_reports_overflow() {
    let s = Array::compact_ndarray(&array![1e200f64]).unwrap().square();
    let r = s.to_ndarray_sub::<Ix1>(&[0..1], &strict(s.read_ctx()));
    println!("all=Error: square(1e200) -> {r:?}");
    assert!(r.is_err(), "square(1e200) gave inf with no Overflow: {r:?}");
}

#[test]
fn f16_add_overflow_reports_overflow() {
    use half::f16;
    let a = Array::compact_ndarray(&array![f16::from_f32(60000.0)]).unwrap();
    let b = Array::compact_ndarray(&array![f16::from_f32(60000.0)]).unwrap();
    let s = a + b;
    let r = s.to_ndarray_sub::<Ix1>(&[0..1], &strict(s.read_ctx()));
    println!("all=Error: f16 60000 + 60000 -> {r:?}");
    assert!(r.is_err(), "f16 60000 + 60000 gave inf with no Overflow: {r:?}");
}

#[test]
fn sum_overflow_reports_overflow() {
    let t = Array::compact_ndarray(&array![1e308f64, 1e308]).unwrap().sum(0usize);
    let r = t.to_ndarray_sub::<Ix0>(&[], &strict(t.read_ctx()));
    println!("all=Error: sum([1e308, 1e308]) -> {r:?}");
    assert!(r.is_err(), "sum([1e308, 1e308]) gave inf with no Overflow: {r:?}");
}

#[test]
fn sum_inf_minus_inf_reports_invalid_default_modes() {
    let t = Array::compact_ndarray(&array![f64::INFINITY, f64::NEG_INFINITY]).unwrap().sum(0usize);
    let r = t.to_ndarray_sub::<Ix0>(&[], &t.read_ctx());
    println!("default modes: sum([inf, -inf]) -> {r:?}");
    assert!(r.is_err(), "sum([inf, -inf]) gave NaN with no InvalidOperation: {r:?}");
}

#[test]
fn product_overflow_reports_overflow() {
    let t = Array::compact_ndarray(&array![1e200f64, 1e200]).unwrap().product(0usize);
    let r = t.to_ndarray_sub::<Ix0>(&[], &strict(t.read_ctx()));
    println!("all=Error: product([1e200, 1e200]) -> {r:?}");
    assert!(r.is_err(), "product([1e200, 1e200]) gave inf with no Overflow: {r:?}");
}
