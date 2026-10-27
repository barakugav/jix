// F022: f16 tan(+-177.5) rounds the f32 result (-66347.4) to f16 -inf, but Tan's error set only
// has InvalidOperation (infinite input), so the overflow is never reported - not even with the
// Overflow mode set to Error. f16 Exp reports the same kind of rounding overflow.
// These tests assert the CORRECT behavior, so they fail while the bug exists.
use jix::Array;
use jix::error::{ArithmeticErrorKind, ArithmeticErrorMode, ArithmeticOpStatus};
use jix::scalar::{Exp, Tan, f16};
use ndarray::{Ix1, array};

#[test]
fn f16_tan_overflow_array_read() {
    let a = Array::compact_ndarray(&array![f16::from_f32(177.5), f16::from_f32(-177.5)])
        .unwrap()
        .tan();
    let mut ctx = a.read_ctx();
    ctx.set_arithmetic_error_mode(ArithmeticErrorKind::Overflow, ArithmeticErrorMode::Error);
    let r = a.to_ndarray_sub::<Ix1>(&[0..2], &ctx);
    println!("tan(177.5f32) = {}, f16::MAX = {}", 177.5f32.tan(), f16::MAX);
    println!("f16 tan([177.5, -177.5]) with Overflow=Error: {r:?}");

    // Control: f16 exp(12) also overflows only by rounding to f16, and is reported.
    let e = Array::compact_ndarray(&array![f16::from_f32(12.0)]).unwrap().exp();
    let mut ectx = e.read_ctx();
    ectx.set_arithmetic_error_mode(ArithmeticErrorKind::Overflow, ArithmeticErrorMode::Error);
    let re = e.to_ndarray_sub::<Ix1>(&[0..1], &ectx);
    println!("f16 exp([12.0]) with Overflow=Error: {re:?}");
    assert!(re.is_err());

    assert!(r.is_err(), "f16 tan overflowed to inf without an Overflow report: {r:?}");
}

#[test]
fn f16_tan_scalar_status_exhaustive() {
    // Every finite f16 whose tan is infinite must report Overflow (as Exp does).
    let mut silent = Vec::new();
    for bits in 0..=u16::MAX {
        let x = f16::from_bits(bits);
        if !x.is_finite() {
            continue;
        }
        let mut st = ArithmeticOpStatus::default();
        let r = Tan::tan(x, &mut st);
        if r.is_infinite() && !st.contains(ArithmeticErrorKind::Overflow) {
            silent.push((x, r));
        }
    }
    let mut st = ArithmeticOpStatus::default();
    let e = Exp::exp(f16::from_f32(12.0), &mut st);
    println!("Exp::exp(f16 12.0) = {e}, Overflow reported: {}", st.contains(ArithmeticErrorKind::Overflow));
    println!("finite f16 inputs whose tan is inf with no Overflow status: {silent:?}");
    assert!(silent.is_empty(), "silent f16 tan overflows: {silent:?}");
}
