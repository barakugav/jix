//! F030 repro (verifier): value-strategy gaps in `jix/src/util/test_util.rs`.
//!
//! `ScalarStrategy` is `pub(crate)`, so this test rebuilds the two strategy expressions verbatim
//! from the public crates they are made of:
//! - `f32::any_strategy()` / `f64::any_strategy()` = `any::<f32>()` / `any::<f64>()`
//!   (test_util.rs:276, :307), documented as "full domain" (test_util.rs:42);
//! - `f16::any_strategy()` = `any::<f32>().prop_map(f16::from_f32)` (test_util.rs:245).
//!
//! Each test asserts what the doc promises (full domain incl. non-finite values; an f16 domain
//! that is mostly finite non-zero values), so it FAILS while the gap exists.
//! Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//!        --test repro_2026_10_29_batch6__f030 -- --nocapture
use jix::scalar::f16;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

const N: usize = 100_000;

fn draw<S: Strategy>(s: &S, runner: &mut TestRunner) -> S::Value {
    s.new_tree(runner).unwrap().current()
}

#[test]
fn f30_f32_f64_any_strategy_full_domain_includes_non_finite() {
    let mut runner = TestRunner::deterministic();
    let s32 = any::<f32>();
    let s64 = any::<f64>();
    let (mut nan32, mut inf32, mut nan64, mut inf64) = (0, 0, 0, 0);
    for _ in 0..N {
        let x = draw(&s32, &mut runner);
        nan32 += x.is_nan() as usize;
        inf32 += x.is_infinite() as usize;
        let y = draw(&s64, &mut runner);
        nan64 += y.is_nan() as usize;
        inf64 += y.is_infinite() as usize;
    }
    eprintln!("any::<f32>(): NaN {nan32}, inf {inf32} of {N}");
    eprintln!("any::<f64>(): NaN {nan64}, inf {inf64} of {N}");
    assert!(
        nan32 > 0 && inf32 > 0 && nan64 > 0 && inf64 > 0,
        "any_strategy is documented as 'full domain' but never yields NaN or +-inf"
    );
}

#[test]
fn f30_f16_any_strategy_mostly_finite_non_zero() {
    let mut runner = TestRunner::deterministic();
    let s = any::<f32>().prop_map(f16::from_f32);
    let (mut inf, mut zero, mut sub, mut finite) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..N {
        let h = draw(&s, &mut runner);
        if h.is_infinite() {
            inf += 1;
        } else if h.to_f32() == 0.0 {
            zero += 1;
        } else {
            finite += 1;
            sub += h.is_normal().then_some(0).unwrap_or(1);
        }
    }
    let pct = |k: usize| 100.0 * k as f64 / N as f64;
    eprintln!(
        "f16 any_strategy: +-inf {:.1}%  +-0 {:.1}%  finite non-zero {:.1}% (of which subnormal {:.1}%)",
        pct(inf),
        pct(zero),
        pct(finite),
        pct(sub)
    );
    assert!(
        finite * 2 >= N,
        "f16 any_strategy: only {:.1}% of draws are finite non-zero",
        pct(finite)
    );
}

/// The f16 `max` reduction proptest (`test_reduction!(max, .., [f16], any_strategy)`,
/// reduction.rs:3790) feeds these values in. Over a 1-D reduce of length 1..=100 (the most
/// heavily weighted branch of `reduction_shape_strategy`, reduction.rs:3571), how often is the
/// expected result a finite, non-zero value (the only case that exercises finite f16 ordering)?
#[test]
fn f30_f16_max_reduction_inputs_mostly_reduce_to_finite() {
    let mut runner = TestRunner::deterministic();
    let s = (1usize..=100).prop_flat_map(|n| {
        proptest::collection::vec(any::<f32>().prop_map(f16::from_f32), n)
    });
    let trials = 20_000;
    let (mut pos_inf, mut zero, mut finite) = (0usize, 0usize, 0usize);
    for _ in 0..trials {
        let v = draw(&s, &mut runner);
        let m = v.iter().copied().reduce(|m, x| if x > m { x } else { m }).unwrap();
        if m.is_infinite() {
            pos_inf += (m > f16::ZERO) as usize;
        } else if m.to_f32() == 0.0 {
            zero += 1;
        } else {
            finite += 1;
        }
    }
    let pct = |k: usize| 100.0 * k as f64 / trials as f64;
    eprintln!(
        "f16 max over 1..=100 elements: +inf {:.1}%  +-0 {:.1}%  finite non-zero {:.1}%",
        pct(pos_inf),
        pct(zero),
        pct(finite)
    );
    assert!(
        finite * 2 >= trials,
        "f16 max reduction proptest: only {:.1}% of 1-D cases have a finite non-zero max",
        pct(finite)
    );
}
