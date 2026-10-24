// Repro for bug-hunt run 2026-10-24-batch4, finding F006 (incomplete K28 fix).
//
// Since c009869 (#87), `Cast<f16> for f64` is `half::f16::from_f64` (jix/src/ops/cast.rs:55).
// jix depends on `half` with `default-features = false` (jix/Cargo.toml:28) and nothing in the
// jix / jix-py dependency graph turns half's `std` feature back on, so half's `convert_fn!`
// (half-2.7.1/src/binary16/arch.rs:14-81) cannot detect F16C / FP16 at run time. Unless the
// target enables `f16c` / `fp16` at compile time (only aarch64-apple-darwin does among the wheel
// targets), `from_f64` runs `f64_to_f16_fallback` (arch.rs:617-679), which drops the low 32
// mantissa bits before rounding: an f64 within 2^-32 (relative) of an f16 halfway point is
// rounded as an exact tie. (With f16c it would do `f as f32` first - the old K28 double rounding.)
//
// The tests assert the CORRECT (single, ties-to-even) rounding, so they fail while the bug exists:
//   cargo test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_24_batch4__f006 -- --nocapture
use jix::scalar::{f16, Cast, Complex};
use jix::Array;
use ndarray::{array, Array1, Ix1};

/// Independent correctly rounded f64 -> f16 (round to nearest, ties to even), returned as f64.
/// Scales x by the f16 quantum of its binade (exact: a power-of-two scaling), rounds to an
/// integer with ties-to-even, scales back. Covers normals, subnormals and overflow to inf.
fn ref_f64_to_f16(x: f64) -> f64 {
    assert!(x.is_finite());
    if x == 0.0 {
        return x;
    }
    let e = x.abs().log2().floor() as i32; // only used to pick the binade; fixed up below
    let e = if 2f64.powi(e) > x.abs() {
        e - 1
    } else if 2f64.powi(e + 1) <= x.abs() {
        e + 1
    } else {
        e
    };
    let quantum_exp = e.max(-14) - 10; // subnormal quantum is 2^-24
    let q = x * 2f64.powi(-quantum_exp); // exact
    let r = q.round_ties_even() * 2f64.powi(quantum_exp);
    if r.abs() > 65504.0 { f64::INFINITY.copysign(x) } else { r }
}

const X: f64 = 1.0 + 1.0 / 2048.0 + 1.0 / 1099511627776.0; // 1 + 2^-11 + 2^-40
const X_WANT: f64 = 1.0 + 1.0 / 1024.0; // 1 + 2^-10

#[test]
fn f64_to_f16_just_above_halfway_normal() {
    assert_eq!(ref_f64_to_f16(X), X_WANT);
    let got: f16 = <f64 as Cast<f16>>::cast(X);
    assert_eq!(got.to_f64(), X_WANT, "<f64 as Cast<f16>>::cast({X:e})");
}

#[test]
fn f64_to_f16_just_above_halfway_subnormal() {
    // 2^-25 is exactly half the smallest f16 subnormal 2^-24; anything above it rounds up.
    let x = 2f64.powi(-25) + 2f64.powi(-60);
    assert_eq!(ref_f64_to_f16(x), 2f64.powi(-24));
    let got: f16 = <f64 as Cast<f16>>::cast(x);
    assert_eq!(got.to_f64(), 2f64.powi(-24), "<f64 as Cast<f16>>::cast({x:e})");
}

#[test]
fn array_cast_f64_to_f16() {
    let a = Array::compact_ndarray(&array![X, -X, 0.5, 2f64.powi(-25) + 2f64.powi(-60)]).unwrap();
    let got = a.cast::<f16>().to_ndarray::<Ix1>().unwrap();
    let got: Vec<f64> = got.iter().map(|v| v.to_f64()).collect();
    assert_eq!(got, vec![X_WANT, -X_WANT, 0.5, 2f64.powi(-24)]);
}

#[test]
fn complex_cast_f64_to_f16() {
    let z = Complex { re: X, im: -X };
    let got: Complex<f16> = <Complex<f64> as Cast<Complex<f16>>>::cast(z);
    assert_eq!((got.re.to_f64(), got.im.to_f64()), (X_WANT, -X_WANT));
}

#[test]
fn array_cast_f64_to_f16_random_sweep() {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let n = 1_000_000usize;
    let v: Array1<f64> = (0..n)
        .map(|_| ((next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 400.0)
        .collect();
    let got = Array::compact_ndarray(&v).unwrap().cast::<f16>().to_ndarray::<Ix1>().unwrap();
    let (mut bad, mut old_bad) = (Vec::new(), 0usize);
    for (x, g) in v.iter().zip(got.iter()) {
        let want = ref_f64_to_f16(*x);
        if f16::from_f32(*x as f32).to_f64() != want {
            old_bad += 1; // pre-#87 formula, for comparison only
        }
        if g.to_f64() != want {
            bad.push((*x, g.to_f64(), want));
        }
    }
    println!(
        "jix f64->f16 mismatches: {} of {n}; first (x, jix, correct): {:?}",
        bad.len(),
        &bad[..bad.len().min(3)]
    );
    println!("(info) pre-#87 formula f16::from_f32(x as f32) mismatches: {old_bad} of {n}");
    assert!(bad.is_empty(), "{} of {n} f64 values cast to the wrong f16", bad.len());
}
