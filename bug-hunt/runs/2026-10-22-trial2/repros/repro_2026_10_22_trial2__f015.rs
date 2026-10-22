// F015 repro (bug-hunt run 2026-10-22-trial2).
//
// `Cast<f16>` for every source type is `f16::from_f32(value as f32)` (jix/src/ops/cast.rs:59).
// For an f64 source that rounds twice (f64 -> f32 -> f16). When the f32 rounding lands exactly on
// an f16 halfway point, ties-to-even then picks the wrong f16 neighbour, 1 ulp away from the
// correctly rounded result (numpy `astype(float16)`, `ref_f64_to_f16` below).
//
// NOTE: `half::f16::from_f64` (half 2.7.1) is NOT a correct oracle and NOT a fix: on x86_64 with
// F16C it also does `f as f32` + vcvtps2ph (same double rounding), and its software fallback
// drops the low 32 mantissa bits from the sticky bit. `ref_f64_to_f16` rounds from all 53 bits.
//
// x = 1 + 2^-11 + 2^-40 lies strictly above the midpoint (1 + 2^-11) of the f16 neighbours 1.0 and
// 1 + 2^-10, so the correctly rounded f16 is 1 + 2^-10 = 1.0009765625. `x as f32` rounds to exactly
// 1 + 2^-11 (the tie), and f32 -> f16 ties-to-even gives 1.0.
//
// The tests assert the CORRECT behavior, so they fail while the bug exists:
//   cargo test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_22_trial2__f015 -- --nocapture
use jix::scalar::{f16, Cast, Complex};
use jix::Array;
use ndarray::{array, Array1, Ix1};

const X: f64 = 1.0 + 1.0 / 2048.0 + 1.0 / 1099511627776.0; // 1 + 2^-11 + 2^-40
const EXPECTED: f64 = 1.0 + 1.0 / 1024.0; // 1 + 2^-10

/// Correctly rounded (ties-to-even) f64 -> f16 for finite `x` whose result is a normal f16
/// or +-inf (|x| >= 2^-14). Rounds the full 53-bit significand to 11 bits in one step.
fn ref_f64_to_f16(x: f64) -> f16 {
    assert!(x.is_finite() && x.abs() >= 2f64.powi(-14));
    let bits = x.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let mut exp = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52); // 53 bits
    let mut q = mant >> 42; // 11 bits
    let r = mant & ((1u64 << 42) - 1);
    let half = 1u64 << 41;
    if r > half || (r == half && q & 1 == 1) {
        q += 1;
    }
    if q == 1 << 11 {
        q >>= 1;
        exp += 1;
    }
    if exp + 15 >= 31 {
        return f16::from_bits(sign | 0x7c00);
    }
    f16::from_bits(sign | (((exp + 15) as u16) << 10) | (q as u16 & 0x3ff))
}

#[test]
fn scalar_cast_f64_to_f16_rounds_once() {
    // Sanity: the reference rounds directly.
    assert_eq!(ref_f64_to_f16(X).to_f64(), EXPECTED);
    let got: f16 = <f64 as Cast<f16>>::cast(X);
    assert_eq!(got.to_f64(), EXPECTED, "Cast<f16> for f64 of {X:e}");
}

#[test]
fn array_cast_f64_to_f16_rounds_once() {
    let a = Array::compact_ndarray(&array![X, -X, 0.5]).unwrap();
    let got = a.cast::<f16>().to_ndarray::<Ix1>().unwrap();
    let got: Vec<f64> = got.iter().map(|v| v.to_f64()).collect();
    assert_eq!(got, vec![EXPECTED, -EXPECTED, 0.5]);
}

#[test]
fn complex_cast_f64_to_f16_rounds_once() {
    let z = Complex { re: X, im: -X };
    let got: Complex<f16> = <Complex<f64> as Cast<Complex<f16>>>::cast(z);
    assert_eq!((got.re.to_f64(), got.im.to_f64()), (EXPECTED, -EXPECTED));
}

#[test]
fn array_cast_f64_to_f16_random_sweep() {
    // Deterministic xorshift, values roughly in +-[0, 200) with full 53-bit mantissas.
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let n = 1_000_000usize;
    let v: Array1<f64> = (0..n)
        .map(|_| {
            let u = (next() >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            (u - 0.5) * 400.0
        })
        .collect();
    let got = Array::compact_ndarray(&v).unwrap().cast::<f16>().to_ndarray::<Ix1>().unwrap();
    let mut bad = Vec::new();
    let (mut checked, mut half_bad) = (0usize, 0usize);
    for (x, g) in v.iter().zip(got.iter()) {
        if x.abs() < 2f64.powi(-14) {
            continue; // f16 subnormal range: not covered by ref_f64_to_f16
        }
        checked += 1;
        let want = ref_f64_to_f16(*x);
        if f16::from_f64(*x).to_bits() != want.to_bits() {
            half_bad += 1;
        }
        if g.to_bits() != want.to_bits() {
            bad.push((*x, g.to_f64(), want.to_f64()));
        }
    }
    println!(
        "jix mismatches: {} of {checked}; first (x, jix, correct): {:?}",
        bad.len(),
        &bad[..bad.len().min(3)]
    );
    println!("(info) half::f16::from_f64 mismatches: {half_bad} of {checked}");
    assert!(bad.is_empty(), "{} of {n} f64 values cast to the wrong f16", bad.len());
}
