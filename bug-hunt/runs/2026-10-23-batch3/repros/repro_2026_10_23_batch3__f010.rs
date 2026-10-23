// Bug hunt 2026-10-23-batch3, finding F010 (verifier repro).
//
// `Pow` for a float base and an integer exponent (op2.rs:1199-1203) computes `a.powi(b as i32)`.
// `powi` is square-and-multiply in the float type itself (compiler-builtins `__powisf2` /
// `__powidf2`) and computes a negative power as `1 / a^|n|`, so:
//   - representable subnormal results of a negative exponent come back as exactly 0 (a^|n|
//     overflows to inf first), with no status bit;
//   - the error grows with |n| (tens to thousands of ulps), while the same values with a float
//     exponent (powf) are within an ulp.
//
// Asserts the CORRECT behavior (results agree with powf within 2 ulps), so it FAILS while the bug
// exists.
use jix::Array;
use ndarray::{array, Ix1};

fn ulps64(a: f64, b: f64) -> u64 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}
fn ulps32(a: f32, b: f32) -> u64 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

#[test]
fn f64_pow_int_exponent_matches_float_exponent() {
    let bases = array![2.0f64, 10.0, 1.0123456789, 0.999];
    let int_exps = array![-1074i32, -309, 100, 10000];
    let float_exps = int_exps.mapv(f64::from);

    let got_int = Array::compact_ndarray(&bases)
        .unwrap()
        .pow(Array::compact_ndarray(&int_exps).unwrap())
        .to_ndarray::<Ix1>()
        .unwrap();
    // The same values through the (f64, f64) impl, i.e. powf.
    let got_float = Array::compact_ndarray(&bases)
        .unwrap()
        .pow(Array::compact_ndarray(&float_exps).unwrap())
        .to_ndarray::<Ix1>()
        .unwrap();
    println!("f64 ** i32: {got_int:?}");
    println!("f64 ** f64: {got_float:?}");
    let mut bad = Vec::new();
    for i in 0..bases.len() {
        let u = ulps64(got_int[i], got_float[i]);
        println!(
            "  {} ** {}: int-exp {:e}, float-exp {:e}, {u} ulps",
            bases[i], int_exps[i], got_int[i], got_float[i]
        );
        // A flush to zero is only 1 ulp away from 5e-324 but is still wrong.
        if u > 2 || (got_int[i] == 0.0) != (got_float[i] == 0.0) {
            bad.push(i);
        }
    }
    assert!(
        bad.is_empty(),
        "f64 ** int disagrees with f64 ** float at indices {bad:?}"
    );
}

#[test]
fn f32_pow_int_exponent_matches_float_exponent() {
    let bases = array![2.0f32, 1.0135511, 1.0 + f32::EPSILON];
    let int_exps = array![-130i32, 100, 1 << 24];

    let got = Array::compact_ndarray(&bases)
        .unwrap()
        .pow(Array::compact_ndarray(&int_exps).unwrap())
        .to_ndarray::<Ix1>()
        .unwrap();
    // Reference: the exact power in f64 (powf, < 1 ulp of f64) rounded once to f32.
    let want = ndarray::Zip::from(&bases)
        .and(&int_exps)
        .map_collect(|&a, &b| f64::from(a).powf(f64::from(b)) as f32);
    println!("f32 ** i32: {got:?}");
    println!("reference : {want:?}");
    let mut bad = Vec::new();
    for i in 0..bases.len() {
        let u = ulps32(got[i], want[i]);
        println!(
            "  {:e} ** {}: got {:e}, want {:e}, {u} ulps",
            bases[i], int_exps[i], got[i], want[i]
        );
        if u > 2 || (got[i] == 0.0) != (want[i] == 0.0) {
            bad.push(i);
        }
    }
    assert!(
        bad.is_empty(),
        "f32 ** int is not within 2 ulps at indices {bad:?}"
    );
}
