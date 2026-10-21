// Verifier repro for F013 (run 2026-10-21-pilot): the C-order stride helpers in jix/src/util/mod.rs
// multiply unchecked. A shape containing a 0 passes every nitems / check_shape_overflow check
// (product == 0), but the product of its NON-zero trailing dims can overflow u64.
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//        --test repro_2026_10_21_pilot__f013 -- --nocapture --test-threads 1
use jix::{Array, ArrayParams};
use ndarray::{Array1, Array3};
use std::panic::{catch_unwind, AssertUnwindSafe};

const BIG: u64 = 1 << 40;

fn report<R: std::fmt::Debug>(name: &str, f: impl FnOnce() -> R) -> bool {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => {
            println!("[{name}] OK: {r:?}");
            true
        }
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            println!("[{name}] PANIC: {msg}");
            false
        }
    }
}

// The hunter's trigger: an empty 1-d array reshaped to [0, 2^40, 2^40].
// nitems 0 == 0 passes, then default_logical_strides(&new_shape) multiplies 2^40 * 2^40.
#[test]
fn reshape_empty_to_huge_trailing_dims() {
    let ok = report("reshape [0] -> [0, 2^40, 2^40]", || {
        let a = Array::compact_ndarray(&Array1::<f32>::zeros(0)).unwrap();
        let r = a.reshape([0u64, BIG, BIG]);
        r.shape().to_vec()
    });
    assert!(ok, "reshape of an empty array panicked instead of Ok / Err");
}

// Control: the same op with a trailing product that fits in u64 is fine.
#[test]
fn reshape_empty_control_fits() {
    let ok = report("reshape [0] -> [0, 2^31, 2^31] (control)", || {
        let a = Array::compact_ndarray(&Array1::<f32>::zeros(0)).unwrap();
        let r = a.reshape([0u64, 1 << 31, 1 << 31]);
        let v = r.to_ndarray::<ndarray::Ix3>();
        (r.shape().to_vec(), v.map(|v| v.shape().to_vec()))
    });
    assert!(ok);
}

// Same shape, built without reshape: broadcast keeps the zero dim, check_shape_overflow accepts it.
// Then materialize / read it.
#[test]
fn broadcast_empty_to_huge_then_compact() {
    let ok = report("broadcast [0,1,1] -> [0,2^40,2^40] .compact()", || {
        let a = Array::compact_ndarray(&Array3::<f32>::zeros((0, 1, 1))).unwrap();
        let b = a.broadcast(&[0, BIG, BIG]);
        let c = b.compact();
        c.map(|c| c.shape().to_vec())
    });
    assert!(ok, "compact of an empty broadcast panicked");
}

#[test]
fn broadcast_empty_to_huge_then_to_ndarray() {
    let ok = report("broadcast [0,1,1] -> [0,2^40,2^40] .to_ndarray()", || {
        let a = Array::compact_ndarray(&Array3::<f32>::zeros((0, 1, 1))).unwrap();
        let b = a.broadcast(&[0, BIG, BIG]);
        b.to_ndarray::<ndarray::Ix3>().map(|v| v.shape().to_vec())
    });
    assert!(ok, "to_ndarray of an empty broadcast panicked");
}

// compact_fn with the same degenerate shape.
#[test]
fn compact_fn_empty_huge() {
    let ok = report("compact_fn [0,2^40,2^40]", || {
        let r = Array::compact_fn_with([0u64, BIG, BIG], ArrayParams::default(), |_| 0f32);
        r.map(|a| a.shape().to_vec())
    });
    assert!(ok, "compact_fn of an empty huge shape panicked");
}

// The validity check is order dependent: try_product short-circuits through a leading 0 but
// overflows on the same dims when the 0 comes last. So [2^40, 2^40, 0] is a clean Err while
// [0, 2^40, 2^40] is accepted (and then panics in debug, above).
#[test]
fn reshape_check_is_order_dependent() {
    let a = Array::compact_ndarray(&Array1::<f32>::zeros(0)).unwrap();
    let r = jix::ops::Reshape::new_array(a, [BIG, BIG, 0u64]);
    println!("[reshape [0] -> [2^40, 2^40, 0]] {:?}", r.as_ref().map(|r| r.shape().to_vec()));
    assert!(r.is_err());
}

// Related (same root: a 0 dim hides an overflowing product): reducing away the 0 dim yields a
// NON-empty array whose shape product does not fit u64 at all.
#[test]
fn related_sum_over_zero_dim_gives_overflowing_shape() {
    let ok = report("broadcast [0,2^40,2^40] .sum(0) .compact()", || {
        let a = Array::compact_ndarray(&Array3::<f32>::zeros((0, 1, 1))).unwrap();
        let s = a.broadcast(&[0, BIG, BIG]).sum(0);
        println!("  sum shape {:?}", s.shape());
        s.compact().map(|c| c.shape().to_vec())
    });
    assert!(ok);
}
