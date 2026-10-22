// Repro for F006 (run 2026-10-22-trial2): MapMultiple::new does not run check_shape_overflow with
// the OUTPUT itemsize (K2's fix covered Op1/Op2/ReductionOp only). A widening closure (u8 -> f64)
// over a broadcast whose shape*1 fits u64 but shape*8 does not is accepted, and a pull read then
// sizes its buffer with an unchecked product(shape) * itemsize (storage/buf.rs:312).
//
// Both tests assert the CORRECT behavior, so they fail while the bug exists:
// - debug build: the first fails its assertion, the second panics with
//   "attempt to multiply with overflow" at src/storage/buf.rs:312.
// - release build: the second test writes past an 8-byte heap allocation (heap overflow) and may
//   crash the process (SIGSEGV) instead of failing an assertion.
use jix::ops::MapMultiple;
use jix::Array;
use jix::error::ErrorKind;
use ndarray::array;

const N: u64 = (1 << 61) + 1; // N * 1 fits u64 (broadcast accepts it); N * 8 does not

#[test]
fn map_multiple_widening_rejected_at_construction() {
    let a = Array::compact_ndarray(&array![7u8]).unwrap().broadcast(&[N]);
    let r = MapMultiple::new(vec![a], |xs: &[u8]| xs[0] as f64);
    match r {
        Ok(_) => panic!("MapMultiple accepted shape [{N}] with f64 output: byte size overflows u64"),
        Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidShapeOperation, "{e}"),
    }
}

#[test]
fn map_multiple_widening_pull_read_errors() {
    let a = Array::compact_ndarray(&array![7u8]).unwrap().broadcast(&[N]);
    // Correct behavior: either construction fails, or the read returns Err. Never a panic,
    // never an Ok over a wrapped-size buffer.
    let storage = match MapMultiple::new(vec![a], |xs: &[u8]| xs[0] as f64) {
        Err(_) => return,
        Ok(s) => s,
    };
    let m = Array::from_storage(storage);
    println!("shape {:?} itemsize {}", m.shape(), m.dtype().itemsize());
    let ctx = m.read_ctx();
    let r = m.to_ndarray_buf(&[0..N], &ctx, None);
    assert!(r.is_err(), "pull read of a {N} x 8-byte array returned Ok");
}

// Same, with the fixed-length-array sequence form (`[Array<S>; 1]`).
#[test]
fn map_multiple_fixed_array_widening_rejected_at_construction() {
    let a = Array::compact_ndarray(&array![7u8]).unwrap().broadcast(&[N]);
    let r = MapMultiple::new([a], |xs: [u8; 1]| xs[0] as u64);
    assert!(
        r.is_err(),
        "MapMultiple accepted shape [{N}] with u64 output: byte size overflows u64"
    );
}

// Control: Map (via Op1::new) already rejects the identical trigger.
#[test]
fn control_map_rejects_same_trigger() {
    let a = Array::compact_ndarray(&array![7u8]).unwrap().broadcast(&[N]);
    let r = jix::ops::Map::new_array(a, |x: u8| x as f64);
    let e = r.err().expect("Map should reject");
    assert_eq!(e.kind(), ErrorKind::InvalidShapeOperation, "{e}");
}
