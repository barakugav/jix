//! F004 (run 2026-10-29-batch6): jix-py `Array.__len__` (jix-py/src/array.rs:417) returns
//! `Ok(*len as usize)` for the u64 length of dim 0. On a 32-bit target (the shipped x86 / armv7 /
//! win32 wheels) a dim >= 2^32 is reduced mod 2^32, and pyo3 only raises OverflowError for a
//! usize above isize::MAX, so `len()` silently returns a wrong small number.
//!
//! jix-py cannot be built or run for 32-bit here, so this test builds the exact core array that
//! `jix.broadcast(jix.compact(np.zeros(1, np.uint8)), (2**32 + 5,))` builds
//! (`broadcast_array_impl` -> `Broadcast::new_array` on an `ArrayAny`, jix-py/src/ops/shape_ops.rs)
//! and then applies the `__len__` expression to its shape.
//!
//! Asserts the CORRECT behavior: the value `__len__` hands to pyo3 must equal `shape[0]` (or
//! `__len__` must fail). Passes on 64-bit; fails on i686:
//!   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//!       --test repro_2026_10_29_batch6__f004 -- --nocapture
use jix::Array;
use jix::ops::Broadcast;
use ndarray::array;

/// Verbatim model of jix-py `Array::__len__` (jix-py/src/array.rs:410-418), with the pyo3
/// `usize -> Py_ssize_t` slot conversion (pyo3 0.30.1 src/impl_/callback.rs:126-130) applied.
/// `Err(())` stands for the OverflowError pyo3 raises.
fn py_len(shape: &[u64]) -> Result<isize, ()> {
    let len = shape.first().ok_or(())?;
    let ret: usize = *len as usize; // jix-py/src/array.rs:417
    isize::try_from(ret).map_err(|_| ()) // pyo3: self.try_into().map_err(|_| PyOverflowError)
}

#[test]
fn len_matches_shape0_for_dim_ge_2_pow_32() {
    let n: u64 = (1u64 << 32) + 5;
    // jix.compact(np.zeros(1, np.uint8)) -> ArrayAny
    let a = Array::compact_ndarray(&array![0u8]).unwrap().into_any();
    // jix.broadcast(a, (2**32 + 5,)) -> broadcast_array_impl -> Broadcast::new_array
    let b = Broadcast::new_array(a, &[n]).expect("core accepts the lazy broadcast").into_any();
    let shape0 = b.shape()[0];
    let size: u64 = b.shape().iter().product();
    let len = py_len(b.shape());
    println!(
        "pointer width {}: shape[0] = {shape0}, size = {size}, len() -> {len:?}",
        usize::BITS
    );
    assert_eq!(shape0, n);
    match len {
        // OverflowError is acceptable (CPython does the same for a length > sys.maxsize).
        Err(()) => {}
        Ok(l) => assert_eq!(
            l as u64, shape0,
            "len() silently returns {l} while shape[0] and size are {shape0}"
        ),
    }
}
