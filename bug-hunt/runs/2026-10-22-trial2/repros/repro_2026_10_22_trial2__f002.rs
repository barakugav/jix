//! F002: a bool `Plain` view over bytes that are not 0/1 is read as Rust `bool` by the reduction
//! fold loops (UB). `Plain::new`'s documented `# Safety` section (jix/src/storage/plain.rs:108-115)
//! only asks for a valid, readable pointer - it says nothing about element validity - and this is
//! exactly what jix-py does for every numpy bool array (jix-py/src/ops/common/operand.rs:184), so
//! `np.uint8([2, 1, 2, 1]).view(bool)` reaches the same state from Python.
//!
//! Asserts the CORRECT behavior (nonzero byte = true, as NumPy reads it); fails while the bug
//! exists. Under Miri (`python jix/run_miri.py --all-features --test repro_2026_10_22_trial2__f002`)
//! it reports the invalid-bool UB.

use jix::Array;
use jix::dtype::{Dtype, ScalarKind};
use jix::storage::Plain;
use jix::ArrayParams;

#[test]
fn all_of_non_01_bool_bytes() {
    // Every byte is nonzero, i.e. every element is truthy.
    let bytes: [u8; 4] = [2, 1, 2, 1];
    // SAFETY (as documented on `Plain::new`): `bytes` outlives the array and all 4 bytes are
    // readable at shape [4], stride 1.
    let storage = unsafe {
        Plain::new(
            &bytes,
            bytes.as_ptr(),
            [4u64],
            &[1usize],
            Dtype::new_scalar(ScalarKind::Bool),
            ArrayParams::default(),
        )
    }
    .unwrap();
    let arr = Array::from_storage(storage).into_typed::<bool>().unwrap();
    let all = arr.view().all(0).to_ndarray::<ndarray::Ix0>().unwrap();
    let sum = arr.view().sum(0).to_ndarray::<ndarray::Ix0>().unwrap();
    println!("all = {}, sum = {}", all[()], sum[()]);
    assert!(all[()], "all() of an all-truthy array returned false");
    assert_eq!(sum[()], 4u64, "sum() counted only the 0x01 bytes");
}
