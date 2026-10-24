// F016: to_ndarray_sub sizes its ndarray with a per-dim `as usize` and only bounds the region
// size on 32-bit, via the (zero) element product. Empty regions with huge non-zero extents get a
// truncated shape on 32-bit and panic inside ndarray on 64-bit.
//
// Each test asserts the CORRECT behavior: either Err, or Ok with the true shape. They fail while
// the bug exists.
//
// x86_64: cargo test --manifest-path jix/Cargo.toml --all-features \
//           --test repro_2026_10_24_batch4__f016 -- --nocapture
// i686:   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//           --test repro_2026_10_24_batch4__f016 -- --nocapture
use jix::Array;
use ndarray::{array, IxDyn};

fn check<T: std::fmt::Debug>(
    what: &str,
    got: Result<ndarray::Array<T, IxDyn>, jix::Error>,
    true_shape: &[u64],
) {
    match got {
        Ok(nd) => {
            let shape: Vec<u64> = nd.shape().iter().map(|&s| s as u64).collect();
            println!("{what}: Ok shape {shape:?}");
            assert_eq!(shape, true_shape, "{what}: Ok with the wrong shape");
        }
        Err(e) => println!("{what}: Err {e}"),
    }
}

// i686: Ok with shape [0, 5] instead of [0, 4294967301].
#[test]
fn empty_array_dim_over_u32_keeps_shape() {
    let big = (1u64 << 32) + 5;
    let a = Array::compact_ndarray(&array![[7u8]]).unwrap().broadcast(&[0, big]);
    assert_eq!(a.shape(), &[0, big]);
    check("to_ndarray [0, 2^32+5]", a.to_ndarray::<IxDyn>(), &[0, big]);
}

// x86_64: panics in ndarray::Array::uninit ("Shape too large, product of non-zero axis lengths
// overflows isize"). i686: Ok with shape [0, 0, 3].
#[test]
fn empty_array_nonzero_product_over_isize_max() {
    let shape = [0, 1u64 << 62, 3];
    let a = Array::compact_ndarray(&array![[[7u8]]]).unwrap().broadcast(&shape);
    assert_eq!(a.shape(), &shape); // a valid jix shape: check_shape_overflow accepts it
    check("to_ndarray [0, 2^62, 3]", a.to_ndarray::<IxDyn>(), &shape);
}

// Not limited to empty arrays: an empty sub-region of a NON-empty array [1, 2^62, 3].
#[test]
fn empty_subregion_of_nonempty_array() {
    let shape = [1, 1u64 << 62, 3];
    let a = Array::compact_ndarray(&array![[[7u8]]]).unwrap().broadcast(&shape);
    let index = [0..0, 0..(1u64 << 62), 0..3];
    check(
        "to_ndarray_sub [0..0, 0..2^62, 0..3]",
        a.to_ndarray_sub::<IxDyn>(&index, &a.read_ctx()),
        &[0, 1 << 62, 3],
    );
}

// Secondary (64-bit only): a non-empty region over isize::MAX bytes. 32-bit returns
// Err(InvalidShapeOperation) from check_region_size; 64-bit has no check and panics with
// "capacity overflow" in the ndarray allocation.
#[test]
fn nonempty_region_over_isize_max_bytes() {
    let a = Array::compact_ndarray(&array![7i32]).unwrap().broadcast(&[1u64 << 61]);
    let got = a.to_ndarray::<IxDyn>();
    match &got {
        Ok(nd) => println!("to_ndarray i32 [2^61]: Ok shape {:?}", nd.shape()),
        Err(e) => println!("to_ndarray i32 [2^61]: Err {e}"),
    }
    assert!(got.is_err(), "a 2^63-byte region cannot be materialized");
}

// The truncated shape propagates: to_plain (built on to_ndarray::<IxDyn>) returns an array whose
// shape differs from its source. i686: shape [0, 5].
#[test]
fn to_plain_keeps_shape() {
    let big = (1u64 << 32) + 5;
    let a = Array::compact_ndarray(&array![[7u8]]).unwrap().broadcast(&[0, big]);
    match a.to_plain() {
        Ok(p) => {
            println!("to_plain [0, 2^32+5]: Ok shape {:?}", p.shape());
            assert_eq!(p.shape(), &[0, big], "to_plain changed the shape");
        }
        Err(e) => println!("to_plain [0, 2^32+5]: Err {e}"),
    }
}
