//! F031 (util-test-infra): `assert_read_rejects_invalid_index` (jix/src/util/test_util.rs:1379)
//! only builds ndim+1 ranges, `0..len+1` and `1..0`, and only permute_dims, remove_dim, sub_dtype
//! and where_op call it. This repro runs the invalid-index classes the helper misses against the
//! storages it never sees, in pull and push mode, and asserts every read fails with InvalidIndex
//! (no panic, no success, no other error kind). It PASSES today: the gap hides no library bug,
//! it is a test-infrastructure gap only.
//!
//! Classes added beyond the helper's three: fewer ranges, empty index (ndim > 0), many more ranges
//! than NDIM_MAX (9 and 16), empty ranges past the end (`len+1..len+1`, `u64::MAX..u64::MAX`),
//! reversed ranges (`5..2`, `u64::MAX..0`), huge ends (`0..u64::MAX`, `len..u64::MAX`,
//! `u64::MAX-1..u64::MAX`), and a valid empty range in one dim combined with an invalid range in
//! another (an "empty region, nothing to read" fast path must not skip validation). Base shapes:
//! 3-d, 3-d with a zero-length dim, 0-d, 8-d (NDIM_MAX).
use std::ops::Range;

use jix::ops::SliceItem;
use jix::scalar::Complex;
use jix::storage::StridedBuf;
use jix::{Array, ArrayParams, ArrayStorage, ReadContext};
use ndarray::{Array3, ArrayD, IxDyn};

fn invalid(shape: &[u64]) -> Vec<Vec<Range<u64>>> {
    let ndim = shape.len();
    let full: Vec<Range<u64>> = shape.iter().map(|&s| 0..s).collect();
    let mut v: Vec<Vec<Range<u64>>> = vec![];
    v.push(full.iter().cloned().chain([0..1]).collect());
    v.push(
        full.iter()
            .cloned()
            .chain((0..9 - ndim.min(9)).map(|_| 0..1))
            .chain([0..1])
            .collect(),
    );
    v.push((0..16).map(|_| 0..1).collect());
    if ndim > 0 {
        v.push(full[..ndim - 1].to_vec());
        v.push(vec![]);
    }
    #[allow(clippy::reversed_empty_ranges)]
    for d in 0..ndim {
        for r in [
            0..shape[d] + 1,
            shape[d] + 1..shape[d] + 1,
            1..0,
            5..2,
            u64::MAX..u64::MAX,
            u64::MAX - 1..u64::MAX,
            0..u64::MAX,
            u64::MAX..0,
            shape[d]..u64::MAX,
        ] {
            let mut x = full.clone();
            x[d] = r.clone();
            v.push(x);
            // A valid empty range in another dim must not short-circuit validation.
            for e in 0..ndim {
                if e != d {
                    let mut y = full.clone();
                    y[d] = r.clone();
                    y[e] = 0..0;
                    v.push(y);
                }
            }
        }
    }
    v
}

fn check(name: &str, storage: &dyn ArrayStorage) -> Vec<String> {
    let shape = storage.shape().to_vec();
    let ndim = shape.len();
    let itemsize = storage.dtype().itemsize() as usize;
    let mut bad = vec![];
    for index in invalid(&shape) {
        let ctx = ReadContext::default();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            storage
                .read_data(&index, &ctx, None)
                .map(|_| ())
                .map_err(|e| e.kind())
        }));
        if !matches!(r, Ok(Err(jix::error::ErrorKind::InvalidIndex))) {
            bad.push(format!("{name}: pull {index:?} for {shape:?}: {r:?}"));
        }
        // Push into a packed buffer with room for 8 elements per dim (every base dim is <= 7).
        let oshape = vec![8usize; ndim];
        let mut dst = vec![0u8; 8usize.pow(ndim as u32) * itemsize];
        let strides: Vec<usize> = (0..ndim)
            .map(|d| itemsize * 8usize.pow((ndim - 1 - d) as u32))
            .collect();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out = unsafe {
                StridedBuf::from_raw_parts_mut(dst.as_mut_ptr(), &oshape, &strides, itemsize)
            };
            storage
                .read_data(&index, &ctx, Some(&mut out))
                .map(|_| ())
                .map_err(|e| e.kind())
        }));
        if !matches!(r, Ok(Err(jix::error::ErrorKind::InvalidIndex))) {
            bad.push(format!("{name}: push {index:?} for {shape:?}: {r:?}"));
        }
    }
    bad
}

#[test]
fn every_storage_rejects_every_invalid_index_class() {
    std::panic::set_hook(Box::new(|_| {}));
    let mut bad = vec![];
    macro_rules! t {
        ($name:expr, $arr:expr) => {{
            let a = $arr;
            bad.extend(check($name, a.storage()));
        }};
    }

    // --- 3-d base, plus storages the hunter's run did not include ---
    let nd = Array3::<f32>::from_shape_fn((5, 6, 7), |(i, j, k)| (i * 100 + j * 10 + k) as f32);
    let ndi = Array3::<i32>::from_shape_fn((5, 6, 7), |(i, j, k)| (i + j + k) as i32);
    let ndc = nd.mapv(|x| Complex::new(x, -x));
    let mut p = ArrayParams::default();
    p.block_shape(&[2, 3, 4]);
    let c = || Array::compact_ndarray_with(&nd, p.clone()).unwrap();
    let ci = || Array::compact_ndarray_with(&ndi, p.clone()).unwrap();
    let cc = || Array::compact_ndarray_with(&ndc, p.clone()).unwrap();
    let pl = || Array::plain_ndarray_view(nd.view()).unwrap();
    let ndt = nd.view().reversed_axes().to_owned();
    let plt = || Array::plain_ndarray_view(ndt.t()).unwrap(); // non-C strides
    let scalar = || {
        Array::from_storage(
            jix::__private::Scalar::new(1.5f32, [5u64, 6, 7], ArrayParams::default()).unwrap(),
        )
    };

    t!("compact", c());
    let cv = c();
    let cvv = cv.view();
    t!("view", cvv.view());
    t!("plain", pl());
    t!("plain_transposed", plt());
    t!("scalar", scalar());
    t!("scalar_plus_compact", scalar() + c());
    t!("neg", -c());
    t!("add", c() + pl());
    t!("cast", c().cast::<f64>());
    t!("cmp", c().greater(pl()));
    t!("clamp", c().clamp(pl(), pl()));
    t!("maximum", c().maximum(pl()));
    t!("sum0", c().sum(0));
    t!("sum_all", c().sum([0, 1, 2]));
    t!("product", c().product(1));
    t!("min", c().min([0, 2]));
    t!("max", c().max(2));
    t!("argmax", ci().argmax(1));
    t!("argmin", ci().argmin(0));
    t!("mean", c().mean(2));
    t!("var", c().var(2, 0.0));
    t!("std", c().std(0, 1.0));
    t!("all", c().greater(pl()).all(1));
    t!("any", c().greater(pl()).any([0, 1]));
    t!(
        "reduce_unordered",
        ci().reduce_unordered(1, |a, b| a.wrapping_add(b))
    );
    t!("permute", c().permute_dims(&[2, 0, 1]));
    t!("flip", c().flip([0, 2]));
    t!("slice", c().slice((.., SliceItem::new(None, None, 2), 1..)));
    t!("reshape", c().reshape([30u64, 7]));
    t!("broadcast", c().slice((0..1, .., ..)).broadcast(&[4, 6, 7]));
    t!("insert_dim", c().insert_dim(1));
    t!("remove_dim", c().slice((.., 2..3, ..)).remove_dim(1));
    t!("tile", c().tile(2, 1));
    t!("repeat", c().repeat(3, 2));
    t!("roll", c().roll(3, 1));
    t!("concat", jix::ops::concatenate((c(), pl()), 1));
    t!("stack", jix::ops::stack((c(), pl()), 2));
    t!(
        "where",
        jix::ops::where_condition(c().greater(pl()), c(), pl())
    );
    t!("into_any", c().into_any());
    t!("into_typed", c().into_any().into_typed::<f32>().unwrap());
    t!("transmute", unsafe { ci().transmute_elements::<u32>() });
    t!("maybe_compact", (-c()).maybe_compact().unwrap());
    t!("map", c().map(|x: f32| x * 2.0));
    t!("map2", jix::ops::map2(c(), pl(), |x: f32, y: f32| x - y));
    t!(
        "map_multiple",
        jix::ops::map_multiple([c(), c(), c()], |xs: [f32; 3]| xs[0] + xs[2])
    );
    t!("bitand", ci() & ci());
    t!("is_nan", c().is_nan());
    t!("count_ones", ci().count_ones());
    t!("sqrt", c().sqrt());
    t!("approx", c().approx_equal(pl(), 1e-3, 1e-3));
    t!("real", cc().real());
    t!("imag", cc().imag());

    // --- archives: heap TypeDyn and mmap ---
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.jix");
    c().write_to_file(&path).unwrap();
    let len = std::fs::metadata(&path).unwrap().len();
    t!(
        "read_from_file",
        Array::read_from_file(&path, ArrayParams::default()).unwrap()
    );
    t!("read_from_file_mmap", unsafe {
        Array::read_from_file_mmap(&path, 0, len, ArrayParams::default(), Default::default())
            .unwrap()
    });

    // --- a zero-length dim ---
    let nd0 = Array3::<f32>::zeros((0, 6, 7));
    let z = || Array::compact_ndarray(&nd0).unwrap();
    let zp = || Array::plain_ndarray_view(nd0.view()).unwrap();
    t!("z_compact", z());
    t!("z_plain", zp());
    t!("z_add", z() + zp());
    t!("z_sum", z().sum(1));
    t!("z_sum0", z().sum(0));
    t!("z_permute", z().permute_dims(&[1, 2, 0]));
    t!("z_concat", jix::ops::concatenate((z(), zp()), 0));
    t!(
        "z_broadcast",
        z().slice((.., 0..1, ..)).broadcast(&[0, 6, 7])
    );

    // --- 0-d ---
    let nd00 = ndarray::arr0(3.0f32);
    let s0 = || Array::compact_ndarray(&nd00).unwrap();
    t!("d0_compact", s0());
    t!("d0_plain", Array::plain_ndarray_view(nd00.view()).unwrap());
    t!(
        "d0_scalar",
        Array::from_storage(
            jix::__private::Scalar::new(1.0f32, [0u64; 0], ArrayParams::default()).unwrap()
        )
    );
    t!("d0_neg", -s0());
    t!(
        "d0_sum_of_1d",
        Array::compact_ndarray(&ndarray::array![1.0f32, 2.0])
            .unwrap()
            .sum(0)
    );
    t!("d0_insert", s0().insert_dim(0));

    // --- 8-d (NDIM_MAX): ndim+1 ranges exceed DimArray capacity ---
    let nd8 = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 1, 2, 1, 2, 1, 2, 1]), |i| i[0] as f32);
    let e = || Array::compact_ndarray(&nd8).unwrap();
    t!("d8_compact", e());
    t!("d8_plain", Array::plain_ndarray_view(nd8.view()).unwrap());
    t!("d8_add", e() + e());
    t!("d8_sum", e().sum(3));
    t!("d8_permute", e().permute_dims(&[7, 6, 5, 4, 3, 2, 1, 0]));
    t!("d8_flip", e().flip(0));
    t!("d8_remove", e().remove_dim(1));
    t!("d8_concat", jix::ops::concatenate((e(), e()), 0));
    t!(
        "d8_where",
        jix::ops::where_condition(e().greater(e()), e(), e())
    );

    let _ = std::panic::take_hook();
    for b in &bad {
        eprintln!("{b}");
    }
    assert!(
        bad.is_empty(),
        "{} invalid reads not rejected with InvalidIndex",
        bad.len()
    );
}
