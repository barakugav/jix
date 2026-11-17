// K193 (run 2026-11-17-scan-106-107-b): on a 32-bit target, since #106 a ZeroCopy / Fusable /
// SinglePass top-level read is not tiled (array/read.rs to_buf_strided_checked), so a NARROWING op
// (a struct field, a map / cast to a smaller type, a comparison giving bool8) over broadcast leaves
// pulls every leaf for the whole region, and each leaf's `check_out_buf` -> `check_region_size`
// rejects region * LEAF itemsize > isize::MAX, although the output fits, the leaf is a 0-stride
// view that allocates nothing, and the same expression read in two halves works.
//
// Each test asserts the CORRECT behavior (Ok with the right values), so it fails while the bug
// exists. On 64-bit there is no region-size check and the tests pass; the two big comparison
// tests are ignored there because they only cost time.
//
// i686: bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//   --test repro_k193 -- --nocapture --test-threads 1

use jix::dtype::Dtyped;
use jix::storage::StridedBuf;
use jix::Array;
use ndarray::{Array1, Array2, Ix1, Ix2};

/// A 64-byte record: `a` is the field we read.
#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Rec64 {
    a: u8,
    rest: [u8; 63],
}

/// 2^25 + 1 records: 2 GiB + 64 virtual bytes of leaf, 32 MiB of u8 output.
const N_REC: u64 = (1 << 25) + 1;

fn rec_src() -> Array1<Rec64> {
    let r = Rec64 {
        a: 7,
        rest: [1; 63],
    };
    Array1::from_elem(1, r)
}

#[test]
fn sub_dtype_field_of_broadcast_struct_reads_whole_region() {
    let src = Array::plain_ndarray(rec_src()).unwrap();
    let f = src.broadcast(&[N_REC]).dtype_sub_field::<u8>("a");
    let ctx = f.read_ctx();

    // Control: the same expression in two halves (1 GiB of virtual leaf each) reads fine.
    let h = N_REC / 2;
    let lo = f.to_ndarray_sub::<Ix1>(&[0..h], &ctx).unwrap();
    let hi = f.to_ndarray_sub::<Ix1>(&[h..N_REC], &ctx).unwrap();
    assert!(lo.iter().chain(hi.iter()).all(|&x| x == 7));

    let whole = f.to_ndarray::<Ix1>();
    eprintln!("sub_dtype field, whole region: {:?}", whole.as_ref().err());
    let whole = whole.unwrap();
    assert_eq!(whole.len() as u64, N_REC);
    assert!(whole.iter().all(|&x| x == 7));
}

#[test]
fn map_of_broadcast_struct_reads_whole_region() {
    let src = Array::plain_ndarray(rec_src()).unwrap();
    let f = src.broadcast(&[N_REC]).map(|r: Rec64| r.a);
    let ctx = f.read_ctx();

    let h = N_REC / 2;
    let lo = f.to_ndarray_sub::<Ix1>(&[0..h], &ctx).unwrap();
    assert!(lo.iter().all(|&x| x == 7));

    let mut buf = vec![0u8; N_REC as usize];
    let r = f.to_buf_slice(&[0..N_REC], &mut buf, &ctx);
    eprintln!("map, whole region: {:?}", r.as_ref().err());
    r.unwrap();
    assert!(buf.iter().all(|&x| x == 7));
}

/// 2^14 x 2^14 = 2^28 elements: 2 GiB of virtual f64 per leaf, 256 MiB of bool8 / u8 output.
const M: u64 = 1 << 14;

fn col() -> Array2<f64> {
    Array2::from_shape_fn((M as usize, 1), |(i, _)| (i % 7) as f64)
}
fn row() -> Array2<f64> {
    Array2::from_shape_fn((1, M as usize), |(_, j)| (j % 5) as f64)
}

#[test]
#[cfg_attr(not(target_pointer_width = "32"), ignore = "32-bit only; slow")]
fn outer_less_and_cast_of_broadcast_f64_read_whole_region() {
    let (c, r) = (col(), row());
    let a = || Array::plain_ndarray_ref(&c).unwrap().broadcast(&[M, M]);
    let b = || Array::plain_ndarray_ref(&r).unwrap().broadcast(&[M, M]);

    let less = a().less(b());
    let ctx = less.read_ctx();
    let got = less.to_ndarray_sub::<Ix2>(&[0..M, 0..M], &ctx);
    eprintln!("less, whole region: {:?}", got.as_ref().err());
    let got = got.unwrap();
    assert!(bool::from(got[[3, 4]]));
    assert!(!bool::from(got[[4, 3]]));
    drop(got);

    let cast = (a() + b()).cast::<u8>();
    let mut buf = vec![0u8; (M * M) as usize];
    let res = cast.to_buf_slice(&[0..M, 0..M], &mut buf, &ctx);
    eprintln!("(a + b).cast::<u8>(), whole region: {:?}", res.as_ref().err());
    res.unwrap();
    assert_eq!(buf[(3 * M + 4) as usize], 3 + 4);
}

/// The jix-py shape of `jix.less(jix.asarray(col), jix.asarray(row)).numpy()`: operands are
/// ArrayAny, implicitly broadcast with `Broadcast::new_array(..).into_any()`, re-typed, the op
/// result is erased again, and `numpy()` calls `to_buf_strided_into` (jix-py/src/array.rs:653).
#[test]
#[cfg_attr(not(target_pointer_width = "32"), ignore = "32-bit only; slow")]
fn python_shaped_less_through_any_reads_whole_region() {
    let a = Array::plain_ndarray(col()).unwrap().into_any();
    let b = Array::plain_ndarray(row()).unwrap().into_any();
    let a = a.broadcast(&[M, M]).into_any().into_typed::<f64>().unwrap();
    let b = b.broadcast(&[M, M]).into_any().into_typed::<f64>().unwrap();
    let res = a.less(b).into_any();
    let ctx = res.read_ctx();

    let mut dst = vec![0u8; (M * M) as usize];
    let r = {
        let mut out = unsafe {
            StridedBuf::from_raw_parts_mut(
                dst.as_mut_ptr(),
                &[M as usize, M as usize],
                &[M as usize, 1],
                1,
            )
        };
        res.to_buf_strided_into(&[0..M, 0..M], &mut out, &ctx)
    };
    eprintln!("python-shaped less, whole region: {:?}", r.as_ref().err());
    r.unwrap();
    assert_eq!(dst[(3 * M + 4) as usize], 1);
    assert_eq!(dst[(4 * M + 3) as usize], 0);
}
