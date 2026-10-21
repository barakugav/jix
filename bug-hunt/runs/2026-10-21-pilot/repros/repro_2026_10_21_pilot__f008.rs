// Verifier repro for F008: `ScalarKind::alignment()` hard-codes 8 for I64/U64/F64/ComplexF64,
// but the i386 SysV psABI (i686 Linux, glibc and musl - shipped manylinux/musllinux x86 wheels)
// aligns these types to 4. Rust (`align_of`) and numpy (`NPY_ALIGNOF` = C11 `_Alignof`) both use
// the ABI value there, so every consumer that compares jix's alignment with the platform's breaks.
//
//   cargo test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_21_pilot__f008 -- --nocapture --test-threads 1
//
// Run natively on x86_64 the `*_on_target` tests pass and `numpy_i686_struct_layouts` fails
// (host-side simulation of the descriptors numpy hands `dtype_from_numpy` on i686).
// Cross-built for i686-unknown-linux-musl (see bug-hunt/runs/2026-10-21-pilot/findings/F008.md)
// the `*_on_target` tests fail too.
use std::path::PathBuf;

use jix::dtype::{Alignment, Dtype, Dtyped};
use jix::scalar::Complex;
use jix::{Array, ArrayParams};

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "repro_2026_10_21_pilot__f008_{}_{name}.jix",
        std::process::id()
    ))
}

/// The `Dtyped` contract (D2): `T::DTYPE.alignment() == align_of::<T>()`.
#[test]
fn scalar_dtype_alignment_matches_abi_on_target() {
    let rows = [
        ("i64", align_of::<i64>(), i64::DTYPE.alignment().as_usize()),
        ("u64", align_of::<u64>(), u64::DTYPE.alignment().as_usize()),
        ("f64", align_of::<f64>(), f64::DTYPE.alignment().as_usize()),
        (
            "Complex<f64>",
            align_of::<Complex<f64>>(),
            Complex::<f64>::DTYPE.alignment().as_usize(),
        ),
    ];
    println!(
        "target_arch={} target_env={}",
        std::env::consts::ARCH,
        if cfg!(target_env = "musl") { "musl" } else { "gnu/other" }
    );
    for (name, abi, jix) in rows {
        println!("  {name:13} align_of={abi} DTYPE.alignment()={jix}");
    }
    for (name, abi, jix) in rows {
        assert_eq!(abi, jix, "{name}: DTYPE alignment differs from align_of");
    }
}

/// `#[derive(Dtyped)]` lays out fields with `align_of`, so on i686 `b` lands at offset 4, the
/// struct is 12 bytes / align 4, while `u64::DTYPE` claims align 8. The archive header written
/// from that dtype is rejected by `Dtype::new_struct` on read.
#[derive(Clone, Copy, Debug, PartialEq, Dtyped)]
#[repr(C)]
struct S {
    a: u8,
    b: u64,
}

#[test]
fn derived_struct_archive_roundtrip_on_target() {
    println!(
        "S: size_of={} align_of={} DTYPE={:?}",
        size_of::<S>(),
        align_of::<S>(),
        S::DTYPE
    );
    let data = ndarray::Array1::from(vec![S { a: 7, b: 0x0102_0304_0506_0708 }; 5]);
    let a = Array::compact_ndarray(&data).unwrap();
    let p = tmp_path("derived");
    a.write_to_file(&p).unwrap();
    let back = Array::read_from_file(&p, ArrayParams::default());
    let _ = std::fs::remove_file(&p);
    println!(
        "read back: {:?}",
        back.as_ref().map(|x| x.dtype().clone()).map_err(|e| e.to_string())
    );
    let back = back.expect("archive written on this target must read back on this target");
    let got = back
        .into_typed::<S>()
        .unwrap()
        .to_ndarray::<ndarray::Ix1>()
        .unwrap();
    assert_eq!(got, data);
}

/// Host-side simulation of `jix-py/src/dtype.rs::dtype_from_numpy` on i686 Linux. numpy computes
/// aligned-struct layouts with the ABI alignment (4 for 'f8'/'u8' there) and passes
/// (fields, offsets, itemsize, alignment) to `Dtype::new_struct`:
///   np.dtype([('a','f8'),('b','i4')], align=True): offsets (0, 8), itemsize 12, alignment 4
///   np.dtype([('a','u1'),('b','u8')], align=True): offsets (0, 4), itemsize 12, alignment 4
#[test]
fn numpy_i686_struct_layouts() {
    let a4 = Alignment::new(4).unwrap();
    let cases: [(&str, Vec<(String, u16, Dtype)>); 2] = [
        (
            "[('a','f8'),('b','i4')] align=True",
            vec![("a".into(), 0, f64::DTYPE), ("b".into(), 8, i32::DTYPE)],
        ),
        (
            "[('a','u1'),('b','u8')] align=True",
            vec![("a".into(), 0, u8::DTYPE), ("b".into(), 4, u64::DTYPE)],
        ),
    ];
    let mut failed = Vec::new();
    for (name, fields) in cases {
        let r = Dtype::new_struct(fields, &[], 12, a4);
        println!(
            "  {name}: {:?}",
            r.as_ref().map(|_| ()).map_err(|e| e.to_string())
        );
        if r.is_err() {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "rejected i686 numpy layouts: {failed:?}");
}

/// Debug builds only: `NdCopier` picks `scalar_fn::<u64>` for (itemsize 8, alignment 8) and
/// `debug_assert_eq!(align_of::<u64>(), 8)` (jix/src/util/nd_copy.rs:90) fails on i686.
#[test]
fn f64_compact_roundtrip_on_target() {
    let data = ndarray::array![1.0f64, 2.0, 3.0];
    let a = Array::compact_ndarray(&data).unwrap();
    assert_eq!(a.to_ndarray::<ndarray::Ix1>().unwrap(), data);
}
