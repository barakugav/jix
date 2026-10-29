// F002 (run 2026-10-29-batch6), UB part. No `unsafe` in this file.
//
// Both structs below have 3 bytes of implicit padding, which #[derive(Dtyped)] must reject at
// compile time ("has padding bytes"). Because the derive's generated code is not hygienic, a name
// in the user's scope makes it accept them, with a DTYPE whose fields claim the padding bytes, and
// jix then reads the uninitialized padding (NdCopier copies 8-byte/align-4 elements as [u32; 2]).
//
// Once F002 is fixed this file must FAIL TO COMPILE (the correct behavior); it is not a
// regression test. Run under Miri:
//   python jix/run_miri.py --all-features --test repro_2026_10_29_batch6__f002_ub -- miri_padded_base_offset

#![allow(dead_code)]

use jix::dtype::Dtyped;

mod padded_via_const {
    // An ordinary user const with a natural name; the derive declares its own `BASE_OFFSET = 0`
    // in the block where it re-emits this field type, so it lays out `a` as [u8; 4], not [u8; 1].
    pub const BASE_OFFSET: u16 = 3;

    #[derive(Copy, Clone, Debug, jix::dtype::Dtyped)]
    #[repr(C)]
    pub struct P {
        pub a: [u8; 4 - BASE_OFFSET as usize], // real: [u8; 1] at 0, then 3 padding bytes
        pub b: u32,                            // real: u32 at 4
    }
}

mod padded_via_assert {
    // A no-op `assert!` in scope disables every check the derive emits.
    macro_rules! assert {
        ($($t:tt)*) => {
            ()
        };
    }

    #[derive(Copy, Clone, Debug, jix::dtype::Dtyped)]
    #[repr(C)]
    pub struct Q {
        pub a: u8, // then 3 padding bytes
        pub b: u32,
    }
}

#[test]
fn padded_base_offset_dtype_matches_layout() {
    let d = padded_via_const::P::DTYPE;
    eprintln!("P: size_of = {}, DTYPE = {d:?}", std::mem::size_of::<padded_via_const::P>());
    let a = &d.fields().unwrap()[0];
    assert_eq!(a.2.itemsize() as usize, 1, "field `a` is [u8; 1] in Rust");
}

#[test]
fn padded_assert_dtype_has_no_padding() {
    let d = padded_via_assert::Q::DTYPE;
    eprintln!("Q: size_of = {}, DTYPE = {d:?}", std::mem::size_of::<padded_via_assert::Q>());
    let covered: usize = d.fields().unwrap().iter().map(|f| f.2.itemsize() as usize).sum();
    assert_eq!(covered, d.itemsize() as usize, "derive accepted a struct with padding");
}

#[test]
fn miri_padded_base_offset() {
    let arr = ndarray::array![
        padded_via_const::P { a: [1], b: 2 },
        padded_via_const::P { a: [3], b: 4 }
    ];
    let plain = jix::Array::plain_ndarray_ref(&arr).unwrap();
    let mut buf = vec![0u8; 16];
    // Push-mode copy of the whole array: reads each element, padding included, as [u32; 2].
    plain
        .to_ndarray_slice(&[0..2], &mut buf, &jix::ReadContext::default())
        .unwrap();
    assert_eq!(buf[0], 1);
    assert_eq!(buf[8], 3);
}
