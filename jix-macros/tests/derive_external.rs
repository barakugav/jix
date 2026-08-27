// `#[derive(Dtyped)]` tests the way a downstream crate uses it.
// This reach some edge cases that cant be used by tests that exists inside the jix crate itself.

#![allow(dead_code)]

use jix_core::dtype::{Dtype, Dtyped};

// ---- #[repr(C)] structs with named fields ----

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Pixel {
    r: u8,
    g: u8,
    b: u8,
}

#[test]
fn repr_c_named_fields() {
    let d = Pixel::DTYPE;
    assert_eq!(d.itemsize(), 3);
    assert_eq!(d.itemsize() as usize, size_of::<Pixel>());
    assert_eq!(d.alignment().as_usize(), align_of::<Pixel>());
    assert!(d.is_aligned());
    assert_eq!(d.shape(), &[]);

    let fields = d.fields().unwrap();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].0, "r");
    assert_eq!(fields[1].0, "g");
    assert_eq!(fields[2].0, "b");
    assert_eq!(
        fields.iter().map(|f| f.1).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Padded {
    a: u8,
    b: f64,
}

#[test]
fn repr_c_inserts_padding() {
    let d = Padded::DTYPE;
    // `b` is pushed to offset 8 by its own alignment, and the struct is rounded up to 16.
    assert_eq!(d.itemsize(), 16);
    assert_eq!(d.itemsize() as usize, size_of::<Padded>());
    assert_eq!(d.alignment().as_usize(), 8);
    let fields = d.fields().unwrap();
    assert_eq!(fields[0].1, 0);
    assert_eq!(fields[1].1, 8);
}

// ---- packed structs ----

#[derive(Copy, Clone, Dtyped)]
#[repr(C, packed)]
struct PackedC {
    a: u8,
    b: u32,
}

// A bare `packed` repr is its own branch in the derive (the ABI-qualified `#[repr(Rust, packed)]`
// spelling is not one of the strings the macro accepts), so the lint has to be allowed here.
#[allow(clippy::repr_packed_without_abi)]
#[derive(Copy, Clone, Dtyped)]
#[repr(packed)]
struct PackedBare {
    a: u8,
    b: u32,
}

#[test]
fn repr_packed_has_no_padding() {
    for d in [PackedC::DTYPE, PackedBare::DTYPE] {
        assert_eq!(d.itemsize(), 5);
        assert_eq!(d.alignment().as_usize(), 1);
        assert!(!d.is_aligned());
        let fields = d.fields().unwrap();
        assert_eq!(fields[0].1, 0);
        assert_eq!(fields[1].1, 1);
    }
    assert_eq!(PackedC::DTYPE.itemsize() as usize, size_of::<PackedC>());
    assert_eq!(
        PackedBare::DTYPE.itemsize() as usize,
        size_of::<PackedBare>()
    );
}

// ---- #[repr(transparent)] newtypes ----

#[derive(Copy, Clone, Dtyped)]
#[repr(transparent)]
struct Meters(f32);

#[test]
fn repr_transparent_forwards_inner_dtype() {
    assert_eq!(Meters::DTYPE, f32::DTYPE);
    assert_eq!(Meters::DTYPE.itemsize() as usize, size_of::<Meters>());
    assert_eq!(Meters::DTYPE.scalar_kind(), f32::DTYPE.scalar_kind());
}

// ---- zero-sized struct ----

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Empty {}

#[test]
fn empty_struct_is_zero_sized() {
    let d = Empty::DTYPE;
    assert_eq!(d.itemsize(), 0);
    assert_eq!(d.itemsize() as usize, size_of::<Empty>());
    assert_eq!(d.fields().unwrap(), &[]);
    assert_eq!(d, Dtype::from_fields(vec![]).unwrap());
}

// ---- array fields ----

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Vertex {
    pos: [f32; 3],
    id: u32,
}

#[test]
fn array_field() {
    let d = Vertex::DTYPE;
    assert_eq!(d.itemsize(), 16);
    assert_eq!(d.itemsize() as usize, size_of::<Vertex>());
    let fields = d.fields().unwrap();
    assert_eq!(fields[0].2, <[f32; 3] as Dtyped>::DTYPE);
    assert_eq!(fields[0].2.shape(), &[3]);
    assert_eq!(fields[1].1, 12);
}

// ---- nested derived structs ----

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Point {
    x: f32,
    y: f32,
}

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Segment {
    start: Point,
    end: Point,
}

#[test]
fn nested_struct_field() {
    let d = Segment::DTYPE;
    assert_eq!(d.itemsize(), 16);
    assert_eq!(d.itemsize() as usize, size_of::<Segment>());
    let fields = d.fields().unwrap();
    assert_eq!(fields[0].2, Point::DTYPE);
    assert_eq!(fields[1].1, 8);
}

// ---- generic transparent newtypes ----
//
// Only the transparent branch supports generics. The named-fields branch declares intermediate
// `const` items for the per-field offsets, and a nested const cannot name a type parameter of the
// item it sits in, so a generic `#[repr(C)]` struct does not compile.

#[derive(Copy, Clone, Dtyped)]
#[repr(transparent)]
struct Tagged<T>(T);

#[test]
fn generic_transparent_newtype() {
    assert_eq!(<Tagged<u8> as Dtyped>::DTYPE, u8::DTYPE);
    assert_eq!(<Tagged<f64> as Dtyped>::DTYPE, f64::DTYPE);
    assert_eq!(<Tagged<Point> as Dtyped>::DTYPE, Point::DTYPE);
    assert_eq!(
        <Tagged<Point> as Dtyped>::DTYPE.itemsize() as usize,
        size_of::<Tagged<Point>>()
    );
}

// ---- the derived type actually works as an array element ----

#[test]
fn derived_struct_round_trips_through_an_array() {
    let pixels = ndarray::array![
        Pixel { r: 1, g: 2, b: 3 },
        Pixel { r: 4, g: 5, b: 6 },
        Pixel { r: 7, g: 8, b: 9 },
    ];
    let arr = jix_core::Array::compact_ndarray(&pixels).unwrap();
    assert_eq!(arr.dtype(), &Pixel::DTYPE);

    let out = arr.to_ndarray().unwrap();
    assert_eq!(out.len(), 3);
    for (a, b) in out.iter().zip(pixels.iter()) {
        assert_eq!((a.r, a.g, a.b), (b.r, b.g, b.b));
    }
}

#[test]
fn derived_struct_field_can_be_read_as_a_sub_array() {
    let pixels = ndarray::array![Pixel { r: 1, g: 2, b: 3 }, Pixel { r: 4, g: 5, b: 6 }];
    let arr = jix_core::Array::compact_ndarray(&pixels).unwrap();
    let greens = arr.dtype_sub_field::<u8>("g").to_ndarray().unwrap();
    assert_eq!(greens.as_slice().unwrap(), &[2, 5]);
}
