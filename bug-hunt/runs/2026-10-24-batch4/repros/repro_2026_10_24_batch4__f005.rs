//! F005 repro: `ElementType` is a safe, unsealed public trait, but storages that hold an element
//! type instance (Plain, BlockTable/Compact, SubDtype, Transmute) take the dtype their unsafe code
//! relies on from `ET::dtype()`, and `element_type_change` / `SubDtype::new` never check that it
//! equals the dtype they were built from. A user impl with no `unsafe` then drives safe calls into
//! an out-of-bounds heap read (Plain::read_data, SubDtype push copy) and into
//! `assert_unchecked(false)` (BlockTable::decoder_config, reached by write_to).
//!
//! Each test asserts the sound outcome: the mismatching element type is rejected with an `Err`
//! (by `into_type` / `SubDtype::new`, or by the read / write). If the fix is to seal the trait,
//! this file no longer compiles - that also counts as fixed.
#![allow(clippy::single_range_in_vec_init)]

use jix::dtype::{Dtype, Dtyped};
use jix::{Array, ElementType};

/// Claims 8-byte elements whatever the storage holds.
#[derive(Clone)]
struct Wide;
static U64: Dtype = <u64 as Dtyped>::DTYPE;
impl ElementType for Wide {
    const DTYPE: Option<Dtype> = None;
    fn from_dtype(_dtype: Dtype) -> Result<Self, jix::Error> {
        Ok(Wide)
    }
    fn dtype(&self) -> &Dtype {
        &U64
    }
}

/// Claims 1-byte elements whatever the storage holds.
#[derive(Clone)]
struct Narrow;
static U8: Dtype = <u8 as Dtyped>::DTYPE;
impl ElementType for Narrow {
    const DTYPE: Option<Dtype> = None;
    fn from_dtype(_dtype: Dtype) -> Result<Self, jix::Error> {
        Ok(Narrow)
    }
    fn dtype(&self) -> &Dtype {
        &U8
    }
}

#[derive(Copy, Clone, Debug, jix::dtype::Dtyped)]
#[repr(C)]
struct Pair {
    x: i32,
    y: i32,
}

/// Plain over a 3-byte u8 buffer, re-tagged as 8-byte elements: Plain::read_data builds a
/// `from_raw_parts` of 2*1 + 8 = 10 bytes over the 3-byte allocation and copies it out.
#[test]
fn plain_wide_element_type_is_rejected() {
    let data = ndarray::array![1u8, 2, 3];
    let mut buf = vec![0u8; 24];
    let res = Array::plain_ndarray_ref(&data)
        .unwrap()
        .into_type::<Wide>()
        .and_then(|w| {
            let ctx = w.read_ctx();
            w.to_ndarray_slice(&[0..3], &mut buf, &ctx)
        });
    println!("plain: {:?} buf={buf:?}", res.as_ref().map_err(|e| e.to_string()));
    assert!(
        res.is_err(),
        "a u8 Plain re-tagged as an 8-byte element type was read with Ok: buf={buf:?}"
    );
}

/// Compact u32 re-tagged as 1-byte elements: write_to reaches
/// `assert_unchecked_eq!(element_type.dtype(), decoder_config.dtype)` in BlockTable::decoder_config.
#[test]
fn compact_narrow_element_type_write_to_is_rejected() {
    let mut bytes = Vec::new();
    let res = Array::compact_ndarray(&ndarray::array![1u32, 2, 3])
        .unwrap()
        .into_type::<Narrow>()
        .and_then(|b| b.write_to(std::io::Cursor::new(&mut bytes)));
    println!("write_to: {:?}", res.as_ref().map_err(|e| e.to_string()));
    assert!(res.is_err(), "a u32 Compact re-tagged as a 1-byte element type was written with Ok");
}

/// SubDtype of the 4-byte field `y` (offset 4) of a 1-element Plain<Pair> (8 bytes) with an
/// element type that claims 8-byte elements: the push copy reads bytes 4..12 of the 8-byte
/// element, i.e. 4 bytes past the allocation.
#[test]
fn sub_dtype_wide_element_type_is_rejected() {
    let pairs = ndarray::array![Pair { x: 1, y: 2 }];
    let mut buf = vec![0u8; 8];
    let res = Array::plain_ndarray_ref(&pairs)
        .and_then(|a| jix::ops::SubDtype::<_, Wide>::new_array(a, "y"))
        .and_then(|s| {
            let ctx = s.read_ctx();
            s.to_ndarray_slice(&[0..1], &mut buf, &ctx)
        });
    println!("sub_dtype: {:?} buf={buf:?}", res.as_ref().map_err(|e| e.to_string()));
    assert!(
        res.is_err(),
        "SubDtype of a 4-byte field with an 8-byte element type was read with Ok: buf={buf:?}"
    );
}
