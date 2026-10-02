//! Asm probe for `jix`'s element-wise pipeline.
//!
//! Each `probe_*` function evaluates one op chain over `Plain` arrays into a packed output, through
//! the public API. The loop of interest is not in the probe function itself but in the pipeline's
//! inner loop it instantiates (`elementwise_pipeline::inner_loop`), which `analyze.py` locates by
//! its op chain and element type.
//!
//! Driven by `analyze.py`, see `../README.md`.

use jix::dtype::Dtyped;
use jix::storage::Plain;
use jix::{Array, ArrayStorage, Dim, ReadContext, Ty};
use ndarray::ArrayView1;

type P<'a, T> = Array<Plain<&'a (), Ty<T>, Dim<1>>>;

fn plain<T: Dtyped>(x: &[T]) -> P<'_, T> {
    Array::plain_ndarray_view(ArrayView1::from(x)).unwrap()
}

fn write<S: ArrayStorage>(x: Array<S>, out: &mut [u8]) {
    let n = x.shape()[0];
    x.to_ndarray_slice(&[0..n], out, &ReadContext::default())
        .unwrap();
}

macro_rules! probe {
    ($($t:ident: $neg:ident, $add:ident, $chain:ident, $longchain:ident;)*) => { $(
        #[unsafe(no_mangle)]
        pub fn $neg(a: &[$t], out: &mut [u8]) {
            write(-plain(a), out)
        }
        #[unsafe(no_mangle)]
        pub fn $add(a: &[$t], b: &[$t], out: &mut [u8]) {
            write(plain(a) + plain(b), out)
        }
        /// `(a + b) * (c - d)`.
        #[unsafe(no_mangle)]
        pub fn $chain(a: &[$t], b: &[$t], c: &[$t], d: &[$t], out: &mut [u8]) {
            write((plain(a) + plain(b)) * (plain(c) - plain(d)), out)
        }
        /// `(a + b) * (c - d) + (e + f) * (g - h)`: more values live at once.
        #[unsafe(no_mangle)]
        pub fn $longchain(xs: [&[$t]; 8], out: &mut [u8]) {
            let [a, b, c, d, e, f, g, h] = xs.map(plain);
            write((a + b) * (c - d) + (e + f) * (g - h), out)
        }
    )* };
}
probe! {
    f32: probe_ew_neg_f32, probe_ew_add_f32, probe_ew_chain_f32, probe_ew_longchain_f32;
    f64: probe_ew_neg_f64, probe_ew_add_f64, probe_ew_chain_f64, probe_ew_longchain_f64;
    i32: probe_ew_neg_i32, probe_ew_add_i32, probe_ew_chain_i32, probe_ew_longchain_i32;
}

// Mixed dtypes: the lanes follow the widest value, not the output.
/// `(a + b).cast::<i32>()` over i64.
#[unsafe(no_mangle)]
pub fn probe_ew_narrow_i32(a: &[i64], b: &[i64], out: &mut [u8]) {
    write((plain(a) + plain(b)).cast::<i32>(), out)
}
/// `a.cast::<i64>() + b`, `a` i32.
#[unsafe(no_mangle)]
pub fn probe_ew_widen_i64(a: &[i32], b: &[i64], out: &mut [u8]) {
    write(plain(a).cast::<i64>() + plain(b), out)
}
/// `(a + b).cast::<f32>()` over f64.
#[unsafe(no_mangle)]
pub fn probe_ew_narrow_f32(a: &[f64], b: &[f64], out: &mut [u8]) {
    write((plain(a) + plain(b)).cast::<f32>(), out)
}
/// `a.cast::<f64>() + b`, `a` f32.
#[unsafe(no_mangle)]
pub fn probe_ew_widen_f64(a: &[f32], b: &[f64], out: &mut [u8]) {
    write(plain(a).cast::<f64>() + plain(b), out)
}

// Fused chains over several dtypes, all different from the output's: values in the chain are
// wider than the output, so lanes sized by the output alone take more registers per value.
/// `(a.cast::<f32>() * s + b.cast::<f32>()).cast::<u8>()`, `a`, `b` u8: an image blend.
#[unsafe(no_mangle)]
pub fn probe_ew_blend_u8(a: &[u8], b: &[u8], s: &[f32], out: &mut [u8]) {
    write(
        (plain(a).cast::<f32>() * plain(s) + plain(b).cast::<f32>()).cast::<u8>(),
        out,
    )
}
/// `a.cast::<f32>().greater(t)`, `a` u8: a threshold mask.
#[unsafe(no_mangle)]
pub fn probe_ew_threshold_bool(a: &[u8], t: &[f32], out: &mut [u8]) {
    write(plain(a).cast::<f32>().greater(plain(t)), out)
}
/// `((a + b) * c).greater(d)` over f64.
#[unsafe(no_mangle)]
pub fn probe_ew_cmpchain_bool(a: &[f64], b: &[f64], c: &[f64], d: &[f64], out: &mut [u8]) {
    write(((plain(a) + plain(b)) * plain(c)).greater(plain(d)), out)
}
/// `a.greater(b) & c.less(d)` over f32.
#[unsafe(no_mangle)]
pub fn probe_ew_range_bool(a: &[f32], b: &[f32], c: &[f32], d: &[f32], out: &mut [u8]) {
    write(plain(a).greater(plain(b)) & plain(c).less(plain(d)), out)
}
/// `where(a.greater(b), x, y) + z`, `a`, `b` f64, others f32 (a top-level `where` reads without
/// the pipeline).
#[unsafe(no_mangle)]
pub fn probe_ew_select_f32(a: &[f64], b: &[f64], xs: [&[f32]; 3], out: &mut [u8]) {
    let [x, y, z] = xs.map(plain);
    write(
        jix::ops::where_condition(plain(a).greater(plain(b)), x, y) + z,
        out,
    )
}
/// `((a + b) * (c - d)).cast::<i16>()` over i64.
#[unsafe(no_mangle)]
pub fn probe_ew_chain_i16(xs: [&[i64]; 4], out: &mut [u8]) {
    let [a, b, c, d] = xs.map(plain);
    write(((a + b) * (c - d)).cast::<i16>(), out)
}
/// `((a + b) * (c - d) + (e + f) * (g - h)).cast::<f32>()` over f64.
#[unsafe(no_mangle)]
pub fn probe_ew_longchain_narrow_f32(xs: [&[f64]; 8], out: &mut [u8]) {
    let [a, b, c, d, e, f, g, h] = xs.map(plain);
    write(((a + b) * (c - d) + (e + f) * (g - h)).cast::<f32>(), out)
}
/// `(a.cast::<f64>() * b.cast::<f64>() + c.cast::<f64>()).cast::<f32>()`, f32: computed in f64.
#[unsafe(no_mangle)]
pub fn probe_ew_fma64_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [u8]) {
    let [a, b, c] = [a, b, c].map(|x| plain(x).cast::<f64>());
    write((a * b + c).cast::<f32>(), out)
}
/// `(a.cast::<i32>() * b.cast::<i32>()).cast::<i16>()`, i8: widened products.
#[unsafe(no_mangle)]
pub fn probe_ew_mul_widen_i16(a: &[i8], b: &[i8], out: &mut [u8]) {
    write(
        (plain(a).cast::<i32>() * plain(b).cast::<i32>()).cast::<i16>(),
        out,
    )
}
