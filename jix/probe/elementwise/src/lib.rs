//! Asm probe for `jix`'s element-wise pipeline.
//!
//! Each `probe_*` function evaluates one op chain over `Plain` arrays into a packed output, through
//! the public API. The loop of interest is not in the probe function itself but in the pipeline's
//! inner loop it instantiates (`elementwise_pipeline::inner_loop`), which `analyze.py` locates by
//! its op chain and element type.
//!
//! Driven by `analyze.py --direction elementwise`, see `../README.md`.

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
    ($($t:ident: $neg:ident, $add:ident, $chain:ident;)*) => { $(
        #[unsafe(no_mangle)]
        pub fn $neg(a: &[$t], out: &mut [u8]) {
            write(-plain(a), out)
        }
        #[unsafe(no_mangle)]
        pub fn $add(a: &[$t], b: &[$t], out: &mut [u8]) {
            write(plain(a) + plain(b), out)
        }
        /// `(a + b) * c - d`.
        #[unsafe(no_mangle)]
        pub fn $chain(a: &[$t], b: &[$t], c: &[$t], d: &[$t], out: &mut [u8]) {
            write((plain(a) + plain(b)) * plain(c) - plain(d), out)
        }
    )* };
}
probe! {
    f32: probe_ew_neg_f32, probe_ew_add_f32, probe_ew_chain_f32;
    f64: probe_ew_neg_f64, probe_ew_add_f64, probe_ew_chain_f64;
    i32: probe_ew_neg_i32, probe_ew_add_i32, probe_ew_chain_i32;
}
