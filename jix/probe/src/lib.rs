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

/// `(a + b) * (c - d)`: an example; add the chains to study here.
#[unsafe(no_mangle)]
pub fn probe_chain_f32(a: &[f32], b: &[f32], c: &[f32], d: &[f32], out: &mut [u8]) {
    write((plain(a) + plain(b)) * (plain(c) - plain(d)), out)
}
