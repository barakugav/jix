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

// Deep fused trees: a balanced tree keeps a value per level live while it evaluates the other
// branch of the level, and each value is `LANES` elements (several vectors). `x()` is the next leaf:
// operands are evaluated left to right. Levels from the leaves: `+ * - + *`.
macro_rules! level1 {
    ($x:ident) => {
        ($x() + $x())
    };
}
macro_rules! level2 {
    ($x:ident) => {
        (level1!($x) * level1!($x))
    };
}
macro_rules! level3 {
    ($x:ident) => {
        (level2!($x) - level2!($x))
    };
}
macro_rules! level4 {
    ($x:ident) => {
        (level3!($x) + level3!($x))
    };
}
macro_rules! level5 {
    ($x:ident) => {
        (level4!($x) * level4!($x))
    };
}

macro_rules! tree {
    ($($name:ident: $t:ident, $level:ident, $n:literal;)*) => { $(
        /// A balanced tree over `$n` leaves.
        #[unsafe(no_mangle)]
        pub fn $name(xs: [&[$t]; $n], out: &mut [u8]) {
            let mut leaves = xs.into_iter().map(plain);
            let mut x = || leaves.next().unwrap();
            write($level!(x), out)
        }
    )* };
}
tree! {
    probe_tree_d2_f32: f32, level2, 4;
    probe_tree_d3_f32: f32, level3, 8;
    probe_tree_d4_f32: f32, level4, 16;
    probe_tree_d5_f32: f32, level5, 32;
    probe_tree_d2_f64: f64, level2, 4;
    probe_tree_d3_f64: f64, level3, 8;
    probe_tree_d4_f64: f64, level4, 16;
    probe_tree_d5_f64: f64, level5, 32;
    probe_tree_d4_i32: i32, level4, 16;
}

/// A left-deep chain over 16 f32 leaves, `((x0 + x1) * x2 + x3) * ...`: two values live at once.
#[unsafe(no_mangle)]
pub fn probe_leftchain_f32(xs: [&[f32]; 16], out: &mut [u8]) {
    let mut leaves = xs.into_iter().map(plain);
    let mut x = || leaves.next().unwrap();
    write(
        ((((((((((((((x() + x()) * x()) + x()) * x()) + x()) * x()) + x()) * x()) + x()) * x())
            + x())
            * x())
            + x())
            * x())
            + x(),
        out,
    )
}
