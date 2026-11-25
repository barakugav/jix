//! EXPERIMENTAL: the scenarios shared by `benches/thread_alloc.rs` and
//! `examples/thread_alloc_probe.rs`, which measure how much reusing scratch buffers between small
//! reads saves. Kept in a subdirectory so Cargo does not pick it up as a bench target of its own.

#![allow(dead_code)]

use std::ops::Range;

use fastrand::Rng;
use jix::storage::{CompactArrayMethods, CompactParams, Filter};
use jix::{Array, CompactArray, Storage};

/// Bytes of scratch buffers the thread keeps in the `thread_pool` mode: enough for every scenario.
pub const RETAIN_BYTES: usize = 64 << 20;

/// Number of random positions each scenario cycles through.
const NPOSITIONS: usize = 4096;

pub const SHAPE: [u64; 2] = [2048, 2048];

/// A random-access workload: an array and the positions to read from it.
pub struct Positions {
    /// One index per read, for `Array::get`.
    pub indices: Vec<Vec<u64>>,
    /// The same positions as 1-element regions, for `to_buf_slice`.
    pub regions: Vec<Vec<Range<u64>>>,
}
impl Positions {
    fn random(shape: &[u64], rng: &mut Rng) -> Self {
        let indices: Vec<Vec<u64>> = (0..NPOSITIONS)
            .map(|_| shape.iter().map(|&len| rng.u64(0..len)).collect())
            .collect();
        let regions = indices
            .iter()
            .map(|index| index.iter().map(|&i| i..i + 1).collect())
            .collect();
        Self { indices, regions }
    }
}

pub trait ScenarioVisitor {
    fn visit<S: Storage>(&mut self, name: &str, array: &Array<S>, positions: &Positions);
}

/// Smooth 2D field with a little noise, so zstd has real work to do (random data would be stored
/// raw and decode as a memcpy, overstating the share of the allocations).
fn data(shape: &[u64], rng: &mut Rng) -> ndarray::Array2<i32> {
    let (n, m) = (shape[0] as usize, shape[1] as usize);
    ndarray::Array2::from_shape_fn((n, m), |(i, j)| {
        let v = (i as f64 * 0.011).sin() * (j as f64 * 0.007).cos() * 10_000.0;
        v as i32 + rng.i32(0..16)
    })
}

fn compact(data: &ndarray::Array2<i32>, params: CompactParams) -> CompactArray<i32> {
    Array::compact_ndarray_with(data, params).unwrap()
}

pub fn block_shape_of(array: &CompactArray<i32>) -> Vec<u32> {
    array.block_shape().to_vec()
}

/// Calls `visitor` on every scenario, building the arrays on the way.
pub fn for_each_scenario(visitor: &mut impl ScenarioVisitor) {
    let mut rng = Rng::with_seed(0x7a11_0c5c_4a7c_b10c);
    let nd = data(&SHAPE, &mut rng);
    let pos2 = Positions::random(&SHAPE, &mut rng);
    let pos1 = Positions::random(&SHAPE[..1], &mut rng);

    // A plain (uncompressed, in-memory) array: no scratch buffer at all, the fixed per-read cost.
    let plain = Array::plain_ndarray(nd.clone()).unwrap();
    visitor.visit("plain", &plain, &pos2);

    // Default params: the block shape is picked from the L1 data cache size, byte shuffle.
    let a = compact(&nd, CompactParams::new());
    visitor.visit(
        &format!("compact block={:?} (auto)", block_shape_of(&a)),
        &a,
        &pos2,
    );

    for block_shape in [[32u32, 32], [256, 256], [512, 512]] {
        let array = compact(&nd, CompactParams::new().block_shape(&block_shape).unwrap());
        visitor.visit(&format!("compact block={block_shape:?}"), &array, &pos2);
    }

    // All zeros: decoding is nearly free, so this bounds the share the allocations can take.
    let zeros = ndarray::Array2::<i32>::zeros(nd.raw_dim());
    for params in [
        CompactParams::new(),
        CompactParams::new().block_shape(&[32, 32]).unwrap(),
    ] {
        let array = compact(&zeros, params);
        let name = format!("compact block={:?} zeros", block_shape_of(&array));
        visitor.visit(&name, &array, &pos2);
    }

    // No filter: the decoder writes straight to the block buffer, one scratch buffer fewer.
    let no_filter = compact(&nd, CompactParams::new().filters(&[]).unwrap());
    visitor.visit(
        &format!("compact block={:?} no_filter", block_shape_of(&no_filter)),
        &no_filter,
        &pos2,
    );

    // Bit shuffle takes one more scratch buffer from the pool than byte shuffle.
    let bit_shuffle = compact(
        &nd,
        CompactParams::new().filters(&[Filter::BitShuffle]).unwrap(),
    );
    visitor.visit(
        &format!(
            "compact block={:?} bit_shuffle",
            block_shape_of(&bit_shuffle)
        ),
        &bit_shuffle,
        &pos2,
    );

    // A lazy op: both inputs decode a block each, plus the op's staging buffers.
    let b = compact(&nd.mapv(|x| x / 3 + 1), CompactParams::new());
    let add = a.view() + b.view();
    visitor.visit("lazy a + b", &add, &pos2);

    // A reduction: one element of the output reads a whole row of the input.
    let row_sum = a.view().sum(1);
    visitor.visit("lazy a.sum(1)", &row_sum, &pos1);
}
