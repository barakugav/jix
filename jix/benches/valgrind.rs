//! Instruction-count benchmarks run under Valgrind by Gungraun, for the pr_report CI job.
//!
//! Counts are stable between runs, unlike wall-clock times, so a base and head run on the same
//! machine can be compared directly. Needs Linux, valgrind and `gungraun-runner` matching the
//! `gungraun` version in Cargo.toml.

mod common;

use std::hint::black_box;
use std::ops::Neg;

use gungraun::{library_benchmark, library_benchmark_group, main};
use jix::storage::CompactParams;
use jix::{Array, CompactArray, PlainArray};
use ndarray::Ix2;

use crate::common::{Profile, create_data};

const SHAPE: [u64; 2] = [256, 256];
const ITEM: [u64; 2] = [100, 100];

fn compact() -> CompactArray<i32> {
    let data = create_data::<i32>(Profile::Smooth, &SHAPE, 0x5cefcbc2fbd7e8c0);
    let params = CompactParams::new().block_shape(&[30, 30]).unwrap();
    Array::compact_ndarray_with(&data, params).unwrap()
}

fn plain() -> PlainArray<i32> {
    let data = create_data::<i32>(Profile::Smooth, &SHAPE, 0x5cefcbc2fbd7e8c0);
    Array::plain_ndarray(data).unwrap()
}

fn plain_pair() -> (PlainArray<i32>, PlainArray<i32>) {
    let a = create_data::<i32>(Profile::Smooth, &SHAPE, 0xcb06da9be056e350);
    let b = create_data::<i32>(Profile::Smooth, &SHAPE, 0x1eb06f0c81f140f3);
    (
        Array::plain_ndarray(a).unwrap(),
        Array::plain_ndarray(b).unwrap(),
    )
}

#[library_benchmark]
#[bench::default(setup = compact)]
fn read_compact_item(a: CompactArray<i32>) -> i32 {
    black_box(a.get(&ITEM).unwrap())
}

#[library_benchmark]
#[bench::default(setup = compact)]
fn read_compact_full(a: CompactArray<i32>) -> ndarray::Array2<i32> {
    black_box(a.to_ndarray::<Ix2>().unwrap())
}

#[library_benchmark]
#[bench::default(setup = plain)]
fn read_plain_item(a: PlainArray<i32>) -> i32 {
    black_box(a.get(&ITEM).unwrap())
}

#[library_benchmark]
#[bench::default(setup = plain)]
fn read_plain_full(a: PlainArray<i32>) -> ndarray::Array2<i32> {
    black_box(a.to_ndarray::<Ix2>().unwrap())
}

#[library_benchmark]
#[bench::default(setup = plain)]
fn op1_neg_plain_item(a: PlainArray<i32>) -> i32 {
    black_box(a.view().neg().get(&ITEM).unwrap())
}

#[library_benchmark]
#[bench::default(setup = plain)]
fn op1_neg_plain_full(a: PlainArray<i32>) -> ndarray::Array2<i32> {
    black_box(a.view().neg().to_ndarray::<Ix2>().unwrap())
}

#[library_benchmark]
#[bench::default(setup = plain_pair)]
fn op2_add_plain_item((a, b): (PlainArray<i32>, PlainArray<i32>)) -> i32 {
    black_box((a.view() + b.view()).get(&ITEM).unwrap())
}

#[library_benchmark]
#[bench::default(setup = plain_pair)]
fn op2_add_plain_full((a, b): (PlainArray<i32>, PlainArray<i32>)) -> ndarray::Array2<i32> {
    black_box((a.view() + b.view()).to_ndarray::<Ix2>().unwrap())
}

library_benchmark_group!(
    name = core,
    benchmarks = [
        read_compact_item,
        read_compact_full,
        read_plain_item,
        read_plain_full,
        op1_neg_plain_item,
        op1_neg_plain_full,
        op2_add_plain_item,
        op2_add_plain_full
    ]
);

main!(library_benchmark_groups = core);
