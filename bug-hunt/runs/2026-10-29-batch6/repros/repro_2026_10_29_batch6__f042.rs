// F042 repro: the float sum of the same data (or of the same .jix bytes) depends on the CPU cache
// sizes the host reports.
//
// `ArrayParams::tune` (jix/src/storage/params.rs:366-367, 452-459) defaults
//   block_size = L1d
//   read_size  = (max(L1d / 2, L2 / 16, block_bytes), L2 / 2)
// from `cpu_cache::cache_sizes()` (sysfs / sysctl, process-global). read_size is not stored in the
// archive, so every reader recomputes it from its own host. The reduction driver cuts the reduced
// stream into tiles sized by `read_shape_heuristic` (read_size), folds each tile pairwise and merges
// tiles sequentially, so the f32 rounding depends on the host.
//
// One process sees one host, so the test emulates the other hosts by passing the exact params
// `tune` would derive on them. `bug-hunt/runs/2026-10-29-batch6/repros/F042.py` does the same with
// the real detection (fake sysfs cache sizes in a private mount namespace) and shows that the
// default-params result on each emulated host equals the row below.
//
// The test asserts the CORRECT behavior (the result does not depend on the host), so it fails
// while the bug exists.

use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::{Array1, Ix0};

const K: u64 = 1024;
const N: usize = 3_000_000;

/// (name, L1d, L2) of a few common hosts.
const HOSTS: [(&str, u64, u64); 4] = [
    ("L1d 48K, L2 2M (e.g. Intel Golden Cove)", 48 * K, 2048 * K),
    ("L1d 32K, L2 512K (e.g. Zen3)", 32 * K, 512 * K),
    ("L1d 32K, L2 1M (e.g. Zen4)", 32 * K, 1024 * K),
    ("L1d 128K, L2 12M (e.g. Apple M1)", 128 * K, 12 * 1024 * K),
];

/// The read_size `ArrayParams::tune` derives for a host (params.rs:453-459).
fn host_read_size(l1: u64, l2: u64, block_bytes: u64) -> (u64, u64) {
    let min = (l1 / 2).max(l2 / 16).max(block_bytes);
    (min, (l2 / 2).max(min))
}

/// Deterministic, roughly normal f32 data (sum of 4 uniforms, centered).
fn data() -> Array1<f32> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    Array1::from_shape_fn(N, |_| next() + next() + next() + next() - 2.0)
}

fn sum_f32<S: jix::storage::ArrayStorageTyped<Item = f32>>(a: Array<S>) -> f32 {
    a.sum(0usize).to_ndarray::<Ix0>().unwrap()[[]]
}

fn report(rows: &[(String, f32)]) -> bool {
    let first = rows[0].1;
    for (name, v) in rows {
        println!("{name:<55} sum = {v:?} (bits {:#010x})", v.to_bits());
    }
    rows.iter().all(|(_, v)| v.to_bits() == first.to_bits())
}

/// The same .jix bytes (block shape fixed by the file), read on different hosts.
#[test]
fn same_file_sums_identically_on_every_host() {
    let x = data();
    // A fixed 48 KiB block (12288 f32), as written on a host with L1d = 48K.
    const BLOCK: u32 = 12288;
    let mut wp = ArrayParams::new();
    wp.block_shape(&[BLOCK]);
    let a = Array::compact_ndarray_with(&x, wp).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    let bytes = buf.into_inner();
    let read = |params: ArrayParams| {
        Array::read_from_reader(Cursor::new(bytes.clone()), None, params, ArchiveValidation::default())
            .unwrap()
            .into_typed::<f32>()
            .unwrap()
    };
    // With a fixed block shape, tune's block_size is the block's byte size.
    let block_bytes = BLOCK as u64 * 4;

    let mut rows = vec![("default params on this host".to_string(), sum_f32(read(ArrayParams::default())))];
    for (name, l1, l2) in HOSTS {
        let mut p = ArrayParams::new();
        p.read_size(host_read_size(l1, l2, block_bytes));
        rows.push((format!("file read on {name}"), sum_f32(read(p))));
    }
    let exact: f64 = x.iter().map(|&v| v as f64).sum();
    println!("exact (f64) = {exact:?}");
    assert!(
        report(&rows),
        "the sum of the same .jix file depends on the reader's CPU cache sizes"
    );
}

/// The same in-memory data (Plain view, as `jix.asarray` builds it) summed on different hosts.
#[test]
fn plain_sum_identical_on_every_host() {
    let x = data();
    let mut rows = vec![(
        "default params on this host".to_string(),
        sum_f32(Array::plain_ndarray_ref(&x).unwrap()),
    )];
    for (name, l1, l2) in HOSTS {
        let mut p = ArrayParams::new();
        p.block_size(l1);
        // With block_size = L1d, tune floors read_size.min at block_size = L1d.
        p.read_size(host_read_size(l1, l2, l1));
        rows.push((format!("plain on {name}"), sum_f32(Array::plain_ndarray_ref_with(&x, p).unwrap())));
    }
    assert!(report(&rows), "the sum of the same data depends on the host's CPU cache sizes");
}
