//! Scratch probe for big-scale reads of `Array<Compact>` (experimental branch, not a real bench).
//!
//! Builds one array, then repeats one read kind for `--secs` seconds and prints min/median time
//! and throughput. One process = one case, so a driver can interleave two builds A/B/A/B.
//!
//! ```text
//! cargo run --release --all-features --example probe_read -- \
//!     --dtype i32 --profile smooth --shape 8192x8192 [--block 64x64] --read full --secs 3
//! ```
//!
//! Read kinds:
//! - `full`  whole array into a preallocated packed buffer (no allocation / page faults timed)
//! - `nd`    whole array through `to_ndarray` (allocates the destination every iteration)
//! - `sub`   one big region, unaligned to the blocks in every dim
//! - `col`   the last dim narrowed to a single index
//! - `row`   the first dim narrowed to a single index
//! - `tiles` the whole array in many aligned 1/16-per-dim tile reads
//! - `sum`   `sum` over all dims (pull-mode tiles through a reduction)
//! - `neg`   `neg()` view into a preallocated buffer (pull-mode tiles through an elementwise op)
//! - `t`     transposed view into a preallocated C-order buffer
//! - `mmap`  like `full`, from a memory-mapped file (warm page cache)

#![allow(clippy::all)] // scratch probe

use std::ops::Range;
use std::time::{Duration, Instant};

use std::ops::Neg;

use jix::{Array, ArrayParams, ReadContext};

struct Args {
    dtype: String,
    profile: String,
    shape: Vec<u64>,
    block: Option<Vec<u32>>,
    read: String,
    secs: f64,
    level: Option<i32>,
    label: String,
}

fn parse_dims<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split('x').map(|p| p.parse().unwrap()).collect()
}

fn parse_args() -> Args {
    let mut args = Args {
        dtype: "i32".into(),
        profile: "smooth".into(),
        shape: vec![8192, 8192],
        block: None,
        read: "full".into(),
        secs: 3.0,
        level: None,
        label: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .unwrap_or_else(|| panic!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--dtype" => args.dtype = val(),
            "--profile" => args.profile = val(),
            "--shape" => args.shape = parse_dims(&val()),
            "--block" => args.block = Some(parse_dims(&val())),
            "--read" => args.read = val(),
            "--secs" => args.secs = val().parse().unwrap(),
            "--level" => args.level = Some(val().parse().unwrap()),
            "--label" => args.label = val(),
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

/// xorshift64*
struct Rng(u64);
impl Rng {
    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in `[0, 1)`.
    #[inline(always)]
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A smooth 1D signal over the flat index, in roughly `[-1000, 1250]`.
#[inline(always)]
fn smooth(i: usize) -> f64 {
    let t = 8.0 * std::f64::consts::PI * (i as f64) / 10000.0;
    let grad = 0.25 * ((i % 10_000_000) as f64) / 10000.0;
    (t.sin() + grad) * 1000.0
}

trait Elem: jix::dtype::Dtyped + Copy + Default + PartialEq + std::fmt::Debug {
    fn from_bits(bits: u64) -> Self;
    fn from_f64(x: f64) -> Self;
    const IS_FLOAT: bool;
}
macro_rules! impl_elem_int {
    ($($t:ty),*) => {$(
        impl Elem for $t {
            #[inline(always)]
            fn from_bits(bits: u64) -> Self { bits as $t }
            #[inline(always)]
            fn from_f64(x: f64) -> Self { x.round() as $t }
            const IS_FLOAT: bool = false;
        }
    )*};
}
impl_elem_int!(u8, u16, i16, i32, i64);
impl Elem for f32 {
    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    const IS_FLOAT: bool = true;
}
impl Elem for f64 {
    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x
    }
    const IS_FLOAT: bool = true;
}

fn gen_data<T: Elem>(profile: &str, n: usize, seed: u64) -> Vec<T> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::with_capacity(n);
    match profile {
        // Uniform random bits; incompressible.
        "random" => out.extend((0..n).map(|_| T::from_bits(rng.next()))),
        // Smooth field. Ints round to a slowly varying sequence; floats keep a noisy mantissa.
        "smooth" => out.extend((0..n).map(|i| T::from_f64(smooth(i)))),
        // Smooth field quantized to 1/8 steps plus a little noise: "measurements".
        "noisy" => out.extend((0..n).map(|i| {
            let x = smooth(i) + rng.unit() * 16.0;
            T::from_f64(if T::IS_FLOAT {
                (x * 8.0).round() / 8.0
            } else {
                x
            })
        })),
        // A handful of distinct small values.
        "lowent" => out.extend((0..n).map(|_| T::from_f64((rng.next() % 8) as f64))),
        // Mostly zeros with a few random small values.
        "sparse" => out.extend((0..n).map(|_| {
            let r = rng.next();
            T::from_f64(if r % 20 == 0 {
                ((r >> 8) % 1000) as f64
            } else {
                0.0
            })
        })),
        // All zeros.
        "zero" => out.resize(n, T::default()),
        other => panic!("unknown profile {other}"),
    }
    out
}

fn as_bytes_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), size_of_val(v)) }
}

/// Repeat `f` for `secs` (at least 3 times) and return the sorted per-iteration times.
fn measure(secs: f64, mut f: impl FnMut()) -> Vec<Duration> {
    f(); // warm up
    let mut times = Vec::new();
    let start = Instant::now();
    while times.len() < 3 || start.elapsed().as_secs_f64() < secs {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    times.sort();
    times
}

macro_rules! run_body {
    ($args:ident, $T:ty) => {{
    let args = $args;
    #[allow(dead_code)] type T = $T;
    let shape = &args.shape;
    let ndim = shape.len();
    let nitems = shape.iter().product::<u64>() as usize;
    let itemsize = size_of::<T>();

    let t0 = Instant::now();
    let data = gen_data::<T>(&args.profile, nitems, 0x9E3779B97F4A7C15);
    let dims = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
    let nd = ndarray::ArrayD::from_shape_vec(dims, data).unwrap();
    let t_gen = t0.elapsed();

    let mut params = ArrayParams::new();
    if let Some(block) = &args.block {
        params = params.block_shape(block).unwrap();
    }
    if let Some(level) = args.level {
        params = params.level(level).unwrap();
    }
    let t0 = Instant::now();
    let array = Array::compact_ndarray_with(&nd, params).unwrap();
    let t_compact = t0.elapsed();

    // Compressed size, via the archive writer.
    let mut archive = Vec::new();
    array
        .write_to(
            std::io::Cursor::new(&mut archive),
            ArrayParams::default(),
            &array.read_ctx(),
        )
        .unwrap();
    let ratio = archive.len() as f64 / (nitems * itemsize) as f64;

    let full: Vec<Range<u64>> = shape.iter().map(|&s| 0..s).collect();
    let ctx = array.read_ctx();

    // (region read, element count) per iteration, and the closure to time.
    let read_region = |ctx: &ReadContext, region: &[Range<u64>], buf: &mut Vec<T>| {
        let n = region.iter().map(|r| r.end - r.start).product::<u64>() as usize;
        array
            .to_buf_slice(region, as_bytes_mut(&mut buf[..n]), ctx)
            .unwrap();
    };
    let mut buf: Vec<T> = vec![T::default(); nitems];

    let (out_items, times): (usize, Vec<Duration>) = match args.read.as_str() {
        "full" => {
            let times = measure(args.secs, || read_region(&ctx, &full, &mut buf));
            assert_eq!(&buf[..], nd.as_slice().unwrap(), "full read mismatch");
            (nitems, times)
        }
        "nd" => {
            let times = measure(args.secs, || {
                let out = array.to_ndarray::<ndarray::IxDyn>().unwrap();
                std::hint::black_box(&out);
            });
            (nitems, times)
        }
        "sub" => {
            let region: Vec<Range<u64>> = shape
                .iter()
                .map(|&s| (s / 7).min(s - 1)..(s - s / 9).max(s / 7 + 1).min(s))
                .collect();
            let n = region.iter().map(|r| r.end - r.start).product::<u64>() as usize;
            let times = measure(args.secs, || read_region(&ctx, &region, &mut buf));
            let slice_info = region
                .iter()
                .map(|r| ndarray::SliceInfoElem::from(r.start as usize..r.end as usize))
                .collect::<Vec<_>>();
            let expected = nd.slice(slice_info.as_slice());
            assert!(
                expected.iter().zip(&buf[..n]).all(|(a, b)| a == b),
                "sub read mismatch"
            );
            (n, times)
        }
        "col" | "row" => {
            let dim = if args.read == "col" { ndim - 1 } else { 0 };
            let mut region = full.clone();
            let k = shape[dim] / 3;
            region[dim] = k..k + 1;
            let n = region.iter().map(|r| r.end - r.start).product::<u64>() as usize;
            let times = measure(args.secs, || read_region(&ctx, &region, &mut buf));
            let expected = nd.index_axis(ndarray::Axis(dim), k as usize);
            assert!(
                expected.iter().zip(&buf[..n]).all(|(a, b)| a == b),
                "{} read mismatch",
                args.read
            );
            (n, times)
        }
        "tiles" => {
            let tile: Vec<u64> = shape.iter().map(|&s| (s / 16).max(1)).collect();
            let mut regions: Vec<Vec<Range<u64>>> = vec![vec![]];
            for dim in 0..ndim {
                regions = regions
                    .into_iter()
                    .flat_map(|r| {
                        let (len, step) = (shape[dim], tile[dim]);
                        (0..len.div_ceil(step)).map(move |i| {
                            let mut r = r.clone();
                            r.push(i * step..((i + 1) * step).min(len));
                            r
                        })
                    })
                    .collect();
            }
            let times = measure(args.secs, || {
                for region in &regions {
                    read_region(&ctx, region, &mut buf);
                }
            });
            (nitems, times)
        }
        "sum" => {
            let dims: Vec<usize> = (0..ndim).collect();
            let times = measure(args.secs, || {
                let out = array.view().sum(dims.as_slice()).to_ndarray::<ndarray::Ix0>().unwrap();
                std::hint::black_box(&out);
            });
            (nitems, times)
        }
        "neg" => {
            let view = array.view().neg();
            let vctx = view.read_ctx();
            let times = measure(args.secs, || {
                view.to_buf_slice(&full, as_bytes_mut(&mut buf[..]), &vctx)
                    .unwrap();
            });
            (nitems, times)
        }
        "t" => {
            let perm: Vec<usize> = (0..ndim).rev().collect();
            let view = array.view().permute_dims(&perm);
            let vctx = view.read_ctx();
            let tfull: Vec<Range<u64>> = full.iter().rev().cloned().collect();
            let times = measure(args.secs, || {
                view.to_buf_slice(&tfull, as_bytes_mut(&mut buf[..]), &vctx)
                    .unwrap();
            });
            let expected = nd.view().reversed_axes();
            assert!(
                expected.iter().zip(&buf[..]).all(|(a, b)| a == b),
                "transposed read mismatch"
            );
            (nitems, times)
        }
        "mmap" => {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("probe.jix");
            std::fs::write(&path, &archive).unwrap();
            let mapped = unsafe {
                Array::read_from_file_mmap(
                    &path,
                    0,
                    archive.len() as u64,
                    ArrayParams::default(),
                    jix::ArchiveValidation::default(),
                )
                .unwrap()
            };
            let mctx = mapped.read_ctx();
            let times = measure(args.secs, || {
                mapped
                    .to_buf_slice(&full, as_bytes_mut(&mut buf[..]), &mctx)
                    .unwrap();
            });
            assert_eq!(&buf[..], nd.as_slice().unwrap(), "mmap read mismatch");
            (nitems, times)
        }
        other => panic!("unknown read kind {other}"),
    };

    let min = times[0].as_secs_f64();
    let med = times[times.len() / 2].as_secs_f64();
    let out_mb = (out_items * itemsize) as f64 / 1e6;
    let src_mb = (nitems * itemsize) as f64 / 1e6;
    println!(
        "{label:<6} {dtype:<4} {profile:<7} {shape:<14} blk={block:<16} {read:<6} ratio={ratio:.3} \
         min={min_ms:9.3}ms med={med_ms:9.3}ms n={n:<4} out={out_mbs:8.1}MB/s arr={arr_mbs:8.1}MB/s \
         (gen {gen:.1}s compact {compact:.1}s)",
        label = args.label,
        dtype = args.dtype,
        profile = args.profile,
        shape = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"),
        block = match &args.block {
            Some(b) => b.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"),
            None => "auto".to_string(),
        },
        read = args.read,
        min_ms = min * 1e3,
        med_ms = med * 1e3,
        n = times.len(),
        out_mbs = out_mb / min,
        arr_mbs = src_mb / min,
        gen = t_gen.as_secs_f64(),
        compact = t_compact.as_secs_f64(),
    );
}};
}

macro_rules! define_run {
    ($name:ident, $t:ty) => {
        fn $name(args: &Args) {
            run_body!(args, $t);
        }
    };
}

define_run!(run_i16, i16);
define_run!(run_i32, i32);
define_run!(run_i64, i64);
define_run!(run_f32, f32);
define_run!(run_f64, f64);

fn main() {
    let args = parse_args();
    match args.dtype.as_str() {
        "i16" => run_i16(&args),
        "i32" => run_i32(&args),
        "i64" => run_i64(&args),
        "f32" => run_f32(&args),
        "f64" => run_f64(&args),
        other => panic!("unsupported dtype {other}"),
    }
}
