<!-- banner -->

# jix benchmarks

jix is a multi-dimensional array library with block-compressed storage and lazy operation chains.
This page measures both against the libraries you would otherwise reach for: `ndarray` in Rust,
NumPy, Blosc2 and Zarr in Python.

- **Machines**: GitHub-hosted runners: `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm` (aarch64).

<!-- versions -->
- **Single threaded**: all benchmarks are run with a single thread to ensure a fair comparison across libraries.

- **Codec settings are matched** across jix, Blosc2 and Zarr: zstd level 3, byte-shuffle.
The array is `[130000, 200]` in Rust and Python - 104 MB as `i32/f32`.

`jix` and `blosc2` always mean the byte-shuffled build - the configuration anyone would actually
use. The unfiltered variants appear only in the compression section, where the filter is the thing
being measured. `jix-plain` is jix over an ordinary uncompressed in-memory buffer: it isolates the
cost of jix's operation machinery from the cost of decompression.

---

## Reading a region

Read a random region (of different sizes) out of a large array without decompressing the whole thing.

#### Python

<!-- plot:read -->

- NumPy is the baseline, we read a slice and `.copy()`
- `block 64x200 / read 16x16` means the arrays are compressed with blocks of shape 64x200 each, and we read sub-regions of shape 16x16.
- `blosc2-chunked` is Blosc2 with `chunks` forced equal to `blocks`, and `blosc2` left to choose for itself (which is what we use for other benchmarks).

<!-- table:read -->

---

#### Rust

<!-- plot:rust_read -->

<!-- table:rust_read -->

---

## Compression

#### compression size


<!-- plot:compress_ratio -->

<!-- table:compress_ratio -->

---

#### compression speed

<!-- plot:compress -->

- note that when we use the byte shuffle filter the compression speed *increases*, although we do an additional step, thats because zstd is able to compress better and faster when there are many consecutive similar bytes.

<!-- table:compress -->

---

## Unary Op - Negate

<!-- plot:negate -->

<!-- table:negate -->

---

<!-- plot:negate_dist -->

<!-- table:negate_dist -->

---

## Binary Op - Add

<!-- plot:add -->

<!-- table:add -->

---

## Reductions

<!-- plot:reduction -->

<!-- table:reduction -->

## Rust Ops

<!-- plot:rust_op -->

<!-- table:rust_op -->

---

## Operation chains

Every step is a binary operation over two source arrays, starting from `a*a`.
Every jix operation returns a lazy view, the whole chain is encoded in the type and runs in a
single pass when output is requested. NumPy and `ndarray` make a full pass per step, so the gap
should widen as the chain gets longer.

<!-- plot:chain -->

- jix is faster than numpy here for long chains of operations because of less memory traffic.
numpy allocates an intermediate full array for each step while jix apply the chain sub region
at a time.
- The `exp/log` case, which compute `log(exp(a) + exp(b))`, is an example in which the math op itself is the bottle neck and fusing does not help.

<!-- table:chain -->

---

### Peak memory

Peak RSS memory while ignoring the input and output arrays.

<!-- plot:chain_memory -->

- numpy allocates full intermediate arrays for each step. While jix and blosc operate in chunks.

<!-- table:chain_memory -->

---

### Rust

In rust operations are fused by the compiler.

<!-- plot:rust_chain -->

<!-- table:rust_chain -->

---

## A reduction inside a broadcast

`a / a.std(dim)` - this is a case in which laziness hurt performance pretty badly.
`a / a.std(dim).plain()` could help.

<!-- plot:rust_normalize -->

<!-- table:rust_normalize -->

<!--
The dim-order section is withheld. The benchmark it relied on measured nothing - ndarray's
contiguity check sorts strides, so its permuted views took the fast path - and the fixed version
has not been run yet. Restore this section, with
<!- - plot:rust_dim_order - -> and <!- - table:rust_dim_order - ->, once a run includes it.
-->

## Reproducing

    python jix/benches/run.py --report          # rust, only the comparison benches
    python jix-py/python/benches/run_all.py     # python

The CI matrix across the two runners is documented in [`CI.md`](../CI.md), along with
`scripts/bench/compare.py` for same-machine A/B against a base ref. Every plot and every table on
this page is generated from the benchmark JSON by `benches/report_bars.py`; nothing is typed by
hand.
