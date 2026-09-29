# jix-probe: static asm analysis of the filter kernels and the element-wise pipeline

Experimental tooling for optimizing the filter kernels in
`jix/src/codec/filter/{byte_shuffle,bit_shuffle}/kernels.rs` across platforms **without running
benchmarks**. The objective is llvm-mca's steady-state throughput of each kernel's hot loops, on
the targeted platforms. Byte-shuffle decode was optimized first and is frozen (`results/fearless`);
encode followed (`--direction encode`, `results/encode-*`). The bit-shuffle bit transpose is analyzed
with `--direction bit` (`results/bit-*`). The element-wise pipeline's inner loop is analyzed with
`--direction elementwise` (`results/elementwise-*`), from a second probe crate, see the last section.

## Why a separate crate and not a `jix` example

A `jix` example would build all of `jix`'s dependencies for the target, including `zstd-sys` (C),
which needs a C cross toolchain and sysroot for every target (and a macOS SDK for
`aarch64-apple-darwin`). Instead, the kernels live in self-contained files (`core`/`std` +
`fearless_simd` only). `jix` uses them as modules, and this crate `#[path]`-includes them. So the probe compiles the exact code `jix` ships, for any rustup target, in
about 2 seconds, with no linker or C compiler. Keep the `kernels.rs` files free of `crate::` dependencies.

## Setup

- Toolchain: `rust-toolchain.toml` at the repo root pins `nightly-2026-09-27` (rustc 1.101,
  **LLVM 23.1.1**), with the `llvm-tools` component (provides `opt`) and all cross targets.
- `cargo install cargo-show-asm` (provides `cargo asm`).
- llvm-mca **23**, matching rustc's LLVM. rustup's llvm-tools does not ship llvm-mca. apt.llvm.org
  is blocked in the cloud sandbox, so take it from the official release tarball:

  ```bash
  curl -sSL -o llvm.tar.xz \
      https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.1/LLVM-23.1.1-Linux-X64.tar.xz
  tar -xJf llvm.tar.xz LLVM-23.1.1-Linux-X64/bin/llvm-mca   # statically linked, ~2GB download
  sudo install LLVM-23.1.1-Linux-X64/bin/llvm-mca /usr/local/bin/llvm-mca-23
  ```

  `analyze.py` looks at `$LLVM_MCA`, then `llvm-mca-<rustc LLVM major>`, then `llvm-mca`, and
  refuses to run with a version mismatch.

## Usage

```bash
python jix/probe/analyze.py                         # decode, all platforms, results/baseline/
python jix/probe/analyze.py --label my-variant      # results/my-variant/
python jix/probe/analyze.py --direction bit --label x --compare bit-fearless
python jix/probe/analyze.py --direction elementwise --label x --compare elementwise-baseline
python jix/probe/analyze.py --platform x86_64-v3 --platform aarch64 --fn decode_4
cd jix/probe && cargo asm --release --lib --target aarch64-unknown-linux-gnu   # list symbols
```

A full run takes about 20 seconds. Outputs go to `results/<label>/`:
- `summary.md` / `summary.json`: the tables.
- `asm/<platform>/<kernel>.s`: the full function asm (gitignored, like `mca/`).
- `mca/...`: the raw llvm-mca reports, with resource pressure and bottleneck analysis
  (gitignored).

## Method

Platforms (see `PLATFORMS` in `analyze.py`):

| platform | target | ISA level | llvm-mca CPUs |
|---|---|---|---|
| x86_64-v2 | x86_64-unknown-linux-gnu | fearless_simd `Sse4_2` level features | sandybridge, btver2 |
| x86_64-v3 | x86_64-unknown-linux-gnu | fearless_simd `Avx2` level features | skylake, alderlake, znver3 |
| x86_64-v4 | x86_64-unknown-linux-gnu | fearless_simd `Avx512` level features (Ice Lake: includes VBMI) | icelake-server, sapphirerapids, znver4 |
| i686 | i686-unknown-linux-gnu | SSE2 (the target baseline; fearless_simd `Sse2`) | skylake |
| aarch64 | aarch64-unknown-linux-gnu | armv8-a + NEON | cortex-a72, neoverse-n1, neoverse-v2 |
| aarch64-apple | aarch64-apple-darwin | apple-m1 | apple-m1 |

x86-64 v1 (SSE2 only), armv7 and ppc64le are out of scope. Windows and musl wheels share the ISA
of their linux counterparts. `jix` runs the decode main loop through `fearless_simd::dispatch!`,
which calls it inside a `#[target_feature(enable = ...)]` function of the detected level (generic
tuning). The probe reproduces each dispatch arm by building with the same features in
`-C target-feature` and passing the matching token (`simd()` in `src/lib.rs`). The baselines
(`baseline`, `encode-baseline`, `bit-*-baseline`) were produced the same way, but with the
feature sets of the former `multiversion` x86-64-v2/v3/v4 clones (`--x86-levels multiversion`).

Compare a run with a previous one with `--compare <label>`, which adds a geomean speedup column.

For each (platform, kernel):
1. `cargo asm --simplify` gives the asm, and the loop tree comes from back-edge branches.
2. `cargo asm --llvm` gives the final LLVM IR. The toolchain's own `opt` prints the IR loop tree
   (`print<loops>`) and exact trip counts (`print<scalar-evolution>`).
3. The trees are matched by shape, which gives every asm loop its trip count.
4. The hot loops are the loops whose total iteration count (their trip count times their
   ancestors', from SCEV evaluated at two concrete input lengths) is linear in the input length:
   the byte-shuffle main loop, or the vectorized loop over groups inside a bit-shuffle pass's loop
   over byte planes. Tails and loops over the itemsize do not scale and are ignored. The slope gives
   the bytes per iteration (`B/iter`), which is measured, not assumed, so a kernel can use any step.
5. One iteration of each hot loop is flattened: inner loops that LLVM did not unroll are repeated
   by their trip count. llvm-mca simulates that trace (100 iterations) on each CPU.

**Metric: cycles per 4096 bytes = sum over the hot loops of cycles per iteration * 4096 / B/iter.
Lower is better.** The per-(platform, CPU) geomean over the kernels (itemsize 2/4/8/16) is the
headline number. The unit is only a scale. What makes kernels with different steps comparable is
that `B/iter` is measured.

SCEV trip counts are evaluated with the arguments bound (input length, and `typesize` / `n_full`
for the runtime-itemsize bit-shuffle passes). Values SCEV cannot see through (e.g. a scalar
extracted from SLP-vectorized setup code) are constant-folded from the IR. The auto-vectorized
bit-shuffle loops also depend on non-argument values, which are approximated, and their `B/iter`
is marked `~`: bounds-check limits in a `umin` are dropped (they do not bind in the steady state),
a remainder size SCEV proves small is taken as 0, and an enclosing loop's add recurrence is
evaluated at its first iteration.

Caveats:
- llvm-mca assumes every load hits L1 and ignores the front end. Branches are not followed.
- **Calls are free in llvm-mca.** Kernels that call something (e.g. `memcpy`) inside the hot loop
  are flagged `*`, and their real cost is higher.
- Steady state of the hot loops only, by design. The per-call prologue and epilogue and the
  tails (`decode_impl_generic`, vector-loop remainders) are neither counted nor analyzed.
- aarch64 stores with base-register writeback are split into a store and an `add` before
  llvm-mca (see the bit-shuffle section).
- Instructions with no scheduling info on a CPU are skipped and listed as warnings (none so far).

## Baseline (auto-vectorized kernels)

Full tables: [`results/baseline/summary.md`](results/baseline/summary.md). Cycles per 4096 bytes:

| platform | cpu | geomean | decode_2 | decode_4 | decode_8 | decode_16 |
|---|---|---:|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | **980** | 387 | 677 | 811 | 4355 |
| x86_64-v2 | btver2 | **1993** | 563 | 1860 | 2307 | 6532 |
| x86_64-v3 | skylake | **1474** | 388 | 516 | 4099 | 5765 |
| x86_64-v3 | alderlake | **871** | 264 | 306 | 2071 | 3435 |
| x86_64-v3 | znver3 | **1078** | 327 | 388 | 4099 | 2603 |
| x86_64-v4 | icelake-server | **634** | 389 | 415 | 964 | 1035 |
| x86_64-v4 | sapphirerapids | **721** | 393 | 394 | 905 | 1926 |
| x86_64-v4 | znver4 | **547** | 388 | 356 | 838 | 775 |
| i686 | skylake | **2330*** | 259 | 4099 | 4100 | 6768* |
| aarch64 | cortex-a72 | **2140** | 803 | 1669 | 2409 | 6503 |
| aarch64 | neoverse-n1 | **1784** | 801 | 1601 | 1604 | 4930 |
| aarch64 | neoverse-v2 | **713** | 243 | 616 | 649 | 2662 |
| aarch64-apple | apple-m1 | **557** | 259 | 599 | 584 | 1063 |

`*` = the hot loop calls a function llvm-mca does not simulate. Here that is only i686 `decode_16`,
which calls `memcpy@PLT` (128 bytes) once per outer iteration, to copy the block it transposed in a
stack buffer out to `dst`. The real cost is higher than shown.

Observations:
- Every kernel steps 128 B/iter (`ITEMSIZE * LANES`). v3 uses ymm and v4 uses zmm in places, but
  one 128-byte iteration is only two 512-bit stores. So the step, not the register width, limits
  wide vectors. A 512-bit variant should use a larger step, which the metric now supports.
- `decode_2` is vectorized everywhere (`punpcklbw` / `zip1`), at 250-400 cycles per 4096 bytes
  (800 on Cortex-A72 / Neoverse N1).
- `decode_4` / `decode_8` are vectorized on some levels only. On x86-64-v3, `decode_8` falls back
  to 128 scalar byte `movzbl`+`movb` pairs per 128 bytes (4099 on Skylake / Zen 3, store-port
  bound). The v2 build of the same code is vectorized (811 on Sandy Bridge). i686 `decode_4/8` are
  scalar too.
- `decode_16` is the worst kernel everywhere except Zen 4. On x86-64-v2 and i686, the
  `for b in 0..16` loop is not unrolled (`loops = ?[16]`) and the data round-trips through the
  stack. Elsewhere it is unrolled, but it transposes through long `tbl` (NEON) or shuffle chains.
- Store bandwidth sets the floor: 4096 bytes need 64 x 512-bit, 128 x 256-bit or 256 x 128-bit
  stores (the same number of loads). A clean transpose should land around 100-300 cycles per
  4096 bytes with 128-bit vectors, and below 100 with 512-bit ones. That is 3-20x better than the
  current geomeans.
- `LANES` is honored: one iteration handles `ITEMSIZE * LANES` bytes, and `decode()` picks
  `LANES = 128 / ITEMSIZE`. A one-off sweep of 256/512/1024-byte steps (not committed) confirmed
  the measured `B/iter` tracks `ITEMSIZE * LANES`. For most kernels it made the auto-vectorized
  code worse: LLVM starts calling `memcpy`/`memset` for the `elms` buffer, or it falls back to
  scalar. Only x86-64-v4 gained: `decode_2` by 30-40% and `decode_8` by 7-22%.
- i686: the `i686-*` targets enable SSE2 (128-bit integer SIMD, but no SSSE3 `pshufb`). The
  `multiversion` targets are x86_64-only, so i686 never gets wider clones.

## fearless_simd kernel (`results/fearless`)

`decode_simd` in `kernels.rs` is generic over the fearless_simd level. Each chunk loads one
native vector (`S::u8s`) per byte plane, then transposes them with log2(ITEMSIZE) rounds of the
portable `interleave` (a perfect shuffle: plane `j` with plane `j + ITEMSIZE / 2`), and stores
ITEMSIZE vectors. Each loop iteration processes chunks until it covers at least 256 bytes. The
`<LANES` tail still goes to `decode_impl_generic`.

| platform | cpu | geomean | vs baseline | decode_2 | decode_4 | decode_8 | decode_16 |
|---|---|---:|---:|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | **429** | 2.29x | 258 | 338 | 513 | 754 |
| x86_64-v2 | btver2 | **761** | 2.62x | 473 | 665 | 858 | 1243 |
| x86_64-v3 | skylake | **569** | 2.59x | 258 | 514 | 770 | 1025 |
| x86_64-v3 | alderlake | **304** | 2.86x | 132 | 292 | 405 | 547 |
| x86_64-v3 | znver3 | **300** | 3.59x | 130 | 275 | 403 | 562 |
| x86_64-v4 | icelake-server | **148** | 4.28x | 74 | 131 | 193 | 257 |
| x86_64-v4 | sapphirerapids | **287** | 2.51x | 132 | 260 | 386 | 513 |
| x86_64-v4 | znver4 | **144** | 3.80x | 67 | 131 | 194 | 257 |
| i686 | skylake | **602** | 3.87x | 258 | 514 | 771 | 1284 |
| aarch64 | cortex-a72 | **1080** | 1.98x | 705 | 993 | 1236 | 1572 |
| aarch64 | neoverse-n1 | **699** | 2.55x | 576 | 624 | 753 | 881 |
| aarch64 | neoverse-v2 | **270** | 2.64x | 209 | 273 | 290 | 322 |
| aarch64-apple | apple-m1 | **244** | 2.29x | 194 | 203 | 260 | 345 |

Every kernel is faster than its baseline on every CPU. Findings while tuning on Ice Lake:
- On AVX-512 (VBMI), `interleave` is two `vpermt2b`, so the loop is port-5 bound at
  log2(ITEMSIZE) shuffles per 64 output bytes: 64 / 128 / 192 / 256 cycles per 4096 bytes. Ice
  Lake and Zen 4 reach that bound. `decode_2` is near the 64-cycle store floor.
- Bounds checks: safe slicing left 1-2 compare+branch pairs per plane in the loop (LLVM does not
  hoist them), which also hid the trip count from SCEV. The loads and stores use
  `get_unchecked` with one SAFETY argument.
- `array::from_fn` in the hot loop was not inlined (a call per iteration). Plain loops over a
  splat-initialized array are used instead.
- Loop form: a counted `for` over chunks beat `while i + lanes <= nitems` (`decode_2` 100 -> 77).
  Manual unrolling gains only about 3% on AVX-512. On NEON (16-byte vectors, 32 B/iter for
  `decode_2`) it matters much more, so the loop is unrolled to at least 256 bytes per iteration
  (sweep: 64/128/256/512 bytes; 512 starts to hurt `decode_16`).
- A fixed `u8x64` type on every level instead of the native one helped small itemsizes on
  128-bit ISAs, but hurt `decode_16` badly (register spills, and `memcpy` on i686) and AVX2 in
  general, so it was dropped in favor of the unroll.
- Sapphire Rapids is 2x behind Ice Lake only because llvm-mca's model gives `vpermt2b` a
  throughput of 2 cycles there.
- AVX2: fearless_simd's `interleave_u8x32` is `vpunpckl/hbw` plus two lane-fixup `vperm2i128`,
  i.e. 2 shuffles per output vector per round. Avoiding the fixups needs in-lane unpacks, which
  the portable API does not expose.

### Round 2 experiments (kernel unchanged except the const `shuffle_steps`)

- `shuffle_steps = const { assert!(ITEMSIZE.is_power_of_two()); ITEMSIZE.ilog2() }`: identical asm
  on all 24 (platform, itemsize) functions.
- Unroll, per-CPU geomean (cycles / 4096 B), `unroll = clamp(CAP / (ITEMSIZE * lanes), 1, U)`:

  | cpu | U=1 | U=2 | U=4 | U=4 cap 256 | U=4 cap 512 | U=4 cap 1024 | cap 256 (kept) | cap 512 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | x86_64-v2/sandybridge | 471 | 448 | 432 | 429 | 437 | 432 | 429 | 432 |
  | x86_64-v2/btver2 | 853 | 824 | 798 | 771 | 813 | 798 | 761 | 793 |
  | x86_64-v3/skylake | 573 | 570 | 568 | 569 | 568 | 568 | 569 | 568 |
  | x86_64-v3/alderlake | 313 | 306 | 303 | 304 | 303 | 303 | 304 | 302 |
  | x86_64-v3/znver3 | 305 | 301 | 299 | 300 | 299 | 299 | 300 | 299 |
  | x86_64-v4/icelake-server | 149 | 148 | 145 | 148 | 145 | 145 | 148 | 145 |
  | x86_64-v4/sapphirerapids | 287 | 286 | 285 | 287 | 286 | 285 | 287 | 286 |
  | x86_64-v4/znver4 | 144 | 144 | 143 | 144 | 143 | 143 | 144 | 143 |
  | i686/skylake | 612 | 607 | 603 | 603 | 604 | 603 | 602 | 607 |
  | aarch64/cortex-a72 | 1182 | 1143 | 1099 | 1105 | 1099 | 1099 | 1080 | 1049 |
  | aarch64/neoverse-n1 | 817 | 745 | 701 | 699 | 706 | 701 | 699 | 707 |
  | aarch64/neoverse-v2 | 278 | 282 | 285 | 285 | 285 | 285 | 270 | 261 |
  | aarch64-apple/apple-m1 | 293 | 255 | 248 | 244 | 248 | 248 | 244 | 244 |

  "cap 256" (unroll until an iteration covers 256 bytes, no count limit) is kept. Fixed U=4 is
  within 1% overall (5% on btver2 / Neoverse V2).
- Inlining the scalar tail into `decode_simd`: the SIMD body's steady state is unchanged
  (register renames; +2% on btver2 `decode_16`). But LLVM auto-vectorizes the tail into a
  never-taken 64-item vector loop plus a bounds-checked scalar loop, so functions grow 2.5-16x
  (x86_64-v4 `decode_16`: 205 -> 3335 instructions). The tail stays in `decode_impl_generic`.
- Swizzle instead of interleave: fearless's `swizzle_dyn` is single-source, so a variant gathers
  one piece of each plane with `combine` (`vinserti64x4` from memory, p0 or p5) and does one
  `vpermb` per output vector. Core loop, cycles / 4096 B for `decode_2`: Ice Lake 84 (vs 74),
  Sapphire Rapids 89 (vs 132), Zen 4 126 (vs 67). It needs ITEMSIZE pieces of 64 / ITEMSIZE
  bytes per output, so it degrades quickly for larger itemsizes (`decode_4`: 323 vs 131 on Ice
  Lake, in a prototype that still had bounds checks). On AVX2/SSE, `swizzle_dyn` on vectors wider than 16 bytes is emulated. Not adopted.
- AVX-512 CPUs without VBMI (Skylake-X, Cascade Lake) take the fearless AVX2 path: 569 on the
  `skylake-avx512` model, vs 801 for the baseline's x86-64-v4 multiversion clone.

## Encode (`--direction encode`)

`results/encode-baseline` is the auto-vectorized `encode_impl::<ITEMSIZE, 128 / ITEMSIZE>` with
the `multiversion` x86 feature sets (`--x86-levels multiversion`, analyzed at the commit that
still had it). `encode_simd` mirrors `decode_simd`: load ITEMSIZE vectors of whole items, transpose,
store one vector per byte plane, same unroll rule (at least 256 bytes per iteration; 128 is worse
everywhere, 512 mixed +-3%).

The transpose is the inverse of decode's. A decode round (`interleave`) is a perfect shuffle of
the chunk's `ITEMSIZE * lanes` bytes, i.e. a rotation of the byte index bits by one, and decode
rotates by log2(ITEMSIZE). So encode is either log2(ITEMSIZE) inverse rounds (`deinterleave`), or
log2(lanes) more decode rounds. Instructions per pair in fearless_simd (interleave / deinterleave):
SSE2 2 / scalar, SSE4.2 2 / 4, AVX2 4 / 6, AVX-512 and NEON 2 / 2.

A per-level cost table picked the best form everywhere, but a `simd.level()` switch was not
wanted. The level-agnostic rule kept: `interleave` is never more expensive than `deinterleave`, so
use it when it needs no more rounds (`lanes <= ITEMSIZE`, i.e. only itemsize 16 on 128-bit ISAs).
It is exact on x86-64-v3/v4 for itemsizes 2/4/8 and on NEON for 2/4/8 (the common cases); vs the
per-level table it loses SSE4.2 `encode_8` (Sandy Bridge 641 -> 778, Jaguar 1050 -> 2273), AVX2
`encode_16` (Skylake 1283 -> 1538), and Cortex-A72 / Neoverse V2 `encode_16`. One manual override:
SSE2 always uses `interleave`, since fearless_simd's SSE2 `deinterleave` is scalar code (without
it, i686 itemsizes 2/4/8 stay scalar at the baseline's speed, geomean 3080 vs 825).

`results/encode-fearless`, cycles per 4096 bytes:

| platform | cpu | geomean | vs encode-baseline | encode_2 | encode_4 | encode_8 | encode_16 |
|---|---|---:|---:|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | **551** | 6.59x | 302 | 522 | 778 | 753 |
| x86_64-v2 | btver2 | **1439** | 2.73x | 929 | 1633 | 2273 | 1242 |
| x86_64-v3 | skylake | **853** | 3.68x | 386 | 770 | 1155 | 1538 |
| x86_64-v3 | alderlake | **438** | 4.11x | 196 | 420 | 581 | 771 |
| x86_64-v3 | znver3 | **430** | 7.99x | 194 | 387 | 580 | 782 |
| x86_64-v4 | icelake-server | **148** | 3.48x | 74 | 131 | 193 | 257 |
| x86_64-v4 | sapphirerapids | **287** | 1.93x | 132 | 260 | 386 | 513 |
| x86_64-v4 | znver4 | **144** | 3.03x | 67 | 131 | 194 | 257 |
| i686 | skylake | **825** | 5.95x | 343 | 1026 | 1026 | 1283 |
| aarch64 | cortex-a72 | **818** | 5.16x | 546 | 626 | 898 | 1459 |
| aarch64 | neoverse-n1 | **815** | 5.68x | 768 | 768 | 864 | 865 |
| aarch64 | neoverse-v2 | **288** | 9.75x | 242 | 241 | 306 | 386 |
| aarch64-apple | apple-m1 | **297** | 12.86x | 249 | 266 | 339 | 344 |

## Bit shuffle (`--direction bit`)

`BitShuffleFilter` used to run three passes: a byte shuffle, a bit transpose of each 8-byte group
with a strided scatter / gather to the 8 bit planes (`trans_bit_byte` / `untrans_bit_byte`,
auto-vectorized), and an outer-axis swap of `8 * itemsize` `memcpy` calls (`*bitrow_eight`).
Their results, analyzed with the former `--direction bit-encode` / `bit-decode`, are kept in
`results/bit-{encode,decode}-{baseline,fearless}` (`baseline`: `multiversion` x86 levels;
`fearless`: the same code under fearless_simd `dispatch!`). Cycles per 4096 bytes of the bit
pass (runtime itemsize, equal for all itemsizes):

| platform | cpu | encode baseline | encode fearless | decode baseline | decode fearless |
|---|---|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | 6951 | 6951 | 4774 | 4774 |
| x86_64-v2 | btver2 | 15045 | 15045 | 7990 | 7990 |
| x86_64-v3 | skylake | 9732 | 9732 | 1829 | 1829 |
| x86_64-v3 | alderlake | 6633 | 6633 | 1815 | 1815 |
| x86_64-v3 | znver3 | 9102 | 9102 | 2527 | 2527 |
| x86_64-v4 | icelake-server | 2579 | 2708 | 849 | 849 |
| x86_64-v4 | sapphirerapids | 2806 | 3141 | 1030 | 1030 |
| x86_64-v4 | znver4 | 2007 | 2085 | 1076 | 1076 |
| i686 | skylake | 11736 | 11736 | 8540 | 8540 |
| aarch64 | cortex-a72 | 12743 | 12743 | 5400 | 5400 |
| aarch64 | neoverse-n1 | 11972 | 11972 | 6757 | 6757 |
| aarch64 | neoverse-v2 | 5523 | 5523 | 3380 | 3380 |
| aarch64-apple | apple-m1 | 6726 | 6726 | 3658 | 3658 |

**Correction**: an earlier version of this section reported the fearless encode pass 5.3-7.7x
faster on AVX-512, and a much cheaper decode pass on aarch64. Both were analyzer bugs, fixed since:
- The asm/IR loop matching paired a vector loop with its scalar remainder loop (sibling loops of
  the same shape were matched by size rank, and a vector loop can be smaller than its scalar
  remainder in IR because of the latter's bounds checks, but larger in asm). Siblings are now
  matched by vector share first. The scalar loop had been credited with the vector loop's trip
  count.
- llvm-mca serializes a loop behind an aarch64 store with base-register writeback
  (`str q0, [x0], #16`): it issues an instruction when all its operands are ready, so the new
  base register waits for the stored data. Real cores split stores into address and data
  micro-ops. The analyzer now rewrites such stores as a plain store plus an `add` (5-7x on a
  shift-bound loop; no effect on the byte-shuffle results).

So the dispatch change was neutral, except 0.89-0.96x on AVX-512 (fearless_simd's Ice Lake level
vs the `multiversion` v4 clone). The expensive part was the byte gather: LLVM vectorized the loop
over groups with one 64-bit lane per group, building each vector with one `vpinsrb` / `ld1 {v.b}`
per byte.

### Current: two byte shuffles and a row transpose (`results/bit-fearless`)

After the byte shuffle, a byte shuffle with itemsize 8 of each byte plane gives 8 rows per plane,
row `k` holding byte `k` of every group. The bit planes are then a bit transpose of each column of
8 bytes, *across* the rows: `transpose8x8_rows` is the recursive block-swap transpose applied to 8
row values (`u64`, or fearless_simd `u64` vectors, byte-wise masks), 12 swaps of 6 ops per 8
vectors. The rows are contiguous, so the loads and stores are plain vector ones, and the result
is directly in the final `(B, 8, G)` layout: no scatter, no gather, no `memcpy` pass. The same
code decodes (the transpose is self-inverse). Portable SIMD is used only in
`transpose_bit_rows_simd` (plus the existing byte-shuffle kernels).

Cycles per 4096 bytes, excluding the element-sized byte shuffle and the tails, common to both.
Old: the bit pass alone (the `memcpy` pass is not counted); new: the per-plane byte shuffle
(`encode_8` / `decode_8`) plus `transpose_bit_rows`:

| platform | cpu | encode old | encode new | | decode old | decode new | |
|---|---|---:|---:|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | 6951 | 778 + 1154 = **1932** | 3.6x | 4774 | 1154 + 513 = **1667** | 2.9x |
| x86_64-v2 | btver2 | 15045 | 2273 + 1717 = **3990** | 3.8x | 7990 | 1717 + 858 = **2575** | 3.1x |
| x86_64-v3 | skylake | 9732 | 1155 + 386 = **1541** | 6.3x | 1829 | 386 + 770 = **1157** | 1.6x |
| x86_64-v3 | alderlake | 6633 | 581 + 388 = **969** | 6.8x | 1815 | 388 + 405 = **793** | 2.3x |
| x86_64-v3 | znver3 | 9102 | 580 + 291 = **871** | 10.5x | 2527 | 291 + 403 = **694** | 3.6x |
| x86_64-v4 | icelake-server | 2708 | 193 + 225 = **418** | 6.5x | 849 | 225 + 193 = **418** | 2.0x |
| x86_64-v4 | sapphirerapids | 3141 | 386 + 242 = **628** | 5.0x | 1030 | 242 + 386 = **628** | 1.6x |
| x86_64-v4 | znver4 | 2085 | 194 + 209 = **403** | 5.2x | 1076 | 209 + 194 = **403** | 2.7x |
| i686 | skylake | 11736 | 1026 + 903 = **1929** | 6.1x | 8540 | 903 + 771 = **1674** | 5.1x |
| aarch64 | cortex-a72 | 12743 | 898 + 1385 = **2283** | 5.6x | 5400 | 1385 + 1236 = **2621** | 2.1x |
| aarch64 | neoverse-n1 | 11972 | 864 + 1477 = **2342** | 5.1x | 6757 | 1477 + 753 = **2230** | 3.0x |
| aarch64 | neoverse-v2 | 5523 | 306 + 664 = **969** | 5.7x | 3380 | 664 + 290 = **954** | 3.5x |
| aarch64-apple | apple-m1 | 6726 | 339 + 776 = **1115** | 6.0x | 3658 | 776 + 260 = **1035** | 3.5x |

Findings:
- Faster on every CPU, 3.6-10.5x (encode) and 1.6-5.1x (decode), before counting the removed
  `memcpy` pass.
- The transpose alone: 209-242 on AVX-512, 291-388 on AVX2, 664-1477 on NEON (72 bitwise ops per
  8 vectors, ALU bound). The byte shuffle with itemsize 8 is now the larger half on x86-64 v3.
- The vertical form halves the ALU work of the horizontal one (the existing 8x8 transpose on one
  `u64` per group, 3 delta swaps of 6 ops per vector): prototyped, the horizontal in-place group
  transpose cost 300-3900, portable SIMD or auto-vectorized.
- Written with iterators (`for (s, m) in [...]`, `(0..8).filter(..)`), LLVM did not unroll the
  swap loops and kept the rows on the stack; plain loops in an `#[inline(always)]` helper are fully
  unrolled.
- The loads use `load_array_ref` on a pointer cast instead of `from_slice`: adding a
  `from_slice`-based prototype to the crate changed the inlining of the frozen byte-shuffle
  kernels (length `unwrap` checks, `encode_2` on NEON 546 -> 962). With `load_array_ref`, the
  byte-shuffle results reproduce exactly.
- AVX-512 CPUs without VBMI (Skylake-X, Cascade Lake) take the AVX2 arm: on the `skylake-avx512`
  model, 1541 (encode) and 1156 (decode), vs 2579 and 849 for the former v4 clone.

## Element-wise pipeline (`--direction elementwise`)

The pipeline is not self-contained like the filter kernels, so it gets its own probe crate,
`elementwise/`, which depends on `jix` itself. Its `probe_*` functions evaluate an op chain over
`Plain` arrays into a packed output through the public API (`to_ndarray_slice`):

| kernel | expression | element types |
|---|---|---|
| `neg` | `-a` | f32, f64, i32 |
| `add` | `a + b` | f32, f64, i32 |
| `chain` | `(a + b) * (c - d)` (`Add`, `Mul`, `Sub`); `(a + b) * c - d` before `elementwise-longchain-*` | f32, f64, i32 |
| `longchain` | `(a + b) * (c - d) + (e + f) * (g - h)`, more values live at once (from `elementwise-longchain-*`) | f32, f64, i32 |

What is analyzed is the pipeline's inner loop that each probe instantiates,
`elementwise_pipeline::inner_loop::<T, LANES, true, true, Pipeline>`: the variant for all operands
(inputs and output) contiguous in the inner dimension. `analyze.py` finds it by its demangled name,
built from the op chain (`ELEMENTWISE_EXPRS`), and binds its `len` argument (elements). Its hot
loop is the main loop over `LANES` elements (`read_bulk::<LANES, true>`, `LANES = 128 / size_of::<T>()`).
The unit is the same, **cycles per 4096 bytes of output**, e.g. 1024 f32 elements whatever the
number of inputs. Not counted: the per-call setup in `to_buf_type_erased` (the pipeline calls the
inner loop once per contiguous run of at most 8192 elements, `Staging::BUFFER_SIZE`) and the
remainder of `len % LANES` elements.

`jix` depends on `zstd-sys` (C). The probe is an rlib that is never linked, so the C code is
"compiled" by a stand-in, `elementwise/fake_cc.sh` (empty objects and archives, set as `CC_<target>`
/ `AR_<target>`): no C cross toolchain or macOS SDK is needed. x86 asm is read in Intel syntax for
this crate, as cargo-show-asm fails to parse some of `jix`'s i686 AT&T output (`rep;movsl`).

The pipeline has no runtime dispatch: x86-64 wheels run it at the SSE2 baseline, which is the
extra `x86_64` platform. The x86_64-v2/v3/v4 rows are the same code compiled with fearless_simd's
level features, i.e. what auto-vectorization gives in a dispatch arm of that level.

### Baseline (`results/elementwise-baseline`)

| platform | cpu | geomean | neg_f32 | add_f32 | chain_f32 | neg_f64 | add_f64 | chain_f64 | neg_i32 | add_i32 | chain_i32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x86_64 (SSE2) | sandybridge | **433** | 291 | 387 | 645 | 291 | 387 | 645 | 283 | 387 | 934 |
| x86_64 (SSE2) | skylake | **362** | 259 | 260 | 518 | 259 | 260 | 518 | 259 | 259 | 1285 |
| x86_64 (SSE2) | znver3 | **402** | 260 | 387 | 645 | 260 | 387 | 645 | 260 | 387 | 647 |
| x86_64-v2 | sandybridge | **416** | 291 | 387 | 645 | 291 | 387 | 645 | 283 | 387 | 644 |
| x86_64-v2 | btver2 | **676** | 435 | 564 | 1076 | 435 | 564 | 1077 | 563 | 563 | 1348 |
| x86_64-v3 | skylake | **167** | 132 | 132 | 262 | 132 | 132 | 262 | 132 | 132 | 279 |
| x86_64-v3 | alderlake | **151** | 103 | 136 | 236 | 103 | 136 | 236 | 103 | 136 | 268 |
| x86_64-v3 | znver3 | **203** | 132 | 195 | 324 | 132 | 195 | 324 | 132 | 195 | 324 |
| x86_64-v4 | icelake-server | **1125** | 961 | 1481 | 2567 | 362 | 574 | 1086 | 953 | 1469 | 2515 |
| x86_64-v4 | sapphirerapids | **3059** | 2476 | 4583 | 6429 | 456 | 1923 | 5635 | 2243 | 4443 | 6513 |
| x86_64-v4 | znver4 | **226** | 198 | 265 | 457 | 88 | 160 | 288 | 169 | 232 | 401 |
| i686 | skylake | **362** | 259 | 260 | 518 | 259 | 260 | 518 | 259 | 259 | 1285 |
| aarch64 | cortex-a72 | **925** | 611 | 898 | 1635 | 611 | 898 | 1635 | 579 | 771 | 1380 |
| aarch64 | neoverse-n1 | **878** | 610 | 802 | 1378 | 610 | 802 | 1378 | 642 | 770 | 1380 |
| aarch64 | neoverse-v2 | **339** | 227 | 292 | 579 | 227 | 292 | 579 | 227 | 323 | 548 |
| aarch64-apple | apple-m1 | **277** | 195 | 260 | 422 | 195 | 260 | 423 | 195 | 259 | 420 |

Findings:
- SSE2, AVX2 and NEON: clean code. The `LANES` elements of `read_bulk` become plain unrolled vector
  loads, ops and stores (e.g. `neg_f32` on AVX2: 4 x (`vxorps` with a memory operand, `vmovups`)
  per 128 bytes; bound by the stores on Skylake). AVX2 is 2.2x faster than the SSE2 build that
  ships today (geomean, Skylake).
- **AVX-512 is broken**: the loop vectorizer vectorizes the already-unrolled main loop again, across
  16 (f32/i32) or 8 (f64) of its iterations, which turns every load into a stride-`LANES` gather
  (`vgatherqps` / `vgatherqpd`). The geomean is 6.7x (Ice Lake) and 18x (Sapphire Rapids) that of
  AVX2 on Skylake. llvm-mca's `znver4` model prices the gathers much lower, which is likely
  optimistic.
- `chain_i32` without SSE4.1 (SSE2 builds, x86_64 and i686) has no `pmulld`: the multiply is
  emulated with `pmuludq` and shuffles, up to 2.5x the cost of `chain_f32` (Skylake).
- f64 costs the same as f32 per byte everywhere except AVX-512 (fewer gathers).

Analyzer changes for this crate (the filter results reproduce exactly):
- asm loops are natural loops (back edges to a dominating block, from a CFG of the asm), not
  just backward branches. On AVX-512, a jump from after the main vector loop back to the
  vector-epilogue setup laid out before it looked like a loop around both; and a rotated loop
  entered in the middle must still be found. On all the asm files then committed, the loop trees are the
  same as with backward branches, except for these two cases.
- x86 vector registers are recognized in Intel syntax too.
- The function's mangled name in the IR is found by demangling all `define`s with `c++filt`
  (binutils demangles Rust v0).

### fearless_simd dispatch and SIMD kernels (`results/elementwise-fearless`)

The design:
- `inner_loop_contiguous` (the all-operands-contiguous variant only) runs its loop inside
  `fearless_simd::dispatch!`, and reads through `ElementwisePipelineImpl::read_bulk_simd`, which
  takes the level's token. Its default forwards to `read_bulk`; only `Op1Pipeline` and
  `Op2Pipeline` override it, and pass the token to their kernel's new `apply_bulk(simd, xs)`. The
  strided/staged variants are unchanged (no dispatch, `read_bulk`).
- `apply_bulk` defaults to the scalar `apply` per element. `Neg`, `Add`, `Sub` and `Mul` get SIMD
  bodies (`ops/simd_kernels.rs`) for f32, f64 and i32, selected by `TypeId` (resolved at compile
  time), so the ops' bounds are unchanged and every other type keeps the scalar kernel.
- Nodes still pass `[T; N]` arrays; the SIMD bodies load them into vectors and store the result
  into an array. **Verified in the asm: along `(a + b) * c - d` the values stay in vector
  registers**, e.g. AVX-512 `chain_f32`: per `zmm`, one `vmovups` load, `vaddps`, `vmulps`,
  `vsubps` with memory operands, one `vmovups` store, no stack traffic; the same on AVX2 and NEON.

How the rows map to what runs: the probe is compiled with each platform's features, which makes
that level's `dispatch!` arm the one inlined into `inner_loop_contiguous` (higher levels stay
separate `vectorize_*` functions). So x86_64-v2/v3/v4 are the Sse4_2/Avx2/Avx512 arms, aarch64
the Neon arm, and x86_64/i686 (no features) the baseline `Sse2` arm, which only CPUs without
SSE4.2 run. **A shipped x86-64 wheel now runs the v3/v4 rows on AVX2/AVX-512 CPUs**, instead of
the SSE2 code of the baseline's `x86_64` row: e.g. Skylake 362 -> 166 (2.2x).

| platform | cpu | geomean | vs baseline | neg_f32 | add_f32 | chain_f32 | neg_f64 | add_f64 | chain_f64 | neg_i32 | add_i32 | chain_i32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x86_64 (Sse2 arm) | sandybridge | **440** | 0.99x | 291 | 387 | 644 | 291 | 387 | 644 | 283 | 387 | 1074 |
| x86_64 (Sse2 arm) | skylake | **362** | 1.00x | 259 | 261 | 517 | 259 | 261 | 517 | 259 | 260 | 1287 |
| x86_64 (Sse2 arm) | znver3 | **405** | 0.99x | 260 | 387 | 644 | 260 | 387 | 644 | 260 | 387 | 694 |
| x86_64-v2 | sandybridge | **416** | 1.00x | 291 | 387 | 644 | 291 | 387 | 644 | 283 | 387 | 644 |
| x86_64-v2 | btver2 | **677** | 1.00x | 436 | 564 | 1076 | 436 | 564 | 1079 | 564 | 564 | 1348 |
| x86_64-v3 | skylake | **166** | 1.01x | 132 | 132 | 263 | 132 | 132 | 263 | 132 | 132 | 263 |
| x86_64-v3 | alderlake | **144** | 1.05x | 103 | 136 | 214 | 103 | 136 | 214 | 103 | 125 | 236 |
| x86_64-v3 | znver3 | **203** | 1.00x | 132 | 195 | 324 | 132 | 195 | 324 | 132 | 195 | 323 |
| x86_64-v4 | icelake-server | **91** | 12.32x | 55 | 77 | 198 | 55 | 77 | 198 | 55 | 76 | 150 |
| x86_64-v4 | sapphirerapids | **75** | 40.95x | 52 | 63 | 119 | 52 | 63 | 119 | 52 | 63 | 150 |
| x86_64-v4 | znver4 | **102** | 2.22x | 66 | 97 | 164 | 66 | 97 | 164 | 66 | 97 | 164 |
| i686 (Sse2 arm) | skylake | **366** | 0.99x | 262 | 264 | 524 | 262 | 264 | 524 | 262 | 262 | 1289 |
| aarch64 | cortex-a72 | **916** | 1.01x | 579 | 899 | 1667 | 579 | 899 | 1667 | 579 | 739 | 1411 |
| aarch64 | neoverse-n1 | **878** | 1.00x | 610 | 834 | 1314 | 610 | 834 | 1314 | 642 | 770 | 1411 |
| aarch64 | neoverse-v2 | **351** | 0.97x | 227 | 323 | 611 | 227 | 323 | 611 | 227 | 323 | 548 |
| aarch64-apple | apple-m1 | **271** | 1.02x | 196 | 260 | 389 | 196 | 260 | 391 | 196 | 260 | 390 |

Findings:
- **AVX-512 fixed**: with the work in vector types, the loop vectorizer no longer re-vectorizes
  the main loop with gathers: 12x (Ice Lake), 41x (Sapphire Rapids), 2.2x (Zen 4) over the
  baseline's AVX-512 code, and now about 2x faster than AVX2 per byte, as expected.
- SSE4.2, AVX2 and NEON: the same as the auto-vectorized code (0.97-1.05x), which was already
  clean. Small losses on Neoverse V2 (`add`, `chain`: 0.9-0.95x, a different instruction order).
- i686 has 8 `xmm` registers: with `LANES` = 128 bytes per operand, the SIMD body's order (all
  loads, then the ops) spilled one vector per iteration (geomean 404, 0.90x). fearless_simd exposes
  the vector widths (`S::f32s::LEN`, ...) but not the register count. Halving `LANES` on 32-bit x86
  fixed it (366, 0.99x); this table is from that version. The lanes are now picked per level,
  see the next section.
- The first version chained the per-type attempts with `Option::or_else`, which is not always
  inlined: on the `Sse2` arm, the emulated i32 multiply stayed behind a call in the hot loop
  (`chain_i32` 2-3x slower). Plain `if let ... return` fixed it.

Analyzer changes: `len` reaches the loop through a stack slot (the dispatch closure captures it by
reference and is passed to the non-inlined higher-level arms), and its IR argument can be
unnamed (`%1`). `IrValues` resolves a `load` from `%len` (or from a slot stored once), and the
kernel's last argument is bound whatever its name.



### Lanes per SIMD level (`results/elementwise-k{1,2,4,8}`)

The contiguous loop takes `CONTIGUOUS_VECTORS` (k) vectors of its level per iteration:
`LANES = k * S::u8s::LEN / size_of::<T>()`, i.e. k * 16 bytes on SSE/NEON, k * 32 on AVX2, k * 64
on AVX-512 (`results/elementwise-fearless`: 128 bytes on every level, 64 on i686). `LANES`
depends on the level, so `inner_loop_level` matches on it as an inline `const` inside the
dispatch arm: only the taken arm is codegened. The strided/staged loops keep their fixed lanes.

**Committed: k = 8, and k = 4 on 32-bit x86** (8 `xmm` registers: k = 8 spills there). The
elements per vector are rounded down to a power of two, so every element size maps to an arm
(before, 1-byte elements on AVX-512 with k = 8, or odd struct sizes, hit `unreachable!()`).

Note: the CPU of the cloud sandbox has only the Skylake-X subset of AVX-512, so fearless_simd
picks its AVX2 level there: the tests run the SSE2, SSE4.2 and AVX2 arms, not the AVX-512 one.

llvm-mca cycles per 4096 output bytes; bold: best, or within 2% of it.

#### f32: geomean of `neg`, `add`, `chain`

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 438 | 438 | 435 | **417** |
| x86_64 | skylake | 334 | 334 | **332** | **327** |
| x86_64 | znver3 | **408** | **408** | **406** | **402** |
| x86_64-v2 | sandybridge | 438 | 438 | 435 | **417** |
| x86_64-v2 | btver2 | 716 | 716 | 696 | **642** |
| x86_64-v3 | skylake | 168 | 168 | **166** | **164** |
| x86_64-v3 | alderlake | 149 | 149 | 144 | **130** |
| x86_64-v3 | znver3 | **204** | **204** | **203** | **201** |
| x86_64-v4 | icelake-server | 94 | 94 | 94 | **87** |
| x86_64-v4 | sapphirerapids | 73 | 73 | 70 | **65** |
| x86_64-v4 | znver4 | **102** | **102** | **101** | **100** |
| i686 | skylake | 412 | 353 | **331** | 371 |
| aarch64 | cortex-a72 | 1576 | 1209 | 1027 | **954** |
| aarch64 | neoverse-n1 | 1352 | 1078 | 900 | **874** |
| aarch64 | neoverse-v2 | 770 | 414 | 464 | **355** |
| aarch64-apple | apple-m1 | 459 | 365 | 292 | **271** |

#### f64: geomean of `neg`, `add`, `chain`

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 438 | 438 | 435 | **417** |
| x86_64 | skylake | 334 | 334 | **332** | **327** |
| x86_64 | znver3 | **408** | **408** | **406** | **402** |
| x86_64-v2 | sandybridge | 438 | 438 | 435 | **417** |
| x86_64-v2 | btver2 | 717 | 717 | 697 | **643** |
| x86_64-v3 | skylake | 168 | 168 | **166** | **164** |
| x86_64-v3 | alderlake | 149 | 149 | 144 | **130** |
| x86_64-v3 | znver3 | **204** | **204** | **203** | **201** |
| x86_64-v4 | icelake-server | 94 | 94 | 94 | **87** |
| x86_64-v4 | sapphirerapids | 73 | 73 | 70 | **65** |
| x86_64-v4 | znver4 | **102** | **102** | **101** | **100** |
| i686 | skylake | 411 | 353 | **331** | 371 |
| aarch64 | cortex-a72 | 1576 | 1209 | 1027 | **954** |
| aarch64 | neoverse-n1 | 1352 | 1078 | 900 | **874** |
| aarch64 | neoverse-v2 | 770 | 414 | 464 | **355** |
| aarch64-apple | apple-m1 | 548 | 367 | 292 | **271** |

#### i32: geomean of `neg`, `add`, `chain`

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 516 | **496** | **487** | **490** |
| x86_64 | skylake | **450** | **448** | **447** | **443** |
| x86_64 | znver3 | 430 | **413** | **411** | **412** |
| x86_64-v2 | sandybridge | 430 | 430 | 428 | **413** |
| x86_64-v2 | btver2 | 835 | 835 | 811 | **754** |
| x86_64-v3 | skylake | 174 | 167 | **166** | **164** |
| x86_64-v3 | alderlake | 153 | 153 | 145 | **136** |
| x86_64-v3 | znver3 | **203** | **203** | **202** | **201** |
| x86_64-v4 | icelake-server | 89 | 86 | 84 | **79** |
| x86_64-v4 | sapphirerapids | 78 | 78 | 77 | **72** |
| x86_64-v4 | znver4 | **102** | **102** | **101** | **100** |
| i686 | skylake | 561 | 490 | **446** | 479 |
| aarch64 | cortex-a72 | 1447 | 1108 | 965 | **845** |
| aarch64 | neoverse-n1 | 1353 | 1107 | 914 | **887** |
| aarch64 | neoverse-v2 | 770 | 414 | 452 | **343** |
| aarch64-apple | apple-m1 | 465 | 363 | 291 | **271** |

#### f32 (f64: within 1%, except Apple M1 at k=1): `neg` (`-a`)

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 325 | 325 | 325 | **291** |
| x86_64 | skylake | **262** | **262** | **262** | **259** |
| x86_64 | znver3 | **264** | **264** | **264** | **260** |
| x86_64-v2 | sandybridge | 325 | 325 | 325 | **291** |
| x86_64-v2 | btver2 | 485 | 485 | 486 | **436** |
| x86_64-v3 | skylake | **132** | **132** | **132** | **130** |
| x86_64-v3 | alderlake | 103 | 103 | 103 | **94** |
| x86_64-v3 | znver3 | **132** | **132** | **132** | **130** |
| x86_64-v4 | icelake-server | 55 | 55 | 55 | **49** |
| x86_64-v4 | sapphirerapids | 52 | 52 | 52 | **47** |
| x86_64-v4 | znver4 | **66** | **66** | **66** | **65** |
| i686 | skylake | 333 | 269 | **262** | 292 |
| aarch64 | cortex-a72 | 1290 | 781 | 646 | **579** |
| aarch64 | neoverse-n1 | 1044 | 780 | 645 | **610** |
| aarch64 | neoverse-v2 | 540 | 270 | 262 | **227** |
| aarch64-apple | apple-m1 | 335 | 270 | 203 | **196** |

#### f32 (f64: within 1%, except Apple M1 at k=1): `add` (`a + b`)

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | **390** | **390** | **390** | **387** |
| x86_64 | skylake | **264** | **265** | **266** | **261** |
| x86_64 | znver3 | **390** | **390** | **390** | **387** |
| x86_64-v2 | sandybridge | **390** | **390** | **390** | **387** |
| x86_64-v2 | btver2 | 615 | 615 | 616 | **564** |
| x86_64-v3 | skylake | **133** | **133** | **132** | **131** |
| x86_64-v3 | alderlake | 126 | 126 | 136 | **116** |
| x86_64-v3 | znver3 | **195** | **195** | **195** | **193** |
| x86_64-v4 | icelake-server | 77 | 77 | 77 | **71** |
| x86_64-v4 | sapphirerapids | 63 | 63 | 63 | **58** |
| x86_64-v4 | znver4 | **97** | **97** | **97** | **97** |
| i686 | skylake | 374 | 305 | **264** | 293 |
| aarch64 | cortex-a72 | 1298 | 1166 | 966 | **899** |
| aarch64 | neoverse-n1 | 1300 | 1036 | **837** | **834** |
| aarch64 | neoverse-v2 | 796 | 398 | 454 | **323** |
| aarch64-apple | apple-m1 | 420 | 337 | 266 | **260** |

#### f32 (f64: within 1%, except Apple M1 at k=1): `chain` (`(a + b) * c - d`)

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 663 | 663 | **650** | **644** |
| x86_64 | skylake | 538 | 538 | **525** | **517** |
| x86_64 | znver3 | 659 | 659 | **648** | **644** |
| x86_64-v2 | sandybridge | 663 | 663 | **650** | **644** |
| x86_64-v2 | btver2 | 1231 | 1231 | 1128 | **1076** |
| x86_64-v3 | skylake | 269 | 269 | **263** | **259** |
| x86_64-v3 | alderlake | 255 | 255 | 214 | **204** |
| x86_64-v3 | znver3 | 329 | 329 | **324** | **322** |
| x86_64-v4 | icelake-server | 198 | 198 | **195** | **193** |
| x86_64-v4 | sapphirerapids | 120 | 119 | 108 | **102** |
| x86_64-v4 | znver4 | **164** | 164 | **163** | **161** |
| i686 | skylake | 563 | 539 | **524** | 597 |
| aarch64 | cortex-a72 | 2337 | 1939 | 1734 | **1667** |
| aarch64 | neoverse-n1 | 1820 | 1550 | 1349 | **1314** |
| aarch64 | neoverse-v2 | 1062 | 658 | 838 | **611** |
| aarch64-apple | apple-m1 | 689 | 535 | 459 | **389** |

#### i32: `neg` (`-a`)

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 310 | 310 | 310 | **283** |
| x86_64 | skylake | **262** | **262** | **262** | **259** |
| x86_64 | znver3 | **264** | **264** | **264** | **260** |
| x86_64-v2 | sandybridge | 310 | 310 | 310 | **283** |
| x86_64-v2 | btver2 | 613 | 613 | 614 | **564** |
| x86_64-v3 | skylake | **132** | **132** | **132** | **130** |
| x86_64-v3 | alderlake | 103 | 103 | 103 | **94** |
| x86_64-v3 | znver3 | **132** | **132** | **132** | **130** |
| x86_64-v4 | icelake-server | 55 | 55 | 55 | **49** |
| x86_64-v4 | sapphirerapids | 52 | 52 | 52 | **47** |
| x86_64-v4 | znver4 | **66** | **66** | **66** | **65** |
| i686 | skylake | 366 | 301 | **262** | 291 |
| aarch64 | cortex-a72 | 1290 | 781 | 646 | **579** |
| aarch64 | neoverse-n1 | 1044 | 780 | **645** | **642** |
| aarch64 | neoverse-v2 | 540 | 270 | 262 | **227** |
| aarch64-apple | apple-m1 | 335 | 270 | 203 | **196** |

#### i32: `add` (`a + b`)

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | **390** | **390** | **390** | **387** |
| x86_64 | skylake | **262** | **263** | **264** | **260** |
| x86_64 | znver3 | **390** | **390** | **390** | **387** |
| x86_64-v2 | sandybridge | **390** | **390** | **390** | **387** |
| x86_64-v2 | btver2 | 614 | 614 | 614 | **564** |
| x86_64-v3 | skylake | **132** | **132** | **132** | **130** |
| x86_64-v3 | alderlake | 125 | 125 | 125 | **116** |
| x86_64-v3 | znver3 | **195** | **195** | **195** | **193** |
| x86_64-v4 | icelake-server | 76 | 76 | 76 | **70** |
| x86_64-v4 | sapphirerapids | 63 | 63 | 63 | **58** |
| x86_64-v4 | znver4 | **97** | **97** | **97** | **97** |
| i686 | skylake | 366 | 301 | **262** | 293 |
| aarch64 | cortex-a72 | 1293 | 1037 | 901 | **739** |
| aarch64 | neoverse-n1 | 1300 | 1036 | 837 | **770** |
| aarch64 | neoverse-v2 | 796 | 398 | 454 | **323** |
| aarch64-apple | apple-m1 | 415 | 334 | **264** | **260** |

#### i32: `chain` (`(a + b) * c - d`)

| platform | cpu | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 1139 | 1014 | **958** | 1074 |
| x86_64 | skylake | 1321 | **1302** | **1292** | **1287** |
| x86_64 | znver3 | 776 | **687** | **675** | 694 |
| x86_64-v2 | sandybridge | 658 | 658 | **648** | **644** |
| x86_64-v2 | btver2 | 1549 | 1549 | 1415 | **1348** |
| x86_64-v3 | skylake | 301 | 270 | **263** | **260** |
| x86_64-v3 | alderlake | 280 | 280 | 236 | **230** |
| x86_64-v3 | znver3 | **327** | **327** | **323** | **322** |
| x86_64-v4 | icelake-server | 167 | 150 | **139** | 145 |
| x86_64-v4 | sapphirerapids | 150 | 150 | 144 | **139** |
| x86_64-v4 | znver4 | **164** | **164** | **162** | **161** |
| i686 | skylake | 1318 | **1300** | **1289** | **1286** |
| aarch64 | cortex-a72 | 1818 | 1681 | 1544 | **1411** |
| aarch64 | neoverse-n1 | 1823 | 1679 | **1415** | **1411** |
| aarch64 | neoverse-v2 | 1062 | 658 | 776 | **548** |
| aarch64-apple | apple-m1 | 724 | 531 | 458 | **390** |

#### Bytes per iteration (f32), as compiled: k * vector, times LLVM's own unrolling on x86

| platform | vector | neg k=1/2/4/8 | add k=1/2/4/8 | chain k=1/2/4/8 |
|---|---:|---|---|---|
| x86_64 | 16 B | 64 / 64 / 64 / 128 | 64 / 64 / 64 / 128 | 32 / 32 / 64 / 128 |
| x86_64-v2 | 16 B | 64 / 64 / 64 / 128 | 64 / 64 / 64 / 128 | 32 / 32 / 64 / 128 |
| x86_64-v3 | 32 B | 128 / 128 / 128 / 256 | 128 / 128 / 128 / 256 | 64 / 64 / 128 / 256 |
| x86_64-v4 | 64 B | 256 / 256 / 256 / 512 | 256 / 256 / 256 / 512 | 128 / 128 / 256 / 512 |
| i686 | 16 B | 16 / 32 / 64 / 128 | 16 / 32 / 64 / 128 | 16 / 32 / 64 / 128 |
| aarch64 | 16 B | 16 / 32 / 64 / 128 | 16 / 32 / 64 / 128 | 16 / 32 / 64 / 128 |
| aarch64-apple | 16 B | 16 / 32 / 64 / 128 | 16 / 32 / 64 / 128 | 16 / 32 / 64 / 128 |


Findings:
- k = 8 is the best, or within 2%, for every type, op and CPU, except:
  - i686 (8 registers): k = 4 is best (geomean f32 331 vs 371 at k = 8, which spills).
  - i32 `chain` on the x86-64 `Sse2` arm (emulated multiply, register-hungry): k = 4
    (Sandy Bridge 958 vs 1074, Zen 3 675 vs 694); and on Ice Lake (139 vs 145).
  - Neoverse V2 is non-monotonic (k = 2 beats k = 4, then k = 8 beats both), likely an artifact
    of the model; k = 8 is the best there anyway.
- Fewer vectors cost most on NEON (LLVM does not unroll there): k = 1 is 1.6-2.2x slower than k = 8.
  On x86, LLVM's own unrolling of small bodies hides most of the difference (see the bytes per
  iteration: `neg` / `add` run 4 vectors per iteration for k = 1, 2 and 4).
- f32 and f64 are the same per byte (the same instructions on twice the elements), except k = 1 on
  Apple M1 (5-24% slower for f64); i32 is too, except `chain` (integer multiply).
- Against the former 128 bytes on every level (`elementwise-fearless`): the same on SSE/NEON
  (k = 8 is 128 bytes there), better on AVX2 (Alder Lake f32 geomean 144 -> 130) and AVX-512
  (Ice Lake 94 -> 87, Sapphire Rapids 73 -> 65), which now run 256 / 512 bytes.

### Register pressure: `chain` = `(a + b) * (c - d)` and `longchain` (`results/elementwise-longchain-*`)

`chain` was changed from `(a + b) * c - d` to `(a + b) * (c - d)`, and `longchain`,
`(a + b) * (c - d) + (e + f) * (g - h)` (8 inputs), added, to keep more vectors live. Compared:
128 bytes per operand on every level (`LanesInfo::LANES`, the first fearless_simd version, i686
included), and k = 4 / k = 8 vectors of the level on every platform (i686 included; committed:
k = 4 there). On the 16-byte levels (SSE, NEON), k = 8 is 128 bytes: the same code.

Cycles per 4096 output bytes, and speedup over 128 B; bold: >= 3% faster, italics: >= 3% slower.

#### f32: geomean of the 4 ops

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 556 | 571 (0.97x) | 556 (1.00x) |
| x86_64 | skylake | 444 | 441 (1.01x) | 444 (1.00x) |
| x86_64 | znver3 | 541 | 528 (1.02x) | 541 (1.00x) |
| x86_64-v2 | sandybridge | 556 | 571 (0.97x) | 556 (1.00x) |
| x86_64-v2 | btver2 | 881 | _924 (0.95x)_ | 881 (1.00x) |
| x86_64-v3 | skylake | 221 | 221 (1.00x) | 218 (1.01x) |
| x86_64-v3 | alderlake | 186 | 186 (1.00x) | **170 (1.10x)** |
| x86_64-v3 | znver3 | 264 | 264 (1.00x) | 262 (1.01x) |
| x86_64-v4 | icelake-server | 139 | 139 (1.00x) | **132 (1.06x)** |
| x86_64-v4 | sapphirerapids | 98 | **95 (1.03x)** | **89 (1.10x)** |
| x86_64-v4 | znver4 | 136 | 135 (1.01x) | 134 (1.01x) |
| i686 | skylake | 515 | **466 (1.11x)** | 515 (1.00x) |
| aarch64 | cortex-a72 | 1291 | _1436 (0.90x)_ | 1291 (1.00x) |
| aarch64 | neoverse-n1 | 1125 | 1117 (1.01x) | 1125 (1.00x) |
| aarch64 | neoverse-v2 | 454 | _595 (0.76x)_ | 454 (1.00x) |
| aarch64-apple | apple-m1 | 350 | _387 (0.90x)_ | 350 (1.00x) |

#### f64: geomean of the 4 ops

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 556 | 571 (0.97x) | 556 (1.00x) |
| x86_64 | skylake | 444 | 441 (1.01x) | 444 (1.00x) |
| x86_64 | znver3 | 541 | 528 (1.02x) | 541 (1.00x) |
| x86_64-v2 | sandybridge | 556 | 571 (0.97x) | 556 (1.00x) |
| x86_64-v2 | btver2 | 885 | _925 (0.96x)_ | 885 (1.00x) |
| x86_64-v3 | skylake | 221 | 221 (1.00x) | 218 (1.01x) |
| x86_64-v3 | alderlake | 186 | 186 (1.00x) | **170 (1.10x)** |
| x86_64-v3 | znver3 | 264 | 264 (1.00x) | 262 (1.01x) |
| x86_64-v4 | icelake-server | 139 | 139 (1.00x) | **132 (1.06x)** |
| x86_64-v4 | sapphirerapids | 98 | **95 (1.03x)** | **89 (1.10x)** |
| x86_64-v4 | znver4 | 136 | 135 (1.01x) | 134 (1.01x) |
| i686 | skylake | 515 | **466 (1.11x)** | 515 (1.00x) |
| aarch64 | cortex-a72 | 1291 | _1436 (0.90x)_ | 1291 (1.00x) |
| aarch64 | neoverse-n1 | 1125 | 1117 (1.01x) | 1125 (1.00x) |
| aarch64 | neoverse-v2 | 454 | _595 (0.76x)_ | 454 (1.00x) |
| aarch64-apple | apple-m1 | 350 | _387 (0.90x)_ | 350 (1.00x) |

#### i32: geomean of the 4 ops

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 702 | 690 (1.02x) | 702 (1.00x) |
| x86_64 | skylake | 687 | 692 (0.99x) | 687 (1.00x) |
| x86_64 | znver3 | 595 | **558 (1.07x)** | 595 (1.00x) |
| x86_64-v2 | sandybridge | 552 | 549 (1.00x) | 552 (1.00x) |
| x86_64-v2 | btver2 | 1057 | _1120 (0.94x)_ | 1057 (1.00x) |
| x86_64-v3 | skylake | 221 | 221 (1.00x) | 218 (1.01x) |
| x86_64-v3 | alderlake | 191 | 191 (1.00x) | **181 (1.06x)** |
| x86_64-v3 | znver3 | 263 | 263 (1.00x) | 261 (1.01x) |
| x86_64-v4 | icelake-server | 123 | 120 (1.02x) | **117 (1.05x)** |
| x86_64-v4 | sapphirerapids | 108 | 106 (1.01x) | **101 (1.07x)** |
| x86_64-v4 | znver4 | 132 | 132 (1.00x) | 131 (1.01x) |
| i686 | skylake | 732 | **692 (1.06x)** | 732 (1.00x) |
| aarch64 | cortex-a72 | 1115 | _1240 (0.90x)_ | 1115 (1.00x) |
| aarch64 | neoverse-n1 | 1155 | 1168 (0.99x) | 1155 (1.00x) |
| aarch64 | neoverse-v2 | 457 | _588 (0.78x)_ | 457 (1.00x) |
| aarch64-apple | apple-m1 | 345 | _379 (0.91x)_ | 345 (1.00x) |

#### f32: `neg` (`-a`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 291 | _325 (0.89x)_ | 291 (1.00x) |
| x86_64 | skylake | 259 | 262 (0.99x) | 259 (1.00x) |
| x86_64 | znver3 | 260 | 264 (0.99x) | 260 (1.00x) |
| x86_64-v2 | sandybridge | 291 | _325 (0.89x)_ | 291 (1.00x) |
| x86_64-v2 | btver2 | 436 | _486 (0.90x)_ | 436 (1.00x) |
| x86_64-v3 | skylake | 132 | 132 (1.00x) | 130 (1.01x) |
| x86_64-v3 | alderlake | 103 | 103 (1.00x) | **94 (1.10x)** |
| x86_64-v3 | znver3 | 132 | 132 (1.00x) | 130 (1.01x) |
| x86_64-v4 | icelake-server | 55 | 55 (1.00x) | **49 (1.13x)** |
| x86_64-v4 | sapphirerapids | 52 | 52 (1.00x) | **47 (1.10x)** |
| x86_64-v4 | znver4 | 66 | 66 (1.00x) | 65 (1.01x) |
| i686 | skylake | 292 | **262 (1.11x)** | 292 (1.00x) |
| aarch64 | cortex-a72 | 579 | _646 (0.90x)_ | 579 (1.00x) |
| aarch64 | neoverse-n1 | 610 | _645 (0.94x)_ | 610 (1.00x) |
| aarch64 | neoverse-v2 | 227 | _262 (0.87x)_ | 227 (1.00x) |
| aarch64-apple | apple-m1 | 196 | _203 (0.97x)_ | 196 (1.00x) |

#### f32: `add` (`a + b`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 387 | 390 (0.99x) | 387 (1.00x) |
| x86_64 | skylake | 261 | 266 (0.98x) | 261 (1.00x) |
| x86_64 | znver3 | 387 | 390 (0.99x) | 387 (1.00x) |
| x86_64-v2 | sandybridge | 387 | 390 (0.99x) | 387 (1.00x) |
| x86_64-v2 | btver2 | 564 | _616 (0.92x)_ | 564 (1.00x) |
| x86_64-v3 | skylake | 132 | 132 (1.00x) | 131 (1.01x) |
| x86_64-v3 | alderlake | 136 | 136 (1.00x) | **116 (1.17x)** |
| x86_64-v3 | znver3 | 195 | 195 (1.00x) | 193 (1.01x) |
| x86_64-v4 | icelake-server | 77 | 77 (1.00x) | **71 (1.09x)** |
| x86_64-v4 | sapphirerapids | 63 | 63 (1.00x) | **58 (1.08x)** |
| x86_64-v4 | znver4 | 97 | 97 (1.00x) | 97 (1.01x) |
| i686 | skylake | 293 | **264 (1.11x)** | 293 (1.00x) |
| aarch64 | cortex-a72 | 899 | _966 (0.93x)_ | 899 (1.00x) |
| aarch64 | neoverse-n1 | 834 | 837 (1.00x) | 834 (1.00x) |
| aarch64 | neoverse-v2 | 323 | _454 (0.71x)_ | 323 (1.00x) |
| aarch64-apple | apple-m1 | 260 | 266 (0.98x) | 260 (1.00x) |

#### f32: `chain` (`(a + b) * (c - d)`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 644 | 650 (0.99x) | 644 (1.00x) |
| x86_64 | skylake | 517 | 524 (0.99x) | 517 (1.00x) |
| x86_64 | znver3 | 644 | 650 (0.99x) | 644 (1.00x) |
| x86_64-v2 | sandybridge | 644 | 650 (0.99x) | 644 (1.00x) |
| x86_64-v2 | btver2 | 1077 | _1130 (0.95x)_ | 1077 (1.00x) |
| x86_64-v3 | skylake | 262 | 262 (1.00x) | 260 (1.01x) |
| x86_64-v3 | alderlake | 218 | 218 (1.00x) | **205 (1.06x)** |
| x86_64-v3 | znver3 | 324 | 324 (1.00x) | 322 (1.01x) |
| x86_64-v4 | icelake-server | 197 | 195 (1.01x) | 193 (1.02x) |
| x86_64-v4 | sapphirerapids | 122 | **109 (1.12x)** | **103 (1.19x)** |
| x86_64-v4 | znver4 | 164 | 162 (1.01x) | 161 (1.02x) |
| i686 | skylake | 582 | **524 (1.11x)** | 582 (1.00x) |
| aarch64 | cortex-a72 | 1666 | _1862 (0.89x)_ | 1666 (1.00x) |
| aarch64 | neoverse-n1 | 1281 | 1286 (1.00x) | 1281 (1.00x) |
| aarch64 | neoverse-v2 | 548 | _776 (0.71x)_ | 548 (1.00x) |
| aarch64-apple | apple-m1 | 390 | _458 (0.85x)_ | 390 (1.00x) |

#### f32: `longchain` (`(a + b) * (c - d) + (e + f) * (g - h)`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 1319 | 1292 (1.02x) | 1319 (1.00x) |
| x86_64 | skylake | 1111 | **1039 (1.07x)** | 1111 (1.00x) |
| x86_64 | znver3 | 1318 | **1164 (1.13x)** | 1318 (1.00x) |
| x86_64-v2 | sandybridge | 1319 | 1292 (1.02x) | 1319 (1.00x) |
| x86_64-v2 | btver2 | 2278 | **2157 (1.06x)** | 2278 (1.00x) |
| x86_64-v3 | skylake | 520 | 520 (1.00x) | 516 (1.01x) |
| x86_64-v3 | alderlake | 395 | 395 (1.00x) | **373 (1.06x)** |
| x86_64-v3 | znver3 | 581 | 581 (1.00x) | 578 (1.01x) |
| x86_64-v4 | icelake-server | 453 | 451 (1.00x) | 449 (1.01x) |
| x86_64-v4 | sapphirerapids | 235 | 229 (1.02x) | **227 (1.03x)** |
| x86_64-v4 | znver4 | 325 | 323 (1.01x) | 321 (1.01x) |
| i686 | skylake | 1416 | **1295 (1.09x)** | 1416 (1.00x) |
| aarch64 | cortex-a72 | 3202 | _3654 (0.88x)_ | 3202 (1.00x) |
| aarch64 | neoverse-n1 | 2465 | **2244 (1.10x)** | 2465 (1.00x) |
| aarch64 | neoverse-v2 | 1060 | _1352 (0.78x)_ | 1060 (1.00x) |
| aarch64-apple | apple-m1 | 753 | _908 (0.83x)_ | 753 (1.00x) |

#### i32: `neg` (`-a`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 283 | _310 (0.91x)_ | 283 (1.00x) |
| x86_64 | skylake | 259 | 262 (0.99x) | 259 (1.00x) |
| x86_64 | znver3 | 260 | 264 (0.99x) | 260 (1.00x) |
| x86_64-v2 | sandybridge | 283 | _310 (0.91x)_ | 283 (1.00x) |
| x86_64-v2 | btver2 | 564 | _614 (0.92x)_ | 564 (1.00x) |
| x86_64-v3 | skylake | 132 | 132 (1.00x) | 130 (1.01x) |
| x86_64-v3 | alderlake | 103 | 103 (1.00x) | **94 (1.10x)** |
| x86_64-v3 | znver3 | 132 | 132 (1.00x) | 130 (1.01x) |
| x86_64-v4 | icelake-server | 55 | 55 (1.00x) | **49 (1.13x)** |
| x86_64-v4 | sapphirerapids | 52 | 52 (1.00x) | **47 (1.10x)** |
| x86_64-v4 | znver4 | 66 | 66 (1.00x) | 65 (1.01x) |
| i686 | skylake | 291 | **262 (1.11x)** | 291 (1.00x) |
| aarch64 | cortex-a72 | 579 | _646 (0.90x)_ | 579 (1.00x) |
| aarch64 | neoverse-n1 | 642 | 645 (0.99x) | 642 (1.00x) |
| aarch64 | neoverse-v2 | 227 | _262 (0.87x)_ | 227 (1.00x) |
| aarch64-apple | apple-m1 | 196 | _203 (0.97x)_ | 196 (1.00x) |

#### i32: `add` (`a + b`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 387 | 390 (0.99x) | 387 (1.00x) |
| x86_64 | skylake | 260 | 264 (0.99x) | 260 (1.00x) |
| x86_64 | znver3 | 387 | 390 (0.99x) | 387 (1.00x) |
| x86_64-v2 | sandybridge | 387 | 390 (0.99x) | 387 (1.00x) |
| x86_64-v2 | btver2 | 564 | _614 (0.92x)_ | 564 (1.00x) |
| x86_64-v3 | skylake | 132 | 132 (1.00x) | 130 (1.01x) |
| x86_64-v3 | alderlake | 125 | 125 (1.00x) | **116 (1.08x)** |
| x86_64-v3 | znver3 | 195 | 195 (1.00x) | 193 (1.01x) |
| x86_64-v4 | icelake-server | 76 | 76 (1.00x) | **70 (1.09x)** |
| x86_64-v4 | sapphirerapids | 63 | 63 (1.00x) | **58 (1.08x)** |
| x86_64-v4 | znver4 | 97 | 97 (1.00x) | 97 (1.01x) |
| i686 | skylake | 293 | **262 (1.12x)** | 293 (1.00x) |
| aarch64 | cortex-a72 | 739 | _901 (0.82x)_ | 739 (1.00x) |
| aarch64 | neoverse-n1 | 770 | _837 (0.92x)_ | 770 (1.00x) |
| aarch64 | neoverse-v2 | 323 | _454 (0.71x)_ | 323 (1.00x) |
| aarch64-apple | apple-m1 | 260 | 264 (0.98x) | 260 (1.00x) |

#### i32: `chain` (`(a + b) * (c - d)`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 1060 | **956 (1.11x)** | 1060 (1.00x) |
| x86_64 | skylake | 1286 | 1292 (1.00x) | 1286 (1.00x) |
| x86_64 | znver3 | 775 | **695 (1.11x)** | 775 (1.00x) |
| x86_64-v2 | sandybridge | 644 | 649 (0.99x) | 644 (1.00x) |
| x86_64-v2 | btver2 | 1347 | _1479 (0.91x)_ | 1347 (1.00x) |
| x86_64-v3 | skylake | 263 | 263 (1.00x) | 260 (1.01x) |
| x86_64-v3 | alderlake | 235 | 235 (1.00x) | **225 (1.05x)** |
| x86_64-v3 | znver3 | 324 | 324 (1.00x) | 322 (1.00x) |
| x86_64-v4 | icelake-server | 166 | **156 (1.07x)** | 162 (1.03x) |
| x86_64-v4 | sapphirerapids | 139 | **134 (1.04x)** | **131 (1.06x)** |
| x86_64-v4 | znver4 | 164 | 162 (1.01x) | 161 (1.02x) |
| i686 | skylake | 1289 | 1290 (1.00x) | 1289 (1.00x) |
| aarch64 | cortex-a72 | 1443 | _1544 (0.93x)_ | 1443 (1.00x) |
| aarch64 | neoverse-n1 | 1442 | 1414 (1.02x) | 1442 (1.00x) |
| aarch64 | neoverse-v2 | 580 | _776 (0.75x)_ | 580 (1.00x) |
| aarch64-apple | apple-m1 | 391 | _458 (0.85x)_ | 391 (1.00x) |

#### i32: `longchain` (`(a + b) * (c - d) + (e + f) * (g - h)`)

| platform | cpu | 128 B | k=4 | k=8 |
|---|---|---:|---:|---:|
| x86_64 | sandybridge | 2088 | **1964 (1.06x)** | 2088 (1.00x) |
| x86_64 | skylake | 2568 | 2572 (1.00x) | 2568 (1.00x) |
| x86_64 | znver3 | 1613 | **1358 (1.19x)** | 1613 (1.00x) |
| x86_64-v2 | sandybridge | 1316 | **1164 (1.13x)** | 1316 (1.00x) |
| x86_64-v2 | btver2 | 2916 | **2824 (1.03x)** | 2916 (1.00x) |
| x86_64-v3 | skylake | 520 | 520 (1.00x) | 516 (1.01x) |
| x86_64-v3 | alderlake | 444 | 444 (1.00x) | 438 (1.01x) |
| x86_64-v3 | znver3 | 581 | 581 (1.00x) | 578 (1.01x) |
| x86_64-v4 | icelake-server | 326 | 323 (1.01x) | _338 (0.97x)_ |
| x86_64-v4 | sapphirerapids | 299 | 293 (1.02x) | 291 (1.03x) |
| x86_64-v4 | znver4 | 292 | 290 (1.01x) | 289 (1.01x) |
| i686 | skylake | 2604 | 2577 (1.01x) | 2604 (1.00x) |
| aarch64 | cortex-a72 | 2500 | _2632 (0.95x)_ | 2500 (1.00x) |
| aarch64 | neoverse-n1 | 2498 | 2439 (1.02x) | 2498 (1.00x) |
| aarch64 | neoverse-v2 | 1028 | _1289 (0.80x)_ | 1028 (1.00x) |
| aarch64-apple | apple-m1 | 709 | _842 (0.84x)_ | 709 (1.00x) |


Findings:
- AVX2 and AVX-512 (k = 8 = 256 / 512 B): k = 8 is the best or equal nearly everywhere, up to 1.17x
  over 128 B (Alder Lake `add`) and 1.19x (Sapphire Rapids f32 `chain`); the only loss is Ice Lake
  i32 `longchain` (0.97x; k = 4 1.01x).
- NEON (32 registers): k = 8 (= 128 B) is best, k = 4 loses up to 0.71x (Neoverse V2), except
  Neoverse N1 `longchain`, where k = 4 is 1.10x (f32).
- SSE on x86-64 (16 registers): `neg` / `add` / f32 `chain` prefer k = 8, but the register-heavy
  ones prefer k = 4: `longchain` (Zen 3 SSE2 arm 1.13x f32, 1.19x i32; Sandy Bridge SSE4.2 1.13x
  i32) and i32 `chain` on the SSE2 arm (emulated multiply, 1.11x). The best k depends on the op.
- i686 (8 registers): k = 4 (committed there) is 1.06x (i32) and 1.11x (f32) over 128 B, geomean.

