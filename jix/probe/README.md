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
- `asm/<platform>/<kernel>.s`: the full function asm. It is committed, so variants can be diffed.
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
| `chain` | `(a + b) * c - d` (`Add`, `Mul`, `Sub`) | f32, f64, i32 |

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
- A backward branch is a loop only if no branch from outside jumps into its body. On AVX-512,
  a jump from after the main vector loop back to the vector-epilogue setup laid out before it
  looked like a loop around both.
- x86 vector registers are recognized in Intel syntax too.
- The function's mangled name in the IR is found by demangling all `define`s with `c++filt`
  (binutils demangles Rust v0).

