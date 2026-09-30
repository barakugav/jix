# jix-probe: static asm analysis of the element-wise pipeline

Experimental tooling for optimizing the inner loop of `jix`'s element-wise pipeline
(`jix/src/storage/elementwise_pipeline.rs` and the op kernels it calls) across platforms **without
running benchmarks**. The objective is llvm-mca's steady-state throughput of the loop's hot loops,
on the targeted platforms. Results are under `results/elementwise-*`, see the sections after
Method.

## Sources of kernels

`analyze.py --source` picks the crate whose kernels are analyzed:

- **`py` (default): the `jix-py` extension crate.** It dispatches every op over every dtype it
  supports, so it instantiates the pipeline for each (op, dtype): nothing to add when an op or a
  SIMD body changes. Its kernels are found in the build: every `inner_loop_contiguous` of a single
  op over leaf operands, named `<op>_<input types>` (`neg_f32`, `add_i32`, `equal_f64`,
  `bitwise_shift_left_i32_u32`), plus the output type when it is not an input's (`cast_f32_i32`).
- **`probe`: this crate.** It depends on `jix` and exports a few `probe_*` functions, each of which
  instantiates the pipeline for one op chain (`chain`, `longchain`, mixed dtypes). For what the
  single ops of `py` do not show: several nodes, and values chained between them in registers.

Either is compiled as an rlib that is never linked, so any rustup target works without a linker.
`jix`'s C dependency `zstd-sys` would need a C cross toolchain and sysroot for every target (and a
macOS SDK for `aarch64-apple-darwin`): it is "compiled" by a stand-in, `fake_cc.sh` (empty objects
and archives, set by `analyze.py` as `CC_<target>` / `AR_<target>`). pyo3's build script only needs
the Python version (`PYO3_CROSS_PYTHON_VERSION`).

## Setup

- Toolchain: `rust-toolchain.toml` at the repo root pins `nightly-2026-09-27` (rustc 1.101,
  **LLVM 23.1.1**), with the `llvm-tools` component (provides `opt`) and all cross targets.
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
python jix/probe/analyze.py                         # jix-py: all ops, all platforms, results/baseline/
python jix/probe/analyze.py --label my-variant      # results/my-variant/
python jix/probe/analyze.py --label x --compare py-baseline
python jix/probe/analyze.py --platform x86_64-v3 --platform aarch64 --fn 'add_*' --fn neg_f32
python jix/probe/analyze.py --source probe          # the op chains of this crate
```

A `jix-py` build takes about 8 minutes per platform (two platforms build at a time), the probe
under a minute.

Outputs go to `results/<label>/`:
- `summary.md`: the tables; `summary.json`: the same, machine readable, read by `--compare`
  (gitignored: `--compare` needs a local run of that label).
- `asm/<platform>/<kernel>.s`: the function's asm (gitignored, like `mca/`).
- `mca/...`: the raw llvm-mca reports, with resource pressure and bottleneck analysis
  (gitignored).

## Method

Platforms (see `PLATFORMS` in `analyze.py`):

| platform | target | ISA level | llvm-mca CPUs |
|---|---|---|---|
| x86_64 | x86_64-unknown-linux-gnu | SSE2 (the target baseline; fearless_simd `Sse2`) | sandybridge, skylake, znver3 |
| x86_64-v2 | x86_64-unknown-linux-gnu | fearless_simd `Sse4_2` level features | sandybridge, btver2 |
| x86_64-v3 | x86_64-unknown-linux-gnu | fearless_simd `Avx2` level features | skylake, alderlake, znver3 |
| x86_64-v4 | x86_64-unknown-linux-gnu | fearless_simd `Avx512` level features (Ice Lake: includes VBMI) | icelake-server, sapphirerapids, znver4 |
| i686 | i686-unknown-linux-gnu | SSE2 (the target baseline; fearless_simd `Sse2`) | skylake |
| aarch64 | aarch64-unknown-linux-gnu | armv8-a + NEON | cortex-a72, neoverse-n1, neoverse-v2 |
| aarch64-apple | aarch64-apple-darwin | apple-m1 | apple-m1 |

armv7 and ppc64le are out of scope. Windows and musl wheels share the ISA of their linux
counterparts. `jix` runs the pipeline's contiguous inner loop through `fearless_simd::dispatch!`,
which calls it inside a `#[target_feature(enable = ...)]` function of the detected level (generic
tuning). The probe reproduces each dispatch arm by building with the same features in
`-C target-feature`: that level's arm is then the one inlined into the loop function. So
x86_64-v2/v3/v4 are the Sse4_2/Avx2/Avx512 arms, aarch64 the Neon arm, and x86_64/i686 (no
features) the `Sse2` arm.

Kernels (the `probe_*` functions of `src/lib.rs`, each evaluating an op chain over `Plain` arrays
into a packed output through the public API, `to_ndarray_slice`):

| kernel | expression | element types |
|---|---|---|
| `neg` | `-a` | f32, f64, i32 |
| `add` | `a + b` | f32, f64, i32 |
| `chain` | `(a + b) * (c - d)` (`Add`, `Mul`, `Sub`); `(a + b) * c - d` before `elementwise-longchain-*` | f32, f64, i32 |
| `longchain` | `(a + b) * (c - d) + (e + f) * (g - h)`, more values live at once (from `elementwise-longchain-*`) | f32, f64, i32 |
| `narrow` | `(a + b).cast()` from i64 / f64 (mixed dtypes, see the last section) | i32, f32 |
| `widen` | `a.cast() + b`, `a` i32 / f32 (mixed dtypes, see the last section) | i64, f64 |

What is analyzed is the pipeline's inner loop an op (chain) instantiates,
`elementwise_pipeline::inner_loop_contiguous::<T, Pipeline>` (the baseline analyzed its
predecessor, `inner_loop::<T, LANES, true, true, Pipeline>`): the variant for all operands (inputs
and output) contiguous in the inner dimension. `analyze.py` finds it by its demangled name (for
the probe, built from the op chain, `ELEMENTWISE_EXPRS`), and binds its `len` argument (elements). Its hot loop is
the main loop over `LANES` elements. Not counted: the per-call setup in `to_buf_type_erased` (the
pipeline calls the inner loop once per contiguous run of at most 8192 elements,
`Staging::BUFFER_SIZE`) and the remainder of `len % LANES` elements. x86 asm is emitted in Intel
syntax.

Compare a run with a previous one with `--compare <label>`, which adds a geomean speedup column.

For each platform, the crate is compiled once with `--emit=asm,llvm-ir`. Its functions' asm is
split by label and their names demangled (`c++filt`, v0 mangling). The IR is reduced to the
kernels (and what they call) by the toolchain's own `opt` (`internalize` + `globaldce`), and `opt`
then prints the IR loop trees (`print<loops>`) and exact trip counts (`print<scalar-evolution>`)
of all the kernels at once. Then for each kernel:
1. The loop tree comes from the natural loops of its asm (back edges to a dominating block).
2. Its IR loop tree and trip counts come from the `opt` output.
3. The trees are matched by shape, which gives every asm loop its trip count. Sibling loops of
   the same shape are matched by vector share, then size: a vector loop can be smaller than its
   scalar remainder in IR (bounds checks) but larger in asm.
4. The hot loops are the loops whose total iteration count (their trip count times their
   ancestors', from SCEV evaluated at two concrete input lengths) is linear in the input length:
   the vector main loop. Tails do not scale and are ignored. The slope gives the bytes per
   iteration (`B/iter`), which is measured, not assumed, so a kernel can use any step.
5. One iteration of each hot loop is flattened: inner loops that LLVM did not unroll are repeated
   by their trip count. llvm-mca simulates that trace (100 iterations) on each CPU.

**Metric: cycles per 4096 bytes of output = sum over the hot loops of cycles per iteration * 4096
/ B/iter. Lower is better.** E.g. 1024 f32 elements whatever the number of inputs. The
per-(platform, CPU) geomean over the kernels is the headline number. The unit is only a scale.
What makes kernels with different steps comparable is that `B/iter` is measured.

SCEV trip counts are evaluated with the `len` argument bound (whatever its IR name: it can be
unnamed, or reach the loop through a stack slot). Values SCEV cannot see through (e.g. a scalar
extracted from SLP-vectorized setup code, or a load from a slot stored once) are constant-folded
from the IR. Trip counts that also depend on non-argument values are approximated, and their
`B/iter` is marked `~`: bounds-check limits in a `umin` are dropped (they do not bind in the
steady state), a remainder size SCEV proves small is taken as 0, and an enclosing loop's add
recurrence is evaluated at its first iteration.

Caveats:
- llvm-mca assumes every load hits L1 and ignores the front end. Branches are not followed.
- **Calls are free in llvm-mca.** Kernels that call something (e.g. `memcpy`) inside the hot loop
  are flagged `*`, and their real cost is higher.
- Steady state of the hot loops only, by design. The per-call prologue and epilogue and the
  tails (vector-loop remainders) are neither counted nor analyzed.
- aarch64 stores with base-register writeback (`str q0, [x0], #16`) are split into a plain store
  and an `add` before llvm-mca. llvm-mca issues an instruction when all its operands are ready, so
  the new base register would wait for the stored data and serialize the loop behind it (5-7x on
  a shift-bound loop); real cores split stores into address and data micro-ops.
- Kernels without a flattenable hot loop are listed only in the warnings.
- Instructions with no scheduling info on a CPU are skipped and listed as warnings (none so far).

## Baseline (`results/elementwise-baseline`)

The pipeline had no runtime dispatch yet: x86-64 wheels ran it at the SSE2 baseline, the `x86_64`
platform. The x86_64-v2/v3/v4 rows are the same code compiled with fearless_simd's level
features, i.e. what auto-vectorization gives in a dispatch arm of that level. The hot loop was the
main loop over `LANES` elements (`read_bulk::<LANES, true>`, `LANES = 128 / size_of::<T>()`).

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

Analyzer changes made for this probe:
- asm loops are natural loops (back edges to a dominating block, from a CFG of the asm), not
  just backward branches. On AVX-512, a jump from after the main vector loop back to the
  vector-epilogue setup laid out before it looked like a loop around both; and a rotated loop
  entered in the middle must still be found. On all the asm files then committed, the loop trees are the
  same as with backward branches, except for these two cases.
- x86 vector registers are recognized in Intel syntax too.
- The function's mangled name in the IR is found by demangling all `define`s with `c++filt`
  (binutils demangles Rust v0).

## fearless_simd dispatch and SIMD kernels (`results/elementwise-fearless`)

The design:
- `inner_loop_contiguous` (the all-operands-contiguous variant only) runs its loop inside
  `fearless_simd::dispatch!`, and reads through `ElementwisePipelineImpl::read_bulk_simd`, which
  takes the level's token. Its default forwards to `read_bulk`; only `Op1Pipeline` and
  `Op2Pipeline` override it, and pass the token to their kernel's new `apply_bulk(simd, xs)`. The
  strided/staged variants are unchanged (no dispatch, `read_bulk`). Later merged into a single
  `read_bulk(simd, offset)` implemented by every node, so the token also passes through `where`,
  `map` and sequences; the strided variants pass the target's static baseline level. The
  contiguous asm was unchanged by the merge.
- `apply_bulk` defaults to the scalar `apply` per element. `Neg`, `Add`, `Sub` and `Mul` get SIMD
  bodies (then in `ops/simd_kernels.rs`, since written in the `simd:` argument of `define_op1!` /
  `define_op2!`) for f32, f64 and i32, selected by `TypeId` (resolved at compile time), so the ops'
  bounds are unchanged and every other type keeps the scalar kernel.
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



## Lanes per SIMD level (`results/elementwise-k{1,2,4,8}`)

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

### f32: geomean of `neg`, `add`, `chain`

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

### f64: geomean of `neg`, `add`, `chain`

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

### i32: geomean of `neg`, `add`, `chain`

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

### f32 (f64: within 1%, except Apple M1 at k=1): `neg` (`-a`)

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

### f32 (f64: within 1%, except Apple M1 at k=1): `add` (`a + b`)

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

### f32 (f64: within 1%, except Apple M1 at k=1): `chain` (`(a + b) * c - d`)

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

### i32: `neg` (`-a`)

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

### i32: `add` (`a + b`)

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

### i32: `chain` (`(a + b) * c - d`)

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

### Bytes per iteration (f32), as compiled: k * vector, times LLVM's own unrolling on x86

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

## Register pressure: `chain` = `(a + b) * (c - d)` and `longchain` (`results/elementwise-longchain-*`)

`chain` was changed from `(a + b) * c - d` to `(a + b) * (c - d)`, and `longchain`,
`(a + b) * (c - d) + (e + f) * (g - h)` (8 inputs), added, to keep more vectors live. Compared:
128 bytes per operand on every level (`LanesInfo::LANES`, the first fearless_simd version, i686
included), and k = 4 / k = 8 vectors of the level on every platform (i686 included; committed:
k = 4 there). On the 16-byte levels (SSE, NEON), k = 8 is 128 bytes: the same code.

Cycles per 4096 output bytes, and speedup over 128 B; bold: >= 3% faster, italics: >= 3% slower.

### f32: geomean of the 4 ops

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

### f64: geomean of the 4 ops

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

### i32: geomean of the 4 ops

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

### f32: `neg` (`-a`)

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

### f32: `add` (`a + b`)

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

### f32: `chain` (`(a + b) * (c - d)`)

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

### f32: `longchain` (`(a + b) * (c - d) + (e + f) * (g - h)`)

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

### i32: `neg` (`-a`)

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

### i32: `add` (`a + b`)

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

### i32: `chain` (`(a + b) * (c - d)`)

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

### i32: `longchain` (`(a + b) * (c - d) + (e + f) * (g - h)`)

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


## Mixed dtypes: lanes from `MAX_ITEMSIZE` (`results/elementwise-mixed*`)

The contiguous lanes are sized by the pipeline's widest value (`ElementwisePipelineImpl::MAX_ITEMSIZE`,
the max over its leaves, nodes and output) instead of the output's itemsize. For same-dtype pipelines
the two are equal, so all the probes above are unchanged (re-checked on i686 and x86_64-v3). New
probes, the only kernels that change are the narrowing ones:

- `narrow_i32`: `(a + b).cast::<i32>()`, `a`, `b` i64; `narrow_f32`: the same over f64 -> f32.
- `widen_i64`: `a.cast::<i64>() + b`, `a` i32; `widen_f64`: the same over f32 -> f64 (the output is
  already the widest: control).

The casts use the default scalar `apply_bulk`, left to LLVM to vectorize. Cycles per 4096 output bytes,
output itemsize (`elementwise-mixed-outsize`) -> `MAX_ITEMSIZE` (`elementwise-mixed`); bold: >= 3%
faster, italics: >= 3% slower. The narrowing kernels halve their output bytes per iteration.

| platform | cpu | `narrow_i32` | `narrow_f32` | `widen_i64` | `widen_f64` |
|---|---|---:|---:|---:|---:|
| x86_64 | sandybridge | 643 -> 646 (0.99x) | 1034 -> 1037 (1.00x) | 475 (same) | 389 (same) |
| x86_64 | skylake | 516 -> 519 (0.99x) | 778 -> 781 (1.00x) | 365 (same) | 283 (same) |
| x86_64 | znver3 | 643 -> 647 (0.99x) | **741 -> 650 (1.14x)** | 388 (same) | 388 (same) |
| x86_64-v2 | sandybridge | 643 -> 646 (0.99x) | 1034 -> 1037 (1.00x) | 387 (same) | 389 (same) |
| x86_64-v2 | btver2 | _1076 -> 1127 (0.95x)_ | 1383 -> 1385 (1.00x) | 564 (same) | 613 (same) |
| x86_64-v3 | skylake | 260 -> 261 (0.99x) | _315 -> 328 (0.96x)_ | 141 (same) | 141 (same) |
| x86_64-v3 | alderlake | **229 -> 169 (1.36x)** | _310 -> 320 (0.97x)_ | 138 (same) | 138 (same) |
| x86_64-v3 | znver3 | **322 -> 196 (1.65x)** | **452 -> 419 (1.08x)** | 226 (same) | 226 (same) |
| x86_64-v4 | icelake-server | 158 -> 163 (0.97x) | 257 -> 259 (0.99x) | 71 (same) | 71 (same) |
| x86_64-v4 | sapphirerapids | _135 -> 140 (0.96x)_ | 195 -> 197 (0.99x) | 69 (same) | 122 (same) |
| x86_64-v4 | znver4 | 195 -> 196 (1.00x) | 385 -> 386 (1.00x) | 129 (same) | 193 (same) |
| i686 | skylake | 520 -> 526 (0.99x) | 781 -> 792 (0.99x) | 391 (same) | 308 (same) |
| aarch64 | cortex-a72 | _1380 -> 1544 (0.89x)_ | _1893 -> 1995 (0.95x)_ | 675 (same) | 1058 (same) |
| aarch64 | neoverse-n1 | 1314 -> 1349 (0.97x) | 1443 -> 1479 (0.98x) | 770 (same) | 898 (same) |
| aarch64 | neoverse-v2 | _548 -> 648 (0.85x)_ | _580 -> 650 (0.89x)_ | 275 (same) | 323 (same) |
| aarch64-apple | apple-m1 | _388 -> 414 (0.94x)_ | 453 -> 460 (0.99x) | 259 (same) | 261 (same) |

With only two inputs, the output-sized lanes (16 vectors per i64 input on SSE) did not spill: LLVM
streams load / add / narrow / store per vector, so nothing forces them all live, and the halved
iteration mostly costs a little more loop overhead (NEON most, which LLVM does not unroll). The wins
are on AVX2 integers (alderlake, znver3). The rule is meant for chains whose wide values must stay
live at once (as in `longchain`, where the same-dtype k was tuned); these probes do not show that case.

## SIMD bodies of the ops (`results/py-*`)

`--source py`, against `results/py-baseline` (the ops' auto-vectorized scalar kernels, before any
`simd:` body but Neg / Add / Sub / Mul in f32 / f64 / i32). Speedup per platform: the geomean of
baseline / new cycles over the platform's CPUs. Bold: >= 3% faster, italics: >= 3% slower.
`results/py-tier0` is the first set of bodies, `results/py-tier0b` the retuned bool outputs and
Round.

Most bodies match the baseline exactly (Add, Sub, Neg, Abs, Square, And / Or / Xor / Not, integer
Maximum / Minimum, Floor, Ceil): LLVM already vectorized the scalar kernel, and the explicit body
gives the same code. They are kept, as the SIMD path of those ops. The notable ones:

| kernel | SIMD body | x86_64 | x86_64-v2 | x86_64-v3 | x86_64-v4 | i686 | aarch64 | aarch64-apple |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bitwise_shift_right_i32` | kept | 1.00x | **11.42x** | 1.00x | 1.00x | 1.00x | **1.05x** | 1.00x |
| `bitwise_shift_left_i32` | kept | 1.02x | **5.98x** | 1.00x | 1.00x | **1.13x** | 1.01x | 1.00x |
| `maximum_f32` | kept | **1.17x** | **2.59x** | **1.16x** | 1.00x | **3.59x** | **1.09x** | **1.17x** |
| `maximum_f64` | kept | **1.17x** | **1.13x** | **3.30x** | 1.00x | **1.71x** | **1.09x** | **1.16x** |
| `div_f32` | kept | **2.17x** | **1.31x** | 1.00x | 1.00x | **2.80x** | 1.00x | 1.00x |
| `div_f64` | kept | 1.00x | **1.03x** | **2.53x** | 1.00x | 1.00x | 1.00x | 1.00x |
| `sqrt_f64` | kept | 1.00x | 1.00x | **1.95x** | 1.00x | 1.00x | 1.00x | 1.00x |
| `count_zeros_i32_u32` | kept | 1.00x | **1.31x** | **1.72x** | 1.00x | **1.34x** | 0.98x | 1.00x |
| `count_ones_i32_u32` | removed | _0.82x_ | **1.18x** | **1.88x** | 1.00x | _0.92x_ | 1.00x | 1.00x |
| `bitwise_rotate_right_i32_u32` | removed | _0.75x_ | _0.57x_ | **1.04x** | 1.00x | _0.69x_ | 1.01x | 1.00x |
| `bitwise_shift_right_u8` | removed | 1.00x | 1.00x | 0.99x | _0.63x_ | 1.00x | _0.96x_ | _0.96x_ |
| `mul_i8` | removed | **1.22x** | **1.05x** | 1.00x | 1.00x | **1.70x** | _0.94x_ | 1.00x |
| `less_f32_bool` | removed | _0.40x_ | _0.37x_ | **1.33x** | 0.99x | _0.72x_ | _0.93x_ | _0.57x_ |
| `less_i64_bool` | removed | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | _0.37x_ | _0.47x_ |
| `is_nan_f64_bool` | removed | _0.48x_ | _0.70x_ | _0.43x_ | **1.07x** | _0.20x_ | _0.47x_ | _0.52x_ |
| `round_f32` | removed | _0.10x_ | _0.69x_ | _0.71x_ | _0.91x_ | _0.24x_ | _0.59x_ | _0.38x_ |

- Kept where faster without being slower anywhere: 32-bit shifts (no per-lane shift on SSE4.2,
  which LLVM scalarizes), float Maximum / Minimum (NaN-propagating with a `select`), float Div,
  f64 Sqrt, CountZeros. Removed where slower on some platform: rotates, 8/16/64-bit shifts,
  CountOnes, i8 Mul / Square.
- `bool` outputs (comparisons, IsNan / IsFinite / IsInfinite) are removed: LLVM packs the
  comparison masks of the scalar kernel into bytes better than the `mask` mode (lanes stored and
  compared to 0; `to_bitmask` then per lane, the first version, was 3-10x slower).
- Round (halves away from zero) is removed: LLVM lowers the scalar `round` better (e.g. one
  `frinta` on NEON) than the body (`trunc(|x| + 0.5 - ulp)` with `x`'s sign). Note that on SSE2 the
  baseline calls `roundf` (free in llvm-mca), so its cost there is understated.
