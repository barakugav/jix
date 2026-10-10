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
python jix/probe/analyze.py --label y --no-build     # re-analyze the previous build (minutes)
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
into a packed output through the public API, `to_ndarray_slice`): every `inner_loop_contiguous`
of the build whose pipeline is not a leaf, named by its expression, with the leaves as their
types: `mul(add(f32,f32),add(f32,f32))`. Expressions longer than 80 characters are named by their
root op, depth, leaves and a hash instead: `mul_d5_32xf32_<hash>`. A chain also instantiates loops
for some of its sub-chains, which are analyzed too.

The probes change with the question studied; the earlier ones are in git history. The sections up
to "Mixed dtypes" name them `<chain>_<type>`:

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
and output) contiguous in the inner dimension. `analyze.py` finds it by its demangled name, and binds its `len` argument (elements). Its hot loop is
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
3. Each IR loop is matched to the asm loop holding its header block, named by the verbose asm
   (`.LBB0_5: # %vector.body`), which gives every asm loop its trip count. A kernel whose loops
   do not all match one to one is reported, without a cost.
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

SCEV trip counts are evaluated exactly, with the `len` argument bound (whatever its IR name: it
can be unnamed, or reach the loop through a stack slot named `%len`). Integer values SCEV cannot
see through (e.g. the length after the SIMD level dispatch) are constant-folded from the IR. A
trip count with any other unknown operand is unknown. A loop with panic exits (bounds checks,
division by zero) has no exact count: its symbolic max count is used, which it runs in the steady
state.

Caveats:
- llvm-mca assumes every load hits L1 and ignores the front end. Branches are not followed.
- **Calls are free in llvm-mca.** Kernels that call something (e.g. libm's `expf`) inside the hot
  loop are flagged `*`, and their real cost is higher. x86-64's register-indirect calls are named
  by the GOT entry last loaded into the register.
- **A kernel must not call a Rust function in any loop** (a closure or helper LLVM did not
  inline, e.g. `half`'s software f16 arithmetic): such calls are listed under "Out-of-line Rust
  calls" in `summary.md`, and `analyze.py` exits with an error.
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
that level's `fearless_simd::dispatch!` arm the one inlined into `inner_loop_contiguous` (higher levels stay
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
unnamed (`%1`). `IrValues` resolves a `load` from `%len`, and the
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

`--source py`. Speedup per platform: the geomean of baseline / new cycles over the platform's
CPUs. Bold: >= 3% faster, italics: >= 3% slower.

Asm loops are matched to IR loops by the IR block names of verbose asm. Before that (matching by
shape, vector share and size), the hot loop measured was often a tail loop, mostly in the
baseline: `results/py-baseline`, `py-tier0`, `py-tier0b`, `py-cast`, `py-cast2` and `py-cast3` are
from that analysis and are not reliable. The current ones:
- `results/py-baseline3`: the ops' auto-vectorized scalar kernels, before any `simd:` body.
- `results/py-head3` / `py-head4`: the kept bodies (`py-head4`: Cast and the shifts revised).
- `results/py-allsimd3`: a Cast body for every integer / float pair.

Most bodies match the baseline exactly (Add, Sub, Neg, Abs, Square, And / Or / Xor / Not, integer
Maximum / Minimum, Floor, Ceil): LLVM already vectorized the scalar kernel, and the explicit body
gives the same code. They are kept, as the SIMD path of those ops. The notable ones:

| kernel | SIMD body | x86_64 | x86_64-v2 | x86_64-v3 | x86_64-v4 | i686 | aarch64 | aarch64-apple |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bitwise_shift_right_i32` | kept | 1.00x | **1.25x** | 1.00x | 1.00x | 1.00x | **1.05x** | 1.00x |
| `maximum_f32` | kept | **1.17x** | **2.59x** | **1.16x** | 1.00x | **3.59x** | **1.09x** | **1.17x** |
| `maximum_f64` | kept | **1.17x** | **1.13x** | **3.30x** | 1.00x | **1.71x** | **1.09x** | **1.16x** |
| `div_f32` | kept | **2.17x** | **1.31x** | 1.00x | 1.00x | **2.80x** | 1.00x | 1.00x |
| `div_f64` | kept | 1.00x | **1.03x** | **2.53x** | 1.00x | 1.00x | 1.00x | 1.00x |
| `sqrt_f32` | kept | 1.00x | **1.14x** | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| `sqrt_f64` | kept | 1.00x | 1.00x | **1.95x** | 1.00x | 1.00x | 1.00x | 1.00x |
| `count_zeros_i32_u32` | kept | 1.00x | **1.31x** | **1.72x** | 1.00x | **1.34x** | 0.98x | 1.00x |
| `bitwise_shift_left_i32` | removed | 1.02x | _0.31x_ | 1.00x | 1.00x | **1.13x** | 1.01x | 1.00x |

- The 32-bit shift left body is removed: LLVM vectorizes the scalar kernel's `x << n` as a multiply
  by `2^n` (`pslld 23` + `cvttps2dq` + `pmulld`) where fearless_simd's per-lane shift extracts the
  lanes on SSE4.2. The bodies removed before (rotates, 8/16/64-bit shifts, CountOnes, i8 Mul /
  Square, `bool` outputs, Round) were judged by the old analysis.

### Cast

`Cast::cast_bulk` (the Cast kernel's `apply_bulk`) has SIMD bodies per pair of types, as steps on
arrays of vectors (`widen`, `narrow`, `bitcast`, `float`, `truncate`). Kept where faster or even
(within 5%) on every platform, even ones too so that a fused chain of ops stays in vectors:

| kernel | SIMD body | x86_64 | x86_64-v2 | x86_64-v3 | x86_64-v4 | i686 | aarch64 | aarch64-apple |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cast_f32_i32` | kept | 1.00x | **6.02x** | **13.69x** | **19.57x** | 1.00x | 0.99x | 1.00x |
| `cast_f32_u32` | kept | 1.00x | **2.82x** | **6.13x** | **43.69x** | 1.00x | 0.99x | 1.00x |
| `cast_f32_i64` | kept | _0.96x_ | 0.98x | **1.58x** | **7.13x** | 1.00x | **1.03x** | **1.53x** |
| `cast_f32_u64` | kept | _0.95x_ | 0.97x | **4.07x** | **10.63x** | 1.01x | **1.03x** | **1.53x** |
| `cast_f64_i64` | kept | 1.00x | 1.00x | **2.14x** | **9.79x** | 1.00x | 1.00x | 1.00x |
| `cast_f64_u64` | kept | 1.00x | 1.00x | **5.53x** | **15.43x** | 1.00x | 1.00x | 1.00x |
| `cast_u64_f64` | kept | 1.00x | **1.13x** | **1.20x** | 1.00x | 1.00x | 1.00x | 1.00x |
| `cast_i32_f32` | kept | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 0.99x | 1.00x |
| `cast_i16_i8` | kept | 1.00x | 1.00x | **1.73x** | 1.00x | 1.00x | _0.92x_ | 1.00x |
| `cast_i32_i16` | kept | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | _0.92x_ | 1.00x |
| `cast_i64_i32` | kept | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | _0.92x_ | 1.00x |
| `cast_i16_i32` | removed | 0.97x | _0.84x_ | 1.00x | 1.00x | 0.99x | **1.10x** | **1.17x** |
| `cast_i8_i32` | removed | **1.20x** | _0.72x_ | _0.55x_ | _0.58x_ | **1.31x** | **1.25x** | **1.13x** |
| `cast_f32_f64` | removed | _0.74x_ | _0.97x_ | 1.00x | 1.00x | _0.67x_ | **1.06x** | **1.17x** |
| `cast_f32_u16` | removed | _0.58x_ | **2.10x** | **3.71x** | **13.56x** | _0.48x_ | 1.00x | 0.97x |
| `cast_f32_i8` | removed | _0.60x_ | **4.49x** | **8.17x** | **10.54x** | _0.58x_ | **1.28x** | **1.48x** |
| `cast_i32_f64` | removed | _0.28x_ | _0.43x_ | _0.38x_ | _0.85x_ | _0.17x_ | **1.13x** | **1.35x** |
| `cast_i64_f64` | removed | 1.00x | _0.95x_ | **3.52x** | 1.00x | 1.00x | 1.00x | 1.00x |
| `cast_f64_i32` | removed | _0.54x_ | _0.72x_ | **1.49x** | **5.90x** | _0.26x_ | **1.72x** | **2.75x** |
| `cast_f64_f32` | removed | 1.00x | 1.00x | 1.00x | _0.77x_ | 1.00x | 1.02x | 1.00x |
| `cast_f32` | removed | 1.00x | 1.00x | 1.00x | 1.00x | _0.34x_ | 1.00x | 1.00x |

- The aarch64 0.92x of the narrowings is the same instructions as the baseline (`uzp1`).
- Removed, as slower on some platform (often much faster on others; a body per SIMD level would
  keep those):
  - Integer widening and f32 -> f64: fearless_simd's `widen` loads a full vector and shifts out
    its high half (`psrldq` / `movhlps`) before `pmovsx` / `cvtps2pd`, where LLVM folds two
    half-width loads into them.
  - Integer to f64 and f64 to narrower integers on SSE2 / SSE4.2: through i64 <-> f64, which
    fearless_simd emulates (magic numbers) where LLVM converts the lanes with `cvtsi2sd` /
    `cvttsd2si`; on AVX2 / AVX-512 the emulation wins.
  - f32 to 8 / 16-bit integers on SSE2: fearless_simd's precise f32 -> u32 is scalarized with
    branches, where LLVM clamps (`minss` / `maxss`) branch-free.
  - f64 -> f32 on AVX-512: `narrow` joins the two halves (`vinsertf32x8`, port 5 like
    `vcvtpd2ps`), where LLVM stores each half.
  - Same-type casts: copies, slower on i686's fallback level.

### Cast bodies per SIMD level (`results/py-levels`)

A Cast body can be limited to fearless_simd levels (`on Sse2 | Neon`; a match on
`Simd::level()` that folds away per `S`). A body is on a level when it is faster or even (within
3%, or the same instructions) on every platform of the level in `py-allsimd3`. Geomean speedup
over `py-baseline3` across all cast kernels: x86_64 1.02x, x86_64-v2 1.07x, x86_64-v3 1.19x,
x86_64-v4 1.35x, i686 1.01x, aarch64 1.08x, aarch64-apple 1.15x. No kernel slower than 0.97x except
the aarch64 0.92x of the one-step narrowings (the same instructions, other registers) and i16 ->
f32 on x86_64 (0.94x, the same instructions; 0.99x on i686).

## Fused chains over several dtypes: `MAX_ITEMSIZE` and `WIDEST_VECTORS` (`results/mi-*`, `results/py-widest`)

The probes of the previous section had two inputs, which never forced the wide values to stay live.
New probes (`src/lib.rs`) chain several nodes whose values are wider than the output, and whose
dtypes all differ from it:

- `cast_u8(add(mul(cast_f32(u8),f32),cast_f32(u8)))`: an image blend, u8 -> f32 -> u8.
- `greater(cast_f32(u8),f32)`: a threshold mask; `greater(mul(add(f64,f64),f64),f64)`: a compare of
  an f64 chain; `and(greater(f32,f32),less(f32,f32))`: a range mask (all to bool).
- `add(where(greater(f64,f64),f32,f32),f32)`: a select on an f64 compare (a top-level `where` does
  not read through the pipeline, so it is nested in an add).
- `cast_i16(mul(add(i64,i64),sub(i64,i64)))`, the f64 `longchain` to f32, an f32 chain computed in
  f64 (`cast_f32(add(mul(cast_f64(f32),cast_f64(f32)),cast_f64(f32)))`) and widened i8 products to
  i16 (`cast_i16(mul(cast_i32(i8),cast_i32(i8)))`).

Same-dtype kernels are identical across all the variants below. Speedup = cycles of the base /
cycles of the new, geomean over the platform's CPUs; bold / italics: > 3% faster / slower; `*`: a
call in the hot loop on either side, always on the output-sized side here (`memcpy` / `memset` of
lane arrays that no longer fit in registers, not counted by llvm-mca: those speedups are lower
bounds).

**`MAX_ITEMSIZE` (`mi-max`) vs lanes by the output (`mi-out`)**, 8 vectors (4 on i686) of the
widest value vs of the output:

| kernel | x86_64-v4 | x86_64-v3 | x86_64-v2 | x86_64 | aarch64-apple | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `add(where(greater(f64,f64),f32,f32),f32)` | **1.10** | **5.64** | 0.98 | 0.98 | _0.94_ | _0.88_ | **1.37** |
| `and(greater(f32,f32),less(f32,f32))` | 1.00 | 0.99 | **1.44** | **1.38** | _0.89_ | _0.97_ | **2.21** |
| `cast_f32(add(f64,f64))` | 0.99 | 0.98 | 1.00 | **1.04** | _0.92_ | _0.95_ | 0.98 |
| `cast_f32(add(mul(add(f64,f64),sub(f64,f64)),mul(add(f64,f64),sub(f64,f64))))` | 1.00 | **1.11** | **1.04** | **1.12** | _0.95_ | _0.93_ | **1.12** |
| `cast_f32(add(mul(cast_f64(f32),cast_f64(f32)),cast_f64(f32)))` | 1.01 | 1.01 | **1.03** | 1.02 | 0.99 | 1.01 | 0.99 |
| `cast_i16(mul(add(i64,i64),sub(i64,i64)))` | 0.99 | **3.98\*** | **4.39\*** | **2.49\*** | **1.48** | **1.40** | 1.00 |
| `cast_i16(mul(cast_i32(i8),cast_i32(i8)))` | 0.99 | 0.99 | 0.99 | **1.03** | _0.93_ | _0.86_ | 1.00 |
| `cast_i32(add(i64,i64))` | 0.97 | 0.99 | 0.97 | 0.99 | _0.85_ | _0.97_ | 0.99 |
| `cast_u8(add(mul(cast_f32(u8),f32),cast_f32(u8)))` | **1.04** | _0.93_ | **1.10** | **1.50\*** | 0.99 | **1.10** | _0.85_ |
| `greater(cast_f32(u8),f32)` | **1.08** | 1.02 | **1.33** | **2.42\*** | 0.99 | _0.92_ | **1.53** |
| `greater(f32,f32)` | **1.19** | **1.09** | **1.36** | **1.32** | _0.84_ | _0.93_ | **1.94** |
| `greater(f64,f64)` | **1.14\*** | **2.16\*** | **3.03\*** | **3.18\*** | _0.83_ | 1.02 | **1.51** |
| `greater(mul(add(f64,f64),f64),f64)` | **2.35\*** | **3.36\*** | **3.66\*** | **3.80\*** | **4.37\*** | **2.51\*** | **1.88** |
| `less(f32,f32)` | **1.19** | **1.09** | **1.38** | **1.32** | _0.84_ | _0.93_ | **1.93** |

On x86 the widest-value lanes win wherever the ratio of widest to output is large: compares to bool
(1.1-3.8x), the i64 -> i16 chain (2.5-4.4x), the select on AVX2 (5.6x: with output lanes, LLVM
gives up on vectorizing it). AVX-512 (32 registers) gains less, but still up to 2.35x. On aarch64,
the rule is too strict: NEON has 32 registers and LLVM does not unroll its loops, so the halved
iteration of `mi-max` costs 5-17% on most chains (except the ones that spill: 1.4-4.4x).

**`WIDEST_VECTORS`**: the lanes are `CONTIGUOUS_VECTORS` vectors of the output, at most
`WIDEST_VECTORS` vectors of the widest value. `WIDEST_VECTORS = 2 * CONTIGUOUS_VECTORS` (`mi-w2`)
for all platforms is worse on x86 (0.71-0.76 on SSE compares to bool, 0.18-0.53 on AVX2 for the
select and the f64 compares, 0.73 on i686), mixed on AVX-512 (0.90-1.07), better on aarch64:
committed for aarch64 only (`mi-new`; x86 / i686 identical to `mi-max`, aarch64 identical to
`mi-w2`). `mi-new` vs `mi-max` on aarch64:

| kernel | aarch64-apple | aarch64 |
|---|---:|---:|
| `add(where(greater(f64,f64),f32,f32),f32)` | **1.06** | **1.14** |
| `and(greater(f32,f32),less(f32,f32))` | **1.09** | **1.04** |
| `cast_f32(add(f64,f64))` | **1.09** | **1.06** |
| `cast_f32(add(mul(add(f64,f64),sub(f64,f64)),mul(add(f64,f64),sub(f64,f64))))` | **1.05** | **1.07** |
| `cast_f32(add(mul(cast_f64(f32),cast_f64(f32)),cast_f64(f32)))` | 1.01 | 0.99 |
| `cast_i16(mul(add(i64,i64),sub(i64,i64)))` | _0.89_ | _0.91_ |
| `cast_i16(mul(cast_i32(i8),cast_i32(i8)))` | **1.08** | **1.16** |
| `cast_i32(add(i64,i64))` | **1.17** | **1.04** |
| `cast_u8(add(mul(cast_f32(u8),f32),cast_f32(u8)))` | 1.01 | **1.03** |
| `greater(cast_f32(u8),f32)` | 1.01 | **1.04** |
| `greater(f32,f32)` | **1.12** | **1.04** |
| `greater(f64,f64)` | **1.13** | 0.97 |
| `greater(mul(add(f64,f64),f64),f64)` | **1.07** | **1.04** |
| `less(f32,f32)` | **1.12** | **1.04** |

Net, `mi-new` vs output lanes (`mi-out`) on aarch64: the spilling chains keep their gains
(`greater(mul(add(f64,f64),f64),f64)` 4.7x on M1, 2.6x on the others; the i64 -> i16 chain 1.3x;
the blend 1.14x on the others), the simple compares are 4-7% behind.

**On the single ops of `jix-py`** (`py-widest` vs `final`, aarch64 only, as nothing else changes):

| platform | CPU | kernels | geomean | faster | slower | p10 | median | p90 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aarch64-apple | apple-m1 | 391 | **1.01x** | 15% | 2% | 1.00 | 1.00 | 1.09 | 0.52 | 2.56 |
| aarch64 | cortex-a72 | 391 | **1.01x** | 17% | 2% | 1.00 | 1.00 | 1.06 | 0.74 | 1.15 |
| aarch64 | neoverse-n1 | 391 | **1.01x** | 12% | 7% | 0.98 | 1.00 | 1.07 | 0.71 | 1.62 |
| aarch64 | neoverse-v2 | 391 | **1.02x** | 15% | 5% | 1.00 | 1.00 | 1.10 | 0.78 | 2.27 |

Faster: compares to bool (1.04-1.13x), narrowing casts (1.08-1.24x), complex128 compares and
casts to bool (1.4-2.6x). Slower: only kernels without a SIMD body that, at 128 output bytes per
iteration, are no longer fully unrolled by LLVM (an inner loop of 8-64 scalar trips remains in
the hot loop): casts to f16 on M1 (`cast_i32_f16` / `cast_u32_f16` 0.52, `cast_f32_f16` 0.62,
`cast_i64_f16` 0.80, `cast_f64_f16` 0.85), `less_f16_bool` / `greater_equal_f16_bool` (0.74-0.81)
and `approx_eq_complex128_bool` (0.78-0.85). SIMD bodies for f16 would remove that cliff.

## Deep fused trees: register spills vs `CONTIGUOUS_VECTORS` (`results/tree-v*`)

Does a deep fused pipeline spill? `read_bulk` evaluates a node's left child to a `[T; LANES]`
value, then its right one: a balanced tree keeps one such value live per level, each
`CONTIGUOUS_VECTORS` vectors. Probes (`src/lib.rs`): balanced trees of depth 2-5 (4-32 leaves,
levels `+ * - + *` from the leaves) in f32, f64 and i32 (depth 4), and as a control a left-deep
chain over 16 f32 leaves (`((x0 + x1) * x2 + x3) * ...`, two values live at once). Swept:
`CONTIGUOUS_VECTORS` (`v`) = 1, 2, 4, 8, 16 on every platform (committed: 8, 4 on i686), lanes from
`MAX_ITEMSIZE` (the same as the output's here). The hot loop's stack accesses are counted by
`analyze.py` (`stack` column): an instruction with an `[rsp` / `[esp` (x86) or `[sp` / `[x29`
(aarch64) operand, split by whether its register operand is a vector one (a spill or a reload,
often folded into the arithmetic: `vmulps ymm1, ymm4, ymmword ptr [rsp + N]`) or a general one
(on x86-64, mostly the leaves' data pointers reloaded once per iteration: 16 or more leaves do not
fit in the general registers).

f64 is within 3% of f32 everywhere (the same instructions per byte), so its rows are left out.

**Summary**: the best `v` per kernel and platform, with the speedup of the best over the committed
`v` (8, i686 4); `S`: the committed `v` spills vector registers.

| kernel | x86_64-v4 | x86_64-v3 | x86_64-v2 | x86_64 | i686 | aarch64 | aarch64-apple |
|---|---:|---:|---:|---:|---:|---:|---:|
| tree d2 f32 | 16 (1.00x) | 8 (1.00x) | 8 (1.00x) | 8 (1.00x) | 4 (1.00x) | 16 (1.04x) | 8 (1.00x) |
| tree d3 f32 | 16 (1.00x) | 8 (1.00x) | 4 (1.03x) S | 4 (1.05x) S | 4 (1.00x) S | 8 (1.00x) | 8 (1.00x) |
| tree d4 f32 | 8 (1.00x) | 4 (1.11x) S | 4 (1.05x) S | 4 (1.11x) S | 4 (1.00x) S | 1 (1.02x) S | 8 (1.00x) S |
| tree d5 f32 | 8 (1.00x) | 4 (1.26x) S | 4 (1.07x) S | 4 (1.11x) S | 8 (1.00x) S | 1 (1.03x) S | 8 (1.00x) S |
| tree d4 i32 | 4 (1.01x) | 4 (1.10x) S | 4 (1.09x) S | 1 (1.09x) S | 4 (1.00x) S | 1 (1.00x) S | 16 (1.10x) S |
| leftchain f32 | 16 (1.07x) | 16 (1.07x) | 8 (1.00x) | 8 (1.00x) | 4 (1.00x) | 4 (1.02x) | 16 (1.02x) |

**Full tables**: cycles per 4096 output bytes (geomean over the platform's CPUs), in parentheses
relative to the best `v` (bold), then the stack accesses per 4096 output bytes, `vector/general`.

#### x86_64-v4 (64-byte vectors, 32 vector registers; cycles: geomean over icelake-server, sapphirerapids, znver4)

| kernel | v=1 | v=2 | v=4 | v=8 | v=16 |
|---|---:|---:|---:|---:|---:|
| tree d2 f32 | 158 (1.08) 0/0 | 158 (1.08) 0/0 | 151 (1.03) 0/0 | 147 (1.00) 0/0 | **147 (1.00) 0/0** |
| tree d3 f32 | 335 (1.05) 0/0 | 326 (1.02) 0/0 | 322 (1.01) 0/0 | 320 (1.00) 0/0 | **319 (1.00) 0/0** |
| tree d4 f32 | 707 (1.02) 0/320 | 697 (1.01) 0/160 | 692 (1.00) 0/80 | **690 (1.00) 0/40** | 690 (1.00) 168/20 |
| tree d5 f32 | 1535 (1.09) 0/1344 | 1417 (1.01) 0/672 | 1410 (1.00) 0/336 | **1409 (1.00) 0/176** | 1461 (1.04) 744/88 |
| tree d4 i32 | 628 (1.03) 0/320 | 619 (1.01) 0/160 | **612 (1.00) 0/80** | 618 (1.01) 0/40 | 621 (1.01) 160/20 |
| leftchain f32 | 857 (1.35) 0/320 | 777 (1.22) 0/160 | 708 (1.11) 0/80 | 679 (1.07) 0/40 | **637 (1.00) 0/20** |

#### x86_64-v3 (32-byte vectors, 16 vector registers; cycles: geomean over skylake, alderlake, znver3)

| kernel | v=1 | v=2 | v=4 | v=8 | v=16 |
|---|---:|---:|---:|---:|---:|
| tree d2 f32 | 277 (1.08) 0/0 | 278 (1.08) 0/0 | 265 (1.03) 0/0 | **258 (1.00) 0/0** | 297 (1.15) 128/0 |
| tree d3 f32 | 534 (1.11) 0/0 | 501 (1.04) 0/0 | 492 (1.02) 0/0 | **481 (1.00) 0/0** | 538 (1.12) 176/0 |
| tree d4 f32 | 1134 (1.15) 0/640 | 1030 (1.05) 0/320 | **984 (1.00) 0/160** | 1092 (1.11) 384/80 | 1299 (1.32) 1024/48 |
| tree d5 f32 | 2726 (1.35) 0/2688 | 2236 (1.10) 0/1344 | **2026 (1.00) 0/672** | 2544 (1.26) 1760/352 | 3024 (1.49) 3312/176 |
| tree d4 i32 | 1177 (1.16) 0/640 | 1068 (1.05) 0/320 | **1017 (1.00) 0/160** | 1122 (1.10) 320/80 | 1312 (1.29) 896/48 |
| leftchain f32 | 1558 (1.63) 0/640 | 1333 (1.39) 0/320 | 1130 (1.18) 0/160 | 1024 (1.07) 0/80 | **956 (1.00) 0/40** |

#### x86_64-v2 (16-byte vectors, 16 vector registers; cycles: geomean over sandybridge, btver2)

| kernel | v=1 | v=2 | v=4 | v=8 | v=16 |
|---|---:|---:|---:|---:|---:|
| tree d2 f32 | 902 (1.08) 0/0 | 902 (1.08) 0/0 | 857 (1.03) 0/0 | **833 (1.00) 0/0** | 978 (1.17) 224/0 |
| tree d3 f32 | 1814 (1.09) 0/0 | 1718 (1.03) 0/0 | **1669 (1.00) 0/0** | 1724 (1.03) 256/0 | 2018 (1.21) 832/0 |
| tree d4 f32 | 3953 (1.08) 0/1280 | 3685 (1.01) 0/640 | **3643 (1.00) 0/320** | 3813 (1.05) 896/192 | 4247 (1.17) 1952/96 |
| tree d5 f32 | 9937 (1.31) 0/5632 | 7916 (1.04) 0/2688 | **7615 (1.00) 256/1344** | 8169 (1.07) 2176/704 | 8766 (1.15) 4288/352 |
| tree d4 i32 | 4286 (1.16) 0/1280 | 3940 (1.06) 0/640 | **3705 (1.00) 0/320** | 4054 (1.09) 960/192 | 4522 (1.22) 1952/96 |
| leftchain f32 | 4565 (1.42) 0/1280 | 3841 (1.20) 0/640 | 3407 (1.06) 0/320 | **3204 (1.00) 0/160** | 3809 (1.19) 1312/96 |

#### x86_64 (16-byte vectors, 16 vector registers; cycles: geomean over sandybridge, skylake, znver3)

| kernel | v=1 | v=2 | v=4 | v=8 | v=16 |
|---|---:|---:|---:|---:|---:|
| tree d2 f32 | 614 (1.03) 0/0 | 615 (1.03) 0/0 | 605 (1.01) 0/0 | **599 (1.00) 0/0** | 688 (1.15) 224/0 |
| tree d3 f32 | 1198 (1.03) 0/0 | 1174 (1.01) 0/0 | **1160 (1.00) 0/0** | 1220 (1.05) 256/0 | 1450 (1.25) 832/0 |
| tree d4 f32 | 2607 (1.09) 0/1280 | 2472 (1.03) 0/640 | **2400 (1.00) 0/320** | 2660 (1.11) 896/192 | 3028 (1.26) 1952/96 |
| tree d5 f32 | 6491 (1.29) 0/5632 | 5217 (1.03) 0/2688 | **5047 (1.00) 256/1344** | 5618 (1.11) 2176/704 | 6290 (1.25) 4288/352 |
| tree d4 i32 | **3914 (1.00) 0/1280** | 3962 (1.01) 0/640 | 3932 (1.00) 128/320 | 4277 (1.09) 1920/192 | 4941 (1.26) 3520/96 |
| leftchain f32 | 3485 (1.55) 0/1280 | 2874 (1.27) 0/640 | 2387 (1.06) 0/320 | **2255 (1.00) 0/160** | 2738 (1.21) 1312/96 |

#### i686 (16-byte vectors, 8 vector registers; cycles: geomean over skylake)

| kernel | v=1 | v=2 | v=4 | v=8 | v=16 |
|---|---:|---:|---:|---:|---:|
| tree d2 f32 | 556 (1.06) 0/0 | 534 (1.02) 0/0 | **524 (1.00) 0/0** | 582 (1.11) 256/0 | 668 (1.28) 608/0 |
| tree d3 f32 | 1723 (1.30) 0/1280 | 1371 (1.03) 0/640 | **1329 (1.00) 512/320** | 1432 (1.08) 1024/192 | 1501 (1.13) 1472/96 |
| tree d4 f32 | 3781 (1.31) 0/3328 | 2915 (1.01) 0/1664 | **2897 (1.00) 1024/960** | 2920 (1.01) 2304/480 | 3044 (1.05) 3136/224 |
| tree d5 f32 | 7885 (1.34) 0/7424 | 6184 (1.05) 512/3840 | 5907 (1.00) 2560/1920 | **5896 (1.00) 4864/960** | 6100 (1.03) 6528/480 |
| tree d4 i32 | 5176 (1.01) 0/3328 | 5157 (1.00) 512/1664 | **5139 (1.00) 2048/896** | 5259 (1.02) 4224/448 | 5590 (1.09) 4352/224 |
| leftchain f32 | 3863 (1.42) 0/3328 | 3139 (1.16) 0/1664 | **2712 (1.00) 0/832** | 2855 (1.05) 1792/416 | 3372 (1.24) 4640/208 |

#### aarch64 (16-byte vectors, 32 vector registers; cycles: geomean over cortex-a72, neoverse-n1, neoverse-v2)

| kernel | v=1 | v=2 | v=4 | v=8 | v=16 |
|---|---:|---:|---:|---:|---:|
| tree d2 f32 | 1651 (1.63) 0/0 | 1424 (1.40) 0/0 | 1229 (1.21) 0/0 | 1053 (1.04) 0/0 | **1015 (1.00) 0/0** |
| tree d3 f32 | 3051 (1.50) 0/0 | 2598 (1.28) 0/0 | 2230 (1.10) 0/0 | **2030 (1.00) 0/0** | 2345 (1.16) 416/0 |
| tree d4 f32 | **4285 (1.00) 0/0** | 4768 (1.11) 0/0 | 4389 (1.02) 192/0 | 4351 (1.02) 544/0 | 4968 (1.16) 1424/0 |
| tree d5 f32 | **8692 (1.00) 0/2048** | 9918 (1.14) 384/896 | 9295 (1.07) 1280/512 | 8960 (1.03) 1504/256 | 10092 (1.16) 3248/160 |
| tree d4 i32 | **3637 (1.00) 0/0** | 4243 (1.17) 0/0 | 3804 (1.05) 0/0 | 3653 (1.00) 64/64 | 3850 (1.06) 368/48 |
| leftchain f32 | 4158 (1.06) 0/0 | 4109 (1.05) 0/0 | **3910 (1.00) 0/0** | 4001 (1.02) 0/0 | 4121 (1.05) 0/0 |

#### aarch64-apple (16-byte vectors, 32 vector registers; cycles: geomean over apple-m1)

| kernel | v=1 | v=2 | v=4 | v=8 | v=16 |
|---|---:|---:|---:|---:|---:|
| tree d2 f32 | 724 (1.86) 0/0 | 531 (1.36) 0/0 | 458 (1.18) 0/0 | **390 (1.00) 0/0** | 419 (1.07) 64/0 |
| tree d3 f32 | 1247 (1.66) 0/0 | 961 (1.28) 0/0 | 908 (1.21) 0/0 | **753 (1.00) 0/0** | 891 (1.18) 352/0 |
| tree d4 f32 | 2496 (1.57) 0/0 | 2335 (1.47) 0/0 | 1679 (1.06) 0/0 | **1591 (1.00) 96/0** | 1923 (1.21) 1104/0 |
| tree d5 f32 | 5445 (1.65) 0/2304 | 4388 (1.33) 0/1152 | 3408 (1.03) 0/576 | **3302 (1.00) 448/352** | 4163 (1.26) 2592/112 |
| tree d4 i32 | 2486 (1.84) 0/0 | 2327 (1.73) 0/0 | 1548 (1.15) 0/0 | 1479 (1.10) 128/32 | **1348 (1.00) 112/48** |
| leftchain f32 | 3546 (2.31) 0/0 | 2413 (1.57) 0/0 | 1753 (1.14) 0/0 | 1576 (1.02) 0/0 | **1538 (1.00) 0/0** |

Observations:

- **Spills do happen, at the committed `v`, from depth 3-4**: on AVX2 and SSE (16 vector registers)
  from depth 4 (SSE: 3), 4.5-25% slower than the best `v` (AVX2 depth 5: 1.26x, 1760 vector stack
  accesses per 4096 B); on i686 (8 registers) from depth 3 already, though there `v` = 4 is still
  about the best. AVX-512 (32 registers) does not spill up to depth 5 at `v` = 8. NEON (32 registers)
  spills at depth 4-5 (M1: 96-448 accesses per 4096 B, generic aarch64: 544-1504), but that costs
  little: `v` = 8 is still the best on M1, within 1.03x of the best on the others.
- **Fewer vectors trade spills for loop overhead**: halving `v` stops the spills on AVX2 / SSE
  (`v` = 4 is the best there for depth 3-5 trees), but costs 3-10% on the shallow trees and 6-18%
  on the left-deep chain, which never spills up to `v` = 8 and prefers 8-16. No single `v` is best
  for all shapes: the best follows the number of live values (tree depth), not the dtype.
- **The cost of a spill is moderate**: the folded reloads are cheap loads from L1, the stores
  compete for the store port; the worst measured case is 1.26x (AVX2, depth 5) at `v` = 8 and 1.49x
  at `v` = 16. Up to `v` = 8 nothing falls off a cliff.
- **`v` = 16 spills on every platform** but AVX-512 (and NEON for depth 2) and is never much better
  (AVX2/AVX-512 left chain 1.07x, M1 i32 1.10x).
- **aarch64 (generic)**: depth 4-5 prefer `v` = 1 (1.02-1.03x over 8), but the shape is irregular
  (`v` = 2 is the worst there); LLVM does not unroll NEON loops, so `v` sets the unroll.
- **General-register stack accesses** (leaf pointers) are a constant per iteration (about the
  leaves beyond ~12 on x86-64, more on i686), so they shrink per byte as `v` grows, and do not
  explain the slowdowns at high `v`.

## Reductions: the contiguous inner loops, and multiversioning them (`results/red-*`)

The inner loops of a reduction (`jix/src/ops/reduction.rs`), for contiguous operands:

- `cell`: a run of items folded into one cell (the reduced axis is the innermost):
  `fold_run_leaf`, `ONE_CELL_LANES` = 16 lane accumulators, merged at the end.
- `init` / `update`: the innermost axis is not reduced: item `i` folded into cell `i` (the first
  item of the cells, or a following one): `fold_across_cells_contiguous`.
- `finalize`: the states turned into the outputs: `finalize_states_contiguous`.

Kernels: `reduce_<op>_<type>_<loop>`, found in `jix-py` (`--fn 'reduce_*'`), cost in cycles per 4096
bytes of items (of cells for `finalize`), counted at the item type's size. LLVM merges identical
functions, so some kernels show under one name only (sum is mean's `cell`, argmin's `init` is
argmax's...). `init` / `finalize` loops that LLVM turned into a `memcpy` (sum, mean, product
`init`) are not measured. Variance / standard deviation `finalize` have two alternative hot loops
(the computing one and a NaN fill for `ddof >= n`, unswitched by LLVM) which are summed: both
sides overstated.

Changes for the analysis (`ddcd150`): the across-cells and finalize loops were inlined into the
dyn-called tile closure, which holds all four loop shapes; they became `#[inline(never)]` functions
with scalar arguments, `len` last. The `cell` loop `while i < end { ...; i += LANES }` has no
SCEV trip count (wraparound) and became a counted loop: per item within -5% / +11% of the original
on AVX2 (LLVM now unrolls the simple ones by 2). The analyzer also finds irreducible loops (a
remainder loop entered at its body and at its exit test), which no elementwise kernel had (their
loop trees are unchanged).

Configurations:

- `red-base`: the shipped code (no multiversioning). The x86_64-v2/v3/v4 platforms compile the
  whole crate with the level's features, which is what the compiler can do at that level, not
  what users run: they run the x86_64 (SSE2) build, `red-base-x86` (the x86_64 build simulated on
  every x86 CPU) is the baseline of the x86 platforms below.
- `red-mv` (`17329f3`, `90313c3`): the three contiguous entry points dispatch on the runtime level
  (`fearless_simd::dispatch!(SimdLevel::new(), ...)`), the loops left to auto-vectorize. Two traps on the way: the
  dispatch captures by reference, and a captured constant (a stride) is reloaded in the
  dispatched arm after the level detection's possible call, no longer a constant (finalize on AVX2
  was 20-30x slower until the strides were computed inside the dispatched expression); and the
  per-lane closures of `cell` need `#[inline(always)]` (f16 called them out of line).

`red-mv` against a build with the level's features for the whole crate (`red-base` x86_64-v2/v3/v4):
geomean 0.99-1.00 (AVX-512), 0.98 (AVX2), 0.99 (SSE4.2): the dispatch reaches what the compiler
does at the level, except a few complex `cell` loops (0.38-0.6: 16 lanes of complex128 fill all
16 xmm registers, the dispatched arm needs one more and spills).

### Speedups

Speedup = shipped cycles / multiversioned cycles, geomean over the platform's CPUs and over the
op's types (range when they differ by more than 15%).

#### `cell`: speedup of multiversioned (`red-mv`) over the shipped code

| op | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1.15 | 0.05 | 0.08 | 1.00 | 0.99 | 1.00 | 1.00 |
| any | 1.15 | 0.05 | 0.08 | 1.00 | 0.99 | 1.00 | 1.00 |
| arg_max | 3.78 (1.00 f16 - 10.95 f32) | 1.38 (0.82 u8 - 3.93 f32) | 1.10 (0.52 u8 - 3.28 f32) | 1.07 (0.98 f64 - 1.20 bool) | 0.99 (0.87 i16 - 1.03 u16) | 1.00 | 0.99 (0.94 i16 - 1.10 f16) |
| arg_min | 3.77 (1.00 f16 - 10.95 f32) | 1.34 (0.80 bool - 3.94 f32) | 1.00 (0.47 bool - 3.28 f32) | 1.04 (0.98 bool - 1.17 i8) | 1.01 (0.87 i16 - 1.30 bool) | 1.00 | 0.96 (0.81 f16 - 1.01 f32) |
| max | 1.38 (0.60 u8 - 4.26 f64) | 1.14 (0.05 u8 - 5.66 u32) | 0.99 (0.08 u8 - 3.77 u32) | 0.99 | 1.00 | 1.00 | 1.00 |
| mean | 2.00 (0.38 i64 - 16.72 f16) | 1.82 (0.71 complex128 - 16.78 f16) | 1.28 (0.74 complex128 - 1.79 bool) | 0.98 (0.72 complex128 - 1.11 u16) | 0.95 (0.74 i32 - 1.00 bool) | 1.00 | 1.26 (0.84 i32 - 3.23 f32) |
| min | 1.41 (0.60 u8 - 4.26 f64) | 1.16 (0.05 u8 - 6.08 u32) | 1.02 (0.08 u8 - 4.08 u32) | 1.01 (0.90 f64 - 1.14 i32) | 1.00 | 1.00 | 1.05 (0.98 i64 - 1.32 i8) |
| product | 2.02 (0.95 f32 - 17.54 f16) | 1.47 (0.39 complex64 - 17.60 f16) | 0.91 (0.33 complex64 - 1.00 f16) | 0.88 (0.25 complex64 - 1.01 f16) | 1.00 | 1.00 | 1.19 (0.87 f64 - 3.23 f32) |
| variance | 1.42 (0.94 u16 - 3.31 i64) | 1.34 (0.89 u8 - 2.96 i64) | 1.16 (0.93 u8 - 2.08 i64) | 0.97 (0.86 bool - 1.11 f64) | 1.32 (0.81 i16 - 3.38 u16) | 1.17 (0.98 complex64 - 1.61 u16) | 1.02 (0.89 bool - 1.13 i16) |
| **geomean** | **2.04** | **1.27** | **1.00** | **0.99** | **1.04** | **1.03** | **1.06** |

#### `update`: speedup of multiversioned (`red-mv`) over the shipped code

| op | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4.06 | 2.19 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| any | 4.06 | 2.19 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| arg_max | 1.35 (1.00 bool - 3.64 f64) | 1.03 (1.00 u64 - 1.24 f32) | 1.03 (1.00 bool - 1.20 f64) | 1.00 | 0.76 (0.39 i64 - 2.22 f64) | 0.77 (0.54 f64 - 1.00 f16) | 0.99 (0.88 f16 - 1.02 bool) |
| arg_min | 1.35 (1.00 bool - 3.60 f64) | 1.03 (1.00 u64 - 1.24 f32) | 1.03 (1.00 bool - 1.20 f64) | 1.00 | 0.75 (0.39 i64 - 2.22 f64) | 0.78 (0.54 f64 - 0.91 f16) | 1.00 |
| max | 5.89 (4.05 i16 - 13.10 i64) | 2.61 (1.67 f16 - 4.02 u32) | 1.39 (0.97 f16 - 2.33 u32) | 1.00 | 1.00 | 1.00 | 0.99 |
| mean | 5.34 (3.77 complex128 - 15.37 f16) | 3.01 (1.97 complex128 - 13.70 f16) | 1.27 (1.00 complex128 - 2.23 i8) | 1.00 | 1.00 | 1.00 | 1.00 |
| min | 5.92 (4.05 i16 - 13.09 i64) | 2.61 (1.69 f16 - 4.02 u32) | 1.42 (0.96 f16 - 2.38 i64) | 1.00 | 1.00 | 1.00 | 0.99 |
| product | 4.19 (2.69 complex64 - 16.08 f16) | 1.97 (0.94 complex128 - 14.32 f16) | 0.94 (0.74 i16 - 1.06 u16) | 1.00 | 1.00 | 1.00 | 1.00 |
| variance | 3.01 (1.64 complex128 - 8.69 i8) | 1.62 (0.69 complex128 - 4.76 i8) | 1.20 (0.97 f32 - 1.87 bool) | 1.00 | 1.00 | 1.00 | 1.00 |
| **geomean** | **3.28** | **1.83** | **1.16** | **1.00** | **0.93** | **0.93** | **1.00** |

#### `finalize`: speedup of multiversioned (`red-mv`) over the shipped code

| op | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4.02 | 2.16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| arg_max | - | - | - | - | - | - | 1.00 |
| max | 4.11 | 2.16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mean | 2.78 (1.57 complex128 - 20.48 f16) | 2.31 (1.07 complex128 - 19.06 f16) | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| product | 4.18 | 2.16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| standard_deviation | 41.40 (8.72 f16 - 75.52 complex64) | 20.08 (8.87 f16 - 30.40 complex64) | 1.00 | 1.00 | 1.00 | 1.02 | 1.00 |
| variance | 25.13 (5.75 f16 - 51.12 complex64) | 12.20 (5.70 f16 - 20.65 complex64) | 1.00 | 1.00 | 1.00 | 1.01 | 1.00 |
| **geomean** | **8.20** | **4.71** | **1.00** | **1.00** | **1.00** | **1.01** | **1.00** |

#### `init`: speedup of multiversioned (`red-mv`) over the shipped code

| op | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | - | - | - | - | - | - | - |
| arg_max | 1.51 (1.00 bool - 5.05 i64) | 1.24 (1.00 bool - 2.77 i64) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| max | - | - | - | - | - | - | - |
| mean | 5.04 (2.93 complex128 - 17.64 f16) | 2.82 (0.92 complex128 - 16.14 f16) | 1.36 (1.00 complex64 - 2.01 i16) | 1.00 | 1.00 | 1.00 | 1.00 |
| product | 2.13 (1.00 f16 - 3.25 complex128) | 1.04 (0.96 complex128 - 1.18 complex64) | 0.81 (0.65 complex64 - 1.00 complex128) | 1.00 | 1.00 | 1.00 | 1.00 |
| variance | 4.33 (1.80 complex64 - 10.38 f16) | 2.95 (1.35 complex64 - 6.68 f16) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **geomean** | **3.42** | **2.22** | **1.08** | **1.00** | **1.00** | **1.00** | **1.00** |

### Throughput of the `cell` loops

Items per cycle, shipped -> multiversioned, on the newest CPU of each platform. For scale: two
loads per cycle of L1 feed 16 f32 per cycle with AVX2 (32 with AVX-512, 8 with SSE / NEON).

| op | type | AVX-512 (sapphirerapids) | AVX2 (alderlake) | SSE4.2 (sandybridge) | SSE2 (skylake) | M1 (apple-m1) | aarch64 (neoverse-v2) | i686 (skylake) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | bool | 20.6 -> 15.4 | 20.6 -> 1.9 | 20.6 -> 1.8 | 20.6 -> 20.6 | 30.8 -> 30.5 | 7.7 -> 7.7 | 14.7 -> 14.7 |
| any | bool | 20.6 -> 15.4 | 20.6 -> 1.9 | 20.6 -> 1.8 | 20.6 -> 20.6 | 30.8 -> 30.5 | 7.7 -> 7.7 | 14.7 -> 14.7 |
| arg_max | bool | 1.1 -> 2.4 | 1.1 -> 0.7 | 0.4 -> 0.5 | 0.6 -> 0.9 | 0.7 -> 0.7 | 1.0 -> 1.0 | 0.2 -> 0.2 |
| arg_max | f32 | 0.2 -> 2.0 | 0.2 -> 0.7 | 0.1 -> 0.3 | 0.2 -> 0.2 | 0.5 -> 0.5 | 0.5 -> 0.5 | 0.1 -> 0.1 |
| arg_max | f64 | 0.2 -> 1.5 | 0.2 -> 0.6 | 0.2 -> 0.2 | 0.2 -> 0.2 | 0.5 -> 0.5 | 0.5 -> 0.5 | 0.1 -> 0.1 |
| arg_max | i16 | 1.1 -> 2.4 | 1.1 -> 0.8 | 0.5 -> 0.3 | 0.7 -> 0.9 | 0.4 -> 0.3 | 0.7 -> 0.7 | 0.3 -> 0.2 |
| arg_max | i32 | 0.5 -> 2.2 | 0.5 -> 0.8 | 0.3 -> 0.5 | 0.6 -> 0.6 | 0.5 -> 0.5 | 0.7 -> 0.7 | 0.2 -> 0.2 |
| arg_max | i64 | 0.6 -> 2.0 | 0.6 -> 0.7 | 0.3 -> 0.4 | 0.6 -> 0.6 | 0.5 -> 0.5 | 0.7 -> 0.7 | 0.2 -> 0.2 |
| arg_max | u8 | 1.1 -> 2.4 | 1.1 -> 0.7 | 0.5 -> 0.3 | 0.9 -> 1.0 | 0.4 -> 0.4 | 0.7 -> 0.7 | 0.2 -> 0.2 |
| arg_min | bool | 0.9 -> 2.4 | 0.9 -> 0.7 | 0.5 -> 0.3 | 1.0 -> 1.0 | 0.7 -> 0.9 | 1.0 -> 1.0 | 0.2 -> 0.2 |
| arg_min | f32 | 0.2 -> 2.0 | 0.2 -> 0.7 | 0.1 -> 0.3 | 0.2 -> 0.2 | 0.5 -> 0.5 | 0.5 -> 0.5 | 0.1 -> 0.1 |
| arg_min | f64 | 0.2 -> 1.5 | 0.2 -> 0.6 | 0.2 -> 0.2 | 0.2 -> 0.2 | 0.5 -> 0.5 | 0.5 -> 0.5 | 0.2 -> 0.1 |
| arg_min | i16 | 1.1 -> 2.4 | 1.1 -> 0.8 | 0.5 -> 0.3 | 0.7 -> 0.9 | 0.4 -> 0.3 | 0.7 -> 0.7 | 0.3 -> 0.2 |
| arg_min | i32 | 0.5 -> 2.2 | 0.5 -> 0.8 | 0.3 -> 0.5 | 0.6 -> 0.6 | 0.5 -> 0.5 | 0.7 -> 0.7 | 0.2 -> 0.2 |
| arg_min | i64 | 0.6 -> 2.0 | 0.6 -> 0.7 | 0.3 -> 0.4 | 0.6 -> 0.6 | 0.5 -> 0.5 | 0.7 -> 0.7 | 0.2 -> 0.2 |
| arg_min | u8 | 1.1 -> 2.4 | 1.1 -> 0.7 | 0.5 -> 0.3 | 0.9 -> 1.0 | 0.4 -> 0.4 | 0.7 -> 0.7 | 0.2 -> 0.2 |
| max | f32 | 1.2 -> 2.6 | 1.2 -> 1.5 | 0.4 -> 1.1 | 1.0 -> 1.0 | 1.6 -> 1.6 | 2.2 -> 2.2 | 0.5 -> 0.5 |
| max | f64 | 0.6 -> 2.3 | 0.6 -> 1.1 | 0.2 -> 0.5 | 0.5 -> 0.4 | 0.9 -> 0.9 | 1.1 -> 1.1 | 0.4 -> 0.4 |
| max | i16 | 15.6 -> 5.3 | 15.6 -> 5.3 | 13.2 -> 13.2 | 15.6 -> 15.6 | 5.2 -> 5.2 | 7.7 -> 7.7 | 11.3 -> 11.3 |
| max | i32 | 2.9 -> 0.5 | 2.9 -> 5.3 | 2.1 -> 5.5 | 2.4 -> 2.4 | 5.2 -> 5.2 | 6.2 -> 6.2 | 2.4 -> 2.4 |
| max | i64 | 1.4 -> 0.3 | 1.4 -> 2.0 | 0.7 -> 0.7 | 1.3 -> 1.3 | 2.8 -> 2.8 | 2.9 -> 2.9 | 0.5 -> 0.5 |
| max | u8 | 20.6 -> 7.9 | 20.6 -> 1.9 | 20.6 -> 1.7 | 20.6 -> 20.6 | 2.0 -> 2.0 | 1.5 -> 1.5 | 14.7 -> 14.7 |
| mean | bool | 2.0 -> 7.0 | 2.0 -> 3.9 | 1.2 -> 1.9 | 1.6 -> 1.6 | 3.9 -> 3.9 | 3.9 -> 3.9 | 0.5 -> 1.1 |
| mean | f32 | 4.5 -> 3.2 | 4.5 -> 3.2 | 3.9 -> 3.9 | 3.5 -> 3.5 | 3.9 -> 3.9 | 6.2 -> 6.2 | 0.7 -> 2.2 |
| mean | f64 | 3.9 -> 3.2 | 3.9 -> 3.2 | 2.0 -> 2.0 | 3.2 -> 3.2 | 3.1 -> 3.1 | 3.9 -> 3.9 | 2.2 -> 2.0 |
| mean | i16 | 2.0 -> 7.0 | 2.0 -> 3.9 | 1.2 -> 1.9 | 1.6 -> 1.6 | 3.9 -> 3.9 | 3.9 -> 3.9 | 0.9 -> 1.3 |
| mean | i32 | 2.0 -> 0.2 | 2.0 -> 3.9 | 1.2 -> 1.9 | 1.6 -> 1.3 | 3.5 -> 2.6 | 5.2 -> 5.2 | 1.6 -> 1.3 |
| mean | i64 | 5.7 -> 0.3 | 5.7 -> 5.2 | 3.0 -> 3.0 | 4.0 -> 4.0 | 4.7 -> 4.7 | 3.9 -> 3.9 | 3.0 -> 3.0 |
| mean | u8 | - | - | - | 1.6 -> 1.6 | - | - | 1.1 |
| min | f32 | 1.2 -> 2.6 | 1.2 -> 1.5 | 0.4 -> 1.1 | 1.0 -> 1.0 | 1.6 -> 1.6 | 2.2 -> 2.2 | 0.5 -> 0.5 |
| min | f64 | 0.6 -> 2.3 | 0.6 -> 1.1 | 0.2 -> 0.5 | 0.5 -> 0.4 | 0.9 -> 0.9 | 1.1 -> 1.1 | 0.4 -> 0.4 |
| min | i16 | 15.6 -> 5.3 | 15.6 -> 5.3 | 13.2 -> 13.2 | 15.6 -> 15.6 | 5.2 -> 5.2 | 7.7 -> 7.7 | 11.3 -> 11.3 |
| min | i32 | 2.9 -> 0.5 | 2.9 -> 5.3 | 1.7 -> 5.5 | 2.0 -> 2.4 | 5.2 -> 5.2 | 6.2 -> 6.2 | 2.0 -> 2.4 |
| min | i64 | 1.4 -> 0.3 | 1.4 -> 2.0 | 0.7 -> 0.7 | 1.3 -> 1.3 | 2.8 -> 2.8 | 2.9 -> 2.9 | 0.5 -> 0.5 |
| min | u8 | 20.6 -> 7.9 | 20.6 -> 1.9 | 20.6 -> 1.7 | 20.6 -> 20.6 | 2.0 -> 2.0 | 1.5 -> 1.5 | 14.7 -> 14.7 |
| product | f32 | 3.5 -> 2.7 | 3.5 -> 2.7 | 2.9 -> 2.9 | 3.5 -> 3.5 | 7.7 -> 7.7 | 5.2 -> 5.2 | 0.7 -> 2.2 |
| product | f64 | 3.5 -> 2.7 | 3.5 -> 2.6 | 2.0 -> 2.0 | 3.2 -> 3.2 | 3.1 -> 3.1 | 3.9 -> 3.9 | 2.2 -> 2.0 |
| product | i16 | 1.0 -> 1.4 | 1.0 -> 1.3 | 1.0 -> 1.0 | 1.0 -> 1.0 | 1.0 -> 1.0 | 2.0 -> 2.0 | 0.3 -> 0.3 |
| product | i32 | 1.0 -> 0.2 | 1.0 -> 1.3 | 1.0 -> 1.0 | 1.0 -> 1.0 | 1.0 -> 1.0 | 1.3 -> 1.3 | 0.3 -> 0.3 |
| product | i64 | 1.0 -> 0.3 | 1.0 -> 1.3 | 1.0 -> 1.0 | 1.0 -> 1.0 | 1.0 -> 1.0 | 2.0 -> 2.0 | 0.3 -> 0.3 |
| product | u8 | 1.0 -> 1.4 | 1.0 -> 1.7 | 1.0 -> 1.0 | 1.0 -> 1.0 | 1.0 -> 1.0 | 2.0 -> 2.0 | 0.3 -> 0.5 |
| variance | bool | 0.0 -> 0.2 | 0.0 -> 0.2 | 0.0 -> 0.1 | 0.1 -> 0.1 | 0.1 -> 0.3 | 0.1 -> 0.2 | 0.1 -> 0.1 |
| variance | f32 | 0.2 -> 0.2 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.1 -> 0.2 | 0.1 -> 0.1 |
| variance | f64 | 0.2 -> 0.2 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.1 -> 0.2 | 0.4 -> 0.4 | 0.2 -> 0.2 | 0.1 -> 0.1 |
| variance | i16 | 0.2 -> 0.2 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.1 -> 0.1 | 0.1 -> 0.1 |
| variance | i32 | 0.2 -> 0.2 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.1 -> 0.1 | 0.1 -> 0.1 |
| variance | i64 | 0.0 -> 0.2 | 0.0 -> 0.2 | 0.0 -> 0.1 | 0.1 -> 0.1 | 0.3 -> 0.3 | 0.2 -> 0.2 | 0.1 -> 0.1 |
| variance | u8 | 0.2 -> 0.2 | 0.2 -> 0.2 | 0.1 -> 0.1 | 0.2 -> 0.1 | 0.1 -> 0.3 | 0.1 -> 0.2 | 0.1 -> 0.1 |

### What the asm shows

The cell loop keeps `ONE_CELL_LANES` = 16 accumulators (an array of states), updated per chunk of
16 items; the compiler is left to vectorize it, across the lanes (SLP) or across the chunks (the
loop vectorizer). It does not do so reliably:

- **Too few accumulators for wide vectors**: sum f32 is 16 lanes = 4 xmm, 2 ymm or 1 zmm
  accumulators, a chain of dependent adds (4 cycles each): 3.2 items per cycle with AVX2 / AVX-512
  (vs 4.5 for the SSE2 code, 16-32 available). The lane count fixes the summation order (results
  identical on every machine), so more accumulators per level would change float results across
  machines.
- **The chunk loop vectorized with gathers**: on AVX-512, integer sum / max / product
  (`vpgatherqd`, a stride-16 gather per accumulator: 0.2-0.5 items per cycle, 5-20x slower than the
  SSE2 code); with AVX2 / SSE4.2, all / any / max over u8 and bool (`vpinsrb` byte by byte, 1.9 vs
  20.6 items per cycle). The same in the whole-crate builds: a cost-model failure on this loop
  shape, not the dispatch.
- **Argmax** keeps an index per lane, computed as `base + i * idx_stride` with a runtime stride:
  64-bit multiplies emulated per lane and iteration (`vpmuludq` x3). Its `update` (an array of
  `(u64, T)` states) is scalar and branchy on x86 at every level.
- **Variance** (Welford: a division per item, and a 24-byte state per lane) is shuffled through
  the stack; 0.1-0.2 items per cycle everywhere.
- **The dispatch changes codegen on single-level targets too** (NEON, and the SSE2 arm): the extra
  closure changes the inlining order, and LLVM's decisions with it: argmax `update` on aarch64 is
  no longer if-converted and vectorized (0.39-0.54 for i64 / f64), product complex64 `cell` on
  SSE2 0.25.
- What multiversioning does deliver: the element-wise-like loops, `update` (min / max / sum 2-6x
  on AVX2 / AVX-512), `finalize` (4.7-8.2x geomean, variance / std 12-41x with AVX's packed division),
  and argmax / NaN-propagating min / max `cell` on AVX-512 (masked compares: 4-11x).

## Reductions: `update_state_bulk` and lanes sized by the SIMD level (`results/rb-*`)

The one-cell fold (`fold_run_leaf`) no longer leaves its lanes to auto-vectorization:

- `ReductionOpKernel::update_state_bulk(states, items, indices, simd)` folds a chunk of `N` items
  into the `N` lane accumulators. The default maps `update_state`; SIMD bodies: sum / mean /
  product (the scalar traits' `update_bulk`: the items cast to the state type, then a vector add /
  mul; scalar for items widened 4x or more on SSE2 / NEON, which have no widening load), min / max
  (the element-wise `maximum_bulk` / `minimum_bulk`), all / any (as `u8` vectors). Argmax / argmin
  and variance keep the default.
- The lanes are `ONE_CELL_VECTORS` vectors of the dispatched level's states (a power of two in
  `8..=SPLIT_THRESHOLD`), except `ReductionOpKernel::ONE_CELL_LANES` fixed ones (argmax / argmin:
  16, 8 on 32-bit x86). A leaf takes any length (the remainder folded one by one), so the
  pairwise split does not align to the lanes.
- The lane indices are an array advanced by `LANES * idx_stride` per chunk (an add, not
  `vpmullq`), only for the kernels that read them (`NEEDS_INDICES`).
- Floats sum differently on different levels (different lanes): accepted.

Variants (`cell` loops only, speedup over the shipped code, as in the previous section; `red-mv`:
dispatch only; `rb-v*`: 2, 4, 8 vectors, before the index / argmax / narrow-sum / all-any
fixes):

| config | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `red-mv` | 2.04 | 1.27 | 1.01 | 0.99 | 1.04 | 1.03 | 1.07 |
| `rb-v4` | 3.70 | 1.97 | 1.16 | 1.12 | 1.23 | 1.26 | 1.42 |
| `rb-v8` | 2.09 | 2.19 | 1.25 | 1.15 | 1.44 | 1.33 | 1.40 |
| `rb-final` | 4.40 | 2.75 | 1.39 | 1.22 | 1.87 | 1.48 | 1.07 |

`rb-v8` -> `rb-final`: the fixes found in the asm along the way:

- **8 vectors is the best on every level** but argmax with AVX-512: 32 `(index, value)` lanes are
  no longer unrolled by LLVM (a 32-trip scalar loop through the stack, 0.03-0.1x): argmax fixed at
  16 lanes. 8 lanes (16-byte vectors with 8 vectors of 16-byte states) are 1.5-2.5x slower for
  narrow items, and 16 spill on i686 (8 registers): 8 there (`rb-i686`: argmax 1.76x geomean on
  i686; `rb-final`'s i686 argmax column predates it, 0.5x).
- **Lane indices** as `base + (i + b) * idx_stride` cost two `vpmullq` per vector: advanced by a
  step instead, argmax 1.3-2.4x. Carried across iterations, an unused index array is not dropped
  by LLVM once too large for registers (all over bool, 256 lanes: 64 `vpaddq` and spills per
  iteration), hence `NEEDS_INDICES`; recomputed per chunk from invariant offsets instead, LLVM
  strength-reduces them into a scalar chain and argmax selects each index with a `cmov` (2-5x
  slower).
- **all / any** converted their bytes back with `!= 0`, a compare and an `andn` per vector: a
  transmute (the bytes are 0 / 1) leaves one `vpand` / `and` per vector (2.5-3.3x).
- **Narrow integer sums** (i8 / u8 / bool / i16 into 64 bits) on SSE2 and NEON: the widening
  shuffle chains are up to 2x slower than the scalar loop: scalar there.

`rb-final`, per op (speedup over the shipped code, geomean over the types, range when they differ
by more than 15%):

| op | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 6.06 | 3.32 | 1.22 | - | 1.22 | 3.21 | - |
| any | 6.06 | 3.32 | 1.22 | - | 1.22 | 3.21 | - |
| arg_max | 6.50 (1.06 f16 - 13.93 f32) | 2.81 (1.06 f16 - 8.44 f32) | 1.39 (0.99 f16 - 3.21 f32) | 1.27 (0.94 f16 - 3.70 f32) | 1.92 (1.00 f32 - 4.37 i8) | 1.50 (0.94 u64 - 2.68 i16) | 0.51 (0.38 u32 - 1.11 f16) |
| arg_min | 6.82 (1.01 f16 - 13.93 f32) | 2.73 (1.01 f16 - 8.46 f32) | 1.36 (0.96 f16 - 3.21 f32) | 1.21 (0.91 f16 - 3.70 f32) | 2.04 (1.00 f64 - 4.37 i8) | 1.57 (0.94 i64 - 2.68 i16) | 0.51 (0.38 u32 - 1.06 f32) |
| max | 6.06 (2.54 i16 - 18.25 i8) | 3.58 (1.66 f16 - 14.36 i8) | 1.71 (0.96 f16 - 4.75 i8) | 1.18 (0.99 f16 - 2.44 i8) | 2.54 (1.00 u64 - 18.89 i8) | 1.93 (1.00 i64 - 17.28 u8) | 1.22 (0.80 i16 - 2.33 f32) |
| mean | 4.42 (2.40 u32 - 16.67 f16) | 2.59 (1.27 u32 - 14.87 f16) | 1.45 (0.99 f16 - 2.12 complex64) | 1.18 (1.00 u32 - 2.16 complex64) | 1.04 (0.60 complex128 - 2.00 f32) | 0.99 (0.89 u32 - 1.15 f32) | 1.46 (0.94 f64 - 6.05 f32) |
| min | 6.12 (2.54 i16 - 15.13 i8) | 3.59 (1.50 f16 - 11.91 i8) | 1.75 (0.91 f16 - 4.94 u32) | 1.16 (0.97 f16 - 1.73 i8) | 2.54 (1.00 u64 - 18.89 u8) | 1.93 (1.00 i64 - 17.28 u8) | 1.32 (0.80 i16 - 2.33 f32) |
| product | 3.61 (2.23 complex128 - 17.37 f16) | 1.81 (0.67 complex64 - 15.39 f16) | 0.97 (0.37 complex64 - 1.47 complex128) | 0.97 (0.26 complex64 - 1.65 f32) | 0.90 (0.29 complex64 - 1.22 f32) | 0.97 (0.31 complex64 - 1.86 complex128) | 1.10 (0.38 complex64 - 6.05 f32) |
| variance | 1.46 (1.00 i16 - 3.40 i64) | 2.60 (1.65 u8 - 6.51 i64) | 1.32 (0.89 f16 - 2.50 i64) | 1.66 (0.85 f16 - 3.21 i64) | 3.41 (1.59 complex64 - 6.07 f16) | 1.68 (1.02 f16 - 2.16 u16) | 2.26 (1.46 f16 - 2.83 i16) |

Below 0.9x remain, all without a SIMD body, whose auto-vectorization changed with the new fold:
complex64 product (0.26-0.67, LLVM's shuffles of the complex multiply), f16 sum / product on M1
(0.7), complex128 sum on M1 (0.6), f16 variance on SSE (0.85-0.89), i16 / i32 max on i686 (0.8).

### `SPLIT_THRESHOLD`, and timings

The static metric cannot see `SPLIT_THRESHOLD`: it costs per leaf (a dispatch, merging the
lanes, the remainder), outside the hot loop, and the analyzer evaluates a leaf as long as the
input. So it was timed on this machine (a Xeon with AVX-512 but not VBMI: fearless_simd's AVX2
level), whole 1-d reductions through the public API, in a throwaway crate: the slope between
2^14 and 2^16 elements (L2), which cancels a fixed ~23 us per call (the reduction's setup,
independent of the op and the length: a separate issue, it dominates reductions of up to ~100K
elements). Float sums with 8 vectors: 512 -> 2048 is 1.3-1.65x, 8192 1.5-2.1x (the vector count
within the VM's noise). 2048 keeps each lane's sequential chain at most 128 items, as NumPy's
pairwise sum.

Final code against the shipped one (`16670ab`), ns per element (the min of two runs):

| op | shipped ns/elem | new ns/elem | speedup |
|---|---:|---:|---:|
| sum_f32 | 0.092 | 0.053 | 1.73x |
| sum_f64 | 0.159 | 0.104 | 1.53x |
| sum_i32 | 0.200 | 0.088 | 2.28x |
| mean_f32 | 0.093 | 0.035 | 2.63x |
| product_f64 | 0.141 | 0.091 | 1.55x |
| max_f32 | 0.324 | 0.098 | 3.30x |
| max_i32 | 0.141 | 0.044 | 3.24x |
| max_u8 | 0.030 | 0.008 | 3.64x |
| all_bool | 0.029 | 0.003 | 9.67x |
| argmax_f32 | 1.625 | 0.224 | 7.26x |
| argmax_i32 | 0.546 | 0.187 | 2.92x |
| sum_i8 | 0.210 | 0.071 | 2.97x |
| max_f64 | 0.547 | 0.204 | 2.68x |
| min_i8 | 0.068 | 0.008 | 8.12x |
| argmax_u8 | 0.768 | 0.153 | 5.03x |
| any_bool | 0.028 | 0.013 | 2.18x |

### Option indices, ArgMax / ArgMin traits, Variance bulk (`results/rb-final2`)

- The stream positions are `Option`s (`init_state` / `update_state`: `Option<u64>`,
  `update_state_bulk`: `Option<[u64; N]>`), `Some` exactly for the kernels that `NEEDS_INDICES`,
  which unwrap them. In the asm, argmax / argmin have the same single `unwrap_failed` call as
  the kernels without indices (the leaf's final `state.unwrap()`, outside the loops): the
  index unwraps are gone.
- `ArgMax` / `ArgMin`: jix scalar traits (`update`, `update_bulk`, `merge_states`) instead of
  `PartialOrd`, for bool, the integers, f16, f32, f64. Unchanged codegen except bool on M1 /
  aarch64 / i686 (0.42-0.65x of the shipped code: LLVM keeps its lanes scalar, `csel` per lane;
  comparing as `u8` does not change it) and f16 (0.85-0.93).
- `Variance::update_bulk`: `update` a field at a time (the states interleave mean, m2, count).
  The `cell` loop: AVX-512 1.46 -> 2.94x the shipped code, the other levels unchanged (it was
  already vectorized there).
- all / any are back to `!= 0`: a compare and an `andn` per vector, 2.3-3.6x slower than the
  transmute, 0.49-0.94x the shipped code on AVX2 / SSE / M1 / i686.

| config | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `red-mv` | 2.04 | 1.27 | 1.00 | 0.99 | 1.04 | 1.03 | 1.07 |
| `rb-final` | 4.40 | 2.75 | 1.38 | 1.23 | 1.87 | 1.48 | 1.08 |
| `rb-final2` | 4.85 | 2.64 | 1.36 | 1.20 | 1.77 | 1.46 | 1.50 |

| op | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 2.69 | 0.94 | 0.65 | 0.80 | 0.49 | 1.99 | 0.79 |
| any | 1.67 | 0.94 | 0.65 | 0.80 | 0.49 | 1.99 | 0.79 |
| arg_max | 6.97 (0.93 f16 - 13.93 f32) | 2.74 (0.91 f16 - 8.44 f32) | 1.35 (0.91 f16 - 3.23 f32) | 1.21 (0.85 f16 - 3.69 f32) | 1.77 (0.42 bool - 4.37 i8) | 1.47 (0.65 bool - 2.68 i16) | 1.80 (0.42 bool - 5.26 f32) |
| arg_min | 6.75 (1.00 f16 - 13.93 f32) | 2.67 (0.98 f16 - 8.46 f32) | 1.32 (0.97 i64 - 3.23 f32) | 1.16 (0.77 bool - 3.66 f32) | 1.77 (0.45 bool - 4.37 i8) | 1.48 (0.65 bool - 2.68 i16) | 1.76 (0.42 bool - 5.38 f32) |
| max | 6.06 (2.54 i16 - 18.25 i8) | 3.58 (1.66 f16 - 14.36 i8) | 1.71 (0.96 f16 - 4.75 i8) | 1.18 (0.99 f16 - 2.44 i8) | 2.54 (1.00 u64 - 18.89 u8) | 1.93 (1.00 u64 - 17.28 i8) | 1.22 (0.80 i16 - 2.33 f32) |
| mean | 4.42 (2.40 u32 - 16.67 f16) | 2.59 (1.27 u32 - 14.87 f16) | 1.45 (0.99 f16 - 2.12 complex64) | 1.18 (1.00 u32 - 2.16 complex64) | 1.04 (0.60 complex128 - 2.00 f32) | 0.99 (0.89 i32 - 1.15 f32) | 1.46 (0.94 f64 - 6.05 f32) |
| min | 6.12 (2.54 i16 - 15.13 i8) | 3.59 (1.50 f16 - 11.91 i8) | 1.75 (0.91 f16 - 4.94 u32) | 1.16 (0.97 f16 - 1.73 i8) | 2.54 (1.00 i64 - 18.89 i8) | 1.93 (1.00 u64 - 17.28 u8) | 1.32 (0.80 i16 - 2.33 f32) |
| product | 3.61 (2.23 complex128 - 17.37 f16) | 1.81 (0.67 complex64 - 15.39 f16) | 0.97 (0.37 complex64 - 1.47 complex128) | 0.97 (0.26 complex64 - 1.65 f32) | 0.90 (0.29 complex64 - 1.22 f32) | 0.98 (0.33 complex64 - 1.86 complex128) | 1.10 (0.38 complex64 - 6.05 f32) |
| variance | 2.94 (0.90 complex128 - 8.05 i64) | 2.55 (1.30 u8 - 6.51 i64) | 1.34 (1.04 u8 - 2.47 i64) | 1.70 (1.10 f16 - 3.21 i64) | 3.41 (1.59 complex64 - 6.05 u16) | 1.71 (1.23 u64 - 2.16 u16) | 2.29 (1.67 f16 - 2.85 bool) |

Timed on this machine as above (AVX2 level), against the shipped code; the cheapest ops (all /
any, under 0.02 ns per element) are within the VM's noise:

| op | shipped ns/elem | new ns/elem | speedup |
|---|---:|---:|---:|
| sum_f32 | 0.064 | 0.060 | 1.06x |
| sum_f64 | 0.133 | 0.088 | 1.52x |
| sum_i32 | 0.214 | 0.076 | 2.83x |
| mean_f32 | 0.069 | 0.047 | 1.49x |
| product_f64 | 0.133 | 0.117 | 1.14x |
| max_f32 | 0.363 | 0.075 | 4.84x |
| max_i32 | 0.142 | 0.057 | 2.47x |
| max_u8 | 0.026 | 0.005 | 5.27x |
| all_bool | 0.017 | 0.007 | 2.43x |
| argmax_f32 | 1.500 | 0.144 | 10.44x |
| argmax_i32 | 0.480 | 0.098 | 4.89x |
| sum_i8 | 0.168 | 0.045 | 3.70x |
| max_f64 | 0.538 | 0.146 | 3.70x |
| min_i8 | 0.061 | 0.008 | 7.67x |
| argmax_u8 | 0.683 | 0.075 | 9.06x |
| any_bool | 0.009 | 0.012 | 0.70x |
| var_f32 | 1.705 | 0.693 | 2.46x |
| var_f64 | 1.721 | 0.687 | 2.50x |
| var_i32 | 1.814 | 0.710 | 2.56x |
| argmax_bool | 0.659 | 0.108 | 6.12x |

The fixed ~23 us per call: `ReadContext::default()` (created by `to_ndarray()`) builds a zstd
decompression context eagerly, 13 us here, even for a `Plain` input that decompresses nothing;
the reduction itself, with a reused context, takes 0.34 us for 1024 items.

### all / any without `!= 0` (`results/aa-B`, `results/aa-E`)

The `u8` SIMD bulk of `rb-final2` (`and` / `or` on bytes, then `!= 0` back to `bool`) costs a
compare and an and-not per vector on top of the `and` (M1: `and` + `cmeq` + `bic`). Variants of the
all / any kernels (no `unsafe`):

- A (`rb-final2`): `u8` vectors via `map_vectors2`, then `!= 0`.
- B: `array_from_fn_inline(|i| states[i] & items[i])` on plain `bool` arrays.
- C / D: a `u8` state (0 / 1) with `u8` vectors / plain arrays, `!= 0` once in finalize. Not
  finished: B is already one load and one `and` / `or` per vector, so C / D can only add the
  finalize pass.
- E (committed): B without the override: `update_state` uses non-short-circuit `state & item`
  (`|` for any), and the default `update_state_bulk` maps it. Same asm as B.

`cell` speedup over the code without multiversioning (all and any are identical):

| config | AVX-512 | AVX2 | SSE4.2 | SSE2 | M1 | aarch64 | i686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (`rb-final2`), all | 2.69 | 0.94 | 0.65 | 0.80 | 0.49 | 1.99 | 0.79 |
| A (`rb-final2`), any | 1.67 | 0.94 | 0.65 | 0.80 | 0.49 | 1.99 | 0.79 |
| B / E | 6.06 | 3.32 | 1.22 | 1.43 | 1.22 | 3.21 | 1.43 |

`update` / `finalize` are unchanged (4.06 / 2.19 on AVX-512 / AVX2, 1.00 elsewhere).

Real timing (this machine, AVX2 level, marginal ns per item with a reused `ReadContext`, 2^14 to
2^17 items): the one-cell loop is L2-bound for every variant (all / any 0.015, shipped all 0.016,
shipped any 0.065); across cells ([n / 1024, 1024] along axis 0) E is 0.028 / 0.029 against
shipped 0.038 / 0.049.

## fearless_simd 1.1.0 (`results/fs*`, `results/pre-pull`)

What 1.1.0 changes for these kernels (its generated code, per level; `CHANGELOG.md` of the crate):
no existing function changed on AVX-512, NEON or the fallback. On x86 below AVX-512:
- SSE2: f32 -> i32 / u32 conversions (`cvttps2dq` and fix-ups, was per lane), `i64` / `u64`
  multiply (`pmuludq` steps, was per lane) and right shift by a scalar.
- SSE4.2 / AVX2: `i64` / `u64` min / max / `simd_lt` / `simd_le`, multiply, right shift by a
  scalar; precise f32 -> u32.
- New: mask <-> vector (`to_vector`), mask widen / narrow (`MaskWiden` / `MaskNarrow`),
  `simd_ne`, float classification (`is_nan`, ...), `round`, compress / expand, concat swizzles.
- Unchanged: shifts by vectors (Shl, the 8 / 16 / 64-bit Shr), rotates, count_ones, int <-> f64
  conversions, widening, the SSE2 `unzip` / `deinterleave` (still per lane: `ByteShuffle`'s
  choice of `interleave` on SSE2 holds). `Level::new()` is unchanged (`try_detect` became
  `#[inline]`), so a captured constant of a dispatch is still reloaded.

Runs: `fs10` / `fs11` the branch with 1.0.0 / 1.1.0 (`fs10` x86 only), `pre-pull` the commit
before the pull (`c8e8ba2`, 1.0.0); `fs-unroll` `fs11` + the source-level unroll below; `fs-new`
/ `fs-none` (on `fs-unroll`): every Cast body on every level + the new bodies / no Cast body and
no 64-bit Mul / Square / Maximum / Minimum / integer Product body; `fs-final` the result, `fs-prod`
its Product gate fixed (below).

**The pull itself** (`arr_ext`, renames, the filters): `pre-pull` = `fs10` on all 2666 kernels.

**The bump alone** (`fs11` over `fs10`; other kernels within 2%):

| kernel | SSE2 | SSE4.2 | AVX2 | i686 |
|---|---:|---:|---:|---:|
| `cast_f32_i32` | 6.07 | 1.00 | 1.00 | 7.28 |
| `cast_f32_u32` | 2.83 | 1.03 | 1.10 | 4.28 |
| `cast_f32_u8` / `u16` | 1.00 | 1.06-1.07 | 1.08-1.11 | 1.00 |
| `square_i64` | 1.33 | 1.36 | 1.11 | 1.17 |
| `mul_i64` | _0.61_ | _0.75_ | 1.06 | 1.19 |
| `reduce_max/min_i64/u64_cell` | 1.00 | 1.23-1.43, _0.93_ (min u64) | 1.00 | 1.00 |
| `reduce_product_*_cell` (integers) | _0.34-0.35_ (32/64-bit) | _0.55_ | 1.00-1.10 | 1.35-1.37, _0.74_ (i64) |

**`map_vectors` / `map_vectors2` unrolled in the source.** The `mul_i64` / product regressions are
not the multiply: with 1.1.0's intrinsic sequence LLVM no longer fully unrolls the bulk helpers'
`for c in 0..N / lanes` loop, so the arrays of vectors stay on the stack (65 stack ops per
iteration, an inner loop of 8). `for_each_unrolled(n, f)` makes the calls `f(0)`, .., `f(31)`
(guarded by `k < n`, folded) in the source; debug builds loop instead (unrolled, the test crate
needs more than 12 GB to compile). `fs-unroll` over `fs11`: only 9 kernels change, all faster:
`mul_i64` 2.11 / 1.78 (SSE2 / SSE4.2), integer `reduce_product_*_cell` 1.57-2.19 (SSE2 /
SSE4.2), 1.27-2.92 (i686).

**Integer products** (`fs-none` over `fs-unroll`, i.e. without over with the body): an `i64`
product as `pmuludq` steps is slower than scalar `imul` on x86-64 SSE2 / SSE4.2 (0.52-0.88), but
1.7-2.6x faster on i686 (no 64-bit `imul`): `update_bulk_simd!(Product, *, scalar_on_x86_64 =
[Sse2, Sse4_2])`. The gate is a chain of `==`: `[..].contains(&level)` did not fold early enough
and the loop came out half `pmuludq` (`fs-final`, fixed in `fs-prod`). The elementwise `mul_i64`
/ `square_i64` bodies stay (1.06-1.36 over no body).

**Removed bodies retried with the new API** (`fs-new` over `fs-unroll`; ranges over the kernels):

| kernels | SSE2 | SSE4.2 | AVX2 | AVX-512 | i686 | aarch64 | M1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| comparisons (8/16-bit) | 1.00-1.03 | 1.00 | 1.00-1.73 | 0.90-0.91 | 1.00-1.03 | 1.00 | 1.00 |
| comparisons (32/64-bit) | 0.70-1.31 | 1.00 | 1.06-1.44 | 0.55-0.75 | 1.00 | 0.36-0.58 | 0.16-0.27 |
| is_nan / is_finite / is_infinite | 0.95-1.65 | 0.86-4.93 | 1.10-6.71 | 0.47-0.77 | 0.99-1.63 | 0.32-0.50 | 0.12-0.32 |
| casts to bool | 0.95-1.03 | 0.97-1.02 | 1.00-1.94 | 0.46-0.87 | 1.00 | 0.32-1.00 | 0.12-1.00 |
| round | 0.79-0.97 | 1.00 | 1.00 | 0.98 | 0.81-0.87 | 0.99-1.00 | 1.00 |

- `bool` outputs: the masks of `size_of::<T>()` vectors narrowed to one of bytes (`MaskNarrow`,
  `packssdw` / `packsswb` on SSE2), `to_vector() & 1`, stored, `!= 0` (folds away). Kept on AVX2
  only (`masks_to_bools`): on SSE2 / SSE4.2 it is mostly LLVM's own code; on AVX-512 LLVM stores
  the compare's `k` mask directly (`vmovdqu8 {k}{z}`) where fearless_simd narrows the masks
  through general registers and `vpinsrw`; on NEON LLVM splits the narrowing of the `vmovn`
  halves into lane moves.
- `round` (`trunc(x + copysign(0.49999997, x))`): no gain, not kept.

**Cast gates** (`fs-new` over `fs-none`, the bar as before: faster or within 3% on every platform
of the level): f32 -> i8 / i16 (3.6x) / u8 (1.15-1.19x) / u16 (1.16-1.48x) now on SSE2 too (its
f32 -> i32 / u32 is SIMD now). The others unchanged: i16 -> f32 on SSE2 is 0.94 on x86-64 (the
same instructions, as before), f32 -> u64 on SSE4.2 0.969.

**Net, `fs-final` / `fs-prod` over `fs10`** (x86; AVX-512, aarch64 and M1 are unchanged): faster
are the casts above, `mul_i64` 1.06-1.33, `square_i64` 1.11-1.36, `reduce_max/min_*64_cell` on
SSE4.2 1.23-1.43, integer products on AVX2 1.02-1.10 / i686 1.74-2.55, and on AVX2 the `bool`
outputs (comparisons 1.06-1.73, classification 1.10-6.71, casts to bool 1.09-1.94). Slower:
`reduce_min_u64_cell` on SSE4.2 0.93 (fearless_simd's `min_u64`), unsigned products on SSE4.2
0.98.
