# jix-probe: static asm analysis of the byte-shuffle decode kernels

Experimental tooling for optimizing the **decode** path of
`jix/src/codec/filter/byte_shuffle/kernels.rs` across platforms **without running benchmarks**.
The objective is llvm-mca's steady-state throughput of each decode kernel's hot loop, on the
targeted platforms. The encode path is out of scope, and it is neither probed nor analyzed.

## Why a separate crate and not a `jix` example

A `jix` example would build all of `jix`'s dependencies for the target, including `zstd-sys` (C),
which needs a C cross toolchain and sysroot for every target (and a macOS SDK for
`aarch64-apple-darwin`). Instead, the kernels live in a self-contained file (`core`/`std` +
the optional `multiversion` attribute only). `jix` uses it as a module, and this crate
`#[path]`-includes it. So the probe compiles the exact code `jix` ships, for any rustup target, in
about 2 seconds, with no linker or C compiler. Keep `kernels.rs` free of `crate::` dependencies.

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
python jix/probe/analyze.py                         # all platforms, results/baseline/
python jix/probe/analyze.py --label my-variant      # results/my-variant/
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
| x86_64-v2 | x86_64-unknown-linux-gnu | multiversion v2 clone features | sandybridge, btver2 |
| x86_64-v3 | x86_64-unknown-linux-gnu | multiversion v3 clone features | skylake, alderlake, znver3 |
| x86_64-v4 | x86_64-unknown-linux-gnu | multiversion v4 clone features | icelake-server, sapphirerapids, znver4 |
| i686 | i686-unknown-linux-gnu | SSE2 | skylake |
| aarch64 | aarch64-unknown-linux-gnu | armv8-a + NEON | cortex-a72, neoverse-n1, neoverse-v2 |
| aarch64-apple | aarch64-apple-darwin | apple-m1 | apple-m1 |

x86-64 v1 (SSE2 only), armv7 and ppc64le are out of scope. Windows and musl wheels share the ISA
of their linux counterparts. For x86-64, the multiversion
clones get exactly `#[target_feature(enable = ...)]` (generic tuning). The probe reproduces each
clone by building without `multiversion` and with the same features in `-C target-feature`.

For each (platform, kernel):
1. `cargo asm --simplify` gives the asm, and the loop tree comes from back-edge branches.
2. `cargo asm --llvm` gives the final LLVM IR. The toolchain's own `opt` prints the IR loop tree
   (`print<loops>`) and exact trip counts (`print<scalar-evolution>`).
3. The trees are matched by shape, which gives every asm loop its trip count.
4. The bytes per outer-loop iteration (`B/iter`) come from the outer loop's SCEV trip count,
   evaluated at a concrete input length. They are not assumed, so a kernel can use any step:
   128 B/iter today, and more for a 512-bit variant or a loop LLVM unrolled.
5. One outer iteration is flattened: inner loops that LLVM did not unroll are repeated by their
   trip count. llvm-mca simulates that trace (100 iterations) on each CPU.

**Metric: cycles per 4096 bytes = cycles per outer iteration * 4096 / B/iter. Lower is better.**
The per-(platform, CPU) geomean over the 4 fixed-itemsize decode kernels (itemsize 2/4/8/16) is
the headline number. The unit is only a scale. What makes kernels with different steps comparable
is that `B/iter` is measured.

Caveats:
- llvm-mca assumes every load hits L1 and ignores the front end. Branches are not followed.
- **Calls are free in llvm-mca.** Kernels that call something (e.g. `memcpy`) inside the hot loop
  are flagged `*`, and their real cost is higher.
- Steady state only: the per-call prologue and epilogue and the `<LANES` tail (handled by
  `decode_impl_generic`) are not counted. This matters more as the step grows.
- Instructions with no scheduling info on a CPU are skipped and listed as warnings (none so far).
- `*_generic` (runtime itemsize: odd sizes, struct dtypes, and the `<LANES` tail) cannot be
  flattened. Only the cycles per iteration of their largest innermost loop are reported, for
  information.

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
