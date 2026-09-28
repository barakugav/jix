# jix-probe: static asm analysis of the byte-shuffle kernels

Experimental tooling for optimizing `jix/src/codec/filter/byte_shuffle/kernels.rs` across
platforms **without running benchmarks**. The objective is llvm-mca's steady-state throughput of
each kernel's hot loop, for every platform jix ships wheels for.

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
python jix/probe/analyze.py --platform x86_64-v3 --platform aarch64 --fn encode_4
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
| x86_64-v1 | x86_64-unknown-linux-gnu | SSE2 (no multiversion / pre-2009 CPUs) | x86-64, skylake |
| x86_64-v2 | x86_64-unknown-linux-gnu | multiversion v2 clone features | sandybridge, btver2 |
| x86_64-v3 | x86_64-unknown-linux-gnu | multiversion v3 clone features | skylake, alderlake, znver3 |
| x86_64-v4 | x86_64-unknown-linux-gnu | multiversion v4 clone features | icelake-server, sapphirerapids, znver4 |
| i686 | i686-unknown-linux-gnu | SSE2 | skylake |
| aarch64 | aarch64-unknown-linux-gnu | armv8-a + NEON | cortex-a72, neoverse-n1, neoverse-v2 |
| aarch64-apple | aarch64-apple-darwin | apple-m1 | apple-m1 |
| armv7 | armv7-unknown-linux-gnueabihf | VFPv3-D16, **no NEON** | cortex-a9, cortex-a57 |
| ppc64le | powerpc64le-unknown-linux-gnu | POWER8 (VSX) | pwr8, pwr9 |

Windows and musl wheels share the ISA of their linux counterparts. For x86-64, the multiversion
clones get exactly `#[target_feature(enable = ...)]` (generic tuning). The probe reproduces each
clone by building without `multiversion` and with the same features in `-C target-feature`.

For each (platform, kernel):
1. `cargo asm --simplify` gives the asm, and the loop tree comes from back-edge branches.
2. `cargo asm --llvm` gives the final LLVM IR. The toolchain's own `opt` prints the IR loop tree
   (`print<loops>`) and exact trip counts (`print<scalar-evolution>`).
3. The trees are matched by shape, which gives every asm loop its trip count. The outer loop's
   SCEV expression is checked to step by `LANES`, so one outer iteration is always 128 bytes.
4. One outer iteration is flattened: inner loops that LLVM did not unroll are repeated by their
   trip count. llvm-mca simulates that trace (100 iterations) on each CPU.

**Metric: cycles per 128 bytes. Lower is better.** The per-(platform, CPU) geomean over the 8
fixed-itemsize kernels (encode/decode x itemsize 2/4/8/16) is the headline number.

Caveats:
- llvm-mca assumes every load hits L1 and ignores the front end. Branches are not followed.
- **Calls are free in llvm-mca.** Kernels that call something (e.g. `memcpy`) inside the hot loop
  are flagged `*`, and their real cost is higher.
- Instructions with no scheduling info on a CPU are skipped and listed as warnings. So far this is
  only ppc `mtctr` on pwr8, once per 128 bytes in `encode_16`, so the effect is negligible.
- `*_generic` (runtime itemsize: odd sizes, struct dtypes, and the `<LANES` tail) cannot be
  flattened. Only the cycles per iteration of their largest innermost loop are reported, for
  information.

## Baseline (auto-vectorized kernels, commit introducing this probe)

Full tables: [`results/baseline/summary.md`](results/baseline/summary.md).

| platform | cpu | geomean | encode | decode | worst kernel |
|---|---|---:|---:|---:|---|
| x86_64-v1 | x86-64 | **94.3** | 123.3 | 72.1 | decode_16 (136.1) |
| x86_64-v1 | skylake | **85.9** | 113.2 | 65.2 | decode_16 (136.1) |
| x86_64-v2 | sandybridge | **59.0** | 113.6 | 30.6 | decode_16 (136.1) |
| x86_64-v2 | btver2 | **87.4** | 122.7 | 62.3 | decode_16 (204.1) |
| x86_64-v3 | skylake | **67.2** | 98.1 | 46.1 | decode_16 (180.2) |
| x86_64-v3 | alderlake | **39.2** | 56.4 | 27.2 | decode_16 (107.3) |
| x86_64-v3 | znver3 | **60.1** | 107.3 | 33.7 | decode_8 (128.1) |
| x86_64-v4 | icelake-server | **17.9** | 16.1 | 19.8 | encode_8 (49.7) |
| x86_64-v4 | sapphirerapids | **19.8** | 17.3 | 22.5 | decode_16 (60.2) |
| x86_64-v4 | znver4 | **15.3** | 13.7 | 17.1 | encode_8 (36.2) |
| i686 | skylake | **105.7** | 153.5 | 72.8 | encode_16 (264.1) |
| aarch64 | cortex-a72 | **93.9** | 131.9 | 66.9 | decode_16 (203.2) |
| aarch64 | neoverse-n1 | **89.8** | 144.7 | 55.8 | encode_16 (189.1) |
| aarch64 | neoverse-v2 | **44.2** | 87.7 | 22.3 | encode_16 (95.2) |
| aarch64-apple | apple-m1 | **45.6** | 119.2 | 17.4 | encode_2 (128.1) |
| armv7 | cortex-a9 | **524.6** | 507.2 | 542.7 | decode_8 (624.0) |
| armv7 | cortex-a57 | **231.8** | 232.6 | 231.0 | encode_2 (256.1) |
| ppc64le | pwr8 | **80.0** | 114.9 | 55.7 | encode_16 (126.1) |
| ppc64le | pwr9 | **68.2** | 99.1 | 46.9 | decode_16 (127.0) |

Observations:
- **Encode is not vectorized at all below AVX-512.** `encode_2/4/8` compile to 128
  `movzbl`+`movb` pairs (or `ldrb`+`strb`, `lbz`+`stb`) per 128 bytes on x86 v1-v3, i686, NEON,
  ppc64le and armv7. They are store-port bound at 128 cycles on 1-store/cycle cores, 64 on
  Alder Lake and 85 on Neoverse V2. Only x86-64-v4 gets `vpshufb`+`vpermt2q`, at 4-12 cycles for
  itemsize 2/4.
- **Decode is only partially vectorized.** `decode_2` becomes `punpcklbw` / `zip1` everywhere
  (8-25 cycles). `decode_4/8` are vectorized on some levels only. `decode_8` on x86-64-v3 is
  *worse* than on v2 (128 vs 25 cycles on Sandy Bridge-class): the v3 build falls back to scalar
  bytes.
- **Itemsize 16 is the worst case almost everywhere.** LLVM does not unroll the `for b in
  0..ITEMSIZE` loop (16 iterations, `loops = ?[16]`). The block round-trips through the stack
  (x86 v1/v2, i686, armv7, ppc64le). i686 `decode_16` and armv7 `encode_16`/`decode_16` even call
  `memcpy` for 128 bytes inside the hot loop, which is not counted.
- **armv7 has no NEON in its baseline**, so everything is scalar at 230-620 cycles. A NEON path
  would need runtime detection, and on arm32 `std::arch::is_arm_feature_detected!` is still
  nightly-only.
- Headroom: a clean SIMD transpose should reach roughly 8-16 cycles per 128 bytes with 128-bit
  vectors (bounded by the 8-16 vector loads and stores per 128 bytes), and less with 256/512-bit.
  That is about 5-10x the current geomean on most platforms.
