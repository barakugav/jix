# llvm-mca summary: `bit-fearless` (bit kernels)

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`

llvm-mca steady-state **cycles per 4096 bytes**: the hot loops (those whose total
iteration count is linear in the input length), one iteration each with inner loops flattened
by their trip counts, weighted by their iterations per byte. Lower is better. `geomean` is the
per-CPU geometric mean over the kernels, the single number to optimize.

- `B/iter`: input bytes per hot-loop iteration (from the SCEV trip counts); `~`: the trip count
  depends on non-argument values (e.g. bounds-check limits), approximated up to a bounded term.
- `instrs`: instructions per hot-loop iteration (flattened trace).
- `loops`: asm loop tree with trip counts; `?` = runtime trip count.
- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include. Values of such
  kernels (and geomeans including them) are flagged `*`: the real cost is higher.

## Overview

| platform | cpu | geomean | transpose_bit_rows |
|---|---|---:|---:|
| x86_64-v2 | sandybridge | **1154** | 1154 |
| x86_64-v2 | btver2 | **1717** | 1717 |
| x86_64-v3 | skylake | **386** | 386 |
| x86_64-v3 | alderlake | **388** | 388 |
| x86_64-v3 | znver3 | **291** | 291 |
| x86_64-v4 | icelake-server | **225** | 225 |
| x86_64-v4 | sapphirerapids | **242** | 242 |
| x86_64-v4 | znver4 | **209** | 209 |
| i686 | skylake | **903** | 903 |
| aarch64 | cortex-a72 | **1385** | 1385 |
| aarch64 | neoverse-n1 | **1477** | 1477 |
| aarch64 | neoverse-v2 | **664** | 664 |
| aarch64-apple | apple-m1 | **776** | 776 |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| transpose_bit_rows | 128 | 107 | ? | - | 1154 | 1717 |
| **geomean** | | | | | **1154** | **1717** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| transpose_bit_rows | 256 | 95 | ? | - | 386 | 388 | 291 |
| **geomean** | | | | | **386** | **388** | **291** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| transpose_bit_rows | 512 | 83 | ? | - | 225 | 242 | 209 |
| **geomean** | | | | | **225** | **242** | **209** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| transpose_bit_rows | 128 | 131 | ? | - | 903 |
| **geomean** | | | | | **903** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| transpose_bit_rows | 128 | 91 | ? | - | 1385 | 1477 | 664 |
| **geomean** | | | | | **1385** | **1477** | **664** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| transpose_bit_rows | 128 | 87 | ? | - | 776 |
| **geomean** | | | | | **776** |
