# Byte-shuffle llvm-mca summary: `fearless`

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`

Fixed-itemsize decode kernels: llvm-mca steady-state **cycles per 4096 bytes** (one
outer-loop iteration, inner loops flattened by their trip counts, scaled by the bytes per
iteration). Lower is better. `geomean` is the per-CPU geometric mean over the fixed-itemsize
kernels, the single number to optimize.

- `B/iter`: bytes per outer-loop iteration (from the SCEV trip count).
- `instrs`: instructions per outer-loop iteration (flattened trace).
- `loops`: asm loop tree with trip counts; `?` = runtime trip count (the outer loop).
- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include. Values of such
  kernels (and geomeans including them) are flagged `*`: the real cost is higher.

## Overview

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

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| decode_2 | 256 | 59 | ? | - | 258 | 473 |
| decode_4 | 256 | 83 | ? | - | 338 | 665 |
| decode_8 | 256 | 107 | ? | - | 513 | 858 |
| decode_16 | 256 | 155 | ? | - | 754 | 1243 |
| **geomean** | | | | | **429** | **761** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| decode_2 | 256 | 35 | ? | - | 258 | 132 | 130 |
| decode_4 | 256 | 51 | ? | - | 514 | 292 | 275 |
| decode_8 | 256 | 67 | ? | - | 770 | 405 | 403 |
| decode_16 | 512 | 195 | ? | - | 1025 | 547 | 562 |
| **geomean** | | | | | **569** | **304** | **300** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| decode_2 | 256 | 17 | ? | - | 74 | 132 | 67 |
| decode_4 | 256 | 23 | ? | - | 131 | 260 | 131 |
| decode_8 | 512 | 55 | ? | - | 193 | 386 | 194 |
| decode_16 | 1024 | 137 | ? | - | 257 | 513 | 257 |
| **geomean** | | | | | **148** | **287** | **144** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| decode_2 | 256 | 59 | ? | - | 258 |
| decode_4 | 256 | 100 | ?[4] | - | 514 |
| decode_8 | 256 | 189 | ?[3 3] | - | 771 |
| decode_16 | 256 | 301 | ?[4] | - | 1284 |
| **geomean** | | | | | **602** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| decode_2 | 256 | 37 | ? | - | 705 | 576 | 209 |
| decode_4 | 256 | 51 | ? | - | 993 | 624 | 273 |
| decode_8 | 256 | 76 | ? | - | 1236 | 753 | 290 |
| decode_16 | 256 | 92 | ? | - | 1572 | 881 | 322 |
| **geomean** | | | | | **1080** | **699** | **270** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| decode_2 | 256 | 37 | ? | - | 194 |
| decode_4 | 256 | 51 | ? | - | 203 |
| decode_8 | 256 | 76 | ? | - | 260 |
| decode_16 | 256 | 92 | ? | - | 345 |
| **geomean** | | | | | **244** |
