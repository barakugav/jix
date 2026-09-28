# Byte-shuffle llvm-mca summary: `encode-fearless`

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

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| encode_2 | 256 | 75 | ? | - | 302 | 929 |
| encode_4 | 256 | 115 | ? | - | 522 | 1633 |
| encode_8 | 256 | 155 | ? | - | 778 | 2273 |
| encode_16 | 256 | 155 | ? | - | 753 | 1242 |
| **geomean** | | | | | **551** | **1439** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| encode_2 | 256 | 43 | ? | - | 386 | 196 | 194 |
| encode_4 | 256 | 75 | ? | - | 770 | 420 | 387 |
| encode_8 | 256 | 116 | ? | - | 1155 | 581 | 580 |
| encode_16 | 512 | 412 | ? | - | 1538 | 771 | 782 |
| **geomean** | | | | | **853** | **438** | **430** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| encode_2 | 256 | 17 | ? | - | 74 | 132 | 67 |
| encode_4 | 256 | 23 | ? | - | 131 | 260 | 131 |
| encode_8 | 512 | 55 | ? | - | 193 | 386 | 194 |
| encode_16 | 1024 | 137 | ? | - | 257 | 513 | 257 |
| **geomean** | | | | | **148** | **287** | **144** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| encode_2 | 256 | 99 | ? | - | 343 |
| encode_4 | 256 | 148 | ?[4] | - | 1026 |
| encode_8 | 256 | 233 | ?[4 4] | - | 1026 |
| encode_16 | 256 | 300 | ?[4] | - | 1283 |
| **geomean** | | | | | **825** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| encode_2 | 256 | 37 | ? | - | 546 | 768 | 242 |
| encode_4 | 256 | 37 | ? | - | 626 | 768 | 241 |
| encode_8 | 256 | 59 | ? | - | 898 | 864 | 306 |
| encode_16 | 256 | 93 | ? | - | 1459 | 865 | 386 |
| **geomean** | | | | | **818** | **815** | **288** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| encode_2 | 256 | 37 | ? | - | 249 |
| encode_4 | 256 | 37 | ? | - | 266 |
| encode_8 | 256 | 59 | ? | - | 339 |
| encode_16 | 256 | 92 | ? | - | 344 |
| **geomean** | | | | | **297** |
