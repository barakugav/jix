# Byte-shuffle llvm-mca summary: `encode-baseline`

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

| platform | cpu | geomean | encode_2 | encode_4 | encode_8 | encode_16 |
|---|---|---:|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | **3634** | 4099 | 4099 | 4099 | 2533 |
| x86_64-v2 | btver2 | **3928** | 4162 | 4162 | 4162 | 3301 |
| x86_64-v3 | skylake | **3140** | 4099 | 4099 | 4099 | 1413 |
| x86_64-v3 | alderlake | **1804** | 2070 | 2070 | 2070 | 1193 |
| x86_64-v3 | znver3 | **3435** | 4098 | 4098 | 4098 | 2022 |
| x86_64-v4 | icelake-server | **516** | 196 | 388 | 1589 | 585 |
| x86_64-v4 | sapphirerapids | **555** | 201 | 393 | 1547 | 776 |
| x86_64-v4 | znver4 | **437** | 134 | 262 | 1159 | 900 |
| i686 | skylake | **4911** | 4099 | 4099 | 4099 | 8452 |
| aarch64 | cortex-a72 | **4222** | 4099 | 4099 | 4099 | 4615 |
| aarch64 | neoverse-n1 | **4631** | 4194 | 4226 | 4290 | 6052 |
| aarch64 | neoverse-v2 | **2808** | 2733 | 2733 | 2733 | 3045 |
| aarch64-apple | apple-m1 | **3816** | 4099 | 4099 | 4099 | 3078 |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| encode_2 | 128 | 260 | ? | - | 4099 | 4162 |
| encode_4 | 128 | 260 | ? | - | 4099 | 4162 |
| encode_8 | 128 | 260 | ? | - | 4099 | 4162 |
| encode_16 | 128 | 203 | ? | - | 2533 | 3301 |
| **geomean** | | | | | **3634** | **3928** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| encode_2 | 128 | 260 | ? | - | 4099 | 2070 | 4098 |
| encode_4 | 128 | 260 | ? | - | 4099 | 2070 | 4098 |
| encode_8 | 128 | 260 | ? | - | 4099 | 2070 | 4098 |
| encode_16 | 128 | 116 | ? | - | 1413 | 1193 | 2022 |
| **geomean** | | | | | **3140** | **1804** | **3435** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| encode_2 | 128 | 15 | ? | - | 196 | 201 | 134 |
| encode_4 | 128 | 27 | ? | - | 388 | 393 | 262 |
| encode_8 | 128 | 123 | ? | - | 1589 | 1547 | 1159 |
| encode_16 | 128 | 47 | ? | - | 585 | 776 | 900 |
| **geomean** | | | | | **516** | **555** | **437** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| encode_2 | 128 | 260 | ? | - | 4099 |
| encode_4 | 128 | 261 | ? | - | 4099 |
| encode_8 | 128 | 266 | ? | - | 4099 |
| encode_16 | 128 | 474 | ? | - | 8452 |
| **geomean** | | | | | **4911** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| encode_2 | 128 | 262 | ? | - | 4099 | 4194 | 2733 |
| encode_4 | 128 | 264 | ? | - | 4099 | 4226 | 2733 |
| encode_8 | 128 | 268 | ? | - | 4099 | 4290 | 2733 |
| encode_16 | 128 | 329 | ? | - | 4615 | 6052 | 3045 |
| **geomean** | | | | | **4222** | **4631** | **2808** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| encode_2 | 128 | 262 | ? | - | 4099 |
| encode_4 | 128 | 264 | ? | - | 4099 |
| encode_8 | 128 | 268 | ? | - | 4099 |
| encode_16 | 128 | 340 | ? | - | 3078 |
| **geomean** | | | | | **3816** |
