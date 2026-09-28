# Byte-shuffle llvm-mca summary: `baseline`

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

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| decode_2 | 128 | 35 | ? | - | 387 | 563 |
| decode_4 | 128 | 68 | ? | - | 677 | 1860 |
| decode_8 | 128 | 75 | ? | - | 811 | 2307 |
| decode_16 | 128 | 408 | ?[16] | - | 4355 | 6532 |
| **geomean** | | | | | **980** | **1993** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | sandybridge | btver2 |
|---|---:|---|---|---:|---:|
| decode_generic | 54 | ?[?[? ?] ?] | - | 17.8 | 64.1 |

Warnings:

- decode_generic: asm and IR loop trees differ, trip counts unknown

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| decode_2 | 128 | 27 | ? | - | 388 | 264 | 327 |
| decode_4 | 128 | 44 | ? | - | 516 | 306 | 388 |
| decode_8 | 128 | 260 | ? | - | 4099 | 2071 | 4099 |
| decode_16 | 128 | 273 | ? | - | 5765 | 3435 | 2603 |
| **geomean** | | | | | **1474** | **871** | **1078** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---|---|---:|---:|---:|
| decode_generic | 104 | ?[? ? ?] | - | 47.6 | 65.4 | 48.5 |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| decode_2 | 128 | 27 | ? | - | 389 | 393 | 388 |
| decode_4 | 128 | 36 | ? | - | 415 | 394 | 356 |
| decode_8 | 128 | 55 | ? | - | 964 | 905 | 838 |
| decode_16 | 128 | 41 | ? | - | 1035 | 1926 | 775 |
| **geomean** | | | | | **634** | **721** | **547** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---|---|---:|---:|---:|
| decode_generic | 179 | ?[? ? ?] | - | 94.3 | 88.8 | 158.8 |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| decode_2 | 128 | 35 | ? | - | 259 |
| decode_4 | 128 | 261 | ? | - | 4099 |
| decode_8 | 128 | 320 | ? | - | 4100 |
| decode_16 | 128 | 479 | ?[16] | `memcpy@PLT` | 6768* |
| **geomean** | | | | | **2330*** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake |
|---|---:|---|---|---:|
| decode_generic | 8 | ?[?] | - | 2.1 |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| decode_2 | 128 | 26 | ? | - | 803 | 801 | 243 |
| decode_4 | 128 | 67 | ? | - | 1669 | 1601 | 616 |
| decode_8 | 128 | 72 | ? | - | 2409 | 1604 | 649 |
| decode_16 | 128 | 175 | ? | - | 6503 | 4930 | 2662 |
| **geomean** | | | | | **2140** | **1784** | **713** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---|---|---:|---:|---:|
| decode_generic | 19 | ?[? ?] | - | 8.1 | 12.1 | 4.1 |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| decode_2 | 128 | 26 | ? | - | 259 |
| decode_4 | 128 | 67 | ? | - | 599 |
| decode_8 | 128 | 70 | ? | - | 584 |
| decode_16 | 128 | 112 | ? | - | 1063 |
| **geomean** | | | | | **557** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | apple-m1 |
|---|---:|---|---|---:|
| decode_generic | 19 | ?[? ?] | - | 4.2 |
