# llvm-mca summary: `aa-B`

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`
- source: `py` (`jix-py`)
- kernels: the pipeline's inner loop for contiguous operands (`inner_loop_contiguous`, the arm of the platform's SIMD level); the bytes are output bytes

llvm-mca steady-state **cycles per 4096 bytes**: the hot loops (those whose total
iteration count is linear in the input length), one iteration each with inner loops flattened
by their trip counts, weighted by their iterations per byte. Lower is better. `geomean` is the
per-CPU geometric mean over the kernels, the single number to optimize.

- `B/iter`: input bytes per hot-loop iteration (from the SCEV trip counts).
- `instrs`: instructions per hot-loop iteration (flattened trace).
- `stack`: of those, the ones accessing the stack (spills / reloads), `vector/general` registers.
- `loops`: asm loop tree with trip counts; `?` = runtime trip count.
- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include. Values of such
  kernels (and geomeans including them) are flagged `*`: the real cost is higher.

## Out-of-line Rust calls

None.

## Overview

| kernel | x86_64 sandybridge | x86_64 skylake | x86_64 znver3 | x86_64-v2 sandybridge | x86_64-v2 btver2 | x86_64-v3 skylake | x86_64-v3 alderlake | x86_64-v3 znver3 | x86_64-v4 icelake-server | x86_64-v4 sapphirerapids | x86_64-v4 znver4 | i686 skylake | aarch64 cortex-a72 | aarch64 neoverse-n1 | aarch64 neoverse-v2 | aarch64-apple apple-m1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reduce_all_bool_cell | 155 | 131 | 132 | 155 | 307 | 66 | 50 | 66 | 33 | 33 | 33 | 196 | 291 | 291 | 131 | 109 |
| reduce_all_bool_finalize | 268 | 268 | 270 | 268 | 458 | 131 | 82 | 132 | 55 | 46 | 66 | 268 | 650 | 650 | 268 | 198 |
| reduce_all_bool_init | 268 | 268 | 270 | 268 | 458 | 131 | 82 | 132 | - | - | - | 268 | - | - | - | - |
| reduce_all_bool_update | 397 | 302 | 396 | 397 | 716 | 132 | 125 | 195 | 76 | 68 | 97 | 301 | 910 | 909 | 398 | 263 |
| reduce_any_bool_cell | 155 | 131 | 132 | 155 | 307 | 66 | 50 | 66 | 33 | 33 | 33 | 196 | 291 | 291 | 131 | 109 |
| reduce_any_bool_update | 397 | 302 | 396 | 397 | 716 | 132 | 125 | 195 | 76 | 68 | 97 | 301 | 910 | 909 | 398 | 263 |
| reduce_arg_max_bool_cell | 7672 | 7219 | 4672 | 5647 | 10516 | 3348 | 2854 | 3110 | 799 | 804 | 589 | 39480 | 11156 | 12047 | 5908 | 13330 |
| reduce_arg_max_bool_init | 8264 | 8264 | 8264 | 8264 | 8253 | 8264 | 4301 | 8264 | 4178 | 4301 | 8264 | 12534 | 10445 | 12493 | 6390 | 6200 |
| reduce_arg_max_bool_update | 16630 | 9789 | 9789 | 16630 | 28815 | 9789 | 10691 | 9789 | 9769 | 10691 | 9789 | 13025 | 18760 | 20767 | 10609 | 11059 |
| **geomean** | **869** | **738** | **749** | **840** | **1432** | **413** | **318** | **447** | **241** | **232** | **274** | **1053** | **1777** | **1857** | **857** | **788** |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 128 | 19 | 0/0 | ? ? | - | 155 | 131 | 132 |
| reduce_all_bool_finalize | 32 | 7 | 0/0 | ? ? ? ? | - | 268 | 268 | 270 |
| reduce_all_bool_init | 32 | 7 | 0/0 | ? ? ? | - | 268 | 268 | 270 |
| reduce_all_bool_update | 32 | 11 | 0/0 | ? ? ? | - | 397 | 302 | 396 |
| reduce_any_bool_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 131 | 132 |
| reduce_any_bool_update | 32 | 11 | 0/0 | ? ? ? | - | 397 | 302 | 396 |
| reduce_arg_max_bool_cell | 16 | 106 | 20/0 | ? ? | - | 7672 | 7219 | 4672 |
| reduce_arg_max_bool_init | 4 | 16 | 0/0 | ? ? | - | 8264 | 8264 | 8264 |
| reduce_arg_max_bool_update | 2 | 28 | 0/0 | ? | - | 16630 | 9789 | 9789 |
| **geomean** | | | | | | **869** | **738** | **749** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | btver2 |
|---|---:|---:|---:|---|---|---:|---:|
| reduce_all_bool_cell | 128 | 19 | 0/0 | ? ? | - | 155 | 307 |
| reduce_all_bool_finalize | 32 | 7 | 0/0 | ? ? ? ? | - | 268 | 458 |
| reduce_all_bool_init | 32 | 7 | 0/0 | ? ? ? | - | 268 | 458 |
| reduce_all_bool_update | 32 | 11 | 0/0 | ? ? ? | - | 397 | 716 |
| reduce_any_bool_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_any_bool_update | 32 | 11 | 0/0 | ? ? ? | - | 397 | 716 |
| reduce_arg_max_bool_cell | 16 | 62 | 12/0 | ? ? | - | 5647 | 10516 |
| reduce_arg_max_bool_init | 4 | 16 | 0/0 | ? ? | - | 8264 | 8253 |
| reduce_arg_max_bool_update | 2 | 28 | 0/0 | ? | - | 16630 | 28815 |
| **geomean** | | | | | | **840** | **1432** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | stack | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 50 | 66 |
| reduce_all_bool_finalize | 128 | 11 | 0/0 | ? ? ? ? | - | 131 | 82 | 132 |
| reduce_all_bool_init | 128 | 11 | 0/0 | ? ? ? | - | 131 | 82 | 132 |
| reduce_all_bool_update | 128 | 15 | 0/0 | ? ? ? | - | 132 | 125 | 195 |
| reduce_any_bool_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 50 | 66 |
| reduce_any_bool_update | 128 | 15 | 0/0 | ? ? ? | - | 132 | 125 | 195 |
| reduce_arg_max_bool_cell | 16 | 29 | 0/0 | ? ? | - | 3348 | 2854 | 3110 |
| reduce_arg_max_bool_init | 4 | 16 | 0/0 | ? ? | - | 8264 | 4301 | 8264 |
| reduce_arg_max_bool_update | 2 | 28 | 0/0 | ? | - | 9789 | 10691 | 9789 |
| **geomean** | | | | | | **413** | **318** | **447** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | stack | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 512 | 11 | 0/0 | ? ? ? | - | 33 | 33 | 33 |
| reduce_all_bool_finalize | 256 | 11 | 0/0 | ? ? ? ? | - | 55 | 46 | 66 |
| reduce_all_bool_update | 256 | 15 | 0/0 | ? ? ? | - | 76 | 68 | 97 |
| reduce_any_bool_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 33 | 33 | 33 |
| reduce_any_bool_update | 256 | 15 | 0/0 | ? ? ? | - | 76 | 68 | 97 |
| reduce_arg_max_bool_cell | 16 | 13 | 0/0 | ? ? | - | 799 | 804 | 589 |
| reduce_arg_max_bool_init | 4 | 16 | 0/0 | ? ? | - | 4178 | 4301 | 8264 |
| reduce_arg_max_bool_update | 2 | 28 | 0/0 | ? | - | 9769 | 10691 | 9789 |
| **geomean** | | | | | | **241** | **232** | **274** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | skylake |
|---|---:|---:|---:|---|---|---:|
| reduce_all_bool_cell | 128 | 33 | 4/0 | ? ? | - | 196 |
| reduce_all_bool_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 |
| reduce_all_bool_init | 32 | 7 | 0/0 | ? ? ? | - | 268 |
| reduce_all_bool_update | 32 | 11 | 0/0 | ? ? ? | - | 301 |
| reduce_any_bool_cell | 128 | 33 | 4/0 | ? ? ? ? | - | 196 |
| reduce_any_bool_update | 32 | 11 | 0/0 | ? ? ? | - | 301 |
| reduce_arg_max_bool_cell | 8 | 145 | 0/84 | ? ? | - | 39480 |
| reduce_arg_max_bool_init | 1 | 8 | 0/0 | ? | - | 12534 |
| reduce_arg_max_bool_update | 1 | 17 | 0/0 | ? | - | 13025 |
| **geomean** | | | | | | **1053** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 128 | 15 | 0/0 | ? ? | - | 291 | 291 | 131 |
| reduce_all_bool_finalize | 32 | 6 | 0/0 | ? ? ? | - | 650 | 650 | 268 |
| reduce_all_bool_update | 32 | 9 | 0/0 | ? ? ? | - | 910 | 909 | 398 |
| reduce_any_bool_cell | 128 | 15 | 0/0 | ? ? | - | 291 | 291 | 131 |
| reduce_any_bool_update | 32 | 9 | 0/0 | ? ? ? | - | 910 | 909 | 398 |
| reduce_arg_max_bool_cell | 16 | 124 | 0/21 | ? ? | - | 11156 | 12047 | 5908 |
| reduce_arg_max_bool_init | 1 | 5 | 0/0 | ? | - | 10445 | 12493 | 6390 |
| reduce_arg_max_bool_update | 1 | 11 | 0/0 | ? | - | 18760 | 20767 | 10609 |
| **geomean** | | | | | | **1777** | **1857** | **857** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | stack | loops | calls | apple-m1 |
|---|---:|---:|---:|---|---|---:|
| reduce_all_bool_cell | 128 | 15 | 0/0 | ? ? | - | 109 |
| reduce_all_bool_finalize | 64 | 6 | 0/0 | ? ? ? | - | 198 |
| reduce_all_bool_update | 64 | 12 | 0/0 | ? ? ? | - | 263 |
| reduce_any_bool_cell | 128 | 15 | 0/0 | ? ? | - | 109 |
| reduce_any_bool_update | 64 | 12 | 0/0 | ? ? ? | - | 263 |
| reduce_arg_max_bool_cell | 16 | 124 | 0/22 | ? ? | - | 13330 |
| reduce_arg_max_bool_init | 8 | 29 | 0/0 | ? ? | - | 6200 |
| reduce_arg_max_bool_update | 1 | 11 | 0/0 | ? | - | 11059 |
| **geomean** | | | | | | **788** |
