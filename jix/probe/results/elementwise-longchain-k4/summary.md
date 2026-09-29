# llvm-mca summary: `elementwise-longchain-k4` (elementwise kernels)

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`
- kernels: the pipeline's inner loop for contiguous operands (`inner_loop_contiguous`, the arm of the platform's SIMD level); the bytes are output bytes

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

| platform | cpu | geomean | neg_f32 | add_f32 | chain_f32 | longchain_f32 | neg_f64 | add_f64 | chain_f64 | longchain_f64 | neg_i32 | add_i32 | chain_i32 | longchain_i32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x86_64 | sandybridge | **609** | 325 | 390 | 650 | 1292 | 325 | 390 | 650 | 1292 | 310 | 390 | 956 | 1964 |
| x86_64 | skylake | **513** | 262 | 266 | 524 | 1039 | 262 | 266 | 524 | 1039 | 262 | 264 | 1292 | 2572 |
| x86_64 | znver3 | **538** | 264 | 390 | 650 | 1164 | 264 | 390 | 650 | 1164 | 264 | 390 | 695 | 1358 |
| x86_64-v2 | sandybridge | **564** | 325 | 390 | 650 | 1292 | 325 | 390 | 650 | 1292 | 310 | 390 | 649 | 1164 |
| x86_64-v2 | btver2 | **985** | 486 | 616 | 1130 | 2157 | 486 | 616 | 1132 | 2158 | 614 | 614 | 1479 | 2824 |
| x86_64-v3 | skylake | **221** | 132 | 132 | 262 | 520 | 132 | 132 | 262 | 520 | 132 | 132 | 263 | 520 |
| x86_64-v3 | alderlake | **188** | 103 | 136 | 218 | 395 | 103 | 136 | 218 | 395 | 103 | 125 | 235 | 444 |
| x86_64-v3 | znver3 | **264** | 132 | 195 | 324 | 581 | 132 | 195 | 324 | 581 | 132 | 195 | 324 | 581 |
| x86_64-v4 | icelake-server | **132** | 55 | 77 | 195 | 451 | 55 | 77 | 195 | 451 | 55 | 76 | 156 | 323 |
| x86_64-v4 | sapphirerapids | **98** | 52 | 63 | 109 | 229 | 52 | 63 | 109 | 229 | 52 | 63 | 134 | 293 |
| x86_64-v4 | znver4 | **134** | 66 | 97 | 162 | 323 | 66 | 97 | 162 | 323 | 66 | 97 | 162 | 290 |
| i686 | skylake | **531** | 262 | 264 | 524 | 1295 | 262 | 264 | 524 | 1295 | 262 | 262 | 1290 | 2577 |
| aarch64 | cortex-a72 | **1367** | 646 | 966 | 1862 | 3654 | 646 | 966 | 1862 | 3654 | 646 | 901 | 1544 | 2632 |
| aarch64 | neoverse-n1 | **1134** | 645 | 837 | 1286 | 2244 | 645 | 837 | 1286 | 2244 | 645 | 837 | 1414 | 2439 |
| aarch64 | neoverse-v2 | **592** | 262 | 454 | 776 | 1352 | 262 | 454 | 776 | 1352 | 262 | 454 | 776 | 1289 |
| aarch64-apple | apple-m1 | **385** | 203 | 266 | 458 | 908 | 203 | 265 | 460 | 910 | 203 | 264 | 458 | 842 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? | - | 325 | 262 | 264 |
| add_f32 | 64 | 19 | ? ? ? ? | - | 390 | 266 | 390 |
| chain_f32 | 64 | 35 | ? ? ? | - | 650 | 524 | 650 |
| longchain_f32 | 64 | 67 | ? ? ? | - | 1292 | 1039 | 1164 |
| neg_f64 | 64 | 15 | ? ? ? ? | - | 325 | 262 | 264 |
| add_f64 | 64 | 19 | ? ? ? ? | - | 390 | 266 | 390 |
| chain_f64 | 64 | 35 | ? ? ? | - | 650 | 524 | 650 |
| longchain_f64 | 64 | 67 | ? ? ? | - | 1292 | 1039 | 1164 |
| neg_i32 | 64 | 19 | ? ? ? ? | - | 310 | 262 | 264 |
| add_i32 | 64 | 19 | ? ? ? ? | - | 390 | 264 | 390 |
| chain_i32 | 64 | 59 | ? ? ? | - | 956 | 1292 | 695 |
| longchain_i32 | 64 | 115 | ? ? ? | - | 1964 | 2572 | 1358 |
| **geomean** | | | | | **609** | **513** | **538** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? | - | 325 | 486 |
| add_f32 | 64 | 19 | ? ? ? ? | - | 390 | 616 |
| chain_f32 | 64 | 35 | ? ? ? | - | 650 | 1130 |
| longchain_f32 | 64 | 67 | ? ? ? | - | 1292 | 2157 |
| neg_f64 | 64 | 15 | ? ? ? ? | - | 325 | 486 |
| add_f64 | 64 | 19 | ? ? ? ? | - | 390 | 616 |
| chain_f64 | 64 | 35 | ? ? ? | - | 650 | 1132 |
| longchain_f64 | 64 | 67 | ? ? ? | - | 1292 | 2158 |
| neg_i32 | 64 | 19 | ? ? ? ? | - | 310 | 614 |
| add_i32 | 64 | 19 | ? ? ? ? | - | 390 | 614 |
| chain_i32 | 64 | 35 | ? ? ? | - | 649 | 1479 |
| longchain_i32 | 64 | 67 | ? ? ? | - | 1164 | 2824 |
| **geomean** | | | | | **564** | **985** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f32 | 128 | 15 | ? ? ? ? ? | - | 132 | 136 | 195 |
| chain_f32 | 128 | 27 | ? ? ? ? | - | 262 | 218 | 324 |
| longchain_f32 | 128 | 51 | ? ? ? ? | - | 520 | 395 | 581 |
| neg_f64 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f64 | 128 | 15 | ? ? ? ? ? | - | 132 | 136 | 195 |
| chain_f64 | 128 | 27 | ? ? ? ? | - | 262 | 218 | 324 |
| longchain_f64 | 128 | 51 | ? ? ? | - | 520 | 395 | 581 |
| neg_i32 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_i32 | 128 | 15 | ? ? ? ? ? | - | 132 | 125 | 195 |
| chain_i32 | 128 | 27 | ? ? ? ? | - | 263 | 235 | 324 |
| longchain_i32 | 128 | 51 | ? ? ? | - | 520 | 444 | 581 |
| **geomean** | | | | | **221** | **188** | **264** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_f32 | 256 | 15 | ? ? ? ? | - | 77 | 63 | 97 |
| chain_f32 | 256 | 27 | ? ? ? ? | - | 195 | 109 | 162 |
| longchain_f32 | 256 | 51 | ? ? ? ? | - | 451 | 229 | 323 |
| neg_f64 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_f64 | 256 | 15 | ? ? ? ? | - | 77 | 63 | 97 |
| chain_f64 | 256 | 27 | ? ? ? ? | - | 195 | 109 | 162 |
| longchain_f64 | 256 | 51 | ? ? ? ? | - | 451 | 229 | 323 |
| neg_i32 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_i32 | 256 | 15 | ? ? ? ? | - | 76 | 63 | 97 |
| chain_i32 | 256 | 27 | ? ? ? ? | - | 156 | 134 | 162 |
| longchain_i32 | 256 | 51 | ? ? ? ? | - | 323 | 293 | 290 |
| **geomean** | | | | | **132** | **98** | **134** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| neg_f32 | 64 | 15 | ? ? ? | - | 262 |
| add_f32 | 64 | 19 | ? ? ? | - | 264 |
| chain_f32 | 64 | 35 | ? ? ? | - | 524 |
| longchain_f32 | 64 | 75 | ? ? ? | - | 1295 |
| neg_f64 | 64 | 15 | ? ? ? | - | 262 |
| add_f64 | 64 | 19 | ? ? ? | - | 264 |
| chain_f64 | 64 | 35 | ? ? ? | - | 524 |
| longchain_f64 | 64 | 75 | ? ? ? | - | 1295 |
| neg_i32 | 64 | 19 | ? ? ? | - | 262 |
| add_i32 | 64 | 19 | ? ? ? | - | 262 |
| chain_i32 | 64 | 61 | ? ? ? | - | 1290 |
| longchain_i32 | 64 | 127 | ? ? ? | - | 2577 |
| **geomean** | | | | | **531** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 64 | 11 | ? ? ? | - | 646 | 645 | 262 |
| add_f32 | 64 | 13 | ? ? ? | - | 966 | 837 | 454 |
| chain_f32 | 64 | 25 | ? ? ? | - | 1862 | 1286 | 776 |
| longchain_f32 | 64 | 49 | ? ? ? | - | 3654 | 2244 | 1352 |
| neg_f64 | 64 | 11 | ? ? ? | - | 646 | 645 | 262 |
| add_f64 | 64 | 13 | ? ? ? | - | 966 | 837 | 454 |
| chain_f64 | 64 | 25 | ? ? ? | - | 1862 | 1286 | 776 |
| longchain_f64 | 64 | 49 | ? ? ? | - | 3654 | 2244 | 1352 |
| neg_i32 | 64 | 11 | ? ? ? | - | 646 | 645 | 262 |
| add_i32 | 64 | 13 | ? ? ? | - | 901 | 837 | 454 |
| chain_i32 | 64 | 25 | ? ? ? | - | 1544 | 1414 | 776 |
| longchain_i32 | 64 | 45 | ? ? ? | - | 2632 | 2439 | 1289 |
| **geomean** | | | | | **1367** | **1134** | **592** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| neg_f32 | 64 | 10 | ? ? ? ? | - | 203 |
| add_f32 | 64 | 12 | ? ? ? ? | - | 266 |
| chain_f32 | 64 | 24 | ? ? ? ? | - | 458 |
| longchain_f32 | 64 | 49 | ? ? ? | - | 908 |
| neg_f64 | 64 | 10 | ? ? ? | - | 203 |
| add_f64 | 64 | 12 | ? ? ? | - | 265 |
| chain_f64 | 64 | 24 | ? ? ? | - | 460 |
| longchain_f64 | 64 | 49 | ? ? ? | - | 910 |
| neg_i32 | 64 | 10 | ? ? ? ? | - | 203 |
| add_i32 | 64 | 12 | ? ? ? ? | - | 264 |
| chain_i32 | 64 | 24 | ? ? ? ? | - | 458 |
| longchain_i32 | 64 | 45 | ? ? ? ? | - | 842 |
| **geomean** | | | | | **385** |
