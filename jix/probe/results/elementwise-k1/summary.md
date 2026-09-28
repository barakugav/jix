# llvm-mca summary: `elementwise-k1` (elementwise kernels)

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

| platform | cpu | geomean | neg_f32 | add_f32 | chain_f32 | neg_f64 | add_f64 | chain_f64 | neg_i32 | add_i32 | chain_i32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x86_64 | sandybridge | **463** | 325 | 390 | 663 | 325 | 390 | 663 | 310 | 390 | 1139 |
| x86_64 | skylake | **369** | 262 | 264 | 538 | 262 | 264 | 538 | 262 | 262 | 1321 |
| x86_64 | znver3 | **415** | 264 | 390 | 659 | 264 | 390 | 659 | 264 | 390 | 776 |
| x86_64-v2 | sandybridge | **435** | 325 | 390 | 663 | 325 | 390 | 663 | 310 | 390 | 658 |
| x86_64-v2 | btver2 | **754** | 485 | 615 | 1231 | 485 | 615 | 1236 | 613 | 614 | 1549 |
| x86_64-v3 | skylake | **170** | 132 | 133 | 269 | 132 | 133 | 269 | 132 | 132 | 301 |
| x86_64-v3 | alderlake | **150** | 103 | 126 | 255 | 103 | 126 | 255 | 103 | 125 | 280 |
| x86_64-v3 | znver3 | **203** | 132 | 195 | 329 | 132 | 195 | 329 | 132 | 195 | 327 |
| x86_64-v4 | icelake-server | **92** | 55 | 77 | 198 | 55 | 77 | 198 | 55 | 76 | 167 |
| x86_64-v4 | sapphirerapids | **75** | 52 | 63 | 120 | 52 | 63 | 120 | 52 | 63 | 150 |
| x86_64-v4 | znver4 | **102** | 66 | 97 | 164 | 66 | 97 | 164 | 66 | 97 | 164 |
| i686 | skylake | **456** | 333 | 374 | 563 | 330 | 374 | 563 | 366 | 366 | 1318 |
| aarch64 | cortex-a72 | **1532** | 1290 | 1298 | 2337 | 1290 | 1298 | 2337 | 1290 | 1293 | 1818 |
| aarch64 | neoverse-n1 | **1352** | 1044 | 1300 | 1820 | 1044 | 1300 | 1820 | 1044 | 1300 | 1823 |
| aarch64 | neoverse-v2 | **770** | 540 | 796 | 1062 | 540 | 796 | 1062 | 540 | 796 | 1062 |
| aarch64-apple | apple-m1 | **489** | 335 | 420 | 689 | 412 | 484 | 827 | 335 | 415 | 724 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? ? | - | 325 | 262 | 264 |
| add_f32 | 64 | 19 | ? ? ? ? ? | - | 390 | 264 | 390 |
| chain_f32 | 32 | 19 | ? ? ? | - | 663 | 538 | 659 |
| neg_f64 | 64 | 15 | ? ? ? ? ? | - | 325 | 262 | 264 |
| add_f64 | 64 | 19 | ? ? ? ? ? | - | 390 | 264 | 390 |
| chain_f64 | 32 | 19 | ? ? ? | - | 663 | 538 | 659 |
| neg_i32 | 64 | 19 | ? ? ? ? ? | - | 310 | 262 | 264 |
| add_i32 | 64 | 19 | ? ? ? ? ? | - | 390 | 262 | 390 |
| chain_i32 | 16 | 17 | ? ? ? | - | 1139 | 1321 | 776 |
| **geomean** | | | | | **463** | **369** | **415** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? ? | - | 325 | 485 |
| add_f32 | 64 | 19 | ? ? ? ? ? | - | 390 | 615 |
| chain_f32 | 32 | 19 | ? ? ? | - | 663 | 1231 |
| neg_f64 | 64 | 15 | ? ? ? ? ? | - | 325 | 485 |
| add_f64 | 64 | 19 | ? ? ? ? ? | - | 390 | 615 |
| chain_f64 | 32 | 19 | ? ? ? | - | 663 | 1236 |
| neg_i32 | 64 | 19 | ? ? ? ? ? | - | 310 | 613 |
| add_i32 | 64 | 19 | ? ? ? ? ? | - | 390 | 614 |
| chain_i32 | 32 | 19 | ? ? ? | - | 658 | 1549 |
| **geomean** | | | | | **435** | **754** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 11 | ? ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f32 | 128 | 15 | ? ? ? ? ? ? | - | 133 | 126 | 195 |
| chain_f32 | 64 | 13 | ? ? ? ? | - | 269 | 255 | 329 |
| neg_f64 | 128 | 11 | ? ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f64 | 128 | 15 | ? ? ? ? ? ? | - | 133 | 126 | 195 |
| chain_f64 | 64 | 13 | ? ? ? ? | - | 269 | 255 | 329 |
| neg_i32 | 128 | 11 | ? ? ? ? ? ? | - | 132 | 103 | 132 |
| add_i32 | 128 | 15 | ? ? ? ? ? ? | - | 132 | 125 | 195 |
| chain_i32 | 64 | 13 | ? ? ? ? | - | 301 | 280 | 327 |
| **geomean** | | | | | **170** | **150** | **203** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 256 | 11 | ? ? ? ? ? | - | 55 | 52 | 66 |
| add_f32 | 256 | 15 | ? ? ? ? ? | - | 77 | 63 | 97 |
| chain_f32 | 128 | 13 | ? ? ? ? | - | 198 | 120 | 164 |
| neg_f64 | 256 | 11 | ? ? ? ? ? | - | 55 | 52 | 66 |
| add_f64 | 256 | 15 | ? ? ? ? ? | - | 77 | 63 | 97 |
| chain_f64 | 128 | 13 | ? ? ? ? | - | 198 | 120 | 164 |
| neg_i32 | 256 | 11 | ? ? ? ? ? | - | 55 | 52 | 66 |
| add_i32 | 256 | 15 | ? ? ? ? ? | - | 76 | 63 | 97 |
| chain_i32 | 128 | 13 | ? ? ? ? | - | 167 | 150 | 164 |
| **geomean** | | | | | **92** | **75** | **102** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| neg_f32 | 16 | 6 | ? ? ? | - | 333 |
| add_f32 | 16 | 7 | ? ? ? | - | 374 |
| chain_f32 | 16 | 11 | ? ? ? | - | 563 |
| neg_f64 | 16 | 6 | ? ? ? | - | 330 |
| add_f64 | 16 | 7 | ? ? ? | - | 374 |
| chain_f64 | 16 | 11 | ? ? ? | - | 563 |
| neg_i32 | 16 | 7 | ? ? ? | - | 366 |
| add_i32 | 16 | 7 | ? ? ? | - | 366 |
| chain_i32 | 16 | 17 | ? ? ? | - | 1318 |
| **geomean** | | | | | **456** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 16 | 6 | ? ? ? | - | 1290 | 1044 | 540 |
| add_f32 | 16 | 7 | ? ? ? | - | 1298 | 1300 | 796 |
| chain_f32 | 16 | 11 | ? ? ? | - | 2337 | 1820 | 1062 |
| neg_f64 | 16 | 6 | ? ? ? | - | 1290 | 1044 | 540 |
| add_f64 | 16 | 7 | ? ? ? | - | 1298 | 1300 | 796 |
| chain_f64 | 16 | 11 | ? ? ? | - | 2337 | 1820 | 1062 |
| neg_i32 | 16 | 6 | ? ? ? | - | 1290 | 1044 | 540 |
| add_i32 | 16 | 7 | ? ? ? | - | 1293 | 1300 | 796 |
| chain_i32 | 16 | 11 | ? ? ? | - | 1818 | 1823 | 1062 |
| **geomean** | | | | | **1532** | **1352** | **770** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| neg_f32 | 16 | 5 | ? ? ? ? | - | 335 |
| add_f32 | 16 | 6 | ? ? ? ? | - | 420 |
| chain_f32 | 16 | 10 | ? ? ? ? | - | 689 |
| neg_f64 | 16 | 6 | ? ? ? | - | 412 |
| add_f64 | 16 | 7 | ? ? ? | - | 484 |
| chain_f64 | 16 | 11 | ? ? ? | - | 827 |
| neg_i32 | 16 | 5 | ? ? ? ? | - | 335 |
| add_i32 | 16 | 6 | ? ? ? ? | - | 415 |
| chain_i32 | 16 | 10 | ? ? ? ? | - | 724 |
| **geomean** | | | | | **489** |
