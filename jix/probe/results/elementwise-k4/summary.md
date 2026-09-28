# llvm-mca summary: `elementwise-k4` (elementwise kernels)

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
| x86_64 | sandybridge | **452** | 325 | 390 | 650 | 325 | 390 | 650 | 310 | 390 | 958 |
| x86_64 | skylake | **367** | 262 | 266 | 525 | 262 | 266 | 525 | 262 | 264 | 1292 |
| x86_64 | znver3 | **407** | 264 | 390 | 648 | 264 | 390 | 648 | 264 | 390 | 675 |
| x86_64-v2 | sandybridge | **433** | 325 | 390 | 650 | 325 | 390 | 650 | 310 | 390 | 648 |
| x86_64-v2 | btver2 | **733** | 486 | 616 | 1128 | 486 | 616 | 1130 | 614 | 614 | 1415 |
| x86_64-v3 | skylake | **166** | 132 | 132 | 263 | 132 | 132 | 263 | 132 | 132 | 263 |
| x86_64-v3 | alderlake | **144** | 103 | 136 | 214 | 103 | 136 | 214 | 103 | 125 | 236 |
| x86_64-v3 | znver3 | **203** | 132 | 195 | 324 | 132 | 195 | 324 | 132 | 195 | 323 |
| x86_64-v4 | icelake-server | **90** | 55 | 77 | 195 | 55 | 77 | 195 | 55 | 76 | 139 |
| x86_64-v4 | sapphirerapids | **73** | 52 | 63 | 108 | 52 | 63 | 108 | 52 | 63 | 144 |
| x86_64-v4 | znver4 | **101** | 66 | 97 | 163 | 66 | 97 | 163 | 66 | 97 | 162 |
| i686 | skylake | **366** | 262 | 264 | 524 | 262 | 264 | 524 | 262 | 262 | 1289 |
| aarch64 | cortex-a72 | **1006** | 646 | 966 | 1734 | 646 | 966 | 1734 | 646 | 901 | 1544 |
| aarch64 | neoverse-n1 | **905** | 645 | 837 | 1349 | 645 | 837 | 1349 | 645 | 837 | 1415 |
| aarch64 | neoverse-v2 | **460** | 262 | 454 | 838 | 262 | 454 | 838 | 262 | 454 | 776 |
| aarch64-apple | apple-m1 | **291** | 203 | 266 | 459 | 203 | 265 | 462 | 203 | 264 | 458 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? | - | 325 | 262 | 264 |
| add_f32 | 64 | 19 | ? ? ? ? | - | 390 | 266 | 390 |
| chain_f32 | 64 | 35 | ? ? ? | - | 650 | 525 | 648 |
| neg_f64 | 64 | 15 | ? ? ? ? | - | 325 | 262 | 264 |
| add_f64 | 64 | 19 | ? ? ? ? | - | 390 | 266 | 390 |
| chain_f64 | 64 | 35 | ? ? ? | - | 650 | 525 | 648 |
| neg_i32 | 64 | 19 | ? ? ? ? | - | 310 | 262 | 264 |
| add_i32 | 64 | 19 | ? ? ? ? | - | 390 | 264 | 390 |
| chain_i32 | 64 | 59 | ? ? ? | - | 958 | 1292 | 675 |
| **geomean** | | | | | **452** | **367** | **407** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? | - | 325 | 486 |
| add_f32 | 64 | 19 | ? ? ? ? | - | 390 | 616 |
| chain_f32 | 64 | 35 | ? ? ? | - | 650 | 1128 |
| neg_f64 | 64 | 15 | ? ? ? ? | - | 325 | 486 |
| add_f64 | 64 | 19 | ? ? ? ? | - | 390 | 616 |
| chain_f64 | 64 | 35 | ? ? ? | - | 650 | 1130 |
| neg_i32 | 64 | 19 | ? ? ? ? | - | 310 | 614 |
| add_i32 | 64 | 19 | ? ? ? ? | - | 390 | 614 |
| chain_i32 | 64 | 35 | ? ? ? | - | 648 | 1415 |
| **geomean** | | | | | **433** | **733** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f32 | 128 | 15 | ? ? ? ? ? | - | 132 | 136 | 195 |
| chain_f32 | 128 | 23 | ? ? ? ? | - | 263 | 214 | 324 |
| neg_f64 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f64 | 128 | 15 | ? ? ? ? ? | - | 132 | 136 | 195 |
| chain_f64 | 128 | 23 | ? ? ? ? | - | 263 | 214 | 324 |
| neg_i32 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_i32 | 128 | 15 | ? ? ? ? ? | - | 132 | 125 | 195 |
| chain_i32 | 128 | 23 | ? ? ? ? | - | 263 | 236 | 323 |
| **geomean** | | | | | **166** | **144** | **203** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_f32 | 256 | 15 | ? ? ? ? | - | 77 | 63 | 97 |
| chain_f32 | 256 | 23 | ? ? ? ? | - | 195 | 108 | 163 |
| neg_f64 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_f64 | 256 | 15 | ? ? ? ? | - | 77 | 63 | 97 |
| chain_f64 | 256 | 23 | ? ? ? ? | - | 195 | 108 | 163 |
| neg_i32 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_i32 | 256 | 15 | ? ? ? ? | - | 76 | 63 | 97 |
| chain_i32 | 256 | 23 | ? ? ? ? | - | 139 | 144 | 162 |
| **geomean** | | | | | **90** | **73** | **101** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| neg_f32 | 64 | 15 | ? ? ? | - | 262 |
| add_f32 | 64 | 19 | ? ? ? | - | 264 |
| chain_f32 | 64 | 35 | ? ? ? | - | 524 |
| neg_f64 | 64 | 15 | ? ? ? | - | 262 |
| add_f64 | 64 | 19 | ? ? ? | - | 264 |
| chain_f64 | 64 | 35 | ? ? ? | - | 524 |
| neg_i32 | 64 | 19 | ? ? ? | - | 262 |
| add_i32 | 64 | 19 | ? ? ? | - | 262 |
| chain_i32 | 64 | 59 | ? ? ? | - | 1289 |
| **geomean** | | | | | **366** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 64 | 11 | ? ? ? | - | 646 | 645 | 262 |
| add_f32 | 64 | 13 | ? ? ? | - | 966 | 837 | 454 |
| chain_f32 | 64 | 25 | ? ? ? | - | 1734 | 1349 | 838 |
| neg_f64 | 64 | 11 | ? ? ? | - | 646 | 645 | 262 |
| add_f64 | 64 | 13 | ? ? ? | - | 966 | 837 | 454 |
| chain_f64 | 64 | 25 | ? ? ? | - | 1734 | 1349 | 838 |
| neg_i32 | 64 | 11 | ? ? ? | - | 646 | 645 | 262 |
| add_i32 | 64 | 13 | ? ? ? | - | 901 | 837 | 454 |
| chain_i32 | 64 | 25 | ? ? ? | - | 1544 | 1415 | 776 |
| **geomean** | | | | | **1006** | **905** | **460** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| neg_f32 | 64 | 10 | ? ? ? ? | - | 203 |
| add_f32 | 64 | 12 | ? ? ? ? | - | 266 |
| chain_f32 | 64 | 24 | ? ? ? ? | - | 459 |
| neg_f64 | 64 | 10 | ? ? ? | - | 203 |
| add_f64 | 64 | 12 | ? ? ? | - | 265 |
| chain_f64 | 64 | 24 | ? ? ? | - | 462 |
| neg_i32 | 64 | 10 | ? ? ? ? | - | 203 |
| add_i32 | 64 | 12 | ? ? ? ? | - | 264 |
| chain_i32 | 64 | 24 | ? ? ? ? | - | 458 |
| **geomean** | | | | | **291** |
