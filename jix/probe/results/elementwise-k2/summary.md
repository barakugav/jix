# llvm-mca summary: `elementwise-k2` (elementwise kernels)

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
| x86_64 | sandybridge | **457** | 325 | 390 | 663 | 325 | 390 | 663 | 310 | 390 | 1014 |
| x86_64 | skylake | **369** | 262 | 265 | 538 | 262 | 265 | 538 | 262 | 263 | 1302 |
| x86_64 | znver3 | **410** | 264 | 390 | 659 | 264 | 390 | 659 | 264 | 390 | 687 |
| x86_64-v2 | sandybridge | **435** | 325 | 390 | 663 | 325 | 390 | 663 | 310 | 390 | 658 |
| x86_64-v2 | btver2 | **754** | 485 | 615 | 1231 | 485 | 615 | 1236 | 613 | 614 | 1549 |
| x86_64-v3 | skylake | **168** | 132 | 133 | 269 | 132 | 133 | 269 | 132 | 132 | 270 |
| x86_64-v3 | alderlake | **150** | 103 | 126 | 255 | 103 | 126 | 255 | 103 | 125 | 280 |
| x86_64-v3 | znver3 | **203** | 132 | 195 | 329 | 132 | 195 | 329 | 132 | 195 | 327 |
| x86_64-v4 | icelake-server | **91** | 55 | 77 | 198 | 55 | 77 | 198 | 55 | 76 | 150 |
| x86_64-v4 | sapphirerapids | **75** | 52 | 63 | 119 | 52 | 63 | 119 | 52 | 63 | 150 |
| x86_64-v4 | znver4 | **102** | 66 | 97 | 164 | 66 | 97 | 164 | 66 | 97 | 164 |
| i686 | skylake | **394** | 269 | 305 | 539 | 269 | 305 | 539 | 301 | 301 | 1300 |
| aarch64 | cortex-a72 | **1174** | 781 | 1166 | 1939 | 781 | 1166 | 1939 | 781 | 1037 | 1681 |
| aarch64 | neoverse-n1 | **1087** | 780 | 1036 | 1550 | 780 | 1036 | 1550 | 780 | 1036 | 1679 |
| aarch64 | neoverse-v2 | **414** | 270 | 398 | 658 | 270 | 398 | 658 | 270 | 398 | 658 |
| aarch64-apple | apple-m1 | **365** | 270 | 337 | 535 | 270 | 338 | 541 | 270 | 334 | 531 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? | - | 325 | 262 | 264 |
| add_f32 | 64 | 19 | ? ? ? ? | - | 390 | 265 | 390 |
| chain_f32 | 32 | 19 | ? ? ? | - | 663 | 538 | 659 |
| neg_f64 | 64 | 15 | ? ? ? ? | - | 325 | 262 | 264 |
| add_f64 | 64 | 19 | ? ? ? ? | - | 390 | 265 | 390 |
| chain_f64 | 32 | 19 | ? ? ? | - | 663 | 538 | 659 |
| neg_i32 | 64 | 19 | ? ? ? ? | - | 310 | 262 | 264 |
| add_i32 | 64 | 19 | ? ? ? ? | - | 390 | 263 | 390 |
| chain_i32 | 32 | 31 | ? ? ? | - | 1014 | 1302 | 687 |
| **geomean** | | | | | **457** | **369** | **410** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| neg_f32 | 64 | 15 | ? ? ? ? | - | 325 | 485 |
| add_f32 | 64 | 19 | ? ? ? ? | - | 390 | 615 |
| chain_f32 | 32 | 19 | ? ? ? | - | 663 | 1231 |
| neg_f64 | 64 | 15 | ? ? ? ? | - | 325 | 485 |
| add_f64 | 64 | 19 | ? ? ? ? | - | 390 | 615 |
| chain_f64 | 32 | 19 | ? ? ? | - | 663 | 1236 |
| neg_i32 | 64 | 19 | ? ? ? ? | - | 310 | 613 |
| add_i32 | 64 | 19 | ? ? ? ? | - | 390 | 614 |
| chain_i32 | 32 | 19 | ? ? ? | - | 658 | 1549 |
| **geomean** | | | | | **435** | **754** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f32 | 128 | 15 | ? ? ? ? ? | - | 133 | 126 | 195 |
| chain_f32 | 64 | 13 | ? ? ? ? | - | 269 | 255 | 329 |
| neg_f64 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_f64 | 128 | 15 | ? ? ? ? ? | - | 133 | 126 | 195 |
| chain_f64 | 64 | 13 | ? ? ? ? | - | 269 | 255 | 329 |
| neg_i32 | 128 | 11 | ? ? ? ? ? | - | 132 | 103 | 132 |
| add_i32 | 128 | 15 | ? ? ? ? ? | - | 132 | 125 | 195 |
| chain_i32 | 64 | 13 | ? ? ? ? | - | 270 | 280 | 327 |
| **geomean** | | | | | **168** | **150** | **203** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_f32 | 256 | 15 | ? ? ? ? | - | 77 | 63 | 97 |
| chain_f32 | 128 | 13 | ? ? ? ? | - | 198 | 119 | 164 |
| neg_f64 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_f64 | 256 | 15 | ? ? ? ? | - | 77 | 63 | 97 |
| chain_f64 | 128 | 13 | ? ? ? ? | - | 198 | 119 | 164 |
| neg_i32 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_i32 | 256 | 15 | ? ? ? ? | - | 76 | 63 | 97 |
| chain_i32 | 128 | 13 | ? ? ? ? | - | 150 | 150 | 164 |
| **geomean** | | | | | **91** | **75** | **102** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| neg_f32 | 32 | 9 | ? ? ? | - | 269 |
| add_f32 | 32 | 11 | ? ? ? | - | 305 |
| chain_f32 | 32 | 19 | ? ? ? | - | 539 |
| neg_f64 | 32 | 9 | ? ? ? | - | 269 |
| add_f64 | 32 | 11 | ? ? ? | - | 305 |
| chain_f64 | 32 | 19 | ? ? ? | - | 539 |
| neg_i32 | 32 | 11 | ? ? ? | - | 301 |
| add_i32 | 32 | 11 | ? ? ? | - | 301 |
| chain_i32 | 32 | 31 | ? ? ? | - | 1300 |
| **geomean** | | | | | **394** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 32 | 8 | ? ? ? | - | 781 | 780 | 270 |
| add_f32 | 32 | 10 | ? ? ? | - | 1166 | 1036 | 398 |
| chain_f32 | 32 | 18 | ? ? ? | - | 1939 | 1550 | 658 |
| neg_f64 | 32 | 8 | ? ? ? | - | 781 | 780 | 270 |
| add_f64 | 32 | 10 | ? ? ? | - | 1166 | 1036 | 398 |
| chain_f64 | 32 | 18 | ? ? ? | - | 1939 | 1550 | 658 |
| neg_i32 | 32 | 8 | ? ? ? | - | 781 | 780 | 270 |
| add_i32 | 32 | 10 | ? ? ? | - | 1037 | 1036 | 398 |
| chain_i32 | 32 | 18 | ? ? ? | - | 1681 | 1679 | 658 |
| **geomean** | | | | | **1174** | **1087** | **414** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| neg_f32 | 32 | 9 | ? ? ? ? | - | 270 |
| add_f32 | 32 | 11 | ? ? ? ? | - | 337 |
| chain_f32 | 32 | 19 | ? ? ? ? | - | 535 |
| neg_f64 | 32 | 9 | ? ? ? | - | 270 |
| add_f64 | 32 | 11 | ? ? ? | - | 338 |
| chain_f64 | 32 | 19 | ? ? ? | - | 541 |
| neg_i32 | 32 | 9 | ? ? ? ? | - | 270 |
| add_i32 | 32 | 11 | ? ? ? ? | - | 334 |
| chain_i32 | 32 | 19 | ? ? ? ? | - | 531 |
| **geomean** | | | | | **365** |
