# llvm-mca summary: `elementwise-fearless` (elementwise kernels)

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

| platform | cpu | geomean | vs elementwise-baseline | neg_f32 | add_f32 | chain_f32 | neg_f64 | add_f64 | chain_f64 | neg_i32 | add_i32 | chain_i32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x86_64 | sandybridge | **440** | 0.99x | 291 | 387 | 644 | 291 | 387 | 644 | 283 | 387 | 1074 |
| x86_64 | skylake | **362** | 1.00x | 259 | 261 | 517 | 259 | 261 | 517 | 259 | 260 | 1287 |
| x86_64 | znver3 | **405** | 0.99x | 260 | 387 | 644 | 260 | 387 | 644 | 260 | 387 | 694 |
| x86_64-v2 | sandybridge | **416** | 1.00x | 291 | 387 | 644 | 291 | 387 | 644 | 283 | 387 | 644 |
| x86_64-v2 | btver2 | **677** | 1.00x | 436 | 564 | 1076 | 436 | 564 | 1079 | 564 | 564 | 1348 |
| x86_64-v3 | skylake | **166** | 1.01x | 132 | 132 | 263 | 132 | 132 | 263 | 132 | 132 | 263 |
| x86_64-v3 | alderlake | **144** | 1.05x | 103 | 136 | 214 | 103 | 136 | 214 | 103 | 125 | 236 |
| x86_64-v3 | znver3 | **203** | 1.00x | 132 | 195 | 324 | 132 | 195 | 324 | 132 | 195 | 323 |
| x86_64-v4 | icelake-server | **91** | 12.32x | 55 | 77 | 198 | 55 | 77 | 198 | 55 | 76 | 150 |
| x86_64-v4 | sapphirerapids | **75** | 40.95x | 52 | 63 | 119 | 52 | 63 | 119 | 52 | 63 | 150 |
| x86_64-v4 | znver4 | **102** | 2.22x | 66 | 97 | 164 | 66 | 97 | 164 | 66 | 97 | 164 |
| i686 | skylake | **404** | 0.90x | 292 | 293 | 597 | 292 | 293 | 597 | 291 | 293 | 1286 |
| aarch64 | cortex-a72 | **916** | 1.01x | 579 | 899 | 1667 | 579 | 899 | 1667 | 579 | 739 | 1411 |
| aarch64 | neoverse-n1 | **878** | 1.00x | 610 | 834 | 1314 | 610 | 834 | 1314 | 642 | 770 | 1411 |
| aarch64 | neoverse-v2 | **351** | 0.97x | 227 | 323 | 611 | 227 | 323 | 611 | 227 | 323 | 548 |
| aarch64-apple | apple-m1 | **271** | 1.02x | 196 | 260 | 389 | 196 | 260 | 391 | 196 | 260 | 390 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 27 | ? ? ? ? | - | 291 | 259 | 260 |
| add_f32 | 128 | 35 | ? ? ? ? | - | 387 | 261 | 387 |
| chain_f32 | 128 | 67 | ? ? ? | - | 644 | 517 | 644 |
| neg_f64 | 128 | 27 | ? ? ? ? | - | 291 | 259 | 260 |
| add_f64 | 128 | 35 | ? ? ? ? | - | 387 | 261 | 387 |
| chain_f64 | 128 | 67 | ? ? ? | - | 644 | 517 | 644 |
| neg_i32 | 128 | 35 | ? ? ? ? | - | 283 | 259 | 260 |
| add_i32 | 128 | 35 | ? ? ? ? | - | 387 | 260 | 387 |
| chain_i32 | 128 | 115 | ? ? ? | - | 1074 | 1287 | 694 |
| **geomean** | | | | | **440** | **362** | **405** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| neg_f32 | 128 | 27 | ? ? ? ? | - | 291 | 436 |
| add_f32 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| chain_f32 | 128 | 67 | ? ? ? | - | 644 | 1076 |
| neg_f64 | 128 | 27 | ? ? ? ? | - | 291 | 436 |
| add_f64 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| chain_f64 | 128 | 67 | ? ? ? | - | 644 | 1079 |
| neg_i32 | 128 | 35 | ? ? ? ? | - | 283 | 564 |
| add_i32 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| chain_i32 | 128 | 67 | ? ? ? | - | 644 | 1348 |
| **geomean** | | | | | **416** | **677** |

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
| neg_f32 | 128 | 30 | ? ? ? | - | 292 |
| add_f32 | 128 | 36 | ? ? ? | - | 293 |
| chain_f32 | 128 | 75 | ? ? ? | - | 597 |
| neg_f64 | 128 | 30 | ? ? ? | - | 292 |
| add_f64 | 128 | 36 | ? ? ? | - | 293 |
| chain_f64 | 128 | 75 | ? ? ? | - | 597 |
| neg_i32 | 128 | 36 | ? ? ? | - | 291 |
| add_i32 | 128 | 36 | ? ? ? | - | 293 |
| chain_i32 | 128 | 135 | ? ? ? | - | 1286 |
| **geomean** | | | | | **404** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| add_f32 | 128 | 26 | ? ? ? | - | 899 | 834 | 323 |
| chain_f32 | 128 | 51 | ? ? ? | - | 1667 | 1314 | 611 |
| neg_f64 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| add_f64 | 128 | 26 | ? ? ? | - | 899 | 834 | 323 |
| chain_f64 | 128 | 51 | ? ? ? | - | 1667 | 1314 | 611 |
| neg_i32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| add_i32 | 128 | 26 | ? ? ? | - | 739 | 770 | 323 |
| chain_i32 | 128 | 52 | ? ? ? | - | 1411 | 1411 | 548 |
| **geomean** | | | | | **916** | **878** | **351** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| neg_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| add_f32 | 128 | 26 | ? ? ? ? | - | 260 |
| chain_f32 | 128 | 52 | ? ? ? ? | - | 389 |
| neg_f64 | 128 | 21 | ? ? ? | - | 196 |
| add_f64 | 128 | 26 | ? ? ? | - | 260 |
| chain_f64 | 128 | 52 | ? ? ? | - | 391 |
| neg_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| add_i32 | 128 | 26 | ? ? ? ? | - | 260 |
| chain_i32 | 128 | 52 | ? ? ? ? | - | 390 |
| **geomean** | | | | | **271** |
