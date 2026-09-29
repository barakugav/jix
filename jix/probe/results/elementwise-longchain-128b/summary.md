# llvm-mca summary: `elementwise-longchain-128b` (elementwise kernels)

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
| x86_64 | sandybridge | **601** | 291 | 387 | 644 | 1319 | 291 | 387 | 644 | 1319 | 283 | 387 | 1060 | 2088 |
| x86_64 | skylake | **514** | 259 | 261 | 517 | 1111 | 259 | 261 | 517 | 1111 | 259 | 260 | 1286 | 2568 |
| x86_64 | znver3 | **558** | 260 | 387 | 644 | 1318 | 260 | 387 | 644 | 1318 | 260 | 387 | 775 | 1613 |
| x86_64-v2 | sandybridge | **555** | 291 | 387 | 644 | 1319 | 291 | 387 | 644 | 1319 | 283 | 387 | 644 | 1316 |
| x86_64-v2 | btver2 | **937** | 436 | 564 | 1077 | 2278 | 436 | 564 | 1078 | 2310 | 564 | 564 | 1347 | 2916 |
| x86_64-v3 | skylake | **221** | 132 | 132 | 262 | 520 | 132 | 132 | 262 | 520 | 132 | 132 | 263 | 520 |
| x86_64-v3 | alderlake | **188** | 103 | 136 | 218 | 395 | 103 | 136 | 218 | 395 | 103 | 125 | 235 | 444 |
| x86_64-v3 | znver3 | **264** | 132 | 195 | 324 | 581 | 132 | 195 | 324 | 581 | 132 | 195 | 324 | 581 |
| x86_64-v4 | icelake-server | **134** | 55 | 77 | 197 | 453 | 55 | 77 | 197 | 453 | 55 | 76 | 166 | 326 |
| x86_64-v4 | sapphirerapids | **101** | 52 | 63 | 122 | 235 | 52 | 63 | 122 | 235 | 52 | 63 | 139 | 299 |
| x86_64-v4 | znver4 | **135** | 66 | 97 | 164 | 325 | 66 | 97 | 164 | 325 | 66 | 97 | 164 | 292 |
| i686 | skylake | **579** | 292 | 293 | 582 | 1416 | 292 | 293 | 582 | 1416 | 291 | 293 | 1289 | 2604 |
| aarch64 | cortex-a72 | **1229** | 579 | 899 | 1666 | 3202 | 579 | 899 | 1666 | 3202 | 579 | 739 | 1443 | 2500 |
| aarch64 | neoverse-n1 | **1135** | 610 | 834 | 1281 | 2465 | 610 | 834 | 1281 | 2465 | 642 | 770 | 1442 | 2498 |
| aarch64 | neoverse-v2 | **455** | 227 | 323 | 548 | 1060 | 227 | 323 | 548 | 1060 | 227 | 323 | 580 | 1028 |
| aarch64-apple | apple-m1 | **348** | 196 | 260 | 390 | 753 | 196 | 260 | 390 | 754 | 196 | 260 | 391 | 709 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 27 | ? ? ? ? | - | 291 | 259 | 260 |
| add_f32 | 128 | 35 | ? ? ? ? | - | 387 | 261 | 387 |
| chain_f32 | 128 | 67 | ? ? ? | - | 644 | 517 | 644 |
| longchain_f32 | 128 | 137 | ? ? ? | - | 1319 | 1111 | 1318 |
| neg_f64 | 128 | 27 | ? ? ? ? | - | 291 | 259 | 260 |
| add_f64 | 128 | 35 | ? ? ? ? | - | 387 | 261 | 387 |
| chain_f64 | 128 | 67 | ? ? ? | - | 644 | 517 | 644 |
| longchain_f64 | 128 | 137 | ? ? ? | - | 1319 | 1111 | 1318 |
| neg_i32 | 128 | 35 | ? ? ? ? | - | 283 | 259 | 260 |
| add_i32 | 128 | 35 | ? ? ? ? | - | 387 | 260 | 387 |
| chain_i32 | 128 | 117 | ? ? ? | - | 1060 | 1286 | 775 |
| longchain_i32 | 128 | 241 | ? ? ? | - | 2088 | 2568 | 1613 |
| **geomean** | | | | | **601** | **514** | **558** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| neg_f32 | 128 | 27 | ? ? ? ? | - | 291 | 436 |
| add_f32 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| chain_f32 | 128 | 67 | ? ? ? | - | 644 | 1077 |
| longchain_f32 | 128 | 137 | ? ? ? | - | 1319 | 2278 |
| neg_f64 | 128 | 27 | ? ? ? ? | - | 291 | 436 |
| add_f64 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| chain_f64 | 128 | 67 | ? ? ? | - | 644 | 1078 |
| longchain_f64 | 128 | 137 | ? ? ? | - | 1319 | 2310 |
| neg_i32 | 128 | 35 | ? ? ? ? | - | 283 | 564 |
| add_i32 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| chain_i32 | 128 | 67 | ? ? ? | - | 644 | 1347 |
| longchain_i32 | 128 | 137 | ? ? ? | - | 1316 | 2916 |
| **geomean** | | | | | **555** | **937** |

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
| chain_f32 | 128 | 15 | ? ? ? ? | - | 197 | 122 | 164 |
| longchain_f32 | 128 | 27 | ? ? ? ? | - | 453 | 235 | 325 |
| neg_f64 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_f64 | 256 | 15 | ? ? ? ? | - | 77 | 63 | 97 |
| chain_f64 | 128 | 15 | ? ? ? ? | - | 197 | 122 | 164 |
| longchain_f64 | 128 | 27 | ? ? ? ? | - | 453 | 235 | 325 |
| neg_i32 | 256 | 11 | ? ? ? ? | - | 55 | 52 | 66 |
| add_i32 | 256 | 15 | ? ? ? ? | - | 76 | 63 | 97 |
| chain_i32 | 128 | 15 | ? ? ? ? | - | 166 | 139 | 164 |
| longchain_i32 | 128 | 27 | ? ? ? ? | - | 326 | 299 | 292 |
| **geomean** | | | | | **134** | **101** | **135** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| neg_f32 | 128 | 30 | ? ? ? | - | 292 |
| add_f32 | 128 | 36 | ? ? ? | - | 293 |
| chain_f32 | 128 | 72 | ? ? ? | - | 582 |
| longchain_f32 | 128 | 152 | ? ? ? | - | 1416 |
| neg_f64 | 128 | 30 | ? ? ? | - | 292 |
| add_f64 | 128 | 36 | ? ? ? | - | 293 |
| chain_f64 | 128 | 72 | ? ? ? | - | 582 |
| longchain_f64 | 128 | 152 | ? ? ? | - | 1416 |
| neg_i32 | 128 | 36 | ? ? ? | - | 291 |
| add_i32 | 128 | 36 | ? ? ? | - | 293 |
| chain_i32 | 128 | 139 | ? ? ? | - | 1289 |
| longchain_i32 | 128 | 288 | ? ? ? | - | 2604 |
| **geomean** | | | | | **579** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| add_f32 | 128 | 26 | ? ? ? | - | 899 | 834 | 323 |
| chain_f32 | 128 | 52 | ? ? ? | - | 1666 | 1281 | 548 |
| longchain_f32 | 128 | 104 | ? ? ? | - | 3202 | 2465 | 1060 |
| neg_f64 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| add_f64 | 128 | 26 | ? ? ? | - | 899 | 834 | 323 |
| chain_f64 | 128 | 52 | ? ? ? | - | 1666 | 1281 | 548 |
| longchain_f64 | 128 | 104 | ? ? ? | - | 3202 | 2465 | 1060 |
| neg_i32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| add_i32 | 128 | 26 | ? ? ? | - | 739 | 770 | 323 |
| chain_i32 | 128 | 52 | ? ? ? | - | 1443 | 1442 | 580 |
| longchain_i32 | 128 | 96 | ? ? ? | - | 2500 | 2498 | 1028 |
| **geomean** | | | | | **1229** | **1135** | **455** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| neg_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| add_f32 | 128 | 26 | ? ? ? ? | - | 260 |
| chain_f32 | 128 | 52 | ? ? ? ? | - | 390 |
| longchain_f32 | 128 | 104 | ? ? ? | - | 753 |
| neg_f64 | 128 | 21 | ? ? ? | - | 196 |
| add_f64 | 128 | 26 | ? ? ? | - | 260 |
| chain_f64 | 128 | 52 | ? ? ? | - | 390 |
| longchain_f64 | 128 | 104 | ? ? ? | - | 754 |
| neg_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| add_i32 | 128 | 26 | ? ? ? ? | - | 260 |
| chain_i32 | 128 | 52 | ? ? ? ? | - | 391 |
| longchain_i32 | 128 | 96 | ? ? ? ? | - | 709 |
| **geomean** | | | | | **348** |
