# llvm-mca summary: `elementwise-k8` (elementwise kernels)

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
| x86_64 | sandybridge | **440** | 291 | 387 | 644 | 291 | 387 | 644 | 283 | 387 | 1074 |
| x86_64 | skylake | **362** | 259 | 261 | 517 | 259 | 261 | 517 | 259 | 260 | 1287 |
| x86_64 | znver3 | **405** | 260 | 387 | 644 | 260 | 387 | 644 | 260 | 387 | 694 |
| x86_64-v2 | sandybridge | **416** | 291 | 387 | 644 | 291 | 387 | 644 | 283 | 387 | 644 |
| x86_64-v2 | btver2 | **677** | 436 | 564 | 1076 | 436 | 564 | 1079 | 564 | 564 | 1348 |
| x86_64-v3 | skylake | **164** | 130 | 131 | 259 | 130 | 131 | 259 | 130 | 130 | 260 |
| x86_64-v3 | alderlake | **132** | 94 | 116 | 204 | 94 | 116 | 204 | 94 | 116 | 230 |
| x86_64-v3 | znver3 | **201** | 130 | 193 | 322 | 130 | 193 | 322 | 130 | 193 | 322 |
| x86_64-v4 | icelake-server | **85** | 49 | 71 | 193 | 49 | 71 | 193 | 49 | 70 | 145 |
| x86_64-v4 | sapphirerapids | **68** | 47 | 58 | 102 | 47 | 58 | 102 | 47 | 58 | 139 |
| x86_64-v4 | znver4 | **100** | 65 | 97 | 161 | 65 | 97 | 161 | 65 | 97 | 161 |
| i686 | skylake | **404** | 292 | 293 | 597 | 292 | 293 | 597 | 291 | 293 | 1286 |
| aarch64 | cortex-a72 | **916** | 579 | 899 | 1667 | 579 | 899 | 1667 | 579 | 739 | 1411 |
| aarch64 | neoverse-n1 | **878** | 610 | 834 | 1314 | 610 | 834 | 1314 | 642 | 770 | 1411 |
| aarch64 | neoverse-v2 | **351** | 227 | 323 | 611 | 227 | 323 | 611 | 227 | 323 | 548 |
| aarch64-apple | apple-m1 | **271** | 196 | 260 | 389 | 196 | 260 | 391 | 196 | 260 | 390 |

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
| neg_f32 | 256 | 19 | ? ? ? ? ? | - | 130 | 94 | 130 |
| add_f32 | 256 | 27 | ? ? ? ? ? | - | 131 | 116 | 193 |
| chain_f32 | 256 | 43 | ? ? ? ? | - | 259 | 204 | 322 |
| neg_f64 | 256 | 19 | ? ? ? ? ? | - | 130 | 94 | 130 |
| add_f64 | 256 | 27 | ? ? ? ? ? | - | 131 | 116 | 193 |
| chain_f64 | 256 | 43 | ? ? ? ? | - | 259 | 204 | 322 |
| neg_i32 | 256 | 19 | ? ? ? ? ? | - | 130 | 94 | 130 |
| add_i32 | 256 | 27 | ? ? ? ? ? | - | 130 | 116 | 193 |
| chain_i32 | 256 | 43 | ? ? ? ? | - | 260 | 230 | 322 |
| **geomean** | | | | | **164** | **132** | **201** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 512 | 19 | ? ? ? ? | - | 49 | 47 | 65 |
| add_f32 | 512 | 27 | ? ? ? ? | - | 71 | 58 | 97 |
| chain_f32 | 512 | 43 | ? ? ? ? | - | 193 | 102 | 161 |
| neg_f64 | 512 | 19 | ? ? ? ? | - | 49 | 47 | 65 |
| add_f64 | 512 | 27 | ? ? ? ? | - | 71 | 58 | 97 |
| chain_f64 | 512 | 43 | ? ? ? ? | - | 193 | 102 | 161 |
| neg_i32 | 512 | 19 | ? ? ? ? | - | 49 | 47 | 65 |
| add_i32 | 512 | 27 | ? ? ? ? | - | 70 | 58 | 97 |
| chain_i32 | 512 | 43 | ? ? ? ? | - | 145 | 139 | 161 |
| **geomean** | | | | | **85** | **68** | **100** |

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
