# llvm-mca summary: `pref0`

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`
- source: `probe` (`jix/probe`)
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

| kernel | x86_64 sandybridge | x86_64 skylake | x86_64 znver3 | x86_64-v2 sandybridge | x86_64-v2 btver2 | x86_64-v3 skylake | x86_64-v3 alderlake | x86_64-v3 znver3 | x86_64-v4 icelake-server | x86_64-v4 sapphirerapids | x86_64-v4 znver4 | i686 skylake | aarch64 cortex-a72 | aarch64 neoverse-n1 | aarch64 neoverse-v2 | aarch64-apple apple-m1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neg_f32 | 291 | 259 | 260 | 291 | 436 | 130 | 94 | 130 | 49 | 47 | 65 | 262 | 579 | 610 | 227 | 196 |
| add_f32 | 387 | 261 | 387 | 387 | 564 | 131 | 116 | 193 | 71 | 58 | 97 | 264 | 899 | 834 | 323 | 260 |
| chain_f32 | 644 | 517 | 644 | 644 | 1077 | 260 | 205 | 322 | 193 | 103 | 161 | 524 | 1666 | 1281 | 548 | 390 |
| longchain_f32 | 1319 | 1111 | 1318 | 1319 | 2278 | 516 | 373 | 578 | 449 | 227 | 321 | 1295 | 3202 | 2465 | 1060 | 753 |
| neg_f64 | 291 | 259 | 260 | 291 | 436 | 130 | 94 | 130 | 49 | 47 | 65 | 262 | 579 | 610 | 227 | 196 |
| add_f64 | 387 | 261 | 387 | 387 | 564 | 131 | 116 | 193 | 71 | 58 | 97 | 264 | 899 | 834 | 323 | 260 |
| chain_f64 | 644 | 517 | 644 | 644 | 1078 | 260 | 205 | 322 | 193 | 103 | 161 | 524 | 1666 | 1281 | 548 | 390 |
| longchain_f64 | 1319 | 1111 | 1318 | 1319 | 2310 | 516 | 373 | 578 | 449 | 227 | 321 | 1295 | 3202 | 2465 | 1060 | 754 |
| neg_i32 | 283 | 259 | 260 | 283 | 564 | 130 | 94 | 130 | 49 | 47 | 65 | 262 | 579 | 642 | 227 | 196 |
| add_i32 | 387 | 260 | 387 | 387 | 564 | 130 | 116 | 193 | 70 | 58 | 97 | 262 | 739 | 770 | 323 | 260 |
| chain_i32 | 1060 | 1286 | 775 | 644 | 1347 | 260 | 225 | 322 | 162 | 131 | 161 | 1290 | 1443 | 1442 | 580 | 391 |
| longchain_i32 | 2088 | 2568 | 1613 | 1316 | 2916 | 516 | 438 | 578 | 338 | 291 | 289 | 2577 | 2500 | 2498 | 1028 | 709 |
| narrow_i32 | 647 | 519 | 647 | 647 | 1128 | 262 | 233 | 324 | 163 | 140 | 196 | 526 | 1544 | 1349 | 648 | 456 |
| widen_i64 | 413 | 324 | 324 | 388 | 675 | 141 | 138 | 206 | 70 | 69 | 109 | 351 | 675 | 706 | 323 | 228 |
| narrow_f32 | 1037 | 781 | 650 | 1037 | 1385 | 328 | 320 | 446 | 259 | 197 | 386 | 792 | 1995 | 1479 | 650 | 460 |
| widen_f64 | 389 | 283 | 388 | 389 | 613 | 141 | 138 | 226 | 71 | 122 | 193 | 308 | 1090 | 834 | 323 | 230 |
| **geomean** | **594** | **494** | **537** | **557** | **927** | **214** | **178** | **268** | **125** | **100** | **147** | **513** | **1229** | **1111** | **456** | **342** |

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
| narrow_i32 | 64 | 35 | ? ? ? ? | - | 647 | 519 | 647 |
| widen_i64 | 128 | 51 | ? ? ? ? | - | 413 | 324 | 324 |
| narrow_f32 | 64 | 43 | ? ? ? | - | 1037 | 781 | 650 |
| widen_f64 | 128 | 35 | ? ? ? | - | 389 | 283 | 388 |
| **geomean** | | | | | **594** | **494** | **537** |

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
| narrow_i32 | 64 | 35 | ? ? ? ? | - | 647 | 1128 |
| widen_i64 | 128 | 35 | ? ? ? ? | - | 388 | 675 |
| narrow_f32 | 64 | 43 | ? ? ? | - | 1037 | 1385 |
| widen_f64 | 128 | 35 | ? ? ? | - | 389 | 613 |
| **geomean** | | | | | **557** | **927** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 256 | 19 | ? ? ? ? ? | - | 130 | 94 | 130 |
| add_f32 | 256 | 27 | ? ? ? ? ? | - | 131 | 116 | 193 |
| chain_f32 | 256 | 51 | ? ? ? ? | - | 260 | 205 | 322 |
| longchain_f32 | 256 | 99 | ? ? ? ? | - | 516 | 373 | 578 |
| neg_f64 | 256 | 19 | ? ? ? ? ? | - | 130 | 94 | 130 |
| add_f64 | 256 | 27 | ? ? ? ? ? | - | 131 | 116 | 193 |
| chain_f64 | 256 | 51 | ? ? ? ? | - | 260 | 205 | 322 |
| longchain_f64 | 256 | 99 | ? ? ? | - | 516 | 373 | 578 |
| neg_i32 | 256 | 19 | ? ? ? ? ? | - | 130 | 94 | 130 |
| add_i32 | 256 | 27 | ? ? ? ? ? | - | 130 | 116 | 193 |
| chain_i32 | 256 | 51 | ? ? ? ? | - | 260 | 225 | 322 |
| longchain_i32 | 256 | 99 | ? ? ? | - | 516 | 438 | 578 |
| narrow_i32 | 128 | 31 | ? ? ? ? ? | - | 262 | 233 | 324 |
| widen_i64 | 256 | 27 | ? ? ? ? ? | - | 141 | 138 | 206 |
| narrow_f32 | 128 | 35 | ? ? ? ? | - | 328 | 320 | 446 |
| widen_f64 | 256 | 27 | ? ? ? ? | - | 141 | 138 | 226 |
| **geomean** | | | | | **214** | **178** | **268** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 512 | 19 | ? ? ? ? | - | 49 | 47 | 65 |
| add_f32 | 512 | 27 | ? ? ? ? | - | 71 | 58 | 97 |
| chain_f32 | 512 | 51 | ? ? ? ? | - | 193 | 103 | 161 |
| longchain_f32 | 512 | 99 | ? ? ? ? | - | 449 | 227 | 321 |
| neg_f64 | 512 | 19 | ? ? ? ? | - | 49 | 47 | 65 |
| add_f64 | 512 | 27 | ? ? ? ? | - | 71 | 58 | 97 |
| chain_f64 | 512 | 51 | ? ? ? ? | - | 193 | 103 | 161 |
| longchain_f64 | 512 | 99 | ? ? ? ? | - | 449 | 227 | 321 |
| neg_i32 | 512 | 19 | ? ? ? ? | - | 49 | 47 | 65 |
| add_i32 | 512 | 27 | ? ? ? ? | - | 70 | 58 | 97 |
| chain_i32 | 512 | 51 | ? ? ? ? | - | 162 | 131 | 161 |
| longchain_i32 | 512 | 99 | ? ? ? ? | - | 338 | 291 | 289 |
| narrow_i32 | 256 | 27 | ? ? ? ? | - | 163 | 140 | 196 |
| widen_i64 | 512 | 27 | ? ? ? ? | - | 70 | 69 | 109 |
| narrow_f32 | 256 | 35 | ? ? ? ? | - | 259 | 197 | 386 |
| widen_f64 | 512 | 27 | ? ? ? ? | - | 71 | 122 | 193 |
| **geomean** | | | | | **125** | **100** | **147** |

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
| narrow_i32 | 32 | 19 | ? ? ? | - | 526 |
| widen_i64 | 64 | 27 | ? ? ? | - | 351 |
| narrow_f32 | 32 | 23 | ? ? ? | - | 792 |
| widen_f64 | 64 | 19 | ? ? ? | - | 308 |
| **geomean** | | | | | **513** |

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
| narrow_i32 | 64 | 26 | ? ? ? | - | 1544 | 1349 | 648 |
| widen_i64 | 128 | 23 | ? ? ? | - | 675 | 706 | 323 |
| narrow_f32 | 64 | 31 | ? ? ? | - | 1995 | 1479 | 650 |
| widen_f64 | 128 | 31 | ? ? ? | - | 1090 | 834 | 323 |
| **geomean** | | | | | **1229** | **1111** | **456** |

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
| narrow_i32 | 64 | 27 | ? ? ? ? | - | 456 |
| widen_i64 | 128 | 23 | ? ? ? ? | - | 228 |
| narrow_f32 | 64 | 31 | ? ? ? ? | - | 460 |
| widen_f64 | 128 | 31 | ? ? ? ? | - | 230 |
| **geomean** | | | | | **342** |
