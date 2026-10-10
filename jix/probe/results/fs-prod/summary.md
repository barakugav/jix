# llvm-mca summary: `fs-prod`

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

| kernel | x86_64 sandybridge | x86_64 skylake | x86_64 znver3 | x86_64-v2 sandybridge | x86_64-v2 btver2 | x86_64-v3 skylake | x86_64-v3 alderlake | x86_64-v3 znver3 | i686 skylake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reduce_product_complex128_cell | 676 | 477 | 302 | 676 | 1031 | 525 | 515 | 450 | 648 |
| reduce_product_complex128_finalize | 268 | 268 | 270 | 268 | 458 | 131 | 82 | 132 | 268 |
| reduce_product_complex128_init | 1041 | 538 | 413 | 1041 | 1236 | 582 | 522 | 455 | 545 |
| reduce_product_complex128_update | 1427 | 792 | 561 | 1427 | 1690 | 844 | 821 | 721 | 792 |
| reduce_product_complex64_cell | 3236 | 2054 | 1158 | 2149 | 2804 | 770 | 537 | 499 | 2057 |
| reduce_product_complex64_finalize | 268 | 268 | 270 | 268 | 458 | 131 | 82 | 132 | 268 |
| reduce_product_complex64_init | 1041 | 538 | 413 | 1572 | 1828 | 518 | 395 | 296 | 561 |
| reduce_product_complex64_update | 1427 | 792 | 561 | 1427 | 1618 | 781 | 596 | 528 | 792 |
| reduce_product_f16_cell | 217094 | 165992 | 122955 | 217094 | 326117 | 16390 | 10266 | 7557 | 189355* |
| reduce_product_f16_finalize | 268 | 268 | 270 | 268 | 458 | 131 | 82 | 132 | 268 |
| reduce_product_f16_update | 215245 | 170312 | 143524 | 215245 | 323809 | 16794 | 11284 | 7680 | 182661 |
| reduce_product_f32_cell | 260 | 132 | 132 | 260 | 307 | 66 | 98 | 66 | 244 |
| reduce_product_f32_finalize | 268 | 268 | 270 | 268 | 458 | 131 | 82 | 132 | 1106 |
| reduce_product_f32_update | 401 | 306 | 398 | 401 | 717 | 133 | 126 | 195 | 305 |
| reduce_product_f64_cell | 260 | 132 | 132 | 260 | 516 | 66 | 98 | 66 | 244 |
| reduce_product_f64_finalize | 268 | 268 | 270 | 268 | 458 | 131 | 82 | 132 | 553 |
| reduce_product_f64_update | 401 | 306 | 398 | 401 | 722 | 133 | 126 | 195 | 305 |
| reduce_product_i16_cell | 2061 | 2061 | 2061 | 2061 | 8204 | 1588 | 1548 | 1739 | 6307 |
| reduce_product_i16_update | 3118 | 2104 | 2775 | 6226 | 7501 | 1622 | 1576 | 1684 | 4562 |
| reduce_product_i32_cell | 1030 | 1030 | 1030 | 1030 | 4102 | 796 | 774 | 869 | 1772 |
| reduce_product_i32_update | 1559 | 1052 | 1388 | 3113 | 3750 | 812 | 788 | 842 | 1946 |
| reduce_product_i64_cell | 515 | 515 | 515 | 515 | 2051 | 350 | 344 | 258 | 716 |
| reduce_product_i64_update | 1555 | 874 | 643 | 1555 | 1871 | 405 | 409 | 273 | 874 |
| reduce_product_i8_cell | 4122 | 4122 | 4122 | 4122 | 16407 | 3177 | 3096 | 3478 | 12365 |
| reduce_product_i8_finalize | 2140 | 2140 | 2161 | 2140 | 3666 | 1050 | 653 | 1052 | 2140 |
| reduce_product_i8_update | 6236 | 4209 | 5550 | 12452 | 15002 | 3244 | 3151 | 3369 | 10455 |
| reduce_product_u16_cell | 2162 | 2063 | 2063 | 2162 | 8207 | 1036 | 1033 | 1353 | 5775 |
| reduce_product_u16_update | 3118 | 2104 | 2775 | 4178 | 5443 | 1050 | 1064 | 1354 | 2811 |
| reduce_product_u32_cell | 1081 | 1032 | 1032 | 1081 | 4104 | 518 | 516 | 676 | 1160 |
| reduce_product_u32_update | 2086 | 1239 | 1024 | 2089 | 2721 | 525 | 532 | 677 | 1234 |
| reduce_product_u8_cell | 4324 | 4127 | 4127 | 4324 | 16415 | 2071 | 2066 | 2706 | 11812 |
| reduce_product_u8_update | 6236 | 4209 | 5550 | 8356 | 10885 | 2099 | 2127 | 2708 | 8356 |
| reduce_sum_f16_cell | 212998 | 165992 | 122955 | 212998 | 328165 | 16390 | 10257 | 8294 | 189355* |
| reduce_product_f16_init | - | - | - | - | - | 12544 | 8745 | 6410 | - |
| **geomean** | **1763** | **1336** | **1275** | **1911** | **3385** | **745** | **642** | **685** | **1958*** |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 477 | 302 |
| reduce_product_complex128_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 268 | 270 |
| reduce_product_complex128_init | 32 | 18 | 0/0 | ? ? | - | 1041 | 538 | 413 |
| reduce_product_complex128_update | 32 | 25 | 0/0 | ? ? | - | 1427 | 792 | 561 |
| reduce_product_complex64_cell | 128 | 190 | 31/0 | ? ? | - | 3236 | 2054 | 1158 |
| reduce_product_complex64_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 268 | 270 |
| reduce_product_complex64_init | 32 | 18 | 0/0 | ? ? | - | 1041 | 538 | 413 |
| reduce_product_complex64_update | 32 | 25 | 0/0 | ? ? | - | 1427 | 792 | 561 |
| reduce_product_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 217094 | 165992 | 122955 |
| reduce_product_f16_finalize | 32 | 7 | 0/0 | ? ? ? ? | - | 268 | 268 | 270 |
| reduce_product_f16_update | 2 | 266 | 0/0 | ? | - | 215245 | 170312 | 143524 |
| reduce_product_f32_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_product_f32_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 268 | 270 |
| reduce_product_f32_update | 32 | 11 | 0/0 | ? ? | - | 401 | 306 | 398 |
| reduce_product_f64_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_product_f64_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 268 | 270 |
| reduce_product_f64_update | 32 | 11 | 0/0 | ? ? | - | 401 | 306 | 398 |
| reduce_product_i16_cell | 32 | 47 | 0/10 | ? ? | - | 2061 | 2061 | 2061 |
| reduce_product_i16_update | 8 | 15 | 0/0 | ? ? | - | 3118 | 2104 | 2775 |
| reduce_product_i32_cell | 64 | 47 | 0/10 | ? ? | - | 1030 | 1030 | 1030 |
| reduce_product_i32_update | 16 | 15 | 0/0 | ? ? | - | 1559 | 1052 | 1388 |
| reduce_product_i64_cell | 128 | 29 | 0/10 | ? ? | - | 515 | 515 | 515 |
| reduce_product_i64_update | 32 | 29 | 0/0 | ? ? | - | 1555 | 874 | 643 |
| reduce_product_i8_cell | 16 | 47 | 0/10 | ? ? | - | 4122 | 4122 | 4122 |
| reduce_product_i8_finalize | 4 | 7 | 0/0 | ? ? ? | - | 2140 | 2140 | 2161 |
| reduce_product_i8_update | 4 | 15 | 0/0 | ? ? | - | 6236 | 4209 | 5550 |
| reduce_product_u16_cell | 32 | 67 | 0/10 | ? ? | - | 2162 | 2063 | 2063 |
| reduce_product_u16_update | 8 | 15 | 0/0 | ? ? | - | 3118 | 2104 | 2775 |
| reduce_product_u32_cell | 64 | 67 | 0/10 | ? ? | - | 1081 | 1032 | 1032 |
| reduce_product_u32_update | 16 | 23 | 0/0 | ? ? | - | 2086 | 1239 | 1024 |
| reduce_product_u8_cell | 16 | 67 | 0/10 | ? ? | - | 4324 | 4127 | 4127 |
| reduce_product_u8_update | 4 | 15 | 0/0 | ? ? | - | 6236 | 4209 | 5550 |
| reduce_sum_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 212998 | 165992 | 122955 |
| **geomean** | | | | | | **1763** | **1336** | **1275** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | btver2 |
|---|---:|---:|---:|---|---|---:|---:|
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 1031 |
| reduce_product_complex128_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 458 |
| reduce_product_complex128_init | 32 | 18 | 0/0 | ? ? | - | 1041 | 1236 |
| reduce_product_complex128_update | 32 | 25 | 0/0 | ? ? | - | 1427 | 1690 |
| reduce_product_complex64_cell | 128 | 175 | 34/0 | ? ? | - | 2149 | 2804 |
| reduce_product_complex64_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 458 |
| reduce_product_complex64_init | 16 | 14 | 0/0 | ? ? | - | 1572 | 1828 |
| reduce_product_complex64_update | 32 | 25 | 0/0 | ? ? | - | 1427 | 1618 |
| reduce_product_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 217094 | 326117 |
| reduce_product_f16_finalize | 32 | 7 | 0/0 | ? ? ? ? | - | 268 | 458 |
| reduce_product_f16_update | 2 | 266 | 0/0 | ? | - | 215245 | 323809 |
| reduce_product_f32_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 307 |
| reduce_product_f32_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 458 |
| reduce_product_f32_update | 32 | 11 | 0/0 | ? ? | - | 401 | 717 |
| reduce_product_f64_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 516 |
| reduce_product_f64_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 | 458 |
| reduce_product_f64_update | 32 | 11 | 0/0 | ? ? | - | 401 | 722 |
| reduce_product_i16_cell | 32 | 47 | 0/10 | ? ? | - | 2061 | 8204 |
| reduce_product_i16_update | 8 | 29 | 0/0 | ? ? | - | 6226 | 7501 |
| reduce_product_i32_cell | 64 | 47 | 0/10 | ? ? | - | 1030 | 4102 |
| reduce_product_i32_update | 16 | 29 | 0/0 | ? ? | - | 3113 | 3750 |
| reduce_product_i64_cell | 128 | 29 | 0/10 | ? ? | - | 515 | 2051 |
| reduce_product_i64_update | 32 | 29 | 0/0 | ? ? | - | 1555 | 1871 |
| reduce_product_i8_cell | 16 | 47 | 0/10 | ? ? | - | 4122 | 16407 |
| reduce_product_i8_finalize | 4 | 7 | 0/0 | ? ? ? | - | 2140 | 3666 |
| reduce_product_i8_update | 4 | 29 | 0/0 | ? ? | - | 12452 | 15002 |
| reduce_product_u16_cell | 32 | 67 | 0/10 | ? ? | - | 2162 | 8207 |
| reduce_product_u16_update | 8 | 21 | 0/0 | ? ? | - | 4178 | 5443 |
| reduce_product_u32_cell | 64 | 67 | 0/10 | ? ? | - | 1081 | 4104 |
| reduce_product_u32_update | 16 | 21 | 0/0 | ? ? | - | 2089 | 2721 |
| reduce_product_u8_cell | 16 | 67 | 0/10 | ? ? | - | 4324 | 16415 |
| reduce_product_u8_update | 4 | 21 | 0/0 | ? ? | - | 8356 | 10885 |
| reduce_sum_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 212998 | 328165 |
| **geomean** | | | | | | **1911** | **3385** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | stack | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_product_complex128_cell | 256 | 129 | 12/0 | ? ? | - | 525 | 515 | 450 |
| reduce_product_complex128_finalize | 128 | 11 | 0/0 | ? ? ? | - | 131 | 82 | 132 |
| reduce_product_complex128_init | 128 | 39 | 0/0 | ? ? | - | 582 | 522 | 455 |
| reduce_product_complex128_update | 64 | 29 | 0/0 | ? ? | - | 844 | 821 | 721 |
| reduce_product_complex64_cell | 256 | 87 | 2/0 | ? ? | - | 770 | 537 | 499 |
| reduce_product_complex64_finalize | 128 | 11 | 0/0 | ? ? ? ? | - | 131 | 82 | 132 |
| reduce_product_complex64_init | 128 | 35 | 0/0 | ? ? ? | - | 518 | 395 | 296 |
| reduce_product_complex64_update | 64 | 27 | 0/0 | ? ? | - | 781 | 596 | 528 |
| reduce_product_f16_cell | 256 | 1717 | 424/0 | 64 ?[128] 32 16 ? | - | 16390 | 10266 | 7557 |
| reduce_product_f16_finalize | 128 | 11 | 0/0 | ? ? ? ? | - | 131 | 82 | 132 |
| reduce_product_f16_init | 4 | 19 | 0/0 | ? | - | 12544 | 8745 | 6410 |
| reduce_product_f16_update | 2 | 13 | 0/0 | ? | - | 16794 | 11284 | 7680 |
| reduce_product_f32_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 98 | 66 |
| reduce_product_f32_finalize | 128 | 11 | 0/0 | ? ? ? ? | - | 131 | 82 | 132 |
| reduce_product_f32_update | 128 | 15 | 0/0 | ? ? ? | - | 133 | 126 | 195 |
| reduce_product_f64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 98 | 66 |
| reduce_product_f64_finalize | 128 | 11 | 0/0 | ? ? ? ? | - | 131 | 82 | 132 |
| reduce_product_f64_update | 128 | 15 | 0/0 | ? ? ? | - | 133 | 126 | 195 |
| reduce_product_i16_cell | 64 | 79 | 2/0 | ? ? ? ? | - | 1588 | 1548 | 1739 |
| reduce_product_i16_update | 32 | 47 | 0/0 | ? ? ? | - | 1622 | 1576 | 1684 |
| reduce_product_i32_cell | 128 | 79 | 2/0 | ? ? ? ? | - | 796 | 774 | 869 |
| reduce_product_i32_update | 64 | 47 | 0/0 | ? ? ? | - | 812 | 788 | 842 |
| reduce_product_i64_cell | 256 | 75 | 0/0 | ? ? ? ? | - | 350 | 344 | 258 |
| reduce_product_i64_update | 128 | 47 | 0/0 | ? ? ? | - | 405 | 409 | 273 |
| reduce_product_i8_cell | 32 | 79 | 2/0 | ? ? ? ? | - | 3177 | 3096 | 3478 |
| reduce_product_i8_finalize | 16 | 11 | 0/0 | ? ? ? ? | - | 1050 | 653 | 1052 |
| reduce_product_i8_update | 16 | 47 | 0/0 | ? ? ? | - | 3244 | 3151 | 3369 |
| reduce_product_u16_cell | 64 | 51 | 0/0 | ? ? ? ? | - | 1036 | 1033 | 1353 |
| reduce_product_u16_update | 32 | 35 | 0/0 | ? ? ? | - | 1050 | 1064 | 1354 |
| reduce_product_u32_cell | 128 | 51 | 0/0 | ? ? ? ? | - | 518 | 516 | 676 |
| reduce_product_u32_update | 64 | 35 | 0/0 | ? ? ? | - | 525 | 532 | 677 |
| reduce_product_u8_cell | 32 | 51 | 0/0 | ? ? ? ? | - | 2071 | 2066 | 2706 |
| reduce_product_u8_update | 16 | 35 | 0/0 | ? ? ? | - | 2099 | 2127 | 2708 |
| reduce_sum_f16_cell | 256 | 1717 | 424/0 | 64 ?[128] 32 16 ? | - | 16390 | 10257 | 8294 |
| **geomean** | | | | | | **745** | **642** | **685** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | skylake |
|---|---:|---:|---:|---|---|---:|
| reduce_product_complex128_cell | 128 | 84 | 20/0 | ? ? | - | 648 |
| reduce_product_complex128_finalize | 32 | 7 | 0/0 | ? ? | - | 268 |
| reduce_product_complex128_init | 32 | 18 | 0/0 | ? ? | - | 545 |
| reduce_product_complex128_update | 32 | 25 | 0/0 | ? ? | - | 792 |
| reduce_product_complex64_cell | 128 | 206 | 51/0 | ? ? | - | 2057 |
| reduce_product_complex64_finalize | 32 | 7 | 0/0 | ? ? | - | 268 |
| reduce_product_complex64_init | 32 | 18 | 0/0 | ? ? | - | 561 |
| reduce_product_complex64_update | 32 | 25 | 0/0 | ? ? | - | 792 |
| reduce_product_f16_cell | 128 | 17204 | 32/520 | 64 ?[64] ? | `memcpy@PLT` | 189355* |
| reduce_product_f16_finalize | 32 | 7 | 0/0 | ? ? ? | - | 268 |
| reduce_product_f16_update | 2 | 272 | 0/6 | ? | - | 182661 |
| reduce_product_f32_cell | 128 | 35 | 4/0 | ? ? | - | 244 |
| reduce_product_f32_finalize | 4 | 5 | 0/0 | ? | - | 1106 |
| reduce_product_f32_update | 32 | 11 | 0/0 | ? ? | - | 305 |
| reduce_product_f64_cell | 128 | 35 | 4/0 | ? ? | - | 244 |
| reduce_product_f64_finalize | 8 | 5 | 0/0 | ? | - | 553 |
| reduce_product_f64_update | 32 | 11 | 0/0 | ? ? | - | 305 |
| reduce_product_i16_cell | 32 | 227 | 0/95 | ? ? | - | 6307 |
| reduce_product_i16_update | 8 | 37 | 0/0 | ? ? | - | 4562 |
| reduce_product_i32_cell | 64 | 122 | 22/0 | ? ? | - | 1772 |
| reduce_product_i32_update | 16 | 33 | 0/0 | ? ? | - | 1946 |
| reduce_product_i64_cell | 128 | 96 | 20/0 | ? ? ? | - | 716 |
| reduce_product_i64_update | 32 | 29 | 0/0 | ? ? | - | 874 |
| reduce_product_i8_cell | 16 | 225 | 0/94 | ? ? | - | 12365 |
| reduce_product_i8_finalize | 4 | 7 | 0/0 | ? ? | - | 2140 |
| reduce_product_i8_update | 4 | 41 | 0/0 | ? ? | - | 10455 |
| reduce_product_u16_cell | 32 | 157 | 0/88 | ? ? | - | 5775 |
| reduce_product_u16_update | 8 | 25 | 0/0 | ? ? | - | 2811 |
| reduce_product_u32_cell | 64 | 78 | 16/0 | ? ? | - | 1160 |
| reduce_product_u32_update | 16 | 23 | 0/0 | ? ? | - | 1234 |
| reduce_product_u8_cell | 16 | 161 | 0/89 | ? ? | - | 11812 |
| reduce_product_u8_update | 4 | 29 | 0/0 | ? ? | - | 8356 |
| reduce_sum_f16_cell | 128 | 17204 | 32/520 | 64 ?[64] ? | `memcpy@PLT` | 189355* |
| **geomean** | | | | | | **1958*** |
