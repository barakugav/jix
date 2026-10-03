# llvm-mca summary: `ri-v4`

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

| kernel | x86_64-v3 skylake | x86_64-v3 alderlake | x86_64-v3 znver3 | x86_64-v4 icelake-server | x86_64-v4 sapphirerapids | x86_64-v4 znver4 |
|---|---:|---:|---:|---:|---:|---:|
| reduce_all_bool_cell | 2307 | 1401 | 2310 | 193 | 65 | 129 |
| reduce_any_bool_cell | 2307 | 1401 | 2310 | 193 | 209 | 160 |
| reduce_arg_max_bool_cell | 9329 | 6948 | 4214 | 2828 | 1166 | 2053 |
| reduce_arg_max_f16_cell | 16860 | 14118 | 14034 | 18710 | 16282 | 16274 |
| reduce_arg_max_f32_cell | 906 | 1035 | 586 | 452 | 452 | 226 |
| reduce_arg_max_f64_cell | 459 | 525 | 264 | 229 | 196 | 132 |
| reduce_arg_max_i16_cell | 973 | 1144 | 1052 | 340 | 397 | 250 |
| reduce_arg_max_i32_cell | 516 | 580 | 438 | 195 | 232 | 132 |
| reduce_arg_max_i64_cell | 386 | 460 | 147 | 177 | 162 | 82 |
| reduce_arg_max_i8_cell | 4183 | 3738 | 3564 | 676 | 791 | 499 |
| reduce_arg_max_u16_cell | 1142 | 1318 | 1060 | 340 | 397 | 250 |
| reduce_arg_max_u32_cell | 575 | 664 | 502 | 195 | 232 | 148 |
| reduce_arg_max_u64_cell | 392 | 459 | 200 | 177 | 162 | 98 |
| reduce_arg_max_u8_cell | 12872 | 9825 | 9114 | 676 | 791 | 499 |
| reduce_arg_min_bool_cell | 11346 | 9825 | 8786 | 676 | 791 | 499 |
| reduce_arg_min_f16_cell | 16929 | 14118 | 13770 | 22166 | 18342 | 18163 |
| reduce_arg_min_f32_cell | 906 | 1035 | 586 | 452 | 452 | 226 |
| reduce_arg_min_f64_cell | 459 | 525 | 264 | 229 | 196 | 132 |
| reduce_arg_min_i16_cell | 973 | 1144 | 1052 | 340 | 397 | 250 |
| reduce_arg_min_i32_cell | 516 | 580 | 438 | 195 | 232 | 132 |
| reduce_arg_min_i64_cell | 386 | 460 | 147 | 177 | 162 | 82 |
| reduce_arg_min_i8_cell | 4183 | 3738 | 3564 | 676 | 791 | 499 |
| reduce_arg_min_u16_cell | 1142 | 1318 | 1060 | 340 | 397 | 250 |
| reduce_arg_min_u32_cell | 575 | 664 | 502 | 195 | 232 | 148 |
| reduce_arg_min_u64_cell | 392 | 459 | 200 | 177 | 162 | 98 |
| reduce_arg_min_u8_cell | 11341 | 9830 | 8699 | 676 | 791 | 499 |
| reduce_max_bool_cell | 2180 | 1274 | 2182 | 34 | 50 | 34 |
| reduce_max_f16_cell | 2116 | 2244 | 1198 | 645 | 615 | 516 |
| reduce_max_f32_cell | 303 | 346 | 259 | 205 | 162 | 193 |
| reduce_max_f64_cell | 303 | 346 | 259 | 205 | 162 | 193 |
| reduce_max_i16_cell | 67 | 99 | 67 | 66 | 66 | 34 |
| reduce_max_i32_cell | 66 | 98 | 66 | 66 | 66 | 34 |
| reduce_max_i64_cell | 162 | 258 | 74 | 160 | 81 | 33 |
| reduce_max_i8_cell | 2180 | 1274 | 2182 | 66 | 66 | 34 |
| reduce_max_u16_cell | 67 | 99 | 67 | 66 | 66 | 34 |
| reduce_max_u32_cell | 66 | 98 | 66 | 66 | 66 | 34 |
| reduce_max_u64_cell | 218 | 266 | 132 | 160 | 81 | 33 |
| reduce_max_u8_cell | 2180 | 1274 | 2182 | 66 | 66 | 34 |
| reduce_mean_bool_cell | 1057 | 1060 | 1567 | 529 | 530 | 783 |
| reduce_mean_complex128_cell | 132 | 132 | 132 | 130 | 130 | 130 |
| reduce_mean_complex64_cell | 292 | 262 | 324 | 66 | 82 | 66 |
| reduce_mean_f16_cell | 16394 | 10259 | 8203 | 10325 | 10250 | 8197 |
| reduce_mean_f32_cell | 130 | 162 | 98 | 66 | 82 | 66 |
| reduce_mean_f64_cell | 130 | 162 | 98 | 65 | 81 | 65 |
| reduce_mean_i16_cell | 526 | 530 | 783 | 264 | 265 | 392 |
| reduce_mean_i32_cell | 264 | 265 | 392 | 132 | 132 | 196 |
| reduce_mean_i64_cell | 66 | 98 | 66 | 33 | 49 | 33 |
| reduce_mean_i8_cell | 1052 | 1060 | 1567 | 529 | 530 | 783 |
| reduce_mean_u16_cell | 529 | 530 | 783 | 264 | 265 | 392 |
| reduce_mean_u32_cell | 264 | 265 | 392 | 132 | 132 | 196 |
| reduce_mean_u8_cell | 1057 | 1060 | 1567 | 529 | 530 | 783 |
| reduce_min_bool_cell | 2180 | 1274 | 2182 | 34 | 50 | 34 |
| reduce_min_f16_cell | 2020 | 2213 | 1222 | 629 | 614 | 515 |
| reduce_min_f32_cell | 303 | 346 | 259 | 205 | 162 | 193 |
| reduce_min_f64_cell | 303 | 346 | 259 | 205 | 162 | 193 |
| reduce_min_i16_cell | 67 | 99 | 67 | 66 | 66 | 34 |
| reduce_min_i32_cell | 66 | 98 | 66 | 66 | 66 | 34 |
| reduce_min_i64_cell | 162 | 258 | 74 | 160 | 81 | 33 |
| reduce_min_i8_cell | 2180 | 1274 | 2182 | 66 | 66 | 34 |
| reduce_min_u16_cell | 67 | 99 | 67 | 66 | 66 | 34 |
| reduce_min_u32_cell | 66 | 98 | 66 | 66 | 66 | 34 |
| reduce_min_u64_cell | 218 | 266 | 132 | 160 | 81 | 33 |
| reduce_min_u8_cell | 2180 | 1274 | 2182 | 66 | 66 | 34 |
| reduce_product_complex128_cell | 389 | 350 | 262 | 617 | 524 | 587 |
| reduce_product_complex64_cell | 1061 | 1030 | 859 | 307 | 276 | 307 |
| reduce_product_f16_cell | 16394 | 10258 | 7561 | 10326 | 10250 | 7365 |
| reduce_product_f32_cell | 130 | 194 | 98 | 66 | 98 | 66 |
| reduce_product_f64_cell | 130 | 194 | 98 | 65 | 97 | 65 |
| reduce_product_i16_cell | 1622 | 1562 | 1684 | 974 | 975 | 393 |
| reduce_product_i32_cell | 812 | 781 | 842 | 487 | 487 | 196 |
| reduce_product_i64_cell | 389 | 389 | 261 | 353 | 273 | 65 |
| reduce_product_i8_cell | 3244 | 3123 | 3369 | 1948 | 1949 | 786 |
| reduce_product_u16_cell | 1175 | 1180 | 1379 | 974 | 975 | 393 |
| reduce_product_u32_cell | 588 | 590 | 689 | 487 | 487 | 196 |
| reduce_product_u8_cell | 2350 | 2360 | 2757 | 1948 | 1949 | 786 |
| reduce_sum_f16_cell | 16394 | 10259 | 8203 | 10326 | 10250 | 8197 |
| reduce_variance_bool_cell | 17715 | 12820 | 11525 | 12995 | 12058 | 11453 |
| reduce_variance_complex128_cell | 1229 | 1068 | 1579 | 1181 | 949 | 1387 |
| reduce_variance_complex64_cell | 2601 | 2045 | 3095 | 2397 | 2214 | 3092 |
| reduce_variance_f16_cell | 14630 | 13839 | 13084 | 6487 | 6026 | 5988 |
| reduce_variance_f32_cell | 2993 | 2849 | 2474 | 3246 | 2605 | 2989 |
| reduce_variance_f64_cell | 1496 | 1423 | 1235 | 1621 | 1300 | 1430 |
| reduce_variance_i16_cell | 5990 | 5711 | 5079 | 5788 | 5217 | 5985 |
| reduce_variance_i32_cell | 2988 | 2847 | 2472 | 3122 | 2605 | 2989 |
| reduce_variance_i64_cell | 1497 | 1423 | 1244 | 1624 | 1302 | 1494 |
| reduce_variance_i8_cell | 11976 | 11423 | 10158 | 12498 | 10936 | 11448 |
| reduce_variance_u16_cell | 5998 | 5471 | 5215 | 6246 | 5478 | 5727 |
| reduce_variance_u32_cell | 2996 | 2854 | 2670 | 3240 | 2725 | 2860 |
| reduce_variance_u64_cell | 1500 | 1427 | 1302 | 1620 | 1363 | 1430 |
| reduce_variance_u8_cell | 18150 | 13286 | 10470 | 11505 | 12544 | 11935 |
| **geomean** | **954** | **956** | **804** | **415** | **396** | **293** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | stack | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 128 | 175 | 130/0 | 8 ? ? | - | 2307 | 1401 | 2310 |
| reduce_any_bool_cell | 128 | 175 | 130/0 | 8 ? ? | - | 2307 | 1401 | 2310 |
| reduce_arg_max_bool_cell | 8 | 36 | 0/0 | ? ? | - | 9329 | 6948 | 4214 |
| reduce_arg_max_f16_cell | 16 | 328 | 0/60 | ? ? | - | 16860 | 14118 | 14034 |
| reduce_arg_max_f32_cell | 64 | 27 | 0/0 | ? ? | - | 906 | 1035 | 586 |
| reduce_arg_max_f64_cell | 64 | 17 | 0/0 | ? ? | - | 459 | 525 | 264 |
| reduce_arg_max_i16_cell | 16 | 13 | 0/0 | ? ? | - | 973 | 1144 | 1052 |
| reduce_arg_max_i32_cell | 32 | 13 | 0/0 | ? ? | - | 516 | 580 | 438 |
| reduce_arg_max_i64_cell | 64 | 13 | 0/0 | ? ? | - | 386 | 460 | 147 |
| reduce_arg_max_i8_cell | 8 | 18 | 0/0 | ? ? | - | 4183 | 3738 | 3564 |
| reduce_arg_max_u16_cell | 16 | 15 | 0/0 | ? ? | - | 1142 | 1318 | 1060 |
| reduce_arg_max_u32_cell | 32 | 15 | 0/0 | ? ? | - | 575 | 664 | 502 |
| reduce_arg_max_u64_cell | 64 | 17 | 0/0 | ? ? | - | 392 | 459 | 200 |
| reduce_arg_max_u8_cell | 8 | 106 | 0/30 | ? ? | - | 12872 | 9825 | 9114 |
| reduce_arg_min_bool_cell | 8 | 102 | 0/30 | ? ? | - | 11346 | 9825 | 8786 |
| reduce_arg_min_f16_cell | 16 | 322 | 0/61 | ? ? | - | 16929 | 14118 | 13770 |
| reduce_arg_min_f32_cell | 64 | 27 | 0/0 | ? ? | - | 906 | 1035 | 586 |
| reduce_arg_min_f64_cell | 64 | 17 | 0/0 | ? ? | - | 459 | 525 | 264 |
| reduce_arg_min_i16_cell | 16 | 13 | 0/0 | ? ? | - | 973 | 1144 | 1052 |
| reduce_arg_min_i32_cell | 32 | 13 | 0/0 | ? ? | - | 516 | 580 | 438 |
| reduce_arg_min_i64_cell | 64 | 13 | 0/0 | ? ? | - | 386 | 460 | 147 |
| reduce_arg_min_i8_cell | 8 | 18 | 0/0 | ? ? | - | 4183 | 3738 | 3564 |
| reduce_arg_min_u16_cell | 16 | 15 | 0/0 | ? ? | - | 1142 | 1318 | 1060 |
| reduce_arg_min_u32_cell | 32 | 15 | 0/0 | ? ? | - | 575 | 664 | 502 |
| reduce_arg_min_u64_cell | 64 | 17 | 0/0 | ? ? | - | 392 | 459 | 200 |
| reduce_arg_min_u8_cell | 8 | 101 | 0/29 | ? ? | - | 11341 | 9830 | 8699 |
| reduce_max_bool_cell | 128 | 161 | 122/0 | 8 ? ? | - | 2180 | 1274 | 2182 |
| reduce_max_f16_cell | 128 | 152 | 6/0 | ? ? | - | 2116 | 2244 | 1198 |
| reduce_max_f32_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_max_f64_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_max_i16_cell | 128 | 7 | 0/0 | ? ? ? ? | - | 67 | 99 | 67 |
| reduce_max_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_max_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 162 | 258 | 74 |
| reduce_max_i8_cell | 128 | 161 | 122/0 | 8 ? ? ? ? | - | 2180 | 1274 | 2182 |
| reduce_max_u16_cell | 128 | 7 | 0/0 | ? ? ? ? | - | 67 | 99 | 67 |
| reduce_max_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_max_u64_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 218 | 266 | 132 |
| reduce_max_u8_cell | 128 | 161 | 122/0 | 8 ? ? ? ? | - | 2180 | 1274 | 2182 |
| reduce_mean_bool_cell | 16 | 11 | 0/0 | ? ? ? ? | - | 1057 | 1060 | 1567 |
| reduce_mean_complex128_cell | 128 | 11 | 0/0 | ? ? | - | 132 | 132 | 132 |
| reduce_mean_complex64_cell | 128 | 45 | 4/0 | ? ? | - | 292 | 262 | 324 |
| reduce_mean_f16_cell | 128 | 861 | 212/0 | 32 ?[64] 16 ? | - | 16394 | 10259 | 8203 |
| reduce_mean_f32_cell | 256 | 11 | 0/0 | ? ? | - | 130 | 162 | 98 |
| reduce_mean_f64_cell | 256 | 11 | 0/0 | ? ? | - | 130 | 162 | 98 |
| reduce_mean_i16_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 526 | 530 | 783 |
| reduce_mean_i32_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_i64_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_mean_i8_cell | 16 | 11 | 0/0 | ? ? ? ? | - | 1052 | 1060 | 1567 |
| reduce_mean_u16_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_mean_u32_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_u8_cell | 16 | 11 | 0/0 | ? ? ? ? | - | 1057 | 1060 | 1567 |
| reduce_min_bool_cell | 128 | 161 | 122/0 | 8 ? ? | - | 2180 | 1274 | 2182 |
| reduce_min_f16_cell | 128 | 149 | 4/0 | ? ? | - | 2020 | 2213 | 1222 |
| reduce_min_f32_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_min_f64_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_min_i16_cell | 128 | 7 | 0/0 | ? ? ? ? | - | 67 | 99 | 67 |
| reduce_min_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_min_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 162 | 258 | 74 |
| reduce_min_i8_cell | 128 | 161 | 122/0 | 8 ? ? ? ? | - | 2180 | 1274 | 2182 |
| reduce_min_u16_cell | 128 | 7 | 0/0 | ? ? ? ? | - | 67 | 99 | 67 |
| reduce_min_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_min_u64_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 218 | 266 | 132 |
| reduce_min_u8_cell | 128 | 161 | 122/0 | 8 ? ? ? ? | - | 2180 | 1274 | 2182 |
| reduce_product_complex128_cell | 128 | 43 | 0/0 | ? ? | - | 389 | 350 | 262 |
| reduce_product_complex64_cell | 128 | 120 | 10/0 | ? ? | - | 1061 | 1030 | 859 |
| reduce_product_f16_cell | 128 | 861 | 212/0 | 32 ?[64] 16 ? | - | 16394 | 10258 | 7561 |
| reduce_product_f32_cell | 256 | 11 | 0/0 | ? ? | - | 130 | 194 | 98 |
| reduce_product_f64_cell | 256 | 11 | 0/0 | ? ? | - | 130 | 194 | 98 |
| reduce_product_i16_cell | 32 | 39 | 0/0 | ? ? ? ? | - | 1622 | 1562 | 1684 |
| reduce_product_i32_cell | 64 | 39 | 0/0 | ? ? ? ? | - | 812 | 781 | 842 |
| reduce_product_i64_cell | 128 | 39 | 0/0 | ? ? ? ? | - | 389 | 389 | 261 |
| reduce_product_i8_cell | 16 | 39 | 0/0 | ? ? ? ? | - | 3244 | 3123 | 3369 |
| reduce_product_u16_cell | 32 | 27 | 0/0 | ? ? ? ? | - | 1175 | 1180 | 1379 |
| reduce_product_u32_cell | 64 | 27 | 0/0 | ? ? ? ? | - | 588 | 590 | 689 |
| reduce_product_u8_cell | 16 | 27 | 0/0 | ? ? ? ? | - | 2350 | 2360 | 2757 |
| reduce_sum_f16_cell | 128 | 861 | 212/0 | 32 ?[64] 16 ? | - | 16394 | 10259 | 8203 |
| reduce_variance_bool_cell | 8 | 51 | 0/0 | ? ? | - | 17715 | 12820 | 11525 |
| reduce_variance_complex128_cell | 128 | 114 | 27/0 | ? ? | - | 1229 | 1068 | 1579 |
| reduce_variance_complex64_cell | 64 | 107 | 24/0 | ? ? | - | 2601 | 2045 | 3095 |
| reduce_variance_f16_cell | 16 | 38 | 0/0 | ? ? | - | 14630 | 13839 | 13084 |
| reduce_variance_f32_cell | 32 | 30 | 0/0 | ? ? | - | 2993 | 2849 | 2474 |
| reduce_variance_f64_cell | 64 | 30 | 0/0 | ? ? | - | 1496 | 1423 | 1235 |
| reduce_variance_i16_cell | 16 | 24 | 0/0 | ? ? | - | 5990 | 5711 | 5079 |
| reduce_variance_i32_cell | 32 | 22 | 0/0 | ? ? | - | 2988 | 2847 | 2472 |
| reduce_variance_i64_cell | 64 | 34 | 0/0 | ? ? | - | 1497 | 1423 | 1244 |
| reduce_variance_i8_cell | 8 | 23 | 0/0 | ? ? | - | 11976 | 11423 | 10158 |
| reduce_variance_u16_cell | 16 | 36 | 0/0 | ? ? | - | 5998 | 5471 | 5215 |
| reduce_variance_u32_cell | 32 | 40 | 0/0 | ? ? | - | 2996 | 2854 | 2670 |
| reduce_variance_u64_cell | 64 | 38 | 0/0 | ? ? | - | 1500 | 1427 | 1302 |
| reduce_variance_u8_cell | 8 | 59 | 0/0 | ? ? | - | 18150 | 13286 | 10470 |
| **geomean** | | | | | | **954** | **956** | **804** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | stack | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 256 | 11 | 0/0 | ? ? | - | 193 | 65 | 129 |
| reduce_any_bool_cell | 256 | 15 | 0/0 | ? ? | - | 193 | 209 | 160 |
| reduce_arg_max_bool_cell | 32 | 17 | 0/0 | ? ? | - | 2828 | 1166 | 2053 |
| reduce_arg_max_f16_cell | 32 | 700 | 3/204 | ?[16] 8 ? | - | 18710 | 16282 | 16274 |
| reduce_arg_max_f32_cell | 128 | 21 | 0/0 | ? ? | - | 452 | 452 | 226 |
| reduce_arg_max_f64_cell | 128 | 17 | 0/0 | ? ? | - | 229 | 196 | 132 |
| reduce_arg_max_i16_cell | 32 | 11 | 0/0 | ? ? | - | 340 | 397 | 250 |
| reduce_arg_max_i32_cell | 64 | 11 | 0/0 | ? ? | - | 195 | 232 | 132 |
| reduce_arg_max_i64_cell | 256 | 21 | 0/0 | ? ? | - | 177 | 162 | 82 |
| reduce_arg_max_i8_cell | 16 | 11 | 0/0 | ? ? | - | 676 | 791 | 499 |
| reduce_arg_max_u16_cell | 32 | 11 | 0/0 | ? ? | - | 340 | 397 | 250 |
| reduce_arg_max_u32_cell | 64 | 11 | 0/0 | ? ? | - | 195 | 232 | 148 |
| reduce_arg_max_u64_cell | 256 | 21 | 0/0 | ? ? | - | 177 | 162 | 98 |
| reduce_arg_max_u8_cell | 16 | 11 | 0/0 | ? ? | - | 676 | 791 | 499 |
| reduce_arg_min_bool_cell | 16 | 11 | 0/0 | ? ? | - | 676 | 791 | 499 |
| reduce_arg_min_f16_cell | 32 | 796 | 3/204 | ?[16] 8 ? | - | 22166 | 18342 | 18163 |
| reduce_arg_min_f32_cell | 128 | 21 | 0/0 | ? ? | - | 452 | 452 | 226 |
| reduce_arg_min_f64_cell | 128 | 17 | 0/0 | ? ? | - | 229 | 196 | 132 |
| reduce_arg_min_i16_cell | 32 | 11 | 0/0 | ? ? | - | 340 | 397 | 250 |
| reduce_arg_min_i32_cell | 64 | 11 | 0/0 | ? ? | - | 195 | 232 | 132 |
| reduce_arg_min_i64_cell | 256 | 21 | 0/0 | ? ? | - | 177 | 162 | 82 |
| reduce_arg_min_i8_cell | 16 | 11 | 0/0 | ? ? | - | 676 | 791 | 499 |
| reduce_arg_min_u16_cell | 32 | 11 | 0/0 | ? ? | - | 340 | 397 | 250 |
| reduce_arg_min_u32_cell | 64 | 11 | 0/0 | ? ? | - | 195 | 232 | 148 |
| reduce_arg_min_u64_cell | 256 | 21 | 0/0 | ? ? | - | 177 | 162 | 98 |
| reduce_arg_min_u8_cell | 16 | 11 | 0/0 | ? ? | - | 676 | 791 | 499 |
| reduce_max_bool_cell | 256 | 7 | 0/0 | ? ? | - | 34 | 50 | 34 |
| reduce_max_f16_cell | 256 | 121 | 24/0 | ?[4] ? | - | 645 | 615 | 516 |
| reduce_max_f32_cell | 256 | 31 | 0/0 | ? ? ? | - | 205 | 162 | 193 |
| reduce_max_f64_cell | 256 | 31 | 0/0 | ? ? ? | - | 205 | 162 | 193 |
| reduce_max_i16_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_max_i32_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_max_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_max_i8_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_max_u16_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_max_u32_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_max_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_max_u8_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_mean_bool_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_mean_complex128_cell | 256 | 19 | 0/0 | ? ? ? | - | 130 | 130 | 130 |
| reduce_mean_complex64_cell | 256 | 7 | 0/0 | ? ? ? | - | 66 | 82 | 66 |
| reduce_mean_f16_cell | 256 | 1693 | 404/0 | 64 ?[128] 32 16 ? | - | 10325 | 10250 | 8197 |
| reduce_mean_f32_cell | 256 | 7 | 0/0 | ? ? ? | - | 66 | 82 | 66 |
| reduce_mean_f64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 81 | 65 |
| reduce_mean_i16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_i32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 132 | 132 | 196 |
| reduce_mean_i64_cell | 512 | 11 | 0/0 | ? ? ? | - | 33 | 49 | 33 |
| reduce_mean_i8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_mean_u16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_u32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 132 | 132 | 196 |
| reduce_mean_u8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_min_bool_cell | 256 | 7 | 0/0 | ? ? | - | 34 | 50 | 34 |
| reduce_min_f16_cell | 256 | 121 | 24/0 | ?[4] ? | - | 629 | 614 | 515 |
| reduce_min_f32_cell | 256 | 31 | 0/0 | ? ? ? | - | 205 | 162 | 193 |
| reduce_min_f64_cell | 256 | 31 | 0/0 | ? ? ? | - | 205 | 162 | 193 |
| reduce_min_i16_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_min_i32_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_min_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_min_i8_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_min_u16_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_min_u32_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_min_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_min_u8_cell | 256 | 7 | 0/0 | ? ? ? ? | - | 66 | 66 | 34 |
| reduce_product_complex128_cell | 256 | 162 | 48/0 | ? ? | - | 617 | 524 | 587 |
| reduce_product_complex64_cell | 256 | 39 | 0/0 | ? ? | - | 307 | 276 | 307 |
| reduce_product_f16_cell | 256 | 1693 | 404/0 | 64 ?[128] 32 16 ? | - | 10326 | 10250 | 7365 |
| reduce_product_f32_cell | 256 | 7 | 0/0 | ? ? ? | - | 66 | 98 | 66 |
| reduce_product_f64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 97 | 65 |
| reduce_product_i16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 974 | 975 | 393 |
| reduce_product_i32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 487 | 487 | 196 |
| reduce_product_i64_cell | 512 | 11 | 0/0 | ? ? ? | - | 353 | 273 | 65 |
| reduce_product_i8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 1948 | 1949 | 786 |
| reduce_product_u16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 974 | 975 | 393 |
| reduce_product_u32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 487 | 487 | 196 |
| reduce_product_u8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 1948 | 1949 | 786 |
| reduce_sum_f16_cell | 256 | 1693 | 404/0 | 64 ?[128] 32 16 ? | - | 10326 | 10250 | 8197 |
| reduce_variance_bool_cell | 8 | 66 | 0/0 | ? ? | - | 12995 | 12058 | 11453 |
| reduce_variance_complex128_cell | 128 | 87 | 0/0 | ? ? | - | 1181 | 949 | 1387 |
| reduce_variance_complex64_cell | 64 | 89 | 0/0 | ? ? | - | 2397 | 2214 | 3092 |
| reduce_variance_f16_cell | 16 | 56 | 0/0 | ? ? | - | 6487 | 6026 | 5988 |
| reduce_variance_f32_cell | 32 | 48 | 0/0 | ? ? | - | 3246 | 2605 | 2989 |
| reduce_variance_f64_cell | 64 | 48 | 0/0 | ? ? | - | 1621 | 1300 | 1430 |
| reduce_variance_i16_cell | 16 | 56 | 0/0 | ? ? | - | 5788 | 5217 | 5985 |
| reduce_variance_i32_cell | 32 | 48 | 0/0 | ? ? | - | 3122 | 2605 | 2989 |
| reduce_variance_i64_cell | 64 | 48 | 0/0 | ? ? | - | 1624 | 1302 | 1494 |
| reduce_variance_i8_cell | 8 | 53 | 0/0 | ? ? | - | 12498 | 10936 | 11448 |
| reduce_variance_u16_cell | 16 | 50 | 0/0 | ? ? | - | 6246 | 5478 | 5727 |
| reduce_variance_u32_cell | 32 | 42 | 0/0 | ? ? | - | 3240 | 2725 | 2860 |
| reduce_variance_u64_cell | 64 | 42 | 0/0 | ? ? | - | 1620 | 1363 | 1430 |
| reduce_variance_u8_cell | 8 | 56 | 0/0 | ? ? | - | 11505 | 12544 | 11935 |
| **geomean** | | | | | | **415** | **396** | **293** |
