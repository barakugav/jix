# llvm-mca summary: `ri-v8`

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
| reduce_all_bool_cell | 2194 | 1364 | 2194 | 826 | 634 | 1082 |
| reduce_any_bool_cell | 2194 | 1364 | 2194 | 873 | 719 | 1155 |
| reduce_arg_max_bool_cell | 2527 | 2511 | 2348 | 20878 | 20580 | 24882 |
| reduce_arg_max_f16_cell | 18966 | 16280 | 16020 | 19595 | 16332 | 18251 |
| reduce_arg_max_f32_cell | 607 | 735 | 525 | 6515 | 6249 | 7378 |
| reduce_arg_max_f64_cell | 350 | 435 | 260 | 2178 | 1907 | 1603 |
| reduce_arg_max_i16_cell | 916 | 1103 | 980 | 8390 | 9230 | 12441 |
| reduce_arg_max_i32_cell | 484 | 593 | 425 | 4102 | 4263 | 5005 |
| reduce_arg_max_i64_cell | 317 | 350 | 180 | 418 | 386 | 258 |
| reduce_arg_max_i8_cell | 2104 | 2381 | 2342 | 20136 | 20635 | 25738 |
| reduce_arg_max_u16_cell | 1087 | 1172 | 1041 | 9734 | 9614 | 12441 |
| reduce_arg_max_u32_cell | 563 | 660 | 490 | 4742 | 4423 | 5027 |
| reduce_arg_max_u64_cell | 367 | 432 | 276 | 418 | 386 | 258 |
| reduce_arg_max_u8_cell | 2342 | 2440 | 2355 | 21107 | 20635 | 25352 |
| reduce_arg_min_bool_cell | 2342 | 2440 | 2355 | 20110 | 20573 | 25865 |
| reduce_arg_min_f16_cell | 22806 | 18342 | 18223 | 23307 | 18455 | 20362 |
| reduce_arg_min_f32_cell | 607 | 735 | 525 | 6515 | 6249 | 7378 |
| reduce_arg_min_f64_cell | 357 | 434 | 261 | 2178 | 1907 | 1603 |
| reduce_arg_min_i16_cell | 916 | 1103 | 980 | 8390 | 9230 | 12441 |
| reduce_arg_min_i32_cell | 484 | 593 | 425 | 4102 | 4263 | 5005 |
| reduce_arg_min_i64_cell | 317 | 350 | 180 | 418 | 386 | 258 |
| reduce_arg_min_i8_cell | 2104 | 2381 | 2342 | 20136 | 20635 | 25738 |
| reduce_arg_min_u16_cell | 1087 | 1172 | 1041 | 8390 | 9230 | 12441 |
| reduce_arg_min_u32_cell | 563 | 660 | 490 | 4102 | 4263 | 5005 |
| reduce_arg_min_u64_cell | 367 | 432 | 276 | 418 | 386 | 258 |
| reduce_arg_min_u8_cell | 2342 | 2440 | 2355 | 20136 | 20635 | 25738 |
| reduce_max_bool_cell | 2130 | 1276 | 2131 | 802 | 655 | 1082 |
| reduce_max_f32_cell | 301 | 352 | 257 | 196 | 161 | 193 |
| reduce_max_f64_cell | 301 | 352 | 257 | 196 | 161 | 193 |
| reduce_max_i16_cell | 1154 | 701 | 1155 | 65 | 65 | 33 |
| reduce_max_i32_cell | 66 | 66 | 66 | 65 | 65 | 33 |
| reduce_max_i64_cell | 130 | 207 | 74 | 81 | 65 | 33 |
| reduce_max_i8_cell | 2130 | 1276 | 2131 | 802 | 683 | 1082 |
| reduce_max_u16_cell | 1154 | 701 | 1155 | 65 | 65 | 33 |
| reduce_max_u32_cell | 66 | 66 | 66 | 65 | 65 | 33 |
| reduce_max_u64_cell | 215 | 258 | 130 | 81 | 65 | 33 |
| reduce_max_u8_cell | 2130 | 1276 | 2131 | 802 | 683 | 1082 |
| reduce_mean_bool_cell | 1041 | 1042 | 1551 | 520 | 521 | 776 |
| reduce_mean_complex128_cell | 130 | 130 | 130 | 65 | 41 | 65 |
| reduce_mean_complex64_cell | 66 | 82 | 66 | 65 | 41 | 65 |
| reduce_mean_f16_cell | 17299 | 11159 | 9412 | 10339 | 10245 | 8247 |
| reduce_mean_f32_cell | 66 | 82 | 66 | 65 | 41 | 65 |
| reduce_mean_f64_cell | 66 | 82 | 66 | 65 | 41 | 65 |
| reduce_mean_i16_cell | 519 | 521 | 776 | 260 | 260 | 388 |
| reduce_mean_i32_cell | 260 | 260 | 388 | 130 | 130 | 194 |
| reduce_mean_i64_cell | 66 | 50 | 66 | 33 | 33 | 33 |
| reduce_mean_i8_cell | 1038 | 1042 | 1551 | 520 | 521 | 776 |
| reduce_mean_u16_cell | 520 | 521 | 776 | 260 | 260 | 388 |
| reduce_mean_u32_cell | 260 | 260 | 388 | 130 | 130 | 194 |
| reduce_mean_u8_cell | 1041 | 1042 | 1551 | 520 | 521 | 776 |
| reduce_min_bool_cell | 2130 | 1276 | 2131 | 802 | 655 | 1082 |
| reduce_min_f32_cell | 301 | 352 | 257 | 196 | 161 | 193 |
| reduce_min_f64_cell | 301 | 352 | 257 | 196 | 161 | 193 |
| reduce_min_i16_cell | 1154 | 701 | 1155 | 65 | 65 | 33 |
| reduce_min_i32_cell | 66 | 66 | 66 | 65 | 65 | 33 |
| reduce_min_i64_cell | 130 | 207 | 74 | 81 | 65 | 33 |
| reduce_min_i8_cell | 2130 | 1276 | 2131 | 802 | 683 | 1082 |
| reduce_min_u16_cell | 1154 | 701 | 1155 | 65 | 65 | 33 |
| reduce_min_u32_cell | 66 | 66 | 66 | 65 | 65 | 33 |
| reduce_min_u64_cell | 215 | 258 | 130 | 81 | 65 | 33 |
| reduce_min_u8_cell | 2130 | 1276 | 2131 | 802 | 683 | 1082 |
| reduce_product_complex128_cell | 525 | 515 | 450 | 258 | 194 | 241 |
| reduce_product_complex64_cell | 770 | 537 | 499 | 194 | 131 | 130 |
| reduce_product_f16_cell | 17220 | 11142 | 8516 | 10339 | 10253 | 7362 |
| reduce_product_f32_cell | 66 | 98 | 66 | 65 | 65 | 65 |
| reduce_product_f64_cell | 66 | 98 | 66 | 65 | 65 | 65 |
| reduce_product_i16_cell | 1610 | 1612 | 1736 | 776 | 776 | 388 |
| reduce_product_i32_cell | 805 | 806 | 868 | 388 | 388 | 194 |
| reduce_product_i64_cell | 402 | 402 | 258 | 194 | 194 | 65 |
| reduce_product_i8_cell | 3219 | 3223 | 3471 | 1552 | 1553 | 777 |
| reduce_product_u16_cell | 1035 | 1034 | 1352 | 776 | 776 | 388 |
| reduce_product_u32_cell | 517 | 517 | 676 | 388 | 388 | 194 |
| reduce_product_u8_cell | 2070 | 2068 | 2703 | 1552 | 1553 | 777 |
| reduce_sum_complex64_cell | 66 | 82 | 66 | 65 | 41 | 65 |
| reduce_sum_f16_cell | 17220 | 11143 | 9333 | 10339 | 10249 | 8247 |
| reduce_variance_bool_cell | 17715 | 12820 | 11525 | 23109 | 21325 | 18458 |
| reduce_variance_complex128_cell | 1229 | 1068 | 1579 | 1973 | 1541 | 1810 |
| reduce_variance_complex64_cell | 2601 | 2045 | 3095 | 4012 | 3115 | 3556 |
| reduce_variance_f16_cell | 14630 | 13839 | 13084 | 12844 | 11616 | 9357 |
| reduce_variance_f32_cell | 2993 | 2849 | 2474 | 5908 | 5718 | 4740 |
| reduce_variance_f64_cell | 1496 | 1423 | 1235 | 2986 | 2761 | 2370 |
| reduce_variance_i16_cell | 5990 | 5711 | 5079 | 12086 | 10967 | 9167 |
| reduce_variance_i32_cell | 2988 | 2847 | 2472 | 5462 | 5203 | 4548 |
| reduce_variance_i64_cell | 1497 | 1423 | 1244 | 2731 | 2602 | 2274 |
| reduce_variance_i8_cell | 11976 | 11423 | 10158 | 24172 | 21934 | 18335 |
| reduce_variance_u16_cell | 5998 | 5471 | 5215 | 11555 | 10662 | 9229 |
| reduce_variance_u32_cell | 2996 | 2854 | 2670 | 5584 | 5139 | 4553 |
| reduce_variance_u64_cell | 1500 | 1427 | 1302 | 2792 | 2569 | 2276 |
| reduce_variance_u8_cell | 18150 | 13286 | 10470 | 23109 | 21325 | 18458 |
| reduce_max_f16_cell | - | - | - | 699 | 612 | 515 |
| reduce_min_f16_cell | - | - | - | 658 | 612 | 515 |
| **geomean** | **877** | **845** | **808** | **962** | **877** | **833** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | stack | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 256 | 354 | 262/0 | 16 ? ? | - | 2194 | 1364 | 2194 |
| reduce_any_bool_cell | 256 | 354 | 262/0 | 16 ? ? | - | 2194 | 1364 | 2194 |
| reduce_arg_max_bool_cell | 16 | 25 | 0/0 | ? ? | - | 2527 | 2511 | 2348 |
| reduce_arg_max_f16_cell | 32 | 700 | 5/204 | ?[16] 8 ? | - | 18966 | 16280 | 16020 |
| reduce_arg_max_f32_cell | 64 | 27 | 0/0 | ? ? | - | 607 | 735 | 525 |
| reduce_arg_max_f64_cell | 128 | 43 | 12/0 | ? ? | - | 350 | 435 | 260 |
| reduce_arg_max_i16_cell | 32 | 21 | 0/0 | ? ? | - | 916 | 1103 | 980 |
| reduce_arg_max_i32_cell | 64 | 23 | 0/0 | ? ? | - | 484 | 593 | 425 |
| reduce_arg_max_i64_cell | 128 | 33 | 5/0 | ? ? | - | 317 | 350 | 180 |
| reduce_arg_max_i8_cell | 16 | 22 | 0/0 | ? ? | - | 2104 | 2381 | 2342 |
| reduce_arg_max_u16_cell | 32 | 23 | 0/0 | ? ? | - | 1087 | 1172 | 1041 |
| reduce_arg_max_u32_cell | 64 | 27 | 0/0 | ? ? | - | 563 | 660 | 490 |
| reduce_arg_max_u64_cell | 128 | 46 | 13/0 | ? ? | - | 367 | 432 | 276 |
| reduce_arg_max_u8_cell | 16 | 24 | 0/0 | ? ? | - | 2342 | 2440 | 2355 |
| reduce_arg_min_bool_cell | 16 | 24 | 0/0 | ? ? | - | 2342 | 2440 | 2355 |
| reduce_arg_min_f16_cell | 32 | 798 | 5/204 | ?[16] 8 ? | - | 22806 | 18342 | 18223 |
| reduce_arg_min_f32_cell | 64 | 27 | 0/0 | ? ? | - | 607 | 735 | 525 |
| reduce_arg_min_f64_cell | 128 | 42 | 12/0 | ? ? | - | 357 | 434 | 261 |
| reduce_arg_min_i16_cell | 32 | 21 | 0/0 | ? ? | - | 916 | 1103 | 980 |
| reduce_arg_min_i32_cell | 64 | 23 | 0/0 | ? ? | - | 484 | 593 | 425 |
| reduce_arg_min_i64_cell | 128 | 33 | 5/0 | ? ? | - | 317 | 350 | 180 |
| reduce_arg_min_i8_cell | 16 | 22 | 0/0 | ? ? | - | 2104 | 2381 | 2342 |
| reduce_arg_min_u16_cell | 32 | 23 | 0/0 | ? ? | - | 1087 | 1172 | 1041 |
| reduce_arg_min_u32_cell | 64 | 27 | 0/0 | ? ? | - | 563 | 660 | 490 |
| reduce_arg_min_u64_cell | 128 | 46 | 13/0 | ? ? | - | 367 | 432 | 276 |
| reduce_arg_min_u8_cell | 16 | 24 | 0/0 | ? ? | - | 2342 | 2440 | 2355 |
| reduce_max_bool_cell | 256 | 334 | 255/0 | 16 ? ? | - | 2130 | 1276 | 2131 |
| reduce_max_f32_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_max_f64_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_max_i16_cell | 256 | 173 | 130/0 | 8 ? ? ? ? | - | 1154 | 701 | 1155 |
| reduce_max_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_max_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 130 | 207 | 74 |
| reduce_max_i8_cell | 256 | 334 | 255/0 | 16 ? ? ? ? | - | 2130 | 1276 | 2131 |
| reduce_max_u16_cell | 256 | 173 | 130/0 | 8 ? ? ? ? | - | 1154 | 701 | 1155 |
| reduce_max_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_max_u64_cell | 256 | 43 | 0/0 | ? ? ? ? | - | 215 | 258 | 130 |
| reduce_max_u8_cell | 256 | 334 | 255/0 | 16 ? ? ? ? | - | 2130 | 1276 | 2131 |
| reduce_mean_bool_cell | 32 | 19 | 0/0 | ? ? ? ? | - | 1041 | 1042 | 1551 |
| reduce_mean_complex128_cell | 256 | 19 | 0/0 | ? ? | - | 130 | 130 | 130 |
| reduce_mean_complex64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 82 | 66 |
| reduce_mean_f16_cell | 256 | 1878 | 543/0 | 8 64 ?[128] 32 16 ? | - | 17299 | 11159 | 9412 |
| reduce_mean_f32_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 82 | 66 |
| reduce_mean_f64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 82 | 66 |
| reduce_mean_i16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 519 | 521 | 776 |
| reduce_mean_i32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_i64_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 50 | 66 |
| reduce_mean_i8_cell | 32 | 19 | 0/0 | ? ? ? ? | - | 1038 | 1042 | 1551 |
| reduce_mean_u16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_mean_u32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_u8_cell | 32 | 19 | 0/0 | ? ? ? ? | - | 1041 | 1042 | 1551 |
| reduce_min_bool_cell | 256 | 334 | 255/0 | 16 ? ? | - | 2130 | 1276 | 2131 |
| reduce_min_f32_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_min_f64_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_min_i16_cell | 256 | 173 | 130/0 | 8 ? ? ? ? | - | 1154 | 701 | 1155 |
| reduce_min_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_min_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 130 | 207 | 74 |
| reduce_min_i8_cell | 256 | 334 | 255/0 | 16 ? ? ? ? | - | 2130 | 1276 | 2131 |
| reduce_min_u16_cell | 256 | 173 | 130/0 | 8 ? ? ? ? | - | 1154 | 701 | 1155 |
| reduce_min_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_min_u64_cell | 256 | 43 | 0/0 | ? ? ? ? | - | 215 | 258 | 130 |
| reduce_min_u8_cell | 256 | 334 | 255/0 | 16 ? ? ? ? | - | 2130 | 1276 | 2131 |
| reduce_product_complex128_cell | 256 | 129 | 12/0 | ? ? | - | 525 | 515 | 450 |
| reduce_product_complex64_cell | 256 | 87 | 2/0 | ? ? | - | 770 | 537 | 499 |
| reduce_product_f16_cell | 256 | 1864 | 538/0 | 8 64 ?[128] 32 16 ? | - | 17220 | 11142 | 8516 |
| reduce_product_f32_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 98 | 66 |
| reduce_product_f64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 98 | 66 |
| reduce_product_i16_cell | 64 | 80 | 2/0 | ? ? ? ? | - | 1610 | 1612 | 1736 |
| reduce_product_i32_cell | 128 | 80 | 2/0 | ? ? ? ? | - | 805 | 806 | 868 |
| reduce_product_i64_cell | 256 | 75 | 0/0 | ? ? ? ? | - | 402 | 402 | 258 |
| reduce_product_i8_cell | 32 | 80 | 2/0 | ? ? ? ? | - | 3219 | 3223 | 3471 |
| reduce_product_u16_cell | 64 | 51 | 0/0 | ? ? ? ? | - | 1035 | 1034 | 1352 |
| reduce_product_u32_cell | 128 | 51 | 0/0 | ? ? ? ? | - | 517 | 517 | 676 |
| reduce_product_u8_cell | 32 | 51 | 0/0 | ? ? ? ? | - | 2070 | 2068 | 2703 |
| reduce_sum_complex64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 82 | 66 |
| reduce_sum_f16_cell | 256 | 1864 | 538/0 | 8 64 ?[128] 32 16 ? | - | 17220 | 11143 | 9333 |
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
| **geomean** | | | | | | **877** | **845** | **808** |

Warnings:

- reduce_max_f16_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_max_f16_cell: no loop whose iteration count is linear in the input length
- reduce_min_f16_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_min_f16_cell: no loop whose iteration count is linear in the input length

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | stack | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 512 | 330 | 240/0 | 16 ? ? | - | 826 | 634 | 1082 |
| reduce_any_bool_cell | 512 | 350 | 258/0 | 16 ? ? | - | 873 | 719 | 1155 |
| reduce_arg_max_bool_cell | 32 | 733 | 5/397 | ?[32] ? | - | 20878 | 20580 | 24882 |
| reduce_arg_max_f16_cell | 64 | 1405 | 5/428 | ?[32] 16 8 ? | - | 19595 | 16332 | 18251 |
| reduce_arg_max_f32_cell | 128 | 943 | 183/328 | ?[32] 16 ? | - | 6515 | 6249 | 7378 |
| reduce_arg_max_f64_cell | 256 | 286 | 105/1 | ? ? | - | 2178 | 1907 | 1603 |
| reduce_arg_max_i16_cell | 64 | 700 | 5/428 | ?[32] ? | - | 8390 | 9230 | 12441 |
| reduce_arg_max_i32_cell | 128 | 656 | 79/302 | ?[32] ? | - | 4102 | 4263 | 5005 |
| reduce_arg_max_i64_cell | 256 | 51 | 0/0 | ? ? | - | 418 | 386 | 258 |
| reduce_arg_max_i8_cell | 32 | 830 | 5/397 | ?[32] ? | - | 20136 | 20635 | 25738 |
| reduce_arg_max_u16_cell | 64 | 732 | 5/428 | ?[32] ? | - | 9734 | 9614 | 12441 |
| reduce_arg_max_u32_cell | 128 | 688 | 79/302 | ?[32] ? | - | 4742 | 4423 | 5027 |
| reduce_arg_max_u64_cell | 256 | 51 | 0/0 | ? ? | - | 418 | 386 | 258 |
| reduce_arg_max_u8_cell | 32 | 798 | 5/397 | ?[32] ? | - | 21107 | 20635 | 25352 |
| reduce_arg_min_bool_cell | 32 | 828 | 5/397 | ?[32] ? | - | 20110 | 20573 | 25865 |
| reduce_arg_min_f16_cell | 64 | 1596 | 5/428 | ?[32] 5[? ?] ? | - | 23307 | 18455 | 20362 |
| reduce_arg_min_f32_cell | 128 | 943 | 183/328 | ?[32] 16 ? | - | 6515 | 6249 | 7378 |
| reduce_arg_min_f64_cell | 256 | 286 | 105/1 | ? ? | - | 2178 | 1907 | 1603 |
| reduce_arg_min_i16_cell | 64 | 700 | 5/428 | ?[32] ? | - | 8390 | 9230 | 12441 |
| reduce_arg_min_i32_cell | 128 | 656 | 79/302 | ?[32] ? | - | 4102 | 4263 | 5005 |
| reduce_arg_min_i64_cell | 256 | 51 | 0/0 | ? ? | - | 418 | 386 | 258 |
| reduce_arg_min_i8_cell | 32 | 830 | 5/397 | ?[32] ? | - | 20136 | 20635 | 25738 |
| reduce_arg_min_u16_cell | 64 | 700 | 5/428 | ?[32] ? | - | 8390 | 9230 | 12441 |
| reduce_arg_min_u32_cell | 128 | 656 | 79/302 | ?[32] ? | - | 4102 | 4263 | 5005 |
| reduce_arg_min_u64_cell | 256 | 51 | 0/0 | ? ? | - | 418 | 386 | 258 |
| reduce_arg_min_u8_cell | 32 | 830 | 5/397 | ?[32] ? | - | 20136 | 20635 | 25738 |
| reduce_max_bool_cell | 512 | 317 | 241/0 | 16 ? ? | - | 802 | 655 | 1082 |
| reduce_max_f16_cell | 512 | 253 | 64/0 | ?[8] 4 ? | - | 699 | 612 | 515 |
| reduce_max_f32_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_max_f64_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_max_i16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_i32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_max_i8_cell | 512 | 317 | 241/0 | 16 ? ? ? ? | - | 802 | 683 | 1082 |
| reduce_max_u16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_max_u8_cell | 512 | 317 | 241/0 | 16 ? ? ? ? | - | 802 | 683 | 1082 |
| reduce_mean_bool_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_mean_complex128_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_complex64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_f16_cell | 512 | 3381 | 808/0 | 128 ?[256] 64 32 16 ? | - | 10339 | 10245 | 8247 |
| reduce_mean_f32_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_f64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_i16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_i32_cell | 256 | 19 | 0/0 | ? ? ? ? | - | 130 | 130 | 194 |
| reduce_mean_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 33 | 33 | 33 |
| reduce_mean_i8_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_mean_u16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_u32_cell | 256 | 19 | 0/0 | ? ? ? ? | - | 130 | 130 | 194 |
| reduce_mean_u8_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_min_bool_cell | 512 | 317 | 241/0 | 16 ? ? | - | 802 | 655 | 1082 |
| reduce_min_f16_cell | 512 | 253 | 64/0 | ?[8] 4 ? | - | 658 | 612 | 515 |
| reduce_min_f32_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_min_f64_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_min_i16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_i32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_min_i8_cell | 512 | 317 | 241/0 | 16 ? ? ? ? | - | 802 | 683 | 1082 |
| reduce_min_u16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_min_u8_cell | 512 | 317 | 241/0 | 16 ? ? ? ? | - | 802 | 683 | 1082 |
| reduce_product_complex128_cell | 512 | 75 | 0/0 | ? ? | - | 258 | 194 | 241 |
| reduce_product_complex64_cell | 512 | 47 | 0/0 | ? ? | - | 194 | 131 | 130 |
| reduce_product_f16_cell | 512 | 3381 | 808/0 | 128 ?[256] 64 32 16 ? | - | 10339 | 10253 | 7362 |
| reduce_product_f32_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 65 | 65 |
| reduce_product_f64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 65 | 65 |
| reduce_product_i16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 776 | 776 | 388 |
| reduce_product_i32_cell | 256 | 19 | 0/0 | ? ? ? ? | - | 388 | 388 | 194 |
| reduce_product_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 194 | 194 | 65 |
| reduce_product_i8_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 1552 | 1553 | 777 |
| reduce_product_u16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 776 | 776 | 388 |
| reduce_product_u32_cell | 256 | 19 | 0/0 | ? ? ? ? | - | 388 | 388 | 194 |
| reduce_product_u8_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 1552 | 1553 | 777 |
| reduce_sum_complex64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_sum_f16_cell | 512 | 3381 | 808/0 | 128 ?[256] 64 32 16 ? | - | 10339 | 10249 | 8247 |
| reduce_variance_bool_cell | 16 | 133 | 4/2 | ? ? | - | 23109 | 21325 | 18458 |
| reduce_variance_complex128_cell | 256 | 182 | 11/3 | ? ? | - | 1973 | 1541 | 1810 |
| reduce_variance_complex64_cell | 128 | 194 | 18/3 | ? ? | - | 4012 | 3115 | 3556 |
| reduce_variance_f16_cell | 32 | 144 | 5/1 | ? ? | - | 12844 | 11616 | 9357 |
| reduce_variance_f32_cell | 64 | 143 | 5/1 | ? ? | - | 5908 | 5718 | 4740 |
| reduce_variance_f64_cell | 128 | 143 | 5/1 | ? ? | - | 2986 | 2761 | 2370 |
| reduce_variance_i16_cell | 32 | 140 | 5/1 | ? ? | - | 12086 | 10967 | 9167 |
| reduce_variance_i32_cell | 64 | 138 | 5/1 | ? ? | - | 5462 | 5203 | 4548 |
| reduce_variance_i64_cell | 128 | 138 | 5/1 | ? ? | - | 2731 | 2602 | 2274 |
| reduce_variance_i8_cell | 16 | 140 | 5/1 | ? ? | - | 24172 | 21934 | 18335 |
| reduce_variance_u16_cell | 32 | 133 | 4/2 | ? ? | - | 11555 | 10662 | 9229 |
| reduce_variance_u32_cell | 64 | 131 | 4/2 | ? ? | - | 5584 | 5139 | 4553 |
| reduce_variance_u64_cell | 128 | 131 | 4/2 | ? ? | - | 2792 | 2569 | 2276 |
| reduce_variance_u8_cell | 16 | 133 | 4/2 | ? ? | - | 23109 | 21325 | 18458 |
| **geomean** | | | | | | **962** | **877** | **833** |
