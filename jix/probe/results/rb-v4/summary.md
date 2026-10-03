# llvm-mca summary: `rb-v4`

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

- x86_64-v2/reduce_arg_max_bool_cell: `_RINvNvXs2_NtNtCshlbG8Qh09U5_13fearless_simd9generated4avx2NtB8_4Avx2NtNtBa_10simd_trait4Simd9vectorize14vectorize_avx2NCINvNtNtCs3SX7E7xyE1H_3jix3ops9reduction13fold_run_leafbNtB1X_12ArgMaxKernelKb1_Es_0TybEECs81xzGGAQJtz_3jix`, `_RINvNvXs2_NtNtCshlbG8Qh09U5_13fearless_simd9generated6avx512NtB8_6Avx512NtNtBa_10simd_trait4Simd9vectorize16vectorize_avx512NCINvNtNtCs3SX7E7xyE1H_3jix3ops9reduction13fold_run_leafbNtB23_12ArgMaxKernelKb1_Es0_0TybEECs81xzGGAQJtz_3jix`
- x86_64-v2/reduce_arg_min_bool_cell: `_RINvNvXs2_NtNtCshlbG8Qh09U5_13fearless_simd9generated4avx2NtB8_4Avx2NtNtBa_10simd_trait4Simd9vectorize14vectorize_avx2NCINvNtNtCs3SX7E7xyE1H_3jix3ops9reduction13fold_run_leafbNtB1X_12ArgMinKernelKb1_Es_0TybEECs81xzGGAQJtz_3jix`, `_RINvNvXs2_NtNtCshlbG8Qh09U5_13fearless_simd9generated6avx512NtB8_6Avx512NtNtBa_10simd_trait4Simd9vectorize16vectorize_avx512NCINvNtNtCs3SX7E7xyE1H_3jix3ops9reduction13fold_run_leafbNtB23_12ArgMinKernelKb1_Es0_0TybEECs81xzGGAQJtz_3jix`
- i686/reduce_all_bool_cell: `_RNvNtCs2TSF2CumT7P_4core6option13unwrap_failed@PLT`
- i686/reduce_any_bool_cell: `_RNvNtCs2TSF2CumT7P_4core6option13unwrap_failed@PLT`
- i686/reduce_max_bool_cell: `_RNvNtCs2TSF2CumT7P_4core6option13unwrap_failed@PLT`
- i686/reduce_min_bool_cell: `_RNvNtCs2TSF2CumT7P_4core6option13unwrap_failed@PLT`

## Overview

| kernel | x86_64 sandybridge | x86_64 skylake | x86_64 znver3 | x86_64-v2 sandybridge | x86_64-v2 btver2 | x86_64-v3 skylake | x86_64-v3 alderlake | x86_64-v3 znver3 | x86_64-v4 icelake-server | x86_64-v4 sapphirerapids | x86_64-v4 znver4 | i686 skylake | aarch64 cortex-a72 | aarch64 neoverse-n1 | aarch64 neoverse-v2 | aarch64-apple apple-m1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reduce_arg_max_bool_cell | 13911 | 9068 | 8653 | - | - | 11853 | 9088 | 5939 | 2957 | 1715 | 2074 | 16507 | 12570 | 10020 | 4997 | 13860 |
| reduce_arg_max_f16_cell | 21302 | 15078 | 12500 | 21302 | 37274 | 15078 | 13356 | 12500 | 18710 | 16284 | 16275 | 21798 | 20797 | 20795 | 10404 | 22548 |
| reduce_arg_max_f32_cell | 2368 | 1517 | 1083 | 3012 | 3594 | 1453 | 1681 | 1112 | 523 | 525 | 347 | 2904 | 3274 | 3593 | 2063 | 2066 |
| reduce_arg_max_f64_cell | 1928 | 1135 | 820 | 2124 | 3210 | 738 | 842 | 518 | 322 | 337 | 226 | 1631 | 1637 | 1796 | 1032 | 1033 |
| reduce_arg_max_i16_cell | 3607 | 2202 | 1715 | 6397 | 10537 | 2821 | 2721 | 2079 | 832 | 859 | 524 | 5678 | 5391 | 4362 | 2191 | 5130 |
| reduce_arg_max_i32_cell | 2852 | 1501 | 1428 | 3421 | 5775 | 1336 | 1376 | 1033 | 460 | 461 | 278 | 2835 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_max_i64_cell | 1426 | 751 | 714 | 1426 | 2086 | 724 | 791 | 415 | 210 | 252 | 169 | 2191 | 1093 | 962 | 468 | 1028 |
| reduce_arg_max_i8_cell | 16978 | 10839 | 9467 | 16978 | 28201 | 6287 | 6241 | 5627 | 1644 | 1715 | 1047 | 18028 | 14879 | 10532 | 5263 | 16415 |
| reduce_arg_max_u16_cell | 3866 | 2394 | 1772 | 6520 | 11302 | 2849 | 2867 | 2276 | 832 | 859 | 524 | 5844 | 5391 | 4365 | 2194 | 4111 |
| reduce_arg_max_u32_cell | 3606 | 2138 | 1432 | 2189 | 2840 | 1423 | 1467 | 1096 | 460 | 461 | 292 | 2964 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_max_u64_cell | 1803 | 1069 | 716 | 1803 | 2118 | 791 | 865 | 472 | 210 | 252 | 186 | 3139 | 1093 | 962 | 468 | 1028 |
| reduce_arg_max_u8_cell | 17495 | 11351 | 8955 | 19021 | 27950 | 6743 | 7020 | 5484 | 1644 | 1715 | 1047 | 19041 | 14874 | 10271 | 5181 | 14116 |
| reduce_arg_min_bool_cell | 18007 | 11351 | 9364 | - | - | 6743 | 7020 | 5484 | 1644 | 1715 | 1047 | 17741 | 14874 | 10271 | 5181 | 14116 |
| reduce_arg_min_f16_cell | 25019 | 17695 | 14405 | 25019 | 43162 | 17695 | 14886 | 14405 | 22166 | 18342 | 18196 | 23831 | 19604 | 19604 | 9807 | 21770 |
| reduce_arg_min_f32_cell | 2368 | 1517 | 1083 | 3113 | 3596 | 1453 | 1681 | 1112 | 523 | 525 | 347 | 2880 | 3274 | 3593 | 2063 | 2066 |
| reduce_arg_min_f64_cell | 2026 | 1163 | 842 | 2187 | 3210 | 738 | 842 | 518 | 322 | 337 | 226 | 1678 | 1637 | 1796 | 1032 | 1033 |
| reduce_arg_min_i16_cell | 3612 | 2199 | 1715 | 6349 | 10537 | 2821 | 2721 | 2079 | 832 | 859 | 524 | 5670 | 5391 | 4362 | 2191 | 5130 |
| reduce_arg_min_i32_cell | 2852 | 1501 | 1428 | 3610 | 5775 | 1336 | 1376 | 1033 | 460 | 461 | 278 | 2712 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_min_i64_cell | 1426 | 751 | 714 | 1426 | 2086 | 724 | 791 | 415 | 210 | 252 | 169 | 2194 | 1093 | 962 | 468 | 1028 |
| reduce_arg_min_i8_cell | 16978 | 10839 | 9467 | 16978 | 28201 | 6287 | 6241 | 5627 | 1644 | 1715 | 1047 | 18028 | 14879 | 10532 | 5263 | 16415 |
| reduce_arg_min_u16_cell | 3950 | 2463 | 1807 | 6520 | 11302 | 2849 | 2867 | 2276 | 832 | 859 | 524 | 5847 | 5391 | 4365 | 2194 | 4111 |
| reduce_arg_min_u32_cell | 2852 | 1501 | 1428 | 2189 | 2840 | 1423 | 1467 | 1096 | 460 | 461 | 292 | 2756 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_min_u64_cell | 1426 | 751 | 714 | 1426 | 2086 | 791 | 865 | 472 | 210 | 252 | 186 | 3205 | 1093 | 962 | 468 | 1028 |
| reduce_arg_min_u8_cell | 18514 | 11095 | 9375 | 18514 | 27950 | 6743 | 7020 | 5484 | 1644 | 1715 | 1047 | 19041 | 14874 | 10271 | 5181 | 14116 |
| reduce_max_f16_cell | 3634 | 3571 | 2253 | 3315 | 6212 | 2116 | 2244 | 1198 | 645 | 615 | 516 | 3532 | 4954 | 4630 | 2196 | 2739 |
| reduce_max_f32_cell | 1860 | 945 | 553 | 1032 | 1668 | 303 | 346 | 259 | 205 | 162 | 193 | 945 | 1542 | 774 | 454 | 519 |
| reduce_max_f64_cell | 1860 | 945 | 553 | 1032 | 1668 | 303 | 346 | 259 | 205 | 162 | 193 | 945 | 1542 | 774 | 454 | 520 |
| reduce_max_i16_cell | 187 | 147 | 132 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 134 | 326 | 326 | 166 | 197 |
| reduce_max_i32_cell | 497 | 433 | 296 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 520 | 326 | 326 | 166 | 197 |
| reduce_max_i64_cell | 753 | 453 | 379 | 753 | 1124 | 162 | 258 | 74 | 160 | 81 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_max_i8_cell | 497 | 433 | 296 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 520 | 326 | 326 | 166 | 197 |
| reduce_max_u16_cell | 328 | 263 | 210 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 261 | 326 | 326 | 166 | 197 |
| reduce_max_u32_cell | 767 | 689 | 423 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 670 | 326 | 326 | 166 | 197 |
| reduce_max_u64_cell | 923 | 566 | 379 | 899 | 1412 | 218 | 266 | 132 | 160 | 81 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_max_u8_cell | 187 | 147 | 132 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 134 | 326 | 326 | 166 | 197 |
| reduce_mean_bool_cell | 5775 | 5146 | 2877 | 5949 | 10808 | 1057 | 1060 | 1567 | 529 | 530 | 783 | 4009 | 5207 | 3640 | 1894 | 3604 |
| reduce_mean_complex128_cell | 260 | 132 | 132 | 260 | 307 | 132 | 132 | 132 | 130 | 130 | 130 | 196 | 418 | 291 | 131 | 163 |
| reduce_mean_complex64_cell | 519 | 263 | 264 | 519 | 614 | 292 | 262 | 324 | 66 | 82 | 66 | 391 | 454 | 580 | 219 | 262 |
| reduce_mean_f16_cell | 213002 | 168015 | 126929 | 213002 | 328138 | 16394 | 10259 | 8203 | 10326 | 10250 | 8197 | 172495 | 172634 | 147473 | 74076 | 4364 |
| reduce_mean_f32_cell | 260 | 291 | 196 | 260 | 371 | 130 | 162 | 98 | 65 | 81 | 65 | 262 | 454 | 326 | 166 | 261 |
| reduce_mean_f64_cell | 260 | 291 | 196 | 260 | 371 | 130 | 162 | 98 | 65 | 81 | 65 | 262 | 454 | 326 | 166 | 325 |
| reduce_mean_i16_cell | 2204 | 2071 | 1569 | 1175 | 1431 | 526 | 530 | 783 | 264 | 265 | 392 | 2071 | 1306 | 1810 | 668 | 709 |
| reduce_mean_i32_cell | 845 | 526 | 525 | 588 | 716 | 264 | 265 | 392 | 132 | 132 | 196 | 525 | 522 | 649 | 268 | 258 |
| reduce_mean_i64_cell | 187 | 131 | 132 | 187 | 371 | 66 | 98 | 66 | 33 | 49 | 33 | 134 | 326 | 326 | 166 | 98 |
| reduce_mean_i8_cell | 6979 | 6211 | 4157 | 6979 | 12344 | 1052 | 1060 | 1567 | 529 | 530 | 783 | 7726 | 6216 | 4163 | 2120 | 3599 |
| reduce_mean_u16_cell | 1687 | 2071 | 1055 | 1175 | 1431 | 529 | 530 | 783 | 264 | 265 | 392 | 2071 | 1306 | 1810 | 668 | 709 |
| reduce_mean_u32_cell | 588 | 524 | 335 | 588 | 716 | 264 | 265 | 392 | 132 | 132 | 196 | 524 | 522 | 649 | 268 | 258 |
| reduce_mean_u8_cell | 5775 | 5146 | 2877 | 5949 | 10808 | 1057 | 1060 | 1567 | 529 | 530 | 783 | 4009 | 5207 | 3640 | 1894 | 3604 |
| reduce_min_f16_cell | 3527 | 3464 | 2253 | 3421 | 6340 | 2020 | 2213 | 1222 | 629 | 614 | 515 | 3617 | 4922 | 4631 | 2196 | 2708 |
| reduce_min_f32_cell | 1860 | 945 | 553 | 1032 | 1668 | 303 | 346 | 259 | 205 | 162 | 193 | 945 | 1542 | 774 | 454 | 519 |
| reduce_min_f64_cell | 1860 | 945 | 553 | 1032 | 1668 | 303 | 346 | 259 | 205 | 162 | 193 | 945 | 1542 | 774 | 454 | 520 |
| reduce_min_i16_cell | 187 | 147 | 132 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 134 | 326 | 326 | 166 | 197 |
| reduce_min_i32_cell | 614 | 519 | 338 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 520 | 326 | 326 | 166 | 197 |
| reduce_min_i64_cell | 753 | 453 | 379 | 753 | 1124 | 162 | 258 | 74 | 160 | 81 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_min_i8_cell | 614 | 519 | 338 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 520 | 326 | 326 | 166 | 197 |
| reduce_min_u16_cell | 328 | 262 | 210 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 262 | 326 | 326 | 166 | 197 |
| reduce_min_u32_cell | 837 | 774 | 465 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 817 | 326 | 326 | 166 | 197 |
| reduce_min_u64_cell | 753 | 453 | 379 | 899 | 1412 | 218 | 266 | 132 | 160 | 81 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_min_u8_cell | 187 | 147 | 132 | 187 | 371 | 66 | 98 | 66 | 65 | 65 | 33 | 134 | 326 | 326 | 166 | 197 |
| reduce_product_complex128_cell | 676 | 477 | 302 | 676 | 1031 | 389 | 350 | 262 | 617 | 524 | 587 | 648 | 901 | 522 | 261 | 434 |
| reduce_product_complex64_cell | 1734 | 1076 | 677 | 1606 | 1768 | 1061 | 1030 | 859 | 307 | 276 | 307 | 2252 | 941 | 1033 | 524 | 714 |
| reduce_product_f16_cell | 217098 | 168015 | 126929 | 217098 | 326090 | 16394 | 10258 | 7561 | 10326 | 10250 | 7365 | 172495 | 172634 | 147475 | 74077 | 4365 |
| reduce_product_f32_cell | 356 | 291 | 196 | 356 | 371 | 130 | 194 | 98 | 65 | 97 | 65 | 262 | 454 | 326 | 198 | 133 |
| reduce_product_f64_cell | 356 | 291 | 196 | 356 | 516 | 130 | 194 | 98 | 65 | 97 | 65 | 262 | 454 | 326 | 198 | 325 |
| reduce_product_i16_cell | 2074 | 2074 | 2074 | 2074 | 8215 | 1622 | 1562 | 1684 | 974 | 975 | 393 | 6464 | 2076 | 6164 | 1044 | 2076 |
| reduce_product_i32_cell | 1037 | 1037 | 1037 | 1037 | 4108 | 812 | 781 | 842 | 487 | 487 | 196 | 3098 | 1293 | 3078 | 906 | 1038 |
| reduce_product_i64_cell | 518 | 518 | 518 | 518 | 2054 | 389 | 389 | 261 | 353 | 273 | 65 | 1551 | 519 | 1541 | 261 | 519 |
| reduce_product_i8_cell | 6467 | 4454 | 4157 | 6467 | 16440 | 3244 | 3123 | 3369 | 1948 | 1949 | 786 | 10967 | 4157 | 12334 | 2094 | 4157 |
| reduce_product_u16_cell | 2074 | 2074 | 2074 | 2074 | 8215 | 1175 | 1180 | 1379 | 974 | 975 | 393 | 5435 | 2076 | 6164 | 1044 | 2076 |
| reduce_product_u32_cell | 1037 | 1037 | 1037 | 1037 | 4108 | 588 | 590 | 689 | 487 | 487 | 196 | 2703 | 1038 | 3078 | 522 | 1038 |
| reduce_product_u8_cell | 5381 | 4157 | 4157 | 5381 | 16440 | 2350 | 2360 | 2757 | 1948 | 1949 | 786 | 5937 | 4157 | 12334 | 2094 | 4157 |
| reduce_sum_f16_cell | 213002 | 168015 | 126929 | 213002 | 328138 | 16394 | 10259 | 8203 | 10326 | 10250 | 8197 | 172495 | 172634 | 147473 | 74076 | 4364 |
| reduce_variance_bool_cell | 45215 | 16538 | 17224 | 45292 | 58972 | 17715 | 12820 | 11525 | 12995 | 12058 | 11453 | 21294 | 135388 | 18949 | 15985 | 9062 |
| reduce_variance_complex128_cell | 5643 | 1271 | 1293 | 5643 | 5736 | 1229 | 1068 | 1579 | 1181 | 949 | 1387 | 1496 | 16465 | 2186 | 1833 | 723 |
| reduce_variance_complex64_cell | 11286 | 2964 | 2652 | 11286 | 12047 | 2601 | 2045 | 3095 | 2397 | 2214 | 3092 | 3166 | 33063 | 4563 | 3983 | 1616 |
| reduce_variance_f16_cell | 73756 | 35863 | 40737 | 73756 | 105244 | 14630 | 13839 | 13084 | 6487 | 6026 | 5988 | 34337 | 95759 | 48655 | 18450 | 4050 |
| reduce_variance_f32_cell | 11309 | 3190 | 2611 | 11311 | 10801 | 2993 | 2849 | 2474 | 3246 | 2605 | 2989 | 3288 | 33832 | 4576 | 3965 | 1641 |
| reduce_variance_f64_cell | 5654 | 1594 | 1457 | 5654 | 5398 | 1496 | 1423 | 1235 | 1621 | 1300 | 1430 | 1563 | 16914 | 1954 | 1805 | 820 |
| reduce_variance_i16_cell | 22625 | 8274 | 6495 | 22628 | 24632 | 5990 | 5711 | 5079 | 5788 | 5217 | 5985 | 6520 | 67681 | 9285 | 8013 | 3407 |
| reduce_variance_i32_cell | 11311 | 3204 | 2611 | 11313 | 12319 | 2988 | 2847 | 2472 | 3122 | 2605 | 2989 | 3532 | 33839 | 4576 | 3999 | 1641 |
| reduce_variance_i64_cell | 5658 | 1576 | 1370 | 5658 | 5974 | 1497 | 1423 | 1244 | 1624 | 1302 | 1494 | 2520 | 16919 | 2287 | 1982 | 820 |
| reduce_variance_i8_cell | 45353 | 23255 | 20076 | 45245 | 47744 | 11976 | 11423 | 10158 | 12498 | 10936 | 11448 | 19692 | 135393 | 21637 | 14479 | 8305 |
| reduce_variance_u16_cell | 22628 | 6682 | 5225 | 22628 | 27425 | 5998 | 5471 | 5215 | 6246 | 5478 | 5727 | 7660 | 67702 | 8415 | 8397 | 3674 |
| reduce_variance_u32_cell | 11319 | 3213 | 2866 | 11311 | 13088 | 2996 | 2854 | 2670 | 3240 | 2725 | 2860 | 3389 | 33843 | 4582 | 3972 | 1760 |
| reduce_variance_u64_cell | 5660 | 1794 | 1496 | 5660 | 7374 | 1500 | 1427 | 1302 | 1620 | 1363 | 1430 | 1928 | 16923 | 2573 | 2033 | 978 |
| reduce_variance_u8_cell | 45322 | 17940 | 14269 | 45297 | 57964 | 18150 | 13286 | 10470 | 11505 | 12544 | 11935 | 16609 | 135393 | 21381 | 14479 | 8387 |
| reduce_all_bool_cell | - | - | - | 326 | 613 | 337 | 324 | 321 | 193 | 65 | 129 | - | 583 | 518 | 390 | 453 |
| reduce_any_bool_cell | - | - | - | 326 | 613 | 337 | 324 | 321 | 193 | 209 | 160 | - | 583 | 518 | 390 | 453 |
| reduce_max_bool_cell | - | - | - | 181 | 357 | 67 | 99 | 67 | 34 | 50 | 34 | - | 326 | 326 | 166 | 133 |
| reduce_min_bool_cell | - | - | - | 181 | 357 | 67 | 99 | 67 | 34 | 50 | 34 | - | 326 | 326 | 166 | 133 |
| **geomean** | **2749** | **1728** | **1363** | **2124** | **3417** | **823** | **883** | **698** | **480** | **457** | **342** | **2424** | **2789** | **2007** | **1023** | **1127** |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_arg_max_bool_cell | 8 | 94 | 0/34 | ? ? | - | 13911 | 9068 | 8653 |
| reduce_arg_max_f16_cell | 16 | 291 | 0/53 | ? ? | - | 21302 | 15078 | 12500 |
| reduce_arg_max_f32_cell | 32 | 50 | 1/8 | ? ? | - | 2368 | 1517 | 1083 |
| reduce_arg_max_f64_cell | 64 | 76 | 3/8 | ? ? | - | 1928 | 1135 | 820 |
| reduce_arg_max_i16_cell | 16 | 35 | 0/0 | ? ? | - | 3607 | 2202 | 1715 |
| reduce_arg_max_i32_cell | 32 | 65 | 0/16 | ? ? | - | 2852 | 1501 | 1428 |
| reduce_arg_max_i64_cell | 64 | 65 | 0/16 | ? ? | - | 1426 | 751 | 714 |
| reduce_arg_max_i8_cell | 8 | 110 | 0/34 | ? ? | - | 16978 | 10839 | 9467 |
| reduce_arg_max_u16_cell | 16 | 38 | 0/0 | ? ? | - | 3866 | 2394 | 1772 |
| reduce_arg_max_u32_cell | 32 | 66 | 0/17 | ? ? | - | 3606 | 2138 | 1432 |
| reduce_arg_max_u64_cell | 64 | 66 | 0/17 | ? ? | - | 1803 | 1069 | 716 |
| reduce_arg_max_u8_cell | 8 | 104 | 0/32 | ? ? | - | 17495 | 11351 | 8955 |
| reduce_arg_min_bool_cell | 8 | 109 | 0/31 | ? ? | - | 18007 | 11351 | 9364 |
| reduce_arg_min_f16_cell | 16 | 337 | 0/64 | ? ? | - | 25019 | 17695 | 14405 |
| reduce_arg_min_f32_cell | 32 | 50 | 1/8 | ? ? | - | 2368 | 1517 | 1083 |
| reduce_arg_min_f64_cell | 64 | 78 | 3/8 | ? ? | - | 2026 | 1163 | 842 |
| reduce_arg_min_i16_cell | 16 | 35 | 0/0 | ? ? | - | 3612 | 2199 | 1715 |
| reduce_arg_min_i32_cell | 32 | 65 | 0/16 | ? ? | - | 2852 | 1501 | 1428 |
| reduce_arg_min_i64_cell | 64 | 65 | 0/16 | ? ? | - | 1426 | 751 | 714 |
| reduce_arg_min_i8_cell | 8 | 110 | 0/34 | ? ? | - | 16978 | 10839 | 9467 |
| reduce_arg_min_u16_cell | 16 | 39 | 0/0 | ? ? | - | 3950 | 2463 | 1807 |
| reduce_arg_min_u32_cell | 32 | 65 | 0/16 | ? ? | - | 2852 | 1501 | 1428 |
| reduce_arg_min_u64_cell | 64 | 65 | 0/16 | ? ? | - | 1426 | 751 | 714 |
| reduce_arg_min_u8_cell | 8 | 109 | 0/31 | ? ? | - | 18514 | 11095 | 9375 |
| reduce_max_f16_cell | 64 | 190 | 0/0 | ? ? | - | 3634 | 3571 | 2253 |
| reduce_max_f32_cell | 64 | 51 | 0/0 | ? ? | - | 1860 | 945 | 553 |
| reduce_max_f64_cell | 64 | 51 | 0/0 | ? ? | - | 1860 | 945 | 553 |
| reduce_max_i16_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 147 | 132 |
| reduce_max_i32_cell | 64 | 27 | 0/0 | ? ? ? | - | 497 | 433 | 296 |
| reduce_max_i64_cell | 64 | 35 | 0/0 | ? ? | - | 753 | 453 | 379 |
| reduce_max_i8_cell | 64 | 27 | 0/0 | ? ? ? ? | - | 497 | 433 | 296 |
| reduce_max_u16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 328 | 263 | 210 |
| reduce_max_u32_cell | 64 | 39 | 0/0 | ? ? ? | - | 767 | 689 | 423 |
| reduce_max_u64_cell | 64 | 35 | 0/0 | ? ? | - | 923 | 566 | 379 |
| reduce_max_u8_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 147 | 132 |
| reduce_mean_bool_cell | 8 | 33 | 0/0 | ? ? | - | 5775 | 5146 | 2877 |
| reduce_mean_complex128_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_mean_complex64_cell | 64 | 19 | 0/0 | ? ? | - | 519 | 263 | 264 |
| reduce_mean_f16_cell | 64 | 8534 | 12/96 | 32 ?[32] ? | - | 213002 | 168015 | 126929 |
| reduce_mean_f32_cell | 128 | 23 | 0/0 | ? ? | - | 260 | 291 | 196 |
| reduce_mean_f64_cell | 128 | 23 | 0/0 | ? ? | - | 260 | 291 | 196 |
| reduce_mean_i16_cell | 16 | 31 | 0/0 | ? ? | - | 2204 | 2071 | 1569 |
| reduce_mean_i32_cell | 32 | 23 | 0/0 | ? ? ? | - | 845 | 526 | 525 |
| reduce_mean_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 131 | 132 |
| reduce_mean_i8_cell | 8 | 40 | 0/0 | ? ? | - | 6979 | 6211 | 4157 |
| reduce_mean_u16_cell | 16 | 19 | 0/0 | ? ? ? | - | 1687 | 2071 | 1055 |
| reduce_mean_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 588 | 524 | 335 |
| reduce_mean_u8_cell | 8 | 33 | 0/0 | ? ? | - | 5775 | 5146 | 2877 |
| reduce_min_f16_cell | 64 | 179 | 0/0 | ? ? | - | 3527 | 3464 | 2253 |
| reduce_min_f32_cell | 64 | 51 | 0/0 | ? ? | - | 1860 | 945 | 553 |
| reduce_min_f64_cell | 64 | 51 | 0/0 | ? ? | - | 1860 | 945 | 553 |
| reduce_min_i16_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 147 | 132 |
| reduce_min_i32_cell | 64 | 31 | 0/0 | ? ? ? | - | 614 | 519 | 338 |
| reduce_min_i64_cell | 64 | 35 | 0/0 | ? ? | - | 753 | 453 | 379 |
| reduce_min_i8_cell | 64 | 31 | 0/0 | ? ? ? ? | - | 614 | 519 | 338 |
| reduce_min_u16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 328 | 262 | 210 |
| reduce_min_u32_cell | 64 | 43 | 0/0 | ? ? ? | - | 837 | 774 | 465 |
| reduce_min_u64_cell | 64 | 35 | 0/0 | ? ? | - | 753 | 453 | 379 |
| reduce_min_u8_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 147 | 132 |
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 477 | 302 |
| reduce_product_complex64_cell | 64 | 57 | 0/0 | ? ? | - | 1734 | 1076 | 677 |
| reduce_product_f16_cell | 64 | 8534 | 12/96 | 32 ?[32] ? | - | 217098 | 168015 | 126929 |
| reduce_product_f32_cell | 128 | 23 | 0/0 | ? ? | - | 356 | 291 | 196 |
| reduce_product_f64_cell | 128 | 23 | 0/0 | ? ? | - | 356 | 291 | 196 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 2074 | 2074 |
| reduce_product_i32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 1037 | 1037 |
| reduce_product_i64_cell | 64 | 11 | 0/0 | ? ? | - | 518 | 518 | 518 |
| reduce_product_i8_cell | 8 | 33 | 0/0 | ? ? | - | 6467 | 4454 | 4157 |
| reduce_product_u16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 2074 | 2074 |
| reduce_product_u32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 1037 | 1037 |
| reduce_product_u8_cell | 8 | 30 | 0/0 | ? ? | - | 5381 | 4157 | 4157 |
| reduce_sum_f16_cell | 64 | 8534 | 12/96 | 32 ?[32] ? | - | 213002 | 168015 | 126929 |
| reduce_variance_bool_cell | 8 | 126 | 17/0 | ? ? | - | 45215 | 16538 | 17224 |
| reduce_variance_complex128_cell | 128 | 140 | 40/0 | ? ? | - | 5643 | 1271 | 1293 |
| reduce_variance_complex64_cell | 64 | 156 | 40/0 | ? ? | - | 11286 | 2964 | 2652 |
| reduce_variance_f16_cell | 16 | 508 | 10/1 | ? ? | - | 73756 | 35863 | 40737 |
| reduce_variance_f32_cell | 32 | 63 | 0/0 | ? ? | - | 11309 | 3190 | 2611 |
| reduce_variance_f64_cell | 64 | 53 | 0/0 | ? ? | - | 5654 | 1594 | 1457 |
| reduce_variance_i16_cell | 16 | 65 | 10/0 | ? ? | - | 22625 | 8274 | 6495 |
| reduce_variance_i32_cell | 32 | 61 | 10/0 | ? ? | - | 11311 | 3204 | 2611 |
| reduce_variance_i64_cell | 64 | 73 | 10/0 | ? ? | - | 5658 | 1576 | 1370 |
| reduce_variance_i8_cell | 8 | 107 | 16/3 | ? ? | - | 45353 | 23255 | 20076 |
| reduce_variance_u16_cell | 16 | 69 | 10/0 | ? ? | - | 22628 | 6682 | 5225 |
| reduce_variance_u32_cell | 32 | 75 | 10/0 | ? ? | - | 11319 | 3213 | 2866 |
| reduce_variance_u64_cell | 64 | 94 | 10/0 | ? ? | - | 5660 | 1794 | 1496 |
| reduce_variance_u8_cell | 8 | 109 | 12/3 | ? ? | - | 45322 | 17940 | 14269 |
| **geomean** | | | | | | **2749** | **1728** | **1363** |

Warnings:

- reduce_all_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_all_bool_cell: no loop whose iteration count is linear in the input length
- reduce_any_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_any_bool_cell: no loop whose iteration count is linear in the input length
- reduce_max_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_max_bool_cell: no loop whose iteration count is linear in the input length
- reduce_min_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_min_bool_cell: no loop whose iteration count is linear in the input length

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | btver2 |
|---|---:|---:|---:|---|---|---:|---:|
| reduce_all_bool_cell | 64 | 19 | 0/0 | ? ? | - | 326 | 613 |
| reduce_any_bool_cell | 64 | 19 | 0/0 | ? ? | - | 326 | 613 |
| reduce_arg_max_f16_cell | 16 | 291 | 0/53 | ? ? | - | 21302 | 37274 |
| reduce_arg_max_f32_cell | 32 | 51 | 0/0 | ? ? | - | 3012 | 3594 |
| reduce_arg_max_f64_cell | 64 | 90 | 0/12 | ? ? | - | 2124 | 3210 |
| reduce_arg_max_i16_cell | 16 | 68 | 0/0 | ? ? | - | 6397 | 10537 |
| reduce_arg_max_i32_cell | 32 | 74 | 2/0 | ? ? | - | 3421 | 5775 |
| reduce_arg_max_i64_cell | 64 | 65 | 0/16 | ? ? | - | 1426 | 2086 |
| reduce_arg_max_i8_cell | 8 | 110 | 0/34 | ? ? | - | 16978 | 28201 |
| reduce_arg_max_u16_cell | 16 | 70 | 0/0 | ? ? | - | 6520 | 11302 |
| reduce_arg_max_u32_cell | 32 | 44 | 0/0 | ? ? | - | 2189 | 2840 |
| reduce_arg_max_u64_cell | 64 | 66 | 0/17 | ? ? | - | 1803 | 2118 |
| reduce_arg_max_u8_cell | 8 | 109 | 0/33 | ? ? | - | 19021 | 27950 |
| reduce_arg_min_f16_cell | 16 | 337 | 0/64 | ? ? | - | 25019 | 43162 |
| reduce_arg_min_f32_cell | 32 | 51 | 0/0 | ? ? | - | 3113 | 3596 |
| reduce_arg_min_f64_cell | 64 | 90 | 0/12 | ? ? | - | 2187 | 3210 |
| reduce_arg_min_i16_cell | 16 | 68 | 0/0 | ? ? | - | 6349 | 10537 |
| reduce_arg_min_i32_cell | 32 | 74 | 2/0 | ? ? | - | 3610 | 5775 |
| reduce_arg_min_i64_cell | 64 | 65 | 0/16 | ? ? | - | 1426 | 2086 |
| reduce_arg_min_i8_cell | 8 | 110 | 0/34 | ? ? | - | 16978 | 28201 |
| reduce_arg_min_u16_cell | 16 | 70 | 0/0 | ? ? | - | 6520 | 11302 |
| reduce_arg_min_u32_cell | 32 | 44 | 0/0 | ? ? | - | 2189 | 2840 |
| reduce_arg_min_u64_cell | 64 | 65 | 0/16 | ? ? | - | 1426 | 2086 |
| reduce_arg_min_u8_cell | 8 | 109 | 0/31 | ? ? | - | 18514 | 27950 |
| reduce_max_bool_cell | 64 | 11 | 0/0 | ? ? | - | 181 | 357 |
| reduce_max_f16_cell | 64 | 154 | 0/0 | ? ? | - | 3315 | 6212 |
| reduce_max_f32_cell | 64 | 42 | 0/0 | ? ? | - | 1032 | 1668 |
| reduce_max_f64_cell | 64 | 42 | 0/0 | ? ? | - | 1032 | 1668 |
| reduce_max_i16_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_max_i32_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 371 |
| reduce_max_i64_cell | 64 | 35 | 0/0 | ? ? ? | - | 753 | 1124 |
| reduce_max_i8_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_max_u16_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_max_u32_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 371 |
| reduce_max_u64_cell | 64 | 35 | 0/0 | ? ? ? | - | 899 | 1412 |
| reduce_max_u8_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_mean_bool_cell | 8 | 34 | 0/0 | ? ? ? | - | 5949 | 10808 |
| reduce_mean_complex128_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 307 |
| reduce_mean_complex64_cell | 64 | 19 | 0/0 | ? ? | - | 519 | 614 |
| reduce_mean_f16_cell | 64 | 8534 | 12/96 | 32 ?[32] ? | - | 213002 | 328138 |
| reduce_mean_f32_cell | 128 | 23 | 0/0 | ? ? | - | 260 | 371 |
| reduce_mean_f64_cell | 128 | 23 | 0/0 | ? ? | - | 260 | 371 |
| reduce_mean_i16_cell | 16 | 11 | 0/0 | ? ? ? | - | 1175 | 1431 |
| reduce_mean_i32_cell | 32 | 11 | 0/0 | ? ? ? | - | 588 | 716 |
| reduce_mean_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 371 |
| reduce_mean_i8_cell | 8 | 40 | 0/0 | ? ? ? | - | 6979 | 12344 |
| reduce_mean_u16_cell | 16 | 11 | 0/0 | ? ? ? | - | 1175 | 1431 |
| reduce_mean_u32_cell | 32 | 11 | 0/0 | ? ? ? | - | 588 | 716 |
| reduce_mean_u8_cell | 8 | 34 | 0/0 | ? ? ? | - | 5949 | 10808 |
| reduce_min_bool_cell | 64 | 11 | 0/0 | ? ? | - | 181 | 357 |
| reduce_min_f16_cell | 64 | 154 | 0/0 | ? ? | - | 3421 | 6340 |
| reduce_min_f32_cell | 64 | 42 | 0/0 | ? ? | - | 1032 | 1668 |
| reduce_min_f64_cell | 64 | 42 | 0/0 | ? ? | - | 1032 | 1668 |
| reduce_min_i16_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_min_i32_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 371 |
| reduce_min_i64_cell | 64 | 35 | 0/0 | ? ? ? | - | 753 | 1124 |
| reduce_min_i8_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_min_u16_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_min_u32_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 371 |
| reduce_min_u64_cell | 64 | 35 | 0/0 | ? ? ? | - | 899 | 1412 |
| reduce_min_u8_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 187 | 371 |
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 1031 |
| reduce_product_complex64_cell | 64 | 55 | 0/0 | ? ? | - | 1606 | 1768 |
| reduce_product_f16_cell | 64 | 8534 | 12/96 | 32 ?[32] ? | - | 217098 | 326090 |
| reduce_product_f32_cell | 128 | 23 | 0/0 | ? ? | - | 356 | 371 |
| reduce_product_f64_cell | 128 | 23 | 0/0 | ? ? | - | 356 | 516 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 8215 |
| reduce_product_i32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 4108 |
| reduce_product_i64_cell | 64 | 11 | 0/0 | ? ? | - | 518 | 2054 |
| reduce_product_i8_cell | 8 | 33 | 0/0 | ? ? | - | 6467 | 16440 |
| reduce_product_u16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 8215 |
| reduce_product_u32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 4108 |
| reduce_product_u8_cell | 8 | 30 | 0/0 | ? ? | - | 5381 | 16440 |
| reduce_sum_f16_cell | 64 | 8534 | 12/96 | 32 ?[32] ? | - | 213002 | 328138 |
| reduce_variance_bool_cell | 8 | 87 | 10/0 | ? ? | - | 45292 | 58972 |
| reduce_variance_complex128_cell | 128 | 145 | 36/0 | ? ? | - | 5643 | 5736 |
| reduce_variance_complex64_cell | 64 | 151 | 39/0 | ? ? | - | 11286 | 12047 |
| reduce_variance_f16_cell | 16 | 508 | 10/1 | ? ? | - | 73756 | 105244 |
| reduce_variance_f32_cell | 32 | 63 | 0/0 | ? ? | - | 11311 | 10801 |
| reduce_variance_f64_cell | 64 | 53 | 0/0 | ? ? | - | 5654 | 5398 |
| reduce_variance_i16_cell | 16 | 64 | 10/0 | ? ? | - | 22628 | 24632 |
| reduce_variance_i32_cell | 32 | 64 | 10/0 | ? ? | - | 11313 | 12319 |
| reduce_variance_i64_cell | 64 | 73 | 10/0 | ? ? | - | 5658 | 5974 |
| reduce_variance_i8_cell | 8 | 62 | 10/0 | ? ? | - | 45245 | 47744 |
| reduce_variance_u16_cell | 16 | 67 | 10/0 | ? ? | - | 22628 | 27425 |
| reduce_variance_u32_cell | 32 | 74 | 10/0 | ? ? | - | 11311 | 13088 |
| reduce_variance_u64_cell | 64 | 96 | 10/0 | ? ? | - | 5660 | 7374 |
| reduce_variance_u8_cell | 8 | 79 | 12/0 | ? ? | - | 45297 | 57964 |
| **geomean** | | | | | | **2124** | **3417** |

Warnings:

- reduce_arg_max_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_arg_max_bool_cell: no loop whose iteration count is linear in the input length
- reduce_arg_max_bool_cell: out-of-line call to a Rust function in a loop: `<fearless_simd[c9fd74fa286dc94b]::generated::avx2::Avx2 as fearless_simd[c9fd74fa286dc94b]::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix[2d4446b3a7e8f4ab]::ops::reduction::fold_run_leaf<bool, jix[2d4446b3a7e8f4ab]::ops::reduction::ArgMaxKernel, true: bool>::{closure#1}, (u64, bool)>`
- reduce_arg_min_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_arg_min_bool_cell: no loop whose iteration count is linear in the input length
- reduce_arg_min_bool_cell: out-of-line call to a Rust function in a loop: `<fearless_simd[c9fd74fa286dc94b]::generated::avx2::Avx2 as fearless_simd[c9fd74fa286dc94b]::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix[2d4446b3a7e8f4ab]::ops::reduction::fold_run_leaf<bool, jix[2d4446b3a7e8f4ab]::ops::reduction::ArgMinKernel, true: bool>::{closure#1}, (u64, bool)>`

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | stack | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 128 | 15 | 0/0 | ? ? | - | 337 | 324 | 321 |
| reduce_any_bool_cell | 128 | 15 | 0/0 | ? ? | - | 337 | 324 | 321 |
| reduce_arg_max_bool_cell | 8 | 54 | 0/0 | ? ? | - | 11853 | 9088 | 5939 |
| reduce_arg_max_f16_cell | 16 | 291 | 0/53 | ? ? | - | 15078 | 13356 | 12500 |
| reduce_arg_max_f32_cell | 32 | 33 | 0/0 | ? ? | - | 1453 | 1681 | 1112 |
| reduce_arg_max_f64_cell | 64 | 35 | 0/0 | ? ? | - | 738 | 842 | 518 |
| reduce_arg_max_i16_cell | 16 | 31 | 0/0 | ? ? | - | 2821 | 2721 | 2079 |
| reduce_arg_max_i32_cell | 32 | 31 | 0/0 | ? ? | - | 1336 | 1376 | 1033 |
| reduce_arg_max_i64_cell | 64 | 31 | 0/0 | ? ? | - | 724 | 791 | 415 |
| reduce_arg_max_i8_cell | 8 | 36 | 0/0 | ? ? | - | 6287 | 6241 | 5627 |
| reduce_arg_max_u16_cell | 16 | 33 | 0/0 | ? ? | - | 2849 | 2867 | 2276 |
| reduce_arg_max_u32_cell | 32 | 33 | 0/0 | ? ? | - | 1423 | 1467 | 1096 |
| reduce_arg_max_u64_cell | 64 | 35 | 0/0 | ? ? | - | 791 | 865 | 472 |
| reduce_arg_max_u8_cell | 8 | 38 | 0/0 | ? ? | - | 6743 | 7020 | 5484 |
| reduce_arg_min_bool_cell | 8 | 38 | 0/0 | ? ? | - | 6743 | 7020 | 5484 |
| reduce_arg_min_f16_cell | 16 | 337 | 0/64 | ? ? | - | 17695 | 14886 | 14405 |
| reduce_arg_min_f32_cell | 32 | 33 | 0/0 | ? ? | - | 1453 | 1681 | 1112 |
| reduce_arg_min_f64_cell | 64 | 35 | 0/0 | ? ? | - | 738 | 842 | 518 |
| reduce_arg_min_i16_cell | 16 | 31 | 0/0 | ? ? | - | 2821 | 2721 | 2079 |
| reduce_arg_min_i32_cell | 32 | 31 | 0/0 | ? ? | - | 1336 | 1376 | 1033 |
| reduce_arg_min_i64_cell | 64 | 31 | 0/0 | ? ? | - | 724 | 791 | 415 |
| reduce_arg_min_i8_cell | 8 | 36 | 0/0 | ? ? | - | 6287 | 6241 | 5627 |
| reduce_arg_min_u16_cell | 16 | 33 | 0/0 | ? ? | - | 2849 | 2867 | 2276 |
| reduce_arg_min_u32_cell | 32 | 33 | 0/0 | ? ? | - | 1423 | 1467 | 1096 |
| reduce_arg_min_u64_cell | 64 | 35 | 0/0 | ? ? | - | 791 | 865 | 472 |
| reduce_arg_min_u8_cell | 8 | 38 | 0/0 | ? ? | - | 6743 | 7020 | 5484 |
| reduce_max_bool_cell | 128 | 7 | 0/0 | ? ? | - | 67 | 99 | 67 |
| reduce_max_f16_cell | 128 | 152 | 6/0 | ? ? | - | 2116 | 2244 | 1198 |
| reduce_max_f32_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_max_f64_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_max_i16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_max_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_max_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 162 | 258 | 74 |
| reduce_max_i8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_max_u16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_max_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_max_u64_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 218 | 266 | 132 |
| reduce_max_u8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
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
| reduce_min_bool_cell | 128 | 7 | 0/0 | ? ? | - | 67 | 99 | 67 |
| reduce_min_f16_cell | 128 | 149 | 4/0 | ? ? | - | 2020 | 2213 | 1222 |
| reduce_min_f32_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_min_f64_cell | 128 | 31 | 0/0 | ? ? | - | 303 | 346 | 259 |
| reduce_min_i16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_min_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_min_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 162 | 258 | 74 |
| reduce_min_i8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_min_u16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_min_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
| reduce_min_u64_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 218 | 266 | 132 |
| reduce_min_u8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 98 | 66 |
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
| **geomean** | | | | | | **823** | **883** | **698** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | stack | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 256 | 11 | 0/0 | ? ? | - | 193 | 65 | 129 |
| reduce_any_bool_cell | 256 | 15 | 0/0 | ? ? | - | 193 | 209 | 160 |
| reduce_arg_max_bool_cell | 16 | 14 | 0/0 | ? ? | - | 2957 | 1715 | 2074 |
| reduce_arg_max_f16_cell | 32 | 703 | 3/204 | ?[16] 8 ? | - | 18710 | 16284 | 16275 |
| reduce_arg_max_f32_cell | 64 | 16 | 0/0 | ? ? | - | 523 | 525 | 347 |
| reduce_arg_max_f64_cell | 128 | 20 | 0/0 | ? ? | - | 322 | 337 | 226 |
| reduce_arg_max_i16_cell | 32 | 14 | 0/0 | ? ? | - | 832 | 859 | 524 |
| reduce_arg_max_i32_cell | 64 | 14 | 0/0 | ? ? | - | 460 | 461 | 278 |
| reduce_arg_max_i64_cell | 128 | 16 | 0/0 | ? ? | - | 210 | 252 | 169 |
| reduce_arg_max_i8_cell | 16 | 14 | 0/0 | ? ? | - | 1644 | 1715 | 1047 |
| reduce_arg_max_u16_cell | 32 | 14 | 0/0 | ? ? | - | 832 | 859 | 524 |
| reduce_arg_max_u32_cell | 64 | 14 | 0/0 | ? ? | - | 460 | 461 | 292 |
| reduce_arg_max_u64_cell | 128 | 16 | 0/0 | ? ? | - | 210 | 252 | 186 |
| reduce_arg_max_u8_cell | 16 | 14 | 0/0 | ? ? | - | 1644 | 1715 | 1047 |
| reduce_arg_min_bool_cell | 16 | 14 | 0/0 | ? ? | - | 1644 | 1715 | 1047 |
| reduce_arg_min_f16_cell | 32 | 797 | 3/204 | ?[16] 8 ? | - | 22166 | 18342 | 18196 |
| reduce_arg_min_f32_cell | 64 | 16 | 0/0 | ? ? | - | 523 | 525 | 347 |
| reduce_arg_min_f64_cell | 128 | 20 | 0/0 | ? ? | - | 322 | 337 | 226 |
| reduce_arg_min_i16_cell | 32 | 14 | 0/0 | ? ? | - | 832 | 859 | 524 |
| reduce_arg_min_i32_cell | 64 | 14 | 0/0 | ? ? | - | 460 | 461 | 278 |
| reduce_arg_min_i64_cell | 128 | 16 | 0/0 | ? ? | - | 210 | 252 | 169 |
| reduce_arg_min_i8_cell | 16 | 14 | 0/0 | ? ? | - | 1644 | 1715 | 1047 |
| reduce_arg_min_u16_cell | 32 | 14 | 0/0 | ? ? | - | 832 | 859 | 524 |
| reduce_arg_min_u32_cell | 64 | 14 | 0/0 | ? ? | - | 460 | 461 | 292 |
| reduce_arg_min_u64_cell | 128 | 16 | 0/0 | ? ? | - | 210 | 252 | 186 |
| reduce_arg_min_u8_cell | 16 | 14 | 0/0 | ? ? | - | 1644 | 1715 | 1047 |
| reduce_max_bool_cell | 256 | 7 | 0/0 | ? ? | - | 34 | 50 | 34 |
| reduce_max_f16_cell | 256 | 121 | 24/0 | ?[4] ? | - | 645 | 615 | 516 |
| reduce_max_f32_cell | 256 | 31 | 0/0 | ? ? ? | - | 205 | 162 | 193 |
| reduce_max_f64_cell | 256 | 31 | 0/0 | ? ? ? | - | 205 | 162 | 193 |
| reduce_max_i16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_i32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_max_i8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_max_u8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_mean_bool_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_mean_complex128_cell | 256 | 19 | 0/0 | ? ? ? | - | 130 | 130 | 130 |
| reduce_mean_complex64_cell | 256 | 7 | 0/0 | ? ? ? | - | 66 | 82 | 66 |
| reduce_mean_f16_cell | 256 | 1693 | 404/0 | 64 ?[128] 32 16 ? | - | 10326 | 10250 | 8197 |
| reduce_mean_f32_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 81 | 65 |
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
| reduce_min_i16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_i32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_min_i8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 160 | 81 | 33 |
| reduce_min_u8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_product_complex128_cell | 256 | 162 | 48/0 | ? ? | - | 617 | 524 | 587 |
| reduce_product_complex64_cell | 256 | 39 | 0/0 | ? ? | - | 307 | 276 | 307 |
| reduce_product_f16_cell | 256 | 1693 | 404/0 | 64 ?[128] 32 16 ? | - | 10326 | 10250 | 7365 |
| reduce_product_f32_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 97 | 65 |
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
| **geomean** | | | | | | **480** | **457** | **342** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | skylake |
|---|---:|---:|---:|---|---|---:|
| reduce_arg_max_bool_cell | 8 | 142 | 17/25 | ? ? | - | 16507 |
| reduce_arg_max_f16_cell | 16 | 472 | 0/207 | ? ? | - | 21798 |
| reduce_arg_max_f32_cell | 32 | 81 | 21/0 | ? ? | - | 2904 |
| reduce_arg_max_f64_cell | 64 | 103 | 33/0 | ? ? | - | 1631 |
| reduce_arg_max_i16_cell | 16 | 65 | 14/0 | ? ? | - | 5678 |
| reduce_arg_max_i32_cell | 32 | 70 | 18/0 | ? ? | - | 2835 |
| reduce_arg_max_i64_cell | 64 | 140 | 46/0 | ? ? | - | 2191 |
| reduce_arg_max_i8_cell | 8 | 183 | 26/46 | ? ? | - | 18028 |
| reduce_arg_max_u16_cell | 16 | 69 | 12/0 | ? ? | - | 5844 |
| reduce_arg_max_u32_cell | 32 | 79 | 20/0 | ? ? | - | 2964 |
| reduce_arg_max_u64_cell | 64 | 200 | 0/109 | ? ? | - | 3139 |
| reduce_arg_max_u8_cell | 8 | 183 | 26/46 | ? ? | - | 19041 |
| reduce_arg_min_bool_cell | 8 | 143 | 17/25 | ? ? | - | 17741 |
| reduce_arg_min_f16_cell | 16 | 551 | 0/229 | ? ? | - | 23831 |
| reduce_arg_min_f32_cell | 32 | 82 | 21/0 | ? ? | - | 2880 |
| reduce_arg_min_f64_cell | 64 | 103 | 33/0 | ? ? | - | 1678 |
| reduce_arg_min_i16_cell | 16 | 67 | 13/0 | ? ? | - | 5670 |
| reduce_arg_min_i32_cell | 32 | 74 | 18/0 | ? ? | - | 2712 |
| reduce_arg_min_i64_cell | 64 | 139 | 49/0 | ? ? | - | 2194 |
| reduce_arg_min_i8_cell | 8 | 183 | 26/46 | ? ? | - | 18028 |
| reduce_arg_min_u16_cell | 16 | 71 | 17/0 | ? ? | - | 5847 |
| reduce_arg_min_u32_cell | 32 | 77 | 21/0 | ? ? | - | 2756 |
| reduce_arg_min_u64_cell | 64 | 208 | 0/118 | ? ? | - | 3205 |
| reduce_arg_min_u8_cell | 8 | 183 | 26/46 | ? ? | - | 19041 |
| reduce_max_f16_cell | 64 | 213 | 27/0 | ? ? | - | 3532 |
| reduce_max_f32_cell | 64 | 51 | 0/0 | ? ? | - | 945 |
| reduce_max_f64_cell | 64 | 51 | 0/0 | ? ? | - | 945 |
| reduce_max_i16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 134 |
| reduce_max_i32_cell | 64 | 30 | 5/0 | ? ? ? | - | 520 |
| reduce_max_i64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_max_i8_cell | 64 | 30 | 5/0 | ? ? ? ? | - | 520 |
| reduce_max_u16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 261 |
| reduce_max_u32_cell | 64 | 45 | 5/0 | ? ? ? | - | 670 |
| reduce_max_u64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_max_u8_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 134 |
| reduce_mean_bool_cell | 8 | 31 | 3/0 | ? ? ? | - | 4009 |
| reduce_mean_complex128_cell | 128 | 27 | 4/0 | ? ? | - | 196 |
| reduce_mean_complex64_cell | 64 | 27 | 4/0 | ? ? | - | 391 |
| reduce_mean_f16_cell | 64 | 8580 | 16/166 | 32 ?[32] ? | - | 172495 |
| reduce_mean_f32_cell | 64 | 11 | 0/0 | ? ? | - | 262 |
| reduce_mean_f64_cell | 64 | 11 | 0/0 | ? ? | - | 262 |
| reduce_mean_i16_cell | 16 | 31 | 0/0 | ? ? ? | - | 2071 |
| reduce_mean_i32_cell | 32 | 23 | 0/0 | ? ? ? | - | 525 |
| reduce_mean_i64_cell | 64 | 11 | 0/0 | ? ? ? | - | 134 |
| reduce_mean_i8_cell | 8 | 51 | 4/0 | ? ? ? | - | 7726 |
| reduce_mean_u16_cell | 16 | 19 | 0/0 | ? ? ? | - | 2071 |
| reduce_mean_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 524 |
| reduce_mean_u8_cell | 8 | 31 | 3/0 | ? ? ? | - | 4009 |
| reduce_min_f16_cell | 64 | 214 | 22/0 | ? ? | - | 3617 |
| reduce_min_f32_cell | 64 | 51 | 0/0 | ? ? | - | 945 |
| reduce_min_f64_cell | 64 | 51 | 0/0 | ? ? | - | 945 |
| reduce_min_i16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 134 |
| reduce_min_i32_cell | 64 | 33 | 2/0 | ? ? ? | - | 520 |
| reduce_min_i64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_min_i8_cell | 64 | 33 | 2/0 | ? ? ? ? | - | 520 |
| reduce_min_u16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 262 |
| reduce_min_u32_cell | 64 | 51 | 4/0 | ? ? ? | - | 817 |
| reduce_min_u64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_min_u8_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 134 |
| reduce_product_complex128_cell | 128 | 84 | 20/0 | ? ? | - | 648 |
| reduce_product_complex64_cell | 64 | 111 | 37/0 | ? ? | - | 2252 |
| reduce_product_f16_cell | 64 | 8580 | 16/166 | 32 ?[32] ? | - | 172495 |
| reduce_product_f32_cell | 64 | 11 | 0/0 | ? ? | - | 262 |
| reduce_product_f64_cell | 64 | 11 | 0/0 | ? ? | - | 262 |
| reduce_product_i16_cell | 16 | 120 | 0/54 | ? ? | - | 6464 |
| reduce_product_i32_cell | 32 | 109 | 0/41 | ? ? | - | 3098 |
| reduce_product_i64_cell | 64 | 96 | 0/39 | ? ? ? | - | 1551 |
| reduce_product_i8_cell | 16 | 172 | 40/0 | ? ? ? | - | 10967 |
| reduce_product_u16_cell | 16 | 83 | 0/45 | ? ? | - | 5435 |
| reduce_product_u32_cell | 32 | 78 | 0/40 | ? ? | - | 2703 |
| reduce_product_u8_cell | 16 | 96 | 26/0 | ? ? ? | - | 5937 |
| reduce_sum_f16_cell | 64 | 8580 | 16/166 | 32 ?[32] ? | - | 172495 |
| reduce_variance_bool_cell | 8 | 154 | 57/4 | ? ? | - | 21294 |
| reduce_variance_complex128_cell | 128 | 167 | 76/2 | ? ? | - | 1496 |
| reduce_variance_complex64_cell | 64 | 175 | 81/2 | ? ? | - | 3166 |
| reduce_variance_f16_cell | 16 | 560 | 47/60 | ? ? | - | 34337 |
| reduce_variance_f32_cell | 32 | 83 | 23/2 | ? ? | - | 3288 |
| reduce_variance_f64_cell | 64 | 72 | 21/2 | ? ? | - | 1563 |
| reduce_variance_i16_cell | 16 | 73 | 31/2 | ? ? | - | 6520 |
| reduce_variance_i32_cell | 32 | 78 | 32/2 | ? ? | - | 3532 |
| reduce_variance_i64_cell | 64 | 106 | 50/18 | ? ? | - | 2520 |
| reduce_variance_i8_cell | 8 | 118 | 39/5 | ? ? | - | 19692 |
| reduce_variance_u16_cell | 16 | 86 | 38/2 | ? ? | - | 7660 |
| reduce_variance_u32_cell | 32 | 91 | 36/2 | ? ? | - | 3389 |
| reduce_variance_u64_cell | 64 | 110 | 37/2 | ? ? | - | 1928 |
| reduce_variance_u8_cell | 8 | 106 | 38/2 | ? ? | - | 16609 |
| **geomean** | | | | | | **2424** |

Warnings:

- reduce_all_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_all_bool_cell: no loop whose iteration count is linear in the input length
- reduce_any_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_any_bool_cell: no loop whose iteration count is linear in the input length
- reduce_max_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_max_bool_cell: no loop whose iteration count is linear in the input length
- reduce_min_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_min_bool_cell: no loop whose iteration count is linear in the input length

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 64 | 17 | 0/0 | ? ? | - | 583 | 518 | 390 |
| reduce_any_bool_cell | 64 | 17 | 0/0 | ? ? | - | 583 | 518 | 390 |
| reduce_arg_max_bool_cell | 8 | 57 | 0/0 | ? ? | - | 12570 | 10020 | 4997 |
| reduce_arg_max_f16_cell | 16 | 225 | 0/8 | ? ? | - | 20797 | 20795 | 10404 |
| reduce_arg_max_f32_cell | 32 | 71 | 0/0 | ? ? | - | 3274 | 3593 | 2063 |
| reduce_arg_max_f64_cell | 64 | 71 | 0/0 | ? ? | - | 1637 | 1796 | 1032 |
| reduce_arg_max_i16_cell | 16 | 51 | 0/0 | ? ? | - | 5391 | 4362 | 2191 |
| reduce_arg_max_i32_cell | 32 | 39 | 0/0 | ? ? | - | 2186 | 1670 | 946 |
| reduce_arg_max_i64_cell | 64 | 39 | 0/0 | ? ? | - | 1093 | 962 | 468 |
| reduce_arg_max_i8_cell | 8 | 60 | 0/1 | ? ? | - | 14879 | 10532 | 5263 |
| reduce_arg_max_u16_cell | 16 | 50 | 0/0 | ? ? | - | 5391 | 4365 | 2194 |
| reduce_arg_max_u32_cell | 32 | 39 | 0/0 | ? ? | - | 2186 | 1670 | 946 |
| reduce_arg_max_u64_cell | 64 | 39 | 0/0 | ? ? | - | 1093 | 962 | 468 |
| reduce_arg_max_u8_cell | 8 | 59 | 0/0 | ? ? | - | 14874 | 10271 | 5181 |
| reduce_arg_min_bool_cell | 8 | 59 | 0/0 | ? ? | - | 14874 | 10271 | 5181 |
| reduce_arg_min_f16_cell | 16 | 229 | 0/8 | ? ? | - | 19604 | 19604 | 9807 |
| reduce_arg_min_f32_cell | 32 | 71 | 0/0 | ? ? | - | 3274 | 3593 | 2063 |
| reduce_arg_min_f64_cell | 64 | 71 | 0/0 | ? ? | - | 1637 | 1796 | 1032 |
| reduce_arg_min_i16_cell | 16 | 51 | 0/0 | ? ? | - | 5391 | 4362 | 2191 |
| reduce_arg_min_i32_cell | 32 | 39 | 0/0 | ? ? | - | 2186 | 1670 | 946 |
| reduce_arg_min_i64_cell | 64 | 39 | 0/0 | ? ? | - | 1093 | 962 | 468 |
| reduce_arg_min_i8_cell | 8 | 60 | 0/1 | ? ? | - | 14879 | 10532 | 5263 |
| reduce_arg_min_u16_cell | 16 | 50 | 0/0 | ? ? | - | 5391 | 4365 | 2194 |
| reduce_arg_min_u32_cell | 32 | 39 | 0/0 | ? ? | - | 2186 | 1670 | 946 |
| reduce_arg_min_u64_cell | 64 | 39 | 0/0 | ? ? | - | 1093 | 962 | 468 |
| reduce_arg_min_u8_cell | 8 | 59 | 0/0 | ? ? | - | 14874 | 10271 | 5181 |
| reduce_max_bool_cell | 64 | 9 | 0/0 | ? ? | - | 326 | 326 | 166 |
| reduce_max_f16_cell | 64 | 162 | 6/0 | ?[4] ? | - | 4954 | 4630 | 2196 |
| reduce_max_f32_cell | 64 | 29 | 0/0 | ? ? | - | 1542 | 774 | 454 |
| reduce_max_f64_cell | 64 | 29 | 0/0 | ? ? | - | 1542 | 774 | 454 |
| reduce_max_i16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_max_i32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_max_i64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_max_i8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_max_u16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_max_u32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_max_u64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_max_u8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_mean_bool_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 5207 | 3640 | 1894 |
| reduce_mean_complex128_cell | 128 | 15 | 0/0 | ? ? ? | - | 418 | 291 | 131 |
| reduce_mean_complex64_cell | 64 | 15 | 0/0 | ? ? ? | - | 454 | 580 | 219 |
| reduce_mean_f16_cell | 64 | 6831 | 6/0 | 32 ?[32] ? | - | 172634 | 147473 | 74076 |
| reduce_mean_f32_cell | 64 | 9 | 0/0 | ? ? ? | - | 454 | 326 | 166 |
| reduce_mean_f64_cell | 64 | 9 | 0/0 | ? ? ? | - | 454 | 326 | 166 |
| reduce_mean_i16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 1306 | 1810 | 668 |
| reduce_mean_i32_cell | 32 | 9 | 0/0 | ? ? ? ? | - | 522 | 649 | 268 |
| reduce_mean_i64_cell | 64 | 9 | 0/0 | ? ? | - | 326 | 326 | 166 |
| reduce_mean_i8_cell | 8 | 11 | 0/0 | ? ? ? ? | - | 6216 | 4163 | 2120 |
| reduce_mean_u16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 1306 | 1810 | 668 |
| reduce_mean_u32_cell | 32 | 9 | 0/0 | ? ? ? ? | - | 522 | 649 | 268 |
| reduce_mean_u8_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 5207 | 3640 | 1894 |
| reduce_min_bool_cell | 64 | 9 | 0/0 | ? ? | - | 326 | 326 | 166 |
| reduce_min_f16_cell | 64 | 162 | 6/0 | ?[4] ? | - | 4922 | 4631 | 2196 |
| reduce_min_f32_cell | 64 | 29 | 0/0 | ? ? | - | 1542 | 774 | 454 |
| reduce_min_f64_cell | 64 | 29 | 0/0 | ? ? | - | 1542 | 774 | 454 |
| reduce_min_i16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_min_i32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_min_i64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_min_i8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_min_u16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_min_u32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_min_u64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_min_u8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 326 | 326 | 166 |
| reduce_product_complex128_cell | 128 | 30 | 0/0 | ? ? | - | 901 | 522 | 261 |
| reduce_product_complex64_cell | 64 | 30 | 0/0 | ? ? | - | 941 | 1033 | 524 |
| reduce_product_f16_cell | 64 | 6831 | 6/0 | 32 ?[32] ? | - | 172634 | 147475 | 74077 |
| reduce_product_f32_cell | 64 | 9 | 0/0 | ? ? | - | 454 | 326 | 198 |
| reduce_product_f64_cell | 64 | 9 | 0/0 | ? ? | - | 454 | 326 | 198 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? ? | - | 2076 | 6164 | 1044 |
| reduce_product_i32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1293 | 3078 | 906 |
| reduce_product_i64_cell | 64 | 15 | 0/0 | ? ? ? | - | 519 | 1541 | 261 |
| reduce_product_i8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 | 12334 | 2094 |
| reduce_product_u16_cell | 16 | 18 | 0/0 | ? ? ? | - | 2076 | 6164 | 1044 |
| reduce_product_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1038 | 3078 | 522 |
| reduce_product_u8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 | 12334 | 2094 |
| reduce_sum_f16_cell | 64 | 6831 | 6/0 | 32 ?[32] ? | - | 172634 | 147473 | 74076 |
| reduce_variance_bool_cell | 8 | 51 | 0/0 | ? ? | - | 135388 | 18949 | 15985 |
| reduce_variance_complex128_cell | 128 | 78 | 6/0 | ? ? | - | 16465 | 2186 | 1833 |
| reduce_variance_complex64_cell | 64 | 82 | 4/0 | ? ? | - | 33063 | 4563 | 3983 |
| reduce_variance_f16_cell | 16 | 401 | 29/0 | ? ? | - | 95759 | 48655 | 18450 |
| reduce_variance_f32_cell | 32 | 34 | 0/0 | ? ? | - | 33832 | 4576 | 3965 |
| reduce_variance_f64_cell | 64 | 31 | 0/0 | ? ? | - | 16914 | 1954 | 1805 |
| reduce_variance_i16_cell | 16 | 40 | 0/0 | ? ? | - | 67681 | 9285 | 8013 |
| reduce_variance_i32_cell | 32 | 38 | 0/0 | ? ? | - | 33839 | 4576 | 3999 |
| reduce_variance_i64_cell | 64 | 37 | 0/0 | ? ? | - | 16919 | 2287 | 1982 |
| reduce_variance_i8_cell | 8 | 50 | 0/0 | ? ? | - | 135393 | 21637 | 14479 |
| reduce_variance_u16_cell | 16 | 50 | 0/0 | ? ? | - | 67702 | 8415 | 8397 |
| reduce_variance_u32_cell | 32 | 45 | 0/0 | ? ? | - | 33843 | 4582 | 3972 |
| reduce_variance_u64_cell | 64 | 48 | 0/0 | ? ? | - | 16923 | 2573 | 2033 |
| reduce_variance_u8_cell | 8 | 50 | 0/0 | ? ? | - | 135393 | 21381 | 14479 |
| **geomean** | | | | | | **2789** | **2007** | **1023** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | stack | loops | calls | apple-m1 |
|---|---:|---:|---:|---|---|---:|
| reduce_all_bool_cell | 64 | 17 | 0/0 | ? ? | - | 453 |
| reduce_any_bool_cell | 64 | 17 | 0/0 | ? ? | - | 453 |
| reduce_arg_max_bool_cell | 8 | 57 | 0/0 | ? ? | - | 13860 |
| reduce_arg_max_f16_cell | 16 | 224 | 0/9 | ? ? | - | 22548 |
| reduce_arg_max_f32_cell | 32 | 71 | 0/0 | ? ? | - | 2066 |
| reduce_arg_max_f64_cell | 64 | 71 | 0/0 | ? ? | - | 1033 |
| reduce_arg_max_i16_cell | 16 | 51 | 0/0 | ? ? | - | 5130 |
| reduce_arg_max_i32_cell | 32 | 39 | 0/0 | ? ? | - | 2056 |
| reduce_arg_max_i64_cell | 64 | 39 | 0/0 | ? ? | - | 1028 |
| reduce_arg_max_i8_cell | 8 | 60 | 0/1 | ? ? | - | 16415 |
| reduce_arg_max_u16_cell | 16 | 50 | 0/0 | ? ? | - | 4111 |
| reduce_arg_max_u32_cell | 32 | 39 | 0/0 | ? ? | - | 2056 |
| reduce_arg_max_u64_cell | 64 | 39 | 0/0 | ? ? | - | 1028 |
| reduce_arg_max_u8_cell | 8 | 60 | 0/1 | ? ? | - | 14116 |
| reduce_arg_min_bool_cell | 8 | 60 | 0/1 | ? ? | - | 14116 |
| reduce_arg_min_f16_cell | 16 | 222 | 0/9 | ? ? | - | 21770 |
| reduce_arg_min_f32_cell | 32 | 71 | 0/0 | ? ? | - | 2066 |
| reduce_arg_min_f64_cell | 64 | 71 | 0/0 | ? ? | - | 1033 |
| reduce_arg_min_i16_cell | 16 | 51 | 0/0 | ? ? | - | 5130 |
| reduce_arg_min_i32_cell | 32 | 39 | 0/0 | ? ? | - | 2056 |
| reduce_arg_min_i64_cell | 64 | 39 | 0/0 | ? ? | - | 1028 |
| reduce_arg_min_i8_cell | 8 | 60 | 0/1 | ? ? | - | 16415 |
| reduce_arg_min_u16_cell | 16 | 50 | 0/0 | ? ? | - | 4111 |
| reduce_arg_min_u32_cell | 32 | 39 | 0/0 | ? ? | - | 2056 |
| reduce_arg_min_u64_cell | 64 | 39 | 0/0 | ? ? | - | 1028 |
| reduce_arg_min_u8_cell | 8 | 60 | 0/1 | ? ? | - | 14116 |
| reduce_max_bool_cell | 64 | 9 | 0/0 | ? ? | - | 133 |
| reduce_max_f16_cell | 64 | 162 | 6/0 | ?[4] ? | - | 2739 |
| reduce_max_f32_cell | 64 | 29 | 0/0 | ? ? | - | 519 |
| reduce_max_f64_cell | 64 | 29 | 0/0 | ? ? | - | 520 |
| reduce_max_i16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_max_i32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_max_i64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_max_i8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_max_u16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_max_u32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_max_u64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_max_u8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_mean_bool_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 3604 |
| reduce_mean_complex128_cell | 128 | 15 | 0/0 | ? ? ? | - | 163 |
| reduce_mean_complex64_cell | 64 | 15 | 0/0 | ? ? ? | - | 262 |
| reduce_mean_f16_cell | 64 | 302 | 6/0 | 32 ?[32] ? | - | 4364 |
| reduce_mean_f32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 261 |
| reduce_mean_f64_cell | 64 | 9 | 0/0 | ? ? ? | - | 325 |
| reduce_mean_i16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 709 |
| reduce_mean_i32_cell | 128 | 28 | 0/0 | ? ? ? ? ? | - | 258 |
| reduce_mean_i64_cell | 256 | 28 | 0/0 | ? ? ? | - | 98 |
| reduce_mean_i8_cell | 8 | 11 | 0/0 | ? ? ? ? | - | 3599 |
| reduce_mean_u16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 709 |
| reduce_mean_u32_cell | 128 | 28 | 0/0 | ? ? ? ? ? | - | 258 |
| reduce_mean_u8_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 3604 |
| reduce_min_bool_cell | 64 | 9 | 0/0 | ? ? | - | 133 |
| reduce_min_f16_cell | 64 | 162 | 6/0 | ?[4] ? | - | 2708 |
| reduce_min_f32_cell | 64 | 29 | 0/0 | ? ? | - | 519 |
| reduce_min_f64_cell | 64 | 29 | 0/0 | ? ? | - | 520 |
| reduce_min_i16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_min_i32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_min_i64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_min_i8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_min_u16_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_min_u32_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_min_u64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_min_u8_cell | 64 | 9 | 0/0 | ? ? ? ? | - | 197 |
| reduce_product_complex128_cell | 128 | 30 | 0/0 | ? ? | - | 434 |
| reduce_product_complex64_cell | 64 | 30 | 0/0 | ? ? | - | 714 |
| reduce_product_f16_cell | 64 | 302 | 6/0 | 32 ?[32] ? | - | 4365 |
| reduce_product_f32_cell | 64 | 9 | 0/0 | ? ? | - | 133 |
| reduce_product_f64_cell | 64 | 9 | 0/0 | ? ? | - | 325 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? ? | - | 2076 |
| reduce_product_i32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1038 |
| reduce_product_i64_cell | 64 | 15 | 0/0 | ? ? ? | - | 519 |
| reduce_product_i8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 |
| reduce_product_u16_cell | 16 | 18 | 0/0 | ? ? ? | - | 2076 |
| reduce_product_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1038 |
| reduce_product_u8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 |
| reduce_sum_f16_cell | 64 | 302 | 6/0 | 32 ?[32] ? | - | 4364 |
| reduce_variance_bool_cell | 8 | 51 | 0/0 | ? ? | - | 9062 |
| reduce_variance_complex128_cell | 128 | 66 | 0/0 | ? ? | - | 723 |
| reduce_variance_complex64_cell | 64 | 74 | 0/0 | ? ? | - | 1616 |
| reduce_variance_f16_cell | 16 | 53 | 0/0 | ? ? | - | 4050 |
| reduce_variance_f32_cell | 32 | 34 | 0/0 | ? ? | - | 1641 |
| reduce_variance_f64_cell | 64 | 32 | 0/0 | ? ? | - | 820 |
| reduce_variance_i16_cell | 16 | 40 | 0/0 | ? ? | - | 3407 |
| reduce_variance_i32_cell | 32 | 38 | 0/0 | ? ? | - | 1641 |
| reduce_variance_i64_cell | 64 | 36 | 0/0 | ? ? | - | 820 |
| reduce_variance_i8_cell | 8 | 50 | 0/0 | ? ? | - | 8305 |
| reduce_variance_u16_cell | 16 | 50 | 0/0 | ? ? | - | 3674 |
| reduce_variance_u32_cell | 32 | 46 | 0/0 | ? ? | - | 1760 |
| reduce_variance_u64_cell | 64 | 48 | 0/0 | ? ? | - | 978 |
| reduce_variance_u8_cell | 8 | 50 | 0/0 | ? ? | - | 8387 |
| **geomean** | | | | | | **1127** |
