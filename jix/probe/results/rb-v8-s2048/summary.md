# llvm-mca summary: `rb-v8-s2048`

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

## Overview

| kernel | x86_64 sandybridge | x86_64 skylake | x86_64 znver3 | x86_64-v2 sandybridge | x86_64-v2 btver2 | x86_64-v3 skylake | x86_64-v3 alderlake | x86_64-v3 znver3 | x86_64-v4 icelake-server | x86_64-v4 sapphirerapids | x86_64-v4 znver4 | i686 skylake | aarch64 cortex-a72 | aarch64 neoverse-n1 | aarch64 neoverse-v2 | aarch64-apple apple-m1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reduce_all_bool_cell | 298 | 259 | 196 | 298 | 563 | 217 | 225 | 193 | 97 | 65 | 64 | 355 | 451 | 451 | 227 | 269 |
| reduce_any_bool_cell | 298 | 259 | 196 | 298 | 563 | 217 | 225 | 193 | 129 | 137 | 96 | 355 | 451 | 451 | 227 | 269 |
| reduce_arg_max_bool_cell | 13911 | 9068 | 8653 | - | - | 5274 | 5509 | 4682 | 21263 | 20900 | 24883 | 16507 | 12570 | 10020 | 4997 | 13860 |
| reduce_arg_max_f16_cell | 21302 | 15078 | 12500 | 21302 | 37274 | 19990 | 16987 | 15895 | 19531 | 16463 | 18060 | 21798 | 20797 | 20795 | 10404 | 22548 |
| reduce_arg_max_f32_cell | 2368 | 1517 | 1083 | 3012 | 3594 | 1416 | 1490 | 1099 | 6597 | 6281 | 7393 | 2904 | 3274 | 3593 | 2063 | 2066 |
| reduce_arg_max_f64_cell | 1928 | 1135 | 820 | 2124 | 3210 | 755 | 807 | 518 | 2098 | 1891 | 1586 | 1631 | 1637 | 1796 | 1032 | 1033 |
| reduce_arg_max_i16_cell | 3607 | 2202 | 1715 | 6397 | 10537 | 2447 | 2589 | 1940 | 8647 | 9421 | 12440 | 5678 | 5391 | 4362 | 2191 | 5130 |
| reduce_arg_max_i32_cell | 2852 | 1501 | 1428 | 3421 | 5775 | 1386 | 1357 | 1003 | 4261 | 4296 | 4974 | 2835 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_max_i64_cell | 1426 | 751 | 714 | 1426 | 2086 | 669 | 742 | 453 | 492 | 434 | 337 | 2191 | 1093 | 962 | 468 | 1028 |
| reduce_arg_max_i8_cell | 16978 | 10839 | 9467 | 16978 | 28201 | 4920 | 5361 | 4293 | 20622 | 20893 | 25738 | 18028 | 14879 | 10532 | 5263 | 16415 |
| reduce_arg_max_u16_cell | 3866 | 2394 | 1772 | 6520 | 11302 | 2518 | 2670 | 2125 | 9960 | 9806 | 12504 | 5844 | 5391 | 4365 | 2194 | 4111 |
| reduce_arg_max_u32_cell | 3606 | 2138 | 1432 | 2189 | 2840 | 1361 | 1430 | 1078 | 4807 | 4487 | 4974 | 2964 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_max_u64_cell | 1803 | 1069 | 716 | 1803 | 2118 | 734 | 799 | 500 | 492 | 434 | 338 | 3139 | 1093 | 962 | 468 | 1028 |
| reduce_arg_max_u8_cell | 17495 | 11351 | 8955 | 19021 | 27950 | 5135 | 5496 | 4483 | 21519 | 20892 | 25352 | 19041 | 14874 | 10271 | 5181 | 14116 |
| reduce_arg_min_bool_cell | 18007 | 11351 | 9364 | - | - | 5135 | 5496 | 4483 | 20749 | 20893 | 25990 | 17741 | 14874 | 10271 | 5181 | 14116 |
| reduce_arg_min_f16_cell | 25019 | 17695 | 14405 | 25019 | 43162 | 23958 | 19107 | 18417 | 23435 | 18584 | 20347 | 23831 | 19604 | 19604 | 9807 | 21770 |
| reduce_arg_min_f32_cell | 2368 | 1517 | 1083 | 3113 | 3596 | 1416 | 1490 | 1099 | 6597 | 6281 | 7393 | 2880 | 3274 | 3593 | 2063 | 2066 |
| reduce_arg_min_f64_cell | 2026 | 1163 | 842 | 2187 | 3210 | 755 | 807 | 518 | 2098 | 1891 | 1586 | 1678 | 1637 | 1796 | 1032 | 1033 |
| reduce_arg_min_i16_cell | 3612 | 2199 | 1715 | 6349 | 10537 | 2447 | 2589 | 1940 | 8647 | 9421 | 12440 | 5670 | 5391 | 4362 | 2191 | 5130 |
| reduce_arg_min_i32_cell | 2852 | 1501 | 1428 | 3610 | 5775 | 1386 | 1357 | 1003 | 4261 | 4296 | 4974 | 2712 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_min_i64_cell | 1426 | 751 | 714 | 1426 | 2086 | 669 | 742 | 453 | 492 | 434 | 337 | 2194 | 1093 | 962 | 468 | 1028 |
| reduce_arg_min_i8_cell | 16978 | 10839 | 9467 | 16978 | 28201 | 4920 | 5361 | 4293 | 20622 | 20893 | 25738 | 18028 | 14879 | 10532 | 5263 | 16415 |
| reduce_arg_min_u16_cell | 3950 | 2463 | 1807 | 6520 | 11302 | 2518 | 2670 | 2125 | 8647 | 9421 | 12440 | 5847 | 5391 | 4365 | 2194 | 4111 |
| reduce_arg_min_u32_cell | 2852 | 1501 | 1428 | 2189 | 2840 | 1361 | 1430 | 1078 | 4261 | 4296 | 4974 | 2756 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_min_u64_cell | 1426 | 751 | 714 | 1426 | 2086 | 734 | 799 | 496 | 492 | 434 | 338 | 3205 | 1093 | 962 | 468 | 1028 |
| reduce_arg_min_u8_cell | 18514 | 11095 | 9375 | 18514 | 27950 | 5135 | 5496 | 4483 | 20622 | 20893 | 25738 | 19041 | 14874 | 10271 | 5181 | 14116 |
| reduce_max_bool_cell | 155 | 131 | 132 | 155 | 307 | 66 | 50 | 66 | 33 | 33 | 33 | 196 | 291 | 291 | 131 | 109 |
| reduce_max_f16_cell | 3493 | 3462 | 2112 | 3321 | 6563 | 1841 | 1940 | 1177 | 699 | 612 | 515 | 3569 | 5532 | 5319 | 2252 | 2780 |
| reduce_max_f32_cell | 1826 | 942 | 544 | 1028 | 1602 | 301 | 352 | 257 | 196 | 161 | 193 | 900 | 1475 | 771 | 387 | 516 |
| reduce_max_f64_cell | 1826 | 941 | 533 | 1028 | 1602 | 301 | 352 | 257 | 196 | 161 | 193 | 880 | 1475 | 771 | 387 | 516 |
| reduce_max_i16_cell | 155 | 131 | 132 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 227 | 291 | 291 | 131 | 109 |
| reduce_max_i32_cell | 462 | 436 | 287 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 502 | 291 | 291 | 131 | 109 |
| reduce_max_i64_cell | 773 | 397 | 384 | 773 | 1139 | 130 | 207 | 74 | 81 | 65 | 33 | 1098 | 514 | 354 | 179 | 184 |
| reduce_max_i8_cell | 462 | 435 | 287 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 549 | 291 | 291 | 131 | 109 |
| reduce_max_u16_cell | 292 | 259 | 190 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 238 | 291 | 291 | 131 | 109 |
| reduce_max_u32_cell | 718 | 686 | 441 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 711 | 291 | 291 | 131 | 109 |
| reduce_max_u64_cell | 1061 | 541 | 384 | 1061 | 1139 | 215 | 258 | 130 | 81 | 65 | 33 | 1098 | 514 | 354 | 179 | 184 |
| reduce_max_u8_cell | 155 | 131 | 132 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 163 | 291 | 291 | 131 | 109 |
| reduce_mean_bool_cell | 5396 | 8215 | 3100 | 2199 | 2455 | 1041 | 1042 | 1551 | 520 | 521 | 776 | 8215 | 3103 | 4124 | 2207 | 2074 |
| reduce_mean_complex128_cell | 260 | 132 | 132 | 260 | 307 | 130 | 130 | 130 | 65 | 41 | 65 | 196 | 418 | 291 | 131 | 163 |
| reduce_mean_complex64_cell | 516 | 292 | 324 | 516 | 724 | 66 | 82 | 66 | 65 | 41 | 65 | 420 | 387 | 578 | 195 | 195 |
| reduce_mean_f16_cell | 212998 | 165992 | 122955 | 212998 | 328165 | 16390 | 10257 | 8294 | 10339 | 10249 | 8247 | 189355* | 172654 | 147369 | 74062 | 4421 |
| reduce_mean_f32_cell | 260 | 132 | 132 | 260 | 307 | 66 | 82 | 66 | 65 | 41 | 65 | 228 | 418 | 291 | 131 | 131 |
| reduce_mean_f64_cell | 260 | 132 | 132 | 260 | 307 | 66 | 82 | 66 | 65 | 41 | 65 | 228 | 418 | 291 | 131 | 163 |
| reduce_mean_i16_cell | 2125 | 2060 | 1553 | 1100 | 1228 | 519 | 521 | 776 | 260 | 260 | 388 | 2316 | 1290 | 1418 | 611 | 694 |
| reduce_mean_i32_cell | 806 | 520 | 519 | 550 | 614 | 260 | 260 | 388 | 130 | 130 | 194 | 566 | 452 | 580 | 219 | 218 |
| reduce_mean_i64_cell | 155 | 131 | 132 | 155 | 307 | 66 | 50 | 66 | 33 | 33 | 33 | 171 | 291 | 291 | 131 | 109 |
| reduce_mean_i8_cell | 6420 | 8218 | 4124 | 2199 | 2455 | 1038 | 1042 | 1551 | 520 | 521 | 776 | 8218 | 3103 | 4124 | 2207 | 2074 |
| reduce_mean_u16_cell | 1613 | 2060 | 1039 | 1100 | 1228 | 520 | 521 | 776 | 260 | 260 | 388 | 2061 | 1290 | 1418 | 611 | 694 |
| reduce_mean_u32_cell | 550 | 518 | 296 | 550 | 614 | 260 | 260 | 388 | 130 | 130 | 194 | 518 | 452 | 580 | 219 | 218 |
| reduce_mean_u8_cell | 5396 | 8215 | 3100 | 2199 | 2455 | 1041 | 1042 | 1551 | 520 | 521 | 776 | 8215 | 3103 | 4124 | 2207 | 2074 |
| reduce_min_bool_cell | 155 | 131 | 132 | 155 | 307 | 66 | 50 | 66 | 33 | 33 | 33 | 196 | 291 | 291 | 131 | 109 |
| reduce_min_f16_cell | 3557 | 3525 | 1987 | 3459 | 6755 | 1827 | 2069 | 1266 | 658 | 612 | 515 | 3526 | 5548 | 5288 | 2252 | 2796 |
| reduce_min_f32_cell | 1826 | 942 | 544 | 1028 | 1602 | 301 | 352 | 257 | 196 | 161 | 193 | 900 | 1475 | 771 | 387 | 516 |
| reduce_min_f64_cell | 1826 | 941 | 533 | 1028 | 1602 | 301 | 352 | 257 | 196 | 161 | 193 | 880 | 1475 | 771 | 387 | 516 |
| reduce_min_i16_cell | 155 | 131 | 132 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 227 | 291 | 291 | 131 | 109 |
| reduce_min_i32_cell | 547 | 515 | 329 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 431 | 291 | 291 | 131 | 109 |
| reduce_min_i64_cell | 773 | 397 | 384 | 773 | 1139 | 130 | 207 | 74 | 81 | 65 | 33 | 1097 | 514 | 354 | 179 | 184 |
| reduce_min_i8_cell | 547 | 515 | 329 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 473 | 291 | 291 | 131 | 109 |
| reduce_min_u16_cell | 291 | 259 | 190 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 280 | 291 | 291 | 131 | 109 |
| reduce_min_u32_cell | 804 | 771 | 474 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 669 | 291 | 291 | 131 | 109 |
| reduce_min_u64_cell | 773 | 397 | 384 | 773 | 1139 | 215 | 258 | 130 | 81 | 65 | 33 | 1097 | 514 | 354 | 179 | 184 |
| reduce_min_u8_cell | 155 | 131 | 132 | 155 | 307 | 66 | 66 | 66 | 65 | 65 | 33 | 163 | 291 | 291 | 131 | 109 |
| reduce_product_complex128_cell | 676 | 477 | 302 | 676 | 1031 | 525 | 515 | 450 | 258 | 194 | 241 | 648 | 901 | 522 | 261 | 434 |
| reduce_product_complex64_cell | 3236 | 2054 | 1158 | 2149 | 2804 | 770 | 537 | 499 | 194 | 131 | 130 | 2057 | 2278 | 2179 | 1059 | 1200 |
| reduce_product_f16_cell | 217094 | 165992 | 122955 | 217094 | 326117 | 16390 | 10266 | 7557 | 10339 | 10253 | 7362 | 189355* | 172654 | 147369 | 74062 | 4454 |
| reduce_product_f32_cell | 260 | 132 | 132 | 260 | 307 | 66 | 98 | 66 | 65 | 65 | 65 | 228 | 418 | 291 | 131 | 109 |
| reduce_product_f64_cell | 260 | 132 | 132 | 260 | 516 | 66 | 98 | 66 | 65 | 65 | 65 | 228 | 418 | 291 | 131 | 163 |
| reduce_product_i16_cell | 2061 | 2061 | 2062 | 2061 | 8204 | 1610 | 1612 | 1736 | 776 | 776 | 388 | 6180 | 2062 | 6154 | 1034 | 2062 |
| reduce_product_i32_cell | 1030 | 1030 | 1031 | 1030 | 4102 | 805 | 806 | 868 | 388 | 388 | 194 | 3151 | 1095 | 3075 | 773 | 1031 |
| reduce_product_i64_cell | 515 | 515 | 515 | 515 | 2051 | 402 | 402 | 258 | 194 | 194 | 65 | 1545 | 516 | 1539 | 259 | 516 |
| reduce_product_i8_cell | 4122 | 4122 | 4124 | 4122 | 16407 | 3219 | 3223 | 3471 | 1552 | 1553 | 777 | 12360 | 4124 | 12308 | 2068 | 4124 |
| reduce_product_u16_cell | 2061 | 2061 | 2062 | 2061 | 8204 | 1035 | 1034 | 1352 | 776 | 776 | 388 | 5775 | 2062 | 6154 | 1034 | 2062 |
| reduce_product_u32_cell | 1030 | 1030 | 1030 | 1030 | 4102 | 517 | 517 | 676 | 388 | 388 | 194 | 2954 | 1031 | 3075 | 517 | 1031 |
| reduce_product_u8_cell | 4122 | 4122 | 4124 | 4122 | 16407 | 2070 | 2068 | 2703 | 1552 | 1553 | 777 | 11825 | 4124 | 12308 | 2068 | 4124 |
| reduce_sum_f16_cell | 212998 | 165992 | 122955 | 212998 | 328165 | 16390 | 10257 | 8294 | 10339 | 10249 | 8247 | 189355* | 172654 | 147369 | 74062 | 4421 |
| reduce_variance_bool_cell | 45215 | 16538 | 17224 | 45292 | 58972 | 17715 | 12820 | 11525 | 23109 | 21325 | 18458 | 21294 | 135388 | 18949 | 15985 | 9062 |
| reduce_variance_complex128_cell | 5643 | 1271 | 1293 | 5643 | 5736 | 1229 | 1068 | 1579 | 1973 | 1541 | 1810 | 1496 | 16465 | 2186 | 1833 | 723 |
| reduce_variance_complex64_cell | 11286 | 2964 | 2652 | 11286 | 12047 | 2601 | 2045 | 3095 | 4012 | 3115 | 3556 | 3166 | 33063 | 4563 | 3983 | 1616 |
| reduce_variance_f16_cell | 73756 | 35863 | 40737 | 73756 | 105244 | 14630 | 13839 | 13084 | 12844 | 11616 | 9357 | 34337 | 95759 | 48655 | 18450 | 4050 |
| reduce_variance_f32_cell | 11309 | 3190 | 2611 | 11311 | 10801 | 2993 | 2849 | 2474 | 5908 | 5718 | 4740 | 3288 | 33832 | 4576 | 3965 | 1641 |
| reduce_variance_f64_cell | 5654 | 1594 | 1457 | 5654 | 5398 | 1496 | 1423 | 1235 | 2986 | 2761 | 2370 | 1563 | 16914 | 1954 | 1805 | 820 |
| reduce_variance_i16_cell | 22625 | 8274 | 6495 | 22628 | 24632 | 5990 | 5711 | 5079 | 12086 | 10967 | 9167 | 6520 | 67681 | 9285 | 8013 | 3407 |
| reduce_variance_i32_cell | 11311 | 3204 | 2611 | 11313 | 12319 | 2988 | 2847 | 2472 | 5462 | 5203 | 4548 | 3532 | 33839 | 4576 | 3999 | 1641 |
| reduce_variance_i64_cell | 5658 | 1576 | 1370 | 5658 | 5974 | 1497 | 1423 | 1244 | 2731 | 2602 | 2274 | 2520 | 16919 | 2287 | 1982 | 820 |
| reduce_variance_i8_cell | 45353 | 23255 | 20076 | 45245 | 47744 | 11976 | 11423 | 10158 | 24172 | 21934 | 18335 | 19692 | 135393 | 21637 | 14479 | 8305 |
| reduce_variance_u16_cell | 22628 | 6682 | 5225 | 22628 | 27425 | 5998 | 5471 | 5215 | 11555 | 10662 | 9229 | 7660 | 67702 | 8415 | 8397 | 3674 |
| reduce_variance_u32_cell | 11319 | 3213 | 2866 | 11311 | 13088 | 2996 | 2854 | 2670 | 5584 | 5139 | 4553 | 3389 | 33843 | 4582 | 3972 | 1760 |
| reduce_variance_u64_cell | 5660 | 1794 | 1496 | 5660 | 7374 | 1500 | 1427 | 1302 | 2792 | 2569 | 2276 | 1928 | 16923 | 2573 | 2033 | 978 |
| reduce_variance_u8_cell | 45322 | 17940 | 14269 | 45297 | 57964 | 18150 | 13286 | 10470 | 23109 | 21325 | 18458 | 16609 | 135393 | 21381 | 14479 | 8387 |
| **geomean** | **2381** | **1524** | **1219** | **1953** | **3101** | **746** | **750** | **658** | **798** | **737** | **647** | **2256*** | **2653** | **1942** | **937** | **957** |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 128 | 35 | 0/0 | ? ? | - | 298 | 259 | 196 |
| reduce_any_bool_cell | 128 | 35 | 0/0 | ? ? | - | 298 | 259 | 196 |
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
| reduce_max_bool_cell | 128 | 19 | 0/0 | ? ? | - | 155 | 131 | 132 |
| reduce_max_f16_cell | 128 | 384 | 9/0 | ? ? | - | 3493 | 3462 | 2112 |
| reduce_max_f32_cell | 128 | 101 | 2/0 | ? ? | - | 1826 | 942 | 544 |
| reduce_max_f64_cell | 128 | 99 | 0/0 | ? ? | - | 1826 | 941 | 533 |
| reduce_max_i16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 131 | 132 |
| reduce_max_i32_cell | 128 | 53 | 2/0 | ? ? ? | - | 462 | 436 | 287 |
| reduce_max_i64_cell | 128 | 71 | 0/9 | ? ? | - | 773 | 397 | 384 |
| reduce_max_i8_cell | 128 | 53 | 2/0 | ? ? ? ? | - | 462 | 435 | 287 |
| reduce_max_u16_cell | 128 | 35 | 0/0 | ? ? ? ? | - | 292 | 259 | 190 |
| reduce_max_u32_cell | 128 | 82 | 6/0 | ? ? ? | - | 718 | 686 | 441 |
| reduce_max_u64_cell | 128 | 71 | 0/9 | ? ? | - | 1061 | 541 | 384 |
| reduce_max_u8_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 131 | 132 |
| reduce_mean_bool_cell | 16 | 51 | 0/0 | ? ? | - | 5396 | 8215 | 3100 |
| reduce_mean_complex128_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_mean_complex64_cell | 128 | 41 | 4/0 | ? ? | - | 516 | 292 | 324 |
| reduce_mean_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 212998 | 165992 | 122955 |
| reduce_mean_f32_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_mean_f64_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_mean_i16_cell | 32 | 59 | 0/0 | ? ? | - | 2125 | 2060 | 1553 |
| reduce_mean_i32_cell | 64 | 43 | 0/0 | ? ? ? | - | 806 | 520 | 519 |
| reduce_mean_i64_cell | 128 | 19 | 0/0 | ? ? ? | - | 155 | 131 | 132 |
| reduce_mean_i8_cell | 16 | 75 | 0/0 | ? ? | - | 6420 | 8218 | 4124 |
| reduce_mean_u16_cell | 32 | 35 | 0/0 | ? ? ? | - | 1613 | 2060 | 1039 |
| reduce_mean_u32_cell | 64 | 27 | 0/0 | ? ? ? | - | 550 | 518 | 296 |
| reduce_mean_u8_cell | 16 | 51 | 0/0 | ? ? | - | 5396 | 8215 | 3100 |
| reduce_min_bool_cell | 128 | 19 | 0/0 | ? ? | - | 155 | 131 | 132 |
| reduce_min_f16_cell | 128 | 371 | 10/0 | ? ? | - | 3557 | 3525 | 1987 |
| reduce_min_f32_cell | 128 | 101 | 2/0 | ? ? | - | 1826 | 942 | 544 |
| reduce_min_f64_cell | 128 | 99 | 0/0 | ? ? | - | 1826 | 941 | 533 |
| reduce_min_i16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 131 | 132 |
| reduce_min_i32_cell | 128 | 61 | 2/0 | ? ? ? | - | 547 | 515 | 329 |
| reduce_min_i64_cell | 128 | 71 | 0/9 | ? ? | - | 773 | 397 | 384 |
| reduce_min_i8_cell | 128 | 61 | 2/0 | ? ? ? ? | - | 547 | 515 | 329 |
| reduce_min_u16_cell | 128 | 35 | 0/0 | ? ? ? ? | - | 291 | 259 | 190 |
| reduce_min_u32_cell | 128 | 88 | 4/0 | ? ? ? | - | 804 | 771 | 474 |
| reduce_min_u64_cell | 128 | 71 | 0/9 | ? ? | - | 773 | 397 | 384 |
| reduce_min_u8_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 131 | 132 |
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 477 | 302 |
| reduce_product_complex64_cell | 128 | 190 | 31/0 | ? ? | - | 3236 | 2054 | 1158 |
| reduce_product_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 217094 | 165992 | 122955 |
| reduce_product_f32_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_product_f64_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_product_i16_cell | 32 | 47 | 0/10 | ? ? | - | 2061 | 2061 | 2062 |
| reduce_product_i32_cell | 64 | 47 | 0/10 | ? ? | - | 1030 | 1030 | 1031 |
| reduce_product_i64_cell | 128 | 31 | 0/10 | ? ? | - | 515 | 515 | 515 |
| reduce_product_i8_cell | 16 | 47 | 0/10 | ? ? | - | 4122 | 4122 | 4124 |
| reduce_product_u16_cell | 32 | 47 | 0/10 | ? ? | - | 2061 | 2061 | 2062 |
| reduce_product_u32_cell | 64 | 49 | 0/10 | ? ? | - | 1030 | 1030 | 1030 |
| reduce_product_u8_cell | 16 | 47 | 0/10 | ? ? | - | 4122 | 4122 | 4124 |
| reduce_sum_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 212998 | 165992 | 122955 |
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
| **geomean** | | | | | | **2381** | **1524** | **1219** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | stack | loops | calls | sandybridge | btver2 |
|---|---:|---:|---:|---|---|---:|---:|
| reduce_all_bool_cell | 128 | 35 | 0/0 | ? ? | - | 298 | 563 |
| reduce_any_bool_cell | 128 | 35 | 0/0 | ? ? | - | 298 | 563 |
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
| reduce_max_bool_cell | 128 | 19 | 0/0 | ? ? | - | 155 | 307 |
| reduce_max_f16_cell | 128 | 328 | 16/0 | ? ? | - | 3321 | 6563 |
| reduce_max_f32_cell | 128 | 82 | 0/0 | ? ? | - | 1028 | 1602 |
| reduce_max_f64_cell | 128 | 82 | 0/0 | ? ? | - | 1028 | 1602 |
| reduce_max_i16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_max_i32_cell | 128 | 19 | 0/0 | ? ? ? | - | 155 | 307 |
| reduce_max_i64_cell | 128 | 71 | 0/9 | ? ? ? | - | 773 | 1139 |
| reduce_max_i8_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_max_u16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_max_u32_cell | 128 | 19 | 0/0 | ? ? ? | - | 155 | 307 |
| reduce_max_u64_cell | 128 | 71 | 0/9 | ? ? ? | - | 1061 | 1139 |
| reduce_max_u8_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_mean_bool_cell | 16 | 19 | 0/0 | ? ? ? | - | 2199 | 2455 |
| reduce_mean_complex128_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 307 |
| reduce_mean_complex64_cell | 128 | 45 | 4/0 | ? ? | - | 516 | 724 |
| reduce_mean_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 212998 | 328165 |
| reduce_mean_f32_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 307 |
| reduce_mean_f64_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 307 |
| reduce_mean_i16_cell | 32 | 19 | 0/0 | ? ? ? | - | 1100 | 1228 |
| reduce_mean_i32_cell | 64 | 19 | 0/0 | ? ? ? | - | 550 | 614 |
| reduce_mean_i64_cell | 128 | 19 | 0/0 | ? ? ? | - | 155 | 307 |
| reduce_mean_i8_cell | 16 | 19 | 0/0 | ? ? ? | - | 2199 | 2455 |
| reduce_mean_u16_cell | 32 | 19 | 0/0 | ? ? ? | - | 1100 | 1228 |
| reduce_mean_u32_cell | 64 | 19 | 0/0 | ? ? ? | - | 550 | 614 |
| reduce_mean_u8_cell | 16 | 19 | 0/0 | ? ? ? | - | 2199 | 2455 |
| reduce_min_bool_cell | 128 | 19 | 0/0 | ? ? | - | 155 | 307 |
| reduce_min_f16_cell | 128 | 335 | 14/0 | ? ? | - | 3459 | 6755 |
| reduce_min_f32_cell | 128 | 82 | 0/0 | ? ? | - | 1028 | 1602 |
| reduce_min_f64_cell | 128 | 82 | 0/0 | ? ? | - | 1028 | 1602 |
| reduce_min_i16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_min_i32_cell | 128 | 19 | 0/0 | ? ? ? | - | 155 | 307 |
| reduce_min_i64_cell | 128 | 71 | 0/9 | ? ? ? | - | 773 | 1139 |
| reduce_min_i8_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_min_u16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_min_u32_cell | 128 | 19 | 0/0 | ? ? ? | - | 155 | 307 |
| reduce_min_u64_cell | 128 | 71 | 0/9 | ? ? ? | - | 773 | 1139 |
| reduce_min_u8_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 1031 |
| reduce_product_complex64_cell | 128 | 175 | 34/0 | ? ? | - | 2149 | 2804 |
| reduce_product_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 217094 | 326117 |
| reduce_product_f32_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 307 |
| reduce_product_f64_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 516 |
| reduce_product_i16_cell | 32 | 47 | 0/10 | ? ? | - | 2061 | 8204 |
| reduce_product_i32_cell | 64 | 47 | 0/10 | ? ? | - | 1030 | 4102 |
| reduce_product_i64_cell | 128 | 31 | 0/10 | ? ? | - | 515 | 2051 |
| reduce_product_i8_cell | 16 | 47 | 0/10 | ? ? | - | 4122 | 16407 |
| reduce_product_u16_cell | 32 | 47 | 0/10 | ? ? | - | 2061 | 8204 |
| reduce_product_u32_cell | 64 | 49 | 0/10 | ? ? | - | 1030 | 4102 |
| reduce_product_u8_cell | 16 | 47 | 0/10 | ? ? | - | 4122 | 16407 |
| reduce_sum_f16_cell | 128 | 17062 | 24/192 | 64 ?[64] ? | - | 212998 | 328165 |
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
| **geomean** | | | | | | **1953** | **3101** |

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
| reduce_all_bool_cell | 256 | 27 | 0/0 | ? ? | - | 217 | 225 | 193 |
| reduce_any_bool_cell | 256 | 27 | 0/0 | ? ? | - | 217 | 225 | 193 |
| reduce_arg_max_bool_cell | 16 | 59 | 0/0 | ? ? | - | 5274 | 5509 | 4682 |
| reduce_arg_max_f16_cell | 32 | 735 | 5/204 | ?[16] 8 ? | - | 19990 | 16987 | 15895 |
| reduce_arg_max_f32_cell | 64 | 65 | 2/0 | ? ? | - | 1416 | 1490 | 1099 |
| reduce_arg_max_f64_cell | 128 | 75 | 8/0 | ? ? | - | 755 | 807 | 518 |
| reduce_arg_max_i16_cell | 32 | 55 | 0/0 | ? ? | - | 2447 | 2589 | 1940 |
| reduce_arg_max_i32_cell | 64 | 62 | 2/0 | ? ? | - | 1386 | 1357 | 1003 |
| reduce_arg_max_i64_cell | 128 | 57 | 0/0 | ? ? | - | 669 | 742 | 453 |
| reduce_arg_max_i8_cell | 16 | 56 | 0/0 | ? ? | - | 4920 | 5361 | 4293 |
| reduce_arg_max_u16_cell | 32 | 57 | 0/0 | ? ? | - | 2518 | 2670 | 2125 |
| reduce_arg_max_u32_cell | 64 | 61 | 0/0 | ? ? | - | 1361 | 1430 | 1078 |
| reduce_arg_max_u64_cell | 128 | 78 | 13/0 | ? ? | - | 734 | 799 | 500 |
| reduce_arg_max_u8_cell | 16 | 58 | 0/0 | ? ? | - | 5135 | 5496 | 4483 |
| reduce_arg_min_bool_cell | 16 | 58 | 0/0 | ? ? | - | 5135 | 5496 | 4483 |
| reduce_arg_min_f16_cell | 32 | 831 | 5/204 | ?[16] 8 ? | - | 23958 | 19107 | 18417 |
| reduce_arg_min_f32_cell | 64 | 65 | 2/0 | ? ? | - | 1416 | 1490 | 1099 |
| reduce_arg_min_f64_cell | 128 | 75 | 8/0 | ? ? | - | 755 | 807 | 518 |
| reduce_arg_min_i16_cell | 32 | 55 | 0/0 | ? ? | - | 2447 | 2589 | 1940 |
| reduce_arg_min_i32_cell | 64 | 62 | 2/0 | ? ? | - | 1386 | 1357 | 1003 |
| reduce_arg_min_i64_cell | 128 | 57 | 0/0 | ? ? | - | 669 | 742 | 453 |
| reduce_arg_min_i8_cell | 16 | 56 | 0/0 | ? ? | - | 4920 | 5361 | 4293 |
| reduce_arg_min_u16_cell | 32 | 57 | 0/0 | ? ? | - | 2518 | 2670 | 2125 |
| reduce_arg_min_u32_cell | 64 | 61 | 0/0 | ? ? | - | 1361 | 1430 | 1078 |
| reduce_arg_min_u64_cell | 128 | 78 | 13/0 | ? ? | - | 734 | 799 | 496 |
| reduce_arg_min_u8_cell | 16 | 58 | 0/0 | ? ? | - | 5135 | 5496 | 4483 |
| reduce_max_bool_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 50 | 66 |
| reduce_max_f16_cell | 256 | 310 | 28/0 | ? ? | - | 1841 | 1940 | 1177 |
| reduce_max_f32_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_max_f64_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_max_i16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_max_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_max_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 130 | 207 | 74 |
| reduce_max_i8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_max_u16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_max_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_max_u64_cell | 256 | 43 | 0/0 | ? ? ? ? | - | 215 | 258 | 130 |
| reduce_max_u8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_mean_bool_cell | 32 | 19 | 0/0 | ? ? ? ? | - | 1041 | 1042 | 1551 |
| reduce_mean_complex128_cell | 256 | 19 | 0/0 | ? ? | - | 130 | 130 | 130 |
| reduce_mean_complex64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 82 | 66 |
| reduce_mean_f16_cell | 256 | 1717 | 424/0 | 64 ?[128] 32 16 ? | - | 16390 | 10257 | 8294 |
| reduce_mean_f32_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 82 | 66 |
| reduce_mean_f64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 82 | 66 |
| reduce_mean_i16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 519 | 521 | 776 |
| reduce_mean_i32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_i64_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 50 | 66 |
| reduce_mean_i8_cell | 32 | 19 | 0/0 | ? ? ? ? | - | 1038 | 1042 | 1551 |
| reduce_mean_u16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_mean_u32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_u8_cell | 32 | 19 | 0/0 | ? ? ? ? | - | 1041 | 1042 | 1551 |
| reduce_min_bool_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 50 | 66 |
| reduce_min_f16_cell | 256 | 307 | 22/0 | ? ? | - | 1827 | 2069 | 1266 |
| reduce_min_f32_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_min_f64_cell | 256 | 59 | 0/0 | ? ? | - | 301 | 352 | 257 |
| reduce_min_i16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_min_i32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_min_i64_cell | 256 | 27 | 0/0 | ? ? ? ? | - | 130 | 207 | 74 |
| reduce_min_i8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_min_u16_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_min_u32_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_min_u64_cell | 256 | 43 | 0/0 | ? ? ? ? | - | 215 | 258 | 130 |
| reduce_min_u8_cell | 256 | 11 | 0/0 | ? ? ? ? | - | 66 | 66 | 66 |
| reduce_product_complex128_cell | 256 | 129 | 12/0 | ? ? | - | 525 | 515 | 450 |
| reduce_product_complex64_cell | 256 | 87 | 2/0 | ? ? | - | 770 | 537 | 499 |
| reduce_product_f16_cell | 256 | 1717 | 424/0 | 64 ?[128] 32 16 ? | - | 16390 | 10266 | 7557 |
| reduce_product_f32_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 98 | 66 |
| reduce_product_f64_cell | 256 | 11 | 0/0 | ? ? | - | 66 | 98 | 66 |
| reduce_product_i16_cell | 64 | 80 | 2/0 | ? ? ? ? | - | 1610 | 1612 | 1736 |
| reduce_product_i32_cell | 128 | 80 | 2/0 | ? ? ? ? | - | 805 | 806 | 868 |
| reduce_product_i64_cell | 256 | 75 | 0/0 | ? ? ? ? | - | 402 | 402 | 258 |
| reduce_product_i8_cell | 32 | 80 | 2/0 | ? ? ? ? | - | 3219 | 3223 | 3471 |
| reduce_product_u16_cell | 64 | 51 | 0/0 | ? ? ? ? | - | 1035 | 1034 | 1352 |
| reduce_product_u32_cell | 128 | 51 | 0/0 | ? ? ? ? | - | 517 | 517 | 676 |
| reduce_product_u8_cell | 32 | 51 | 0/0 | ? ? ? ? | - | 2070 | 2068 | 2703 |
| reduce_sum_f16_cell | 256 | 1717 | 424/0 | 64 ?[128] 32 16 ? | - | 16390 | 10257 | 8294 |
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
| **geomean** | | | | | | **746** | **750** | **658** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | stack | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 512 | 19 | 0/0 | ? ? | - | 97 | 65 | 64 |
| reduce_any_bool_cell | 512 | 27 | 0/0 | ? ? | - | 129 | 137 | 96 |
| reduce_arg_max_bool_cell | 32 | 741 | 5/397 | ?[32] ? | - | 21263 | 20900 | 24883 |
| reduce_arg_max_f16_cell | 64 | 1413 | 5/426 | ?[32] 16 8 ? | - | 19531 | 16463 | 18060 |
| reduce_arg_max_f32_cell | 128 | 943 | 174/328 | ?[32] 16 ? | - | 6597 | 6281 | 7393 |
| reduce_arg_max_f64_cell | 256 | 280 | 92/2 | ? ? | - | 2098 | 1891 | 1586 |
| reduce_arg_max_i16_cell | 64 | 715 | 5/428 | ?[32] ? | - | 8647 | 9421 | 12440 |
| reduce_arg_max_i32_cell | 128 | 662 | 72/302 | ?[32] ? | - | 4261 | 4296 | 4974 |
| reduce_arg_max_i64_cell | 256 | 56 | 0/0 | ? ? | - | 492 | 434 | 337 |
| reduce_arg_max_i8_cell | 32 | 839 | 5/397 | ?[32] ? | - | 20622 | 20893 | 25738 |
| reduce_arg_max_u16_cell | 64 | 747 | 5/430 | ?[32] ? | - | 9960 | 9806 | 12504 |
| reduce_arg_max_u32_cell | 128 | 694 | 72/302 | ?[32] ? | - | 4807 | 4487 | 4974 |
| reduce_arg_max_u64_cell | 256 | 56 | 0/0 | ? ? | - | 492 | 434 | 338 |
| reduce_arg_max_u8_cell | 32 | 807 | 5/397 | ?[32] ? | - | 21519 | 20892 | 25352 |
| reduce_arg_min_bool_cell | 32 | 836 | 5/397 | ?[32] ? | - | 20749 | 20893 | 25990 |
| reduce_arg_min_f16_cell | 64 | 1605 | 5/426 | ?[32] 5[? ?] ? | - | 23435 | 18584 | 20347 |
| reduce_arg_min_f32_cell | 128 | 943 | 174/328 | ?[32] 16 ? | - | 6597 | 6281 | 7393 |
| reduce_arg_min_f64_cell | 256 | 280 | 92/2 | ? ? | - | 2098 | 1891 | 1586 |
| reduce_arg_min_i16_cell | 64 | 715 | 5/428 | ?[32] ? | - | 8647 | 9421 | 12440 |
| reduce_arg_min_i32_cell | 128 | 662 | 72/302 | ?[32] ? | - | 4261 | 4296 | 4974 |
| reduce_arg_min_i64_cell | 256 | 56 | 0/0 | ? ? | - | 492 | 434 | 337 |
| reduce_arg_min_i8_cell | 32 | 839 | 5/397 | ?[32] ? | - | 20622 | 20893 | 25738 |
| reduce_arg_min_u16_cell | 64 | 715 | 5/428 | ?[32] ? | - | 8647 | 9421 | 12440 |
| reduce_arg_min_u32_cell | 128 | 662 | 72/302 | ?[32] ? | - | 4261 | 4296 | 4974 |
| reduce_arg_min_u64_cell | 256 | 56 | 0/0 | ? ? | - | 492 | 434 | 338 |
| reduce_arg_min_u8_cell | 32 | 839 | 5/397 | ?[32] ? | - | 20622 | 20893 | 25738 |
| reduce_max_bool_cell | 512 | 11 | 0/0 | ? ? | - | 33 | 33 | 33 |
| reduce_max_f16_cell | 512 | 253 | 64/0 | ?[8] 4 ? | - | 699 | 612 | 515 |
| reduce_max_f32_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_max_f64_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_max_i16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_i32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_max_i8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_max_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_max_u8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_mean_bool_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_mean_complex128_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_complex64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_f16_cell | 512 | 3381 | 808/0 | 128 ?[256] 64 32 16 ? | - | 10339 | 10249 | 8247 |
| reduce_mean_f32_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_f64_cell | 512 | 11 | 0/0 | ? ? ? | - | 65 | 41 | 65 |
| reduce_mean_i16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_i32_cell | 256 | 19 | 0/0 | ? ? ? ? | - | 130 | 130 | 194 |
| reduce_mean_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 33 | 33 | 33 |
| reduce_mean_i8_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_mean_u16_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 260 | 260 | 388 |
| reduce_mean_u32_cell | 256 | 19 | 0/0 | ? ? ? ? | - | 130 | 130 | 194 |
| reduce_mean_u8_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 520 | 521 | 776 |
| reduce_min_bool_cell | 512 | 11 | 0/0 | ? ? | - | 33 | 33 | 33 |
| reduce_min_f16_cell | 512 | 253 | 64/0 | ?[8] 4 ? | - | 658 | 612 | 515 |
| reduce_min_f32_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_min_f64_cell | 512 | 59 | 0/0 | ? ? ? | - | 196 | 161 | 193 |
| reduce_min_i16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_i32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_min_i8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u16_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
| reduce_min_u64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 81 | 65 | 33 |
| reduce_min_u8_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 65 | 65 | 33 |
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
| **geomean** | | | | | | **798** | **737** | **647** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | skylake |
|---|---:|---:|---:|---|---|---:|
| reduce_all_bool_cell | 128 | 56 | 17/0 | ? ? | - | 355 |
| reduce_any_bool_cell | 128 | 56 | 17/0 | ? ? | - | 355 |
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
| reduce_max_bool_cell | 128 | 33 | 4/0 | ? ? | - | 196 |
| reduce_max_f16_cell | 128 | 434 | 56/0 | ? ? | - | 3569 |
| reduce_max_f32_cell | 128 | 112 | 18/0 | ? ? | - | 900 |
| reduce_max_f64_cell | 128 | 111 | 19/0 | ? ? | - | 880 |
| reduce_max_i16_cell | 128 | 35 | 4/0 | ? ? ? ? | - | 227 |
| reduce_max_i32_cell | 128 | 70 | 34/0 | ? ? ? | - | 502 |
| reduce_max_i64_cell | 128 | 172 | 19/15 | ? ? ? | - | 1098 |
| reduce_max_i8_cell | 128 | 77 | 36/0 | ? ? ? ? | - | 549 |
| reduce_max_u16_cell | 128 | 39 | 6/0 | ? ? ? ? | - | 238 |
| reduce_max_u32_cell | 128 | 100 | 36/0 | ? ? ? | - | 711 |
| reduce_max_u64_cell | 128 | 172 | 19/15 | ? ? ? | - | 1098 |
| reduce_max_u8_cell | 128 | 25 | 4/0 | ? ? ? ? | - | 163 |
| reduce_mean_bool_cell | 16 | 63 | 6/0 | ? ? ? | - | 8215 |
| reduce_mean_complex128_cell | 128 | 27 | 4/0 | ? ? | - | 196 |
| reduce_mean_complex64_cell | 128 | 55 | 20/0 | ? ? | - | 420 |
| reduce_mean_f16_cell | 128 | 17204 | 32/520 | 64 ?[64] ? | `memcpy@PLT` | 189355* |
| reduce_mean_f32_cell | 128 | 33 | 4/0 | ? ? | - | 228 |
| reduce_mean_f64_cell | 128 | 33 | 4/0 | ? ? | - | 228 |
| reduce_mean_i16_cell | 32 | 75 | 6/0 | ? ? ? | - | 2316 |
| reduce_mean_i32_cell | 64 | 49 | 6/0 | ? ? ? | - | 566 |
| reduce_mean_i64_cell | 128 | 29 | 4/0 | ? ? ? | - | 171 |
| reduce_mean_i8_cell | 16 | 91 | 6/0 | ? ? ? | - | 8218 |
| reduce_mean_u16_cell | 32 | 47 | 8/0 | ? ? ? | - | 2061 |
| reduce_mean_u32_cell | 64 | 37 | 6/0 | ? ? ? | - | 518 |
| reduce_mean_u8_cell | 16 | 63 | 6/0 | ? ? ? | - | 8215 |
| reduce_min_bool_cell | 128 | 33 | 4/0 | ? ? | - | 196 |
| reduce_min_f16_cell | 128 | 413 | 44/0 | ? ? | - | 3526 |
| reduce_min_f32_cell | 128 | 112 | 18/0 | ? ? | - | 900 |
| reduce_min_f64_cell | 128 | 111 | 19/0 | ? ? | - | 880 |
| reduce_min_i16_cell | 128 | 35 | 4/0 | ? ? ? ? | - | 227 |
| reduce_min_i32_cell | 128 | 67 | 16/0 | ? ? ? | - | 431 |
| reduce_min_i64_cell | 128 | 171 | 19/17 | ? ? ? | - | 1097 |
| reduce_min_i8_cell | 128 | 71 | 16/0 | ? ? ? ? | - | 473 |
| reduce_min_u16_cell | 128 | 43 | 6/0 | ? ? ? ? | - | 280 |
| reduce_min_u32_cell | 128 | 96 | 24/0 | ? ? ? | - | 669 |
| reduce_min_u64_cell | 128 | 171 | 19/17 | ? ? ? | - | 1097 |
| reduce_min_u8_cell | 128 | 25 | 4/0 | ? ? ? ? | - | 163 |
| reduce_product_complex128_cell | 128 | 84 | 20/0 | ? ? | - | 648 |
| reduce_product_complex64_cell | 128 | 206 | 51/0 | ? ? | - | 2057 |
| reduce_product_f16_cell | 128 | 17204 | 32/520 | 64 ?[64] ? | `memcpy@PLT` | 189355* |
| reduce_product_f32_cell | 128 | 33 | 4/0 | ? ? | - | 228 |
| reduce_product_f64_cell | 128 | 33 | 4/0 | ? ? | - | 228 |
| reduce_product_i16_cell | 32 | 224 | 0/89 | ? ? | - | 6180 |
| reduce_product_i32_cell | 64 | 224 | 0/89 | ? ? | - | 3151 |
| reduce_product_i64_cell | 128 | 200 | 0/88 | ? ? ? | - | 1545 |
| reduce_product_i8_cell | 16 | 225 | 0/91 | ? ? | - | 12360 |
| reduce_product_u16_cell | 32 | 155 | 0/86 | ? ? | - | 5775 |
| reduce_product_u32_cell | 64 | 159 | 0/89 | ? ? | - | 2954 |
| reduce_product_u8_cell | 16 | 172 | 0/103 | ? ? | - | 11825 |
| reduce_sum_f16_cell | 128 | 17204 | 32/520 | 64 ?[64] ? | `memcpy@PLT` | 189355* |
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
| **geomean** | | | | | | **2256*** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 128 | 31 | 0/0 | ? ? | - | 451 | 451 | 227 |
| reduce_any_bool_cell | 128 | 31 | 0/0 | ? ? | - | 451 | 451 | 227 |
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
| reduce_max_bool_cell | 128 | 15 | 0/0 | ? ? | - | 291 | 291 | 131 |
| reduce_max_f16_cell | 128 | 326 | 20/0 | ?[8] 4 ? | - | 5532 | 5319 | 2252 |
| reduce_max_f32_cell | 128 | 55 | 0/0 | ? ? | - | 1475 | 771 | 387 |
| reduce_max_f64_cell | 128 | 55 | 0/0 | ? ? | - | 1475 | 771 | 387 |
| reduce_max_i16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_max_i32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_max_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 514 | 354 | 179 |
| reduce_max_i8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_max_u16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_max_u32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_max_u64_cell | 128 | 23 | 0/0 | ? ? ? | - | 514 | 354 | 179 |
| reduce_max_u8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_mean_bool_cell | 16 | 34 | 0/0 | ? ? ? ? | - | 3103 | 4124 | 2207 |
| reduce_mean_complex128_cell | 128 | 15 | 0/0 | ? ? ? | - | 418 | 291 | 131 |
| reduce_mean_complex64_cell | 128 | 27 | 0/0 | ? ? ? | - | 387 | 578 | 195 |
| reduce_mean_f16_cell | 128 | 13655 | 12/0 | 64 ?[64] ? | - | 172654 | 147369 | 74062 |
| reduce_mean_f32_cell | 128 | 15 | 0/0 | ? ? ? | - | 418 | 291 | 131 |
| reduce_mean_f64_cell | 128 | 15 | 0/0 | ? ? ? | - | 418 | 291 | 131 |
| reduce_mean_i16_cell | 32 | 23 | 0/0 | ? ? ? ? | - | 1290 | 1418 | 611 |
| reduce_mean_i32_cell | 64 | 15 | 0/0 | ? ? ? ? | - | 452 | 580 | 219 |
| reduce_mean_i64_cell | 128 | 15 | 0/0 | ? ? ? | - | 291 | 291 | 131 |
| reduce_mean_i8_cell | 16 | 34 | 0/0 | ? ? ? ? | - | 3103 | 4124 | 2207 |
| reduce_mean_u16_cell | 32 | 23 | 0/0 | ? ? ? ? | - | 1290 | 1418 | 611 |
| reduce_mean_u32_cell | 64 | 15 | 0/0 | ? ? ? ? | - | 452 | 580 | 219 |
| reduce_mean_u8_cell | 16 | 34 | 0/0 | ? ? ? ? | - | 3103 | 4124 | 2207 |
| reduce_min_bool_cell | 128 | 15 | 0/0 | ? ? | - | 291 | 291 | 131 |
| reduce_min_f16_cell | 128 | 326 | 20/0 | ?[8] 4 ? | - | 5548 | 5288 | 2252 |
| reduce_min_f32_cell | 128 | 55 | 0/0 | ? ? | - | 1475 | 771 | 387 |
| reduce_min_f64_cell | 128 | 55 | 0/0 | ? ? | - | 1475 | 771 | 387 |
| reduce_min_i16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_min_i32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_min_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 514 | 354 | 179 |
| reduce_min_i8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_min_u16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_min_u32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_min_u64_cell | 128 | 23 | 0/0 | ? ? ? | - | 514 | 354 | 179 |
| reduce_min_u8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 291 | 291 | 131 |
| reduce_product_complex128_cell | 128 | 30 | 0/0 | ? ? | - | 901 | 522 | 261 |
| reduce_product_complex64_cell | 128 | 148 | 18/0 | ? ? | - | 2278 | 2179 | 1059 |
| reduce_product_f16_cell | 128 | 13655 | 12/0 | 64 ?[64] ? | - | 172654 | 147369 | 74062 |
| reduce_product_f32_cell | 128 | 15 | 0/0 | ? ? | - | 418 | 291 | 131 |
| reduce_product_f64_cell | 128 | 15 | 0/0 | ? ? | - | 418 | 291 | 131 |
| reduce_product_i16_cell | 32 | 35 | 0/0 | ? ? ? | - | 2062 | 6154 | 1034 |
| reduce_product_i32_cell | 64 | 28 | 0/0 | ? ? ? | - | 1095 | 3075 | 773 |
| reduce_product_i64_cell | 128 | 27 | 0/0 | ? ? ? | - | 516 | 1539 | 259 |
| reduce_product_i8_cell | 16 | 35 | 0/0 | ? ? ? | - | 4124 | 12308 | 2068 |
| reduce_product_u16_cell | 32 | 34 | 0/0 | ? ? ? | - | 2062 | 6154 | 1034 |
| reduce_product_u32_cell | 64 | 27 | 0/0 | ? ? ? | - | 1031 | 3075 | 517 |
| reduce_product_u8_cell | 16 | 34 | 0/0 | ? ? ? | - | 4124 | 12308 | 2068 |
| reduce_sum_f16_cell | 128 | 13655 | 12/0 | 64 ?[64] ? | - | 172654 | 147369 | 74062 |
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
| **geomean** | | | | | | **2653** | **1942** | **937** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | stack | loops | calls | apple-m1 |
|---|---:|---:|---:|---|---|---:|
| reduce_all_bool_cell | 128 | 31 | 0/0 | ? ? | - | 269 |
| reduce_any_bool_cell | 128 | 31 | 0/0 | ? ? | - | 269 |
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
| reduce_max_bool_cell | 128 | 15 | 0/0 | ? ? | - | 109 |
| reduce_max_f16_cell | 128 | 326 | 20/0 | ?[8] 4 ? | - | 2780 |
| reduce_max_f32_cell | 128 | 55 | 0/0 | ? ? | - | 516 |
| reduce_max_f64_cell | 128 | 55 | 0/0 | ? ? | - | 516 |
| reduce_max_i16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_max_i32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_max_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 184 |
| reduce_max_i8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_max_u16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_max_u32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_max_u64_cell | 128 | 23 | 0/0 | ? ? ? | - | 184 |
| reduce_max_u8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_mean_bool_cell | 16 | 34 | 0/0 | ? ? ? ? | - | 2074 |
| reduce_mean_complex128_cell | 128 | 15 | 0/0 | ? ? ? | - | 163 |
| reduce_mean_complex64_cell | 128 | 27 | 0/0 | ? ? ? | - | 195 |
| reduce_mean_f16_cell | 128 | 606 | 20/0 | 64 ?[64] ? | - | 4421 |
| reduce_mean_f32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 131 |
| reduce_mean_f64_cell | 128 | 15 | 0/0 | ? ? ? | - | 163 |
| reduce_mean_i16_cell | 32 | 23 | 0/0 | ? ? ? ? | - | 694 |
| reduce_mean_i32_cell | 64 | 15 | 0/0 | ? ? ? ? | - | 218 |
| reduce_mean_i64_cell | 128 | 15 | 0/0 | ? ? ? | - | 109 |
| reduce_mean_i8_cell | 16 | 34 | 0/0 | ? ? ? ? | - | 2074 |
| reduce_mean_u16_cell | 32 | 23 | 0/0 | ? ? ? ? | - | 694 |
| reduce_mean_u32_cell | 64 | 15 | 0/0 | ? ? ? ? | - | 218 |
| reduce_mean_u8_cell | 16 | 34 | 0/0 | ? ? ? ? | - | 2074 |
| reduce_min_bool_cell | 128 | 15 | 0/0 | ? ? | - | 109 |
| reduce_min_f16_cell | 128 | 326 | 20/0 | ?[8] 4 ? | - | 2796 |
| reduce_min_f32_cell | 128 | 55 | 0/0 | ? ? | - | 516 |
| reduce_min_f64_cell | 128 | 55 | 0/0 | ? ? | - | 516 |
| reduce_min_i16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_min_i32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_min_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 184 |
| reduce_min_i8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_min_u16_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_min_u32_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_min_u64_cell | 128 | 23 | 0/0 | ? ? ? | - | 184 |
| reduce_min_u8_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 109 |
| reduce_product_complex128_cell | 128 | 30 | 0/0 | ? ? | - | 434 |
| reduce_product_complex64_cell | 128 | 119 | 0/0 | ? ? | - | 1200 |
| reduce_product_f16_cell | 128 | 606 | 20/0 | 64 ?[64] ? | - | 4454 |
| reduce_product_f32_cell | 128 | 15 | 0/0 | ? ? | - | 109 |
| reduce_product_f64_cell | 128 | 15 | 0/0 | ? ? | - | 163 |
| reduce_product_i16_cell | 32 | 35 | 0/0 | ? ? ? | - | 2062 |
| reduce_product_i32_cell | 64 | 28 | 0/0 | ? ? ? | - | 1031 |
| reduce_product_i64_cell | 128 | 27 | 0/0 | ? ? ? | - | 516 |
| reduce_product_i8_cell | 16 | 35 | 0/0 | ? ? ? | - | 4124 |
| reduce_product_u16_cell | 32 | 34 | 0/0 | ? ? ? | - | 2062 |
| reduce_product_u32_cell | 64 | 27 | 0/0 | ? ? ? | - | 1031 |
| reduce_product_u8_cell | 16 | 34 | 0/0 | ? ? ? | - | 4124 |
| reduce_sum_f16_cell | 128 | 606 | 20/0 | 64 ?[64] ? | - | 4421 |
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
| **geomean** | | | | | | **957** |
