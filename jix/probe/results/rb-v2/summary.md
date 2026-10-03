# llvm-mca summary: `rb-v2`

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
| reduce_arg_max_bool_cell | 13911 | 9068 | 8653 | - | - | 11853 | 9088 | 5939 | 3620 | 3441 | 2796 | 16507 | 12570 | 10020 | 4997 | 13860 |
| reduce_arg_max_f16_cell | 21302 | 15078 | 12500 | 21302 | 37274 | 15078 | 13356 | 12500 | 14950 | 12741 | 12137 | 21798 | 20797 | 20795 | 10404 | 22548 |
| reduce_arg_max_f32_cell | 2368 | 1517 | 1083 | 3012 | 3594 | 1453 | 1681 | 1112 | 788 | 662 | 392 | 2904 | 3274 | 3593 | 2063 | 2066 |
| reduce_arg_max_f64_cell | 1928 | 1135 | 820 | 2124 | 3210 | 738 | 842 | 518 | 395 | 345 | 226 | 1631 | 1637 | 1796 | 1032 | 1033 |
| reduce_arg_max_i16_cell | 3607 | 2202 | 1715 | 6397 | 10537 | 2821 | 2721 | 2079 | 868 | 965 | 604 | 5678 | 5391 | 4362 | 2191 | 5130 |
| reduce_arg_max_i32_cell | 2852 | 1501 | 1428 | 3421 | 5775 | 1336 | 1376 | 1033 | 434 | 483 | 302 | 2835 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_max_i64_cell | 1426 | 751 | 714 | 1426 | 2086 | 724 | 791 | 415 | 262 | 277 | 192 | 2191 | 1093 | 962 | 468 | 1028 |
| reduce_arg_max_i8_cell | 16978 | 10839 | 9467 | 16978 | 28201 | 6287 | 6241 | 5627 | 2632 | 2683 | 2488 | 18028 | 14879 | 10532 | 5263 | 16415 |
| reduce_arg_max_u16_cell | 3866 | 2394 | 1772 | 6520 | 11302 | 2849 | 2867 | 2276 | 868 | 965 | 604 | 5844 | 5391 | 4365 | 2194 | 4111 |
| reduce_arg_max_u32_cell | 3606 | 2138 | 1432 | 2189 | 2840 | 1423 | 1467 | 1096 | 434 | 483 | 302 | 2964 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_max_u64_cell | 1803 | 1069 | 716 | 1803 | 2118 | 791 | 865 | 472 | 262 | 277 | 195 | 3139 | 1093 | 962 | 468 | 1028 |
| reduce_arg_max_u8_cell | 17495 | 11351 | 8955 | 19021 | 27950 | 6743 | 7020 | 5484 | 2632 | 2683 | 2488 | 19041 | 14874 | 10271 | 5181 | 14116 |
| reduce_arg_min_bool_cell | 18007 | 11351 | 9364 | - | - | 6743 | 7020 | 5484 | 2632 | 2683 | 2488 | 17741 | 14874 | 10271 | 5181 | 14116 |
| reduce_arg_min_f16_cell | 25019 | 17695 | 14405 | 25019 | 43162 | 17695 | 14886 | 14405 | 17185 | 13850 | 13645 | 23831 | 19604 | 19604 | 9807 | 21770 |
| reduce_arg_min_f32_cell | 2368 | 1517 | 1083 | 3113 | 3596 | 1453 | 1681 | 1112 | 788 | 662 | 392 | 2880 | 3274 | 3593 | 2063 | 2066 |
| reduce_arg_min_f64_cell | 2026 | 1163 | 842 | 2187 | 3210 | 738 | 842 | 518 | 395 | 345 | 226 | 1678 | 1637 | 1796 | 1032 | 1033 |
| reduce_arg_min_i16_cell | 3612 | 2199 | 1715 | 6349 | 10537 | 2821 | 2721 | 2079 | 868 | 965 | 604 | 5670 | 5391 | 4362 | 2191 | 5130 |
| reduce_arg_min_i32_cell | 2852 | 1501 | 1428 | 3610 | 5775 | 1336 | 1376 | 1033 | 434 | 483 | 302 | 2712 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_min_i64_cell | 1426 | 751 | 714 | 1426 | 2086 | 724 | 791 | 415 | 262 | 277 | 192 | 2194 | 1093 | 962 | 468 | 1028 |
| reduce_arg_min_i8_cell | 16978 | 10839 | 9467 | 16978 | 28201 | 6287 | 6241 | 5627 | 2632 | 2683 | 2488 | 18028 | 14879 | 10532 | 5263 | 16415 |
| reduce_arg_min_u16_cell | 3950 | 2463 | 1807 | 6520 | 11302 | 2849 | 2867 | 2276 | 868 | 965 | 604 | 5847 | 5391 | 4365 | 2194 | 4111 |
| reduce_arg_min_u32_cell | 2852 | 1501 | 1428 | 2189 | 2840 | 1423 | 1467 | 1096 | 434 | 483 | 302 | 2756 | 2186 | 1670 | 946 | 2056 |
| reduce_arg_min_u64_cell | 1426 | 751 | 714 | 1426 | 2086 | 791 | 865 | 472 | 262 | 277 | 195 | 3205 | 1093 | 962 | 468 | 1028 |
| reduce_arg_min_u8_cell | 18514 | 11095 | 9375 | 18514 | 27950 | 6743 | 7020 | 5484 | 2632 | 2683 | 2488 | 19041 | 14874 | 10271 | 5181 | 14116 |
| reduce_max_f16_cell | 3514 | 3387 | 2068 | 3471 | 6026 | 1869 | 2313 | 1354 | 901 | 932 | 549 | 3690 | 5132 | 4236 | 3340 | 3602 |
| reduce_max_f32_cell | 1932 | 1165 | 594 | 1039 | 1801 | 458 | 520 | 264 | 354 | 292 | 196 | 1229 | 1801 | 1032 | 780 | 906 |
| reduce_max_f64_cell | 1860 | 945 | 553 | 1032 | 1668 | 458 | 520 | 264 | 354 | 292 | 196 | 945 | 1542 | 774 | 454 | 520 |
| reduce_max_i16_cell | 155 | 131 | 132 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 180 | 396 | 396 | 268 | 393 |
| reduce_max_i32_cell | 573 | 525 | 399 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 463 | 396 | 396 | 268 | 393 |
| reduce_max_i64_cell | 753 | 453 | 379 | 753 | 1124 | 324 | 388 | 132 | 320 | 161 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_max_i8_cell | 573 | 525 | 399 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 463 | 396 | 396 | 268 | 393 |
| reduce_max_u16_cell | 294 | 262 | 263 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 394 | 396 | 396 | 268 | 393 |
| reduce_max_u32_cell | 844 | 698 | 527 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 698 | 396 | 396 | 268 | 393 |
| reduce_max_u64_cell | 923 | 566 | 379 | 899 | 1412 | 389 | 480 | 196 | 320 | 161 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_max_u8_cell | 155 | 131 | 132 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 180 | 396 | 396 | 268 | 393 |
| reduce_mean_bool_cell | 5775 | 5146 | 2877 | 5949 | 10808 | 1052 | 714 | 809 | 529 | 530 | 783 | 4009 | 5207 | 3640 | 1894 | 3604 |
| reduce_mean_complex128_cell | 260 | 132 | 132 | 260 | 307 | 132 | 132 | 132 | 132 | 132 | 132 | 196 | 418 | 291 | 131 | 163 |
| reduce_mean_complex64_cell | 519 | 263 | 264 | 519 | 614 | 263 | 264 | 264 | 516 | 260 | 260 | 391 | 454 | 580 | 219 | 262 |
| reduce_mean_f16_cell | 215039 | 167964 | 127387 | 215039 | 328210 | 16400 | 10275 | 8209 | 10314 | 10258 | 8201 | 173075 | 172841 | 147747 | 74166 | 3089 |
| reduce_mean_f32_cell | 387 | 515 | 388 | 387 | 387 | 258 | 321 | 194 | 129 | 161 | 97 | 524 | 652 | 396 | 268 | 521 |
| reduce_mean_f64_cell | 260 | 291 | 196 | 260 | 371 | 258 | 321 | 194 | 129 | 161 | 97 | 262 | 454 | 326 | 166 | 325 |
| reduce_mean_i16_cell | 2204 | 2071 | 1569 | 1175 | 1431 | 526 | 530 | 783 | 264 | 265 | 392 | 2071 | 1306 | 1810 | 668 | 709 |
| reduce_mean_i32_cell | 845 | 526 | 525 | 588 | 716 | 264 | 265 | 392 | 132 | 132 | 196 | 525 | 522 | 649 | 268 | 258 |
| reduce_mean_i64_cell | 187 | 131 | 132 | 187 | 371 | 66 | 193 | 66 | 33 | 97 | 33 | 134 | 326 | 326 | 166 | 98 |
| reduce_mean_i8_cell | 6979 | 6211 | 4157 | 6979 | 12344 | 2156 | 2115 | 2729 | 529 | 530 | 783 | 7726 | 6216 | 4163 | 2120 | 3599 |
| reduce_mean_u16_cell | 1687 | 2071 | 1055 | 1175 | 1431 | 529 | 530 | 783 | 264 | 265 | 392 | 2071 | 1306 | 1810 | 668 | 709 |
| reduce_mean_u32_cell | 588 | 524 | 335 | 588 | 716 | 264 | 265 | 392 | 132 | 132 | 196 | 524 | 522 | 649 | 268 | 258 |
| reduce_mean_u8_cell | 5775 | 5146 | 2877 | 5949 | 10808 | 1052 | 714 | 809 | 529 | 530 | 783 | 4009 | 5207 | 3640 | 1894 | 3604 |
| reduce_min_f16_cell | 3555 | 3427 | 1879 | 3473 | 5897 | 1679 | 2097 | 1353 | 756 | 749 | 506 | 3558 | 5262 | 4236 | 3340 | 3602 |
| reduce_min_f32_cell | 1932 | 1165 | 594 | 1039 | 1801 | 458 | 520 | 264 | 354 | 292 | 196 | 1229 | 1801 | 1032 | 780 | 906 |
| reduce_min_f64_cell | 1860 | 945 | 553 | 1032 | 1668 | 458 | 520 | 264 | 354 | 292 | 196 | 945 | 1542 | 774 | 454 | 520 |
| reduce_min_i16_cell | 155 | 131 | 132 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 180 | 396 | 396 | 268 | 393 |
| reduce_min_i32_cell | 658 | 526 | 399 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 526 | 396 | 396 | 268 | 393 |
| reduce_min_i64_cell | 753 | 453 | 379 | 753 | 1124 | 324 | 388 | 132 | 320 | 161 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_min_i8_cell | 658 | 526 | 399 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 526 | 396 | 396 | 268 | 393 |
| reduce_min_u16_cell | 390 | 389 | 263 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 394 | 396 | 396 | 268 | 393 |
| reduce_min_u32_cell | 911 | 781 | 527 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 782 | 396 | 396 | 268 | 393 |
| reduce_min_u64_cell | 753 | 453 | 379 | 899 | 1412 | 389 | 480 | 196 | 320 | 161 | 33 | 1376 | 581 | 454 | 262 | 325 |
| reduce_min_u8_cell | 155 | 131 | 132 | 155 | 307 | 66 | 193 | 66 | 65 | 97 | 33 | 180 | 396 | 396 | 268 | 393 |
| reduce_product_complex128_cell | 676 | 477 | 302 | 676 | 1031 | 389 | 350 | 262 | 389 | 350 | 262 | 648 | 901 | 522 | 261 | 434 |
| reduce_product_complex64_cell | 1734 | 1076 | 677 | 1606 | 1768 | 779 | 777 | 650 | 1146 | 1028 | 858 | 2252 | 941 | 1033 | 524 | 714 |
| reduce_product_f16_cell | 217106 | 167964 | 127387 | 217106 | 326162 | 16400 | 10276 | 7441 | 10314 | 10257 | 7369 | 173075 | 172841 | 147749 | 74170 | 3091 |
| reduce_product_f32_cell | 643 | 515 | 388 | 643 | 307 | 258 | 385 | 194 | 129 | 193 | 97 | 524 | 652 | 397 | 396 | 265 |
| reduce_product_f64_cell | 356 | 291 | 196 | 356 | 516 | 258 | 385 | 194 | 129 | 193 | 97 | 262 | 454 | 326 | 198 | 325 |
| reduce_product_i16_cell | 2074 | 2074 | 2074 | 2074 | 8215 | 2340 | 2350 | 1830 | 1930 | 1931 | 393 | 6464 | 2076 | 6164 | 1044 | 2076 |
| reduce_product_i32_cell | 1037 | 1037 | 1037 | 1037 | 4108 | 1171 | 1175 | 915 | 965 | 965 | 196 | 3098 | 1293 | 3078 | 906 | 1038 |
| reduce_product_i64_cell | 518 | 518 | 518 | 518 | 2054 | 615 | 616 | 456 | 704 | 545 | 97 | 1551 | 519 | 1541 | 261 | 519 |
| reduce_product_i8_cell | 6467 | 4454 | 4157 | 6467 | 16440 | 5011 | 5011 | 3475 | 3860 | 3862 | 786 | 10967 | 4157 | 12334 | 2094 | 4157 |
| reduce_product_u16_cell | 2074 | 2074 | 2074 | 2074 | 8215 | 2335 | 2337 | 2053 | 1930 | 1931 | 393 | 5435 | 2076 | 6164 | 1044 | 2076 |
| reduce_product_u32_cell | 1037 | 1037 | 1037 | 1037 | 4108 | 1167 | 1169 | 1027 | 965 | 965 | 196 | 2703 | 1038 | 3078 | 522 | 1038 |
| reduce_product_u8_cell | 5381 | 4157 | 4157 | 5381 | 16440 | 2129 | 2130 | 1743 | 3860 | 3862 | 786 | 5937 | 4157 | 12334 | 2094 | 4157 |
| reduce_sum_f16_cell | 215039 | 167964 | 127387 | 215039 | 328210 | 16400 | 10275 | 8209 | 10314 | 10258 | 8201 | 173075 | 172841 | 147747 | 74166 | - |
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
| reduce_all_bool_cell | - | - | - | - | - | - | - | - | 208 | 98 | 130 | 396 | 1164 | 781 | 780 | 905 |
| reduce_any_bool_cell | - | - | - | - | - | - | - | - | 273 | 289 | 161 | 396 | 1164 | 781 | 780 | 905 |
| reduce_max_bool_cell | - | - | - | - | - | - | - | - | 44 | 99 | 35 | 180 | 396 | 396 | 268 | 109 |
| reduce_min_bool_cell | - | - | - | - | - | - | - | - | 44 | 99 | 35 | 180 | 396 | 396 | 268 | 109 |
| **geomean** | **2781** | **1766** | **1411** | **2323** | **3652** | **998** | **1170** | **794** | **606** | **617** | **379** | **2291** | **2956** | **2105** | **1158** | **1265** |

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
| reduce_max_f16_cell | 32 | 90 | 0/0 | ? ? | - | 3514 | 3387 | 2068 |
| reduce_max_f32_cell | 32 | 27 | 0/0 | ? ? | - | 1932 | 1165 | 594 |
| reduce_max_f64_cell | 64 | 51 | 0/0 | ? ? | - | 1860 | 945 | 553 |
| reduce_max_i16_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 131 | 132 |
| reduce_max_i32_cell | 32 | 15 | 0/0 | ? ? ? | - | 573 | 525 | 399 |
| reduce_max_i64_cell | 64 | 35 | 0/0 | ? ? | - | 753 | 453 | 379 |
| reduce_max_i8_cell | 32 | 15 | 0/0 | ? ? ? ? | - | 573 | 525 | 399 |
| reduce_max_u16_cell | 64 | 15 | 0/0 | ? ? ? ? | - | 294 | 262 | 263 |
| reduce_max_u32_cell | 32 | 21 | 0/0 | ? ? ? | - | 844 | 698 | 527 |
| reduce_max_u64_cell | 64 | 35 | 0/0 | ? ? | - | 923 | 566 | 379 |
| reduce_max_u8_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 131 | 132 |
| reduce_mean_bool_cell | 8 | 33 | 0/0 | ? ? | - | 5775 | 5146 | 2877 |
| reduce_mean_complex128_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 132 | 132 |
| reduce_mean_complex64_cell | 64 | 19 | 0/0 | ? ? | - | 519 | 263 | 264 |
| reduce_mean_f16_cell | 32 | 4270 | 6/48 | 16 ?[16] ? | - | 215039 | 167964 | 127387 |
| reduce_mean_f32_cell | 128 | 19 | 0/0 | ? ? ? | - | 387 | 515 | 388 |
| reduce_mean_f64_cell | 128 | 23 | 0/0 | ? ? | - | 260 | 291 | 196 |
| reduce_mean_i16_cell | 16 | 31 | 0/0 | ? ? | - | 2204 | 2071 | 1569 |
| reduce_mean_i32_cell | 32 | 23 | 0/0 | ? ? ? | - | 845 | 526 | 525 |
| reduce_mean_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 131 | 132 |
| reduce_mean_i8_cell | 8 | 40 | 0/0 | ? ? | - | 6979 | 6211 | 4157 |
| reduce_mean_u16_cell | 16 | 19 | 0/0 | ? ? ? | - | 1687 | 2071 | 1055 |
| reduce_mean_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 588 | 524 | 335 |
| reduce_mean_u8_cell | 8 | 33 | 0/0 | ? ? | - | 5775 | 5146 | 2877 |
| reduce_min_f16_cell | 32 | 85 | 0/0 | ? ? | - | 3555 | 3427 | 1879 |
| reduce_min_f32_cell | 32 | 27 | 0/0 | ? ? | - | 1932 | 1165 | 594 |
| reduce_min_f64_cell | 64 | 51 | 0/0 | ? ? | - | 1860 | 945 | 553 |
| reduce_min_i16_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 131 | 132 |
| reduce_min_i32_cell | 32 | 17 | 0/0 | ? ? ? | - | 658 | 526 | 399 |
| reduce_min_i64_cell | 64 | 35 | 0/0 | ? ? | - | 753 | 453 | 379 |
| reduce_min_i8_cell | 32 | 17 | 0/0 | ? ? ? ? | - | 658 | 526 | 399 |
| reduce_min_u16_cell | 64 | 19 | 0/0 | ? ? ? ? | - | 390 | 389 | 263 |
| reduce_min_u32_cell | 32 | 23 | 0/0 | ? ? ? | - | 911 | 781 | 527 |
| reduce_min_u64_cell | 64 | 35 | 0/0 | ? ? | - | 753 | 453 | 379 |
| reduce_min_u8_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 131 | 132 |
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 477 | 302 |
| reduce_product_complex64_cell | 64 | 57 | 0/0 | ? ? | - | 1734 | 1076 | 677 |
| reduce_product_f16_cell | 32 | 4270 | 6/48 | 16 ?[16] ? | - | 217106 | 167964 | 127387 |
| reduce_product_f32_cell | 128 | 19 | 0/0 | ? ? ? | - | 643 | 515 | 388 |
| reduce_product_f64_cell | 128 | 23 | 0/0 | ? ? | - | 356 | 291 | 196 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 2074 | 2074 |
| reduce_product_i32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 1037 | 1037 |
| reduce_product_i64_cell | 64 | 11 | 0/0 | ? ? | - | 518 | 518 | 518 |
| reduce_product_i8_cell | 8 | 33 | 0/0 | ? ? | - | 6467 | 4454 | 4157 |
| reduce_product_u16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 2074 | 2074 |
| reduce_product_u32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 1037 | 1037 |
| reduce_product_u8_cell | 8 | 30 | 0/0 | ? ? | - | 5381 | 4157 | 4157 |
| reduce_sum_f16_cell | 32 | 4270 | 6/48 | 16 ?[16] ? | - | 215039 | 167964 | 127387 |
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
| **geomean** | | | | | | **2781** | **1766** | **1411** |

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
| reduce_max_f16_cell | 32 | 75 | 0/0 | ? ? | - | 3471 | 6026 |
| reduce_max_f32_cell | 32 | 22 | 0/0 | ? ? | - | 1039 | 1801 |
| reduce_max_f64_cell | 64 | 42 | 0/0 | ? ? | - | 1032 | 1668 |
| reduce_max_i16_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_max_i32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_max_i64_cell | 64 | 35 | 0/0 | ? ? ? | - | 753 | 1124 |
| reduce_max_i8_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_max_u16_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_max_u32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_max_u64_cell | 64 | 35 | 0/0 | ? ? ? | - | 899 | 1412 |
| reduce_max_u8_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_mean_bool_cell | 8 | 34 | 0/0 | ? ? ? | - | 5949 | 10808 |
| reduce_mean_complex128_cell | 128 | 19 | 0/0 | ? ? | - | 260 | 307 |
| reduce_mean_complex64_cell | 64 | 19 | 0/0 | ? ? | - | 519 | 614 |
| reduce_mean_f16_cell | 32 | 4270 | 6/48 | 16 ?[16] ? | - | 215039 | 328210 |
| reduce_mean_f32_cell | 128 | 19 | 0/0 | ? ? ? | - | 387 | 387 |
| reduce_mean_f64_cell | 128 | 23 | 0/0 | ? ? | - | 260 | 371 |
| reduce_mean_i16_cell | 16 | 11 | 0/0 | ? ? ? | - | 1175 | 1431 |
| reduce_mean_i32_cell | 32 | 11 | 0/0 | ? ? ? | - | 588 | 716 |
| reduce_mean_i64_cell | 128 | 23 | 0/0 | ? ? ? | - | 187 | 371 |
| reduce_mean_i8_cell | 8 | 40 | 0/0 | ? ? ? | - | 6979 | 12344 |
| reduce_mean_u16_cell | 16 | 11 | 0/0 | ? ? ? | - | 1175 | 1431 |
| reduce_mean_u32_cell | 32 | 11 | 0/0 | ? ? ? | - | 588 | 716 |
| reduce_mean_u8_cell | 8 | 34 | 0/0 | ? ? ? | - | 5949 | 10808 |
| reduce_min_f16_cell | 32 | 73 | 0/0 | ? ? | - | 3473 | 5897 |
| reduce_min_f32_cell | 32 | 22 | 0/0 | ? ? | - | 1039 | 1801 |
| reduce_min_f64_cell | 64 | 42 | 0/0 | ? ? | - | 1032 | 1668 |
| reduce_min_i16_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_min_i32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_min_i64_cell | 64 | 35 | 0/0 | ? ? ? | - | 753 | 1124 |
| reduce_min_i8_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_min_u16_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_min_u32_cell | 128 | 19 | 0/0 | ? ? ? ? | - | 155 | 307 |
| reduce_min_u64_cell | 64 | 35 | 0/0 | ? ? ? | - | 899 | 1412 |
| reduce_min_u8_cell | 128 | 19 | 0/0 | ? ? ? ? ? | - | 155 | 307 |
| reduce_product_complex128_cell | 128 | 55 | 0/0 | ? ? | - | 676 | 1031 |
| reduce_product_complex64_cell | 64 | 55 | 0/0 | ? ? | - | 1606 | 1768 |
| reduce_product_f16_cell | 32 | 4270 | 6/48 | 16 ?[16] ? | - | 217106 | 326162 |
| reduce_product_f32_cell | 128 | 19 | 0/0 | ? ? ? | - | 643 | 307 |
| reduce_product_f64_cell | 128 | 23 | 0/0 | ? ? | - | 356 | 516 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 8215 |
| reduce_product_i32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 4108 |
| reduce_product_i64_cell | 64 | 11 | 0/0 | ? ? | - | 518 | 2054 |
| reduce_product_i8_cell | 8 | 33 | 0/0 | ? ? | - | 6467 | 16440 |
| reduce_product_u16_cell | 16 | 19 | 0/0 | ? ? | - | 2074 | 8215 |
| reduce_product_u32_cell | 32 | 19 | 0/0 | ? ? | - | 1037 | 4108 |
| reduce_product_u8_cell | 8 | 30 | 0/0 | ? ? | - | 5381 | 16440 |
| reduce_sum_f16_cell | 32 | 4270 | 6/48 | 16 ?[16] ? | - | 215039 | 328210 |
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
| **geomean** | | | | | | **2323** | **3652** |

Warnings:

- reduce_all_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_all_bool_cell: no loop whose iteration count is linear in the input length
- reduce_any_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_any_bool_cell: no loop whose iteration count is linear in the input length
- reduce_arg_max_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_arg_max_bool_cell: no loop whose iteration count is linear in the input length
- reduce_arg_max_bool_cell: out-of-line call to a Rust function in a loop: `<fearless_simd[c9fd74fa286dc94b]::generated::avx2::Avx2 as fearless_simd[c9fd74fa286dc94b]::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix[2d4446b3a7e8f4ab]::ops::reduction::fold_run_leaf<bool, jix[2d4446b3a7e8f4ab]::ops::reduction::ArgMaxKernel, true: bool>::{closure#1}, (u64, bool)>`
- reduce_arg_min_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_arg_min_bool_cell: no loop whose iteration count is linear in the input length
- reduce_arg_min_bool_cell: out-of-line call to a Rust function in a loop: `<fearless_simd[c9fd74fa286dc94b]::generated::avx2::Avx2 as fearless_simd[c9fd74fa286dc94b]::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix[2d4446b3a7e8f4ab]::ops::reduction::fold_run_leaf<bool, jix[2d4446b3a7e8f4ab]::ops::reduction::ArgMinKernel, true: bool>::{closure#1}, (u64, bool)>`
- reduce_max_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_max_bool_cell: no loop whose iteration count is linear in the input length
- reduce_min_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_min_bool_cell: no loop whose iteration count is linear in the input length

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | stack | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---:|---|---|---:|---:|---:|
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
| reduce_max_f16_cell | 64 | 71 | 0/0 | ? ? | - | 1869 | 2313 | 1354 |
| reduce_max_f32_cell | 64 | 17 | 0/0 | ? ? | - | 458 | 520 | 264 |
| reduce_max_f64_cell | 64 | 17 | 0/0 | ? ? | - | 458 | 520 | 264 |
| reduce_max_i16_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_max_i32_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_max_i64_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 324 | 388 | 132 |
| reduce_max_i8_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_max_u16_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_max_u32_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_max_u64_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 389 | 480 | 196 |
| reduce_max_u8_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_mean_bool_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 1052 | 714 | 809 |
| reduce_mean_complex128_cell | 128 | 11 | 0/0 | ? ? | - | 132 | 132 | 132 |
| reduce_mean_complex64_cell | 64 | 19 | 0/0 | ? ? | - | 263 | 264 | 264 |
| reduce_mean_f16_cell | 64 | 429 | 102/0 | 16 ?[32] ? | - | 16400 | 10275 | 8209 |
| reduce_mean_f32_cell | 256 | 11 | 0/0 | ? ? ? | - | 258 | 321 | 194 |
| reduce_mean_f64_cell | 256 | 11 | 0/0 | ? ? ? | - | 258 | 321 | 194 |
| reduce_mean_i16_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 526 | 530 | 783 |
| reduce_mean_i32_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_i64_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_mean_i8_cell | 8 | 16 | 0/0 | ? ? ? ? | - | 2156 | 2115 | 2729 |
| reduce_mean_u16_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_mean_u32_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_u8_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 1052 | 714 | 809 |
| reduce_min_f16_cell | 64 | 71 | 0/0 | ? ? | - | 1679 | 2097 | 1353 |
| reduce_min_f32_cell | 64 | 17 | 0/0 | ? ? | - | 458 | 520 | 264 |
| reduce_min_f64_cell | 64 | 17 | 0/0 | ? ? | - | 458 | 520 | 264 |
| reduce_min_i16_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_min_i32_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_min_i64_cell | 128 | 15 | 0/0 | ? ? ? ? | - | 324 | 388 | 132 |
| reduce_min_i8_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_min_u16_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_min_u32_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_min_u64_cell | 128 | 23 | 0/0 | ? ? ? ? | - | 389 | 480 | 196 |
| reduce_min_u8_cell | 256 | 11 | 0/0 | ? ? ? ? ? | - | 66 | 193 | 66 |
| reduce_product_complex128_cell | 128 | 43 | 0/0 | ? ? | - | 389 | 350 | 262 |
| reduce_product_complex64_cell | 64 | 41 | 0/0 | ? ? | - | 779 | 777 | 650 |
| reduce_product_f16_cell | 64 | 429 | 102/0 | 16 ?[32] ? | - | 16400 | 10276 | 7441 |
| reduce_product_f32_cell | 256 | 11 | 0/0 | ? ? ? | - | 258 | 385 | 194 |
| reduce_product_f64_cell | 256 | 11 | 0/0 | ? ? ? | - | 258 | 385 | 194 |
| reduce_product_i16_cell | 16 | 21 | 0/0 | ? ? ? ? | - | 2340 | 2350 | 1830 |
| reduce_product_i32_cell | 32 | 21 | 0/0 | ? ? ? ? | - | 1171 | 1175 | 915 |
| reduce_product_i64_cell | 64 | 21 | 0/0 | ? ? ? ? | - | 615 | 616 | 456 |
| reduce_product_i8_cell | 32 | 105 | 0/0 | ? ? ? ? ? | - | 5011 | 5011 | 3475 |
| reduce_product_u16_cell | 16 | 15 | 0/0 | ? ? ? ? | - | 2335 | 2337 | 2053 |
| reduce_product_u32_cell | 32 | 15 | 0/0 | ? ? ? ? | - | 1167 | 1169 | 1027 |
| reduce_product_u8_cell | 32 | 52 | 0/0 | ? ? ? ? ? | - | 2129 | 2130 | 1743 |
| reduce_sum_f16_cell | 64 | 429 | 102/0 | 16 ?[32] ? | - | 16400 | 10275 | 8209 |
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
| **geomean** | | | | | | **998** | **1170** | **794** |

Warnings:

- reduce_all_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_all_bool_cell: no loop whose iteration count is linear in the input length
- reduce_any_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_any_bool_cell: no loop whose iteration count is linear in the input length
- reduce_max_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_max_bool_cell: no loop whose iteration count is linear in the input length
- reduce_min_bool_cell: asm loops not matched to IR loops by block name, trip counts unknown
- reduce_min_bool_cell: no loop whose iteration count is linear in the input length

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | stack | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 256 | 11 | 0/0 | ? ? | - | 208 | 98 | 130 |
| reduce_any_bool_cell | 256 | 17 | 0/0 | ? ? | - | 273 | 289 | 161 |
| reduce_arg_max_bool_cell | 8 | 17 | 0/0 | ? ? | - | 3620 | 3441 | 2796 |
| reduce_arg_max_f16_cell | 16 | 283 | 0/44 | ? ? | - | 14950 | 12741 | 12137 |
| reduce_arg_max_f32_cell | 64 | 22 | 0/0 | ? ? | - | 788 | 662 | 392 |
| reduce_arg_max_f64_cell | 128 | 22 | 0/0 | ? ? | - | 395 | 345 | 226 |
| reduce_arg_max_i16_cell | 16 | 10 | 0/0 | ? ? | - | 868 | 965 | 604 |
| reduce_arg_max_i32_cell | 32 | 10 | 0/0 | ? ? | - | 434 | 483 | 302 |
| reduce_arg_max_i64_cell | 128 | 18 | 0/0 | ? ? | - | 262 | 277 | 192 |
| reduce_arg_max_i8_cell | 8 | 12 | 0/0 | ? ? | - | 2632 | 2683 | 2488 |
| reduce_arg_max_u16_cell | 16 | 10 | 0/0 | ? ? | - | 868 | 965 | 604 |
| reduce_arg_max_u32_cell | 32 | 10 | 0/0 | ? ? | - | 434 | 483 | 302 |
| reduce_arg_max_u64_cell | 128 | 18 | 0/0 | ? ? | - | 262 | 277 | 195 |
| reduce_arg_max_u8_cell | 8 | 12 | 0/0 | ? ? | - | 2632 | 2683 | 2488 |
| reduce_arg_min_bool_cell | 8 | 12 | 0/0 | ? ? | - | 2632 | 2683 | 2488 |
| reduce_arg_min_f16_cell | 16 | 319 | 0/50 | ? ? | - | 17185 | 13850 | 13645 |
| reduce_arg_min_f32_cell | 64 | 22 | 0/0 | ? ? | - | 788 | 662 | 392 |
| reduce_arg_min_f64_cell | 128 | 22 | 0/0 | ? ? | - | 395 | 345 | 226 |
| reduce_arg_min_i16_cell | 16 | 10 | 0/0 | ? ? | - | 868 | 965 | 604 |
| reduce_arg_min_i32_cell | 32 | 10 | 0/0 | ? ? | - | 434 | 483 | 302 |
| reduce_arg_min_i64_cell | 128 | 18 | 0/0 | ? ? | - | 262 | 277 | 192 |
| reduce_arg_min_i8_cell | 8 | 12 | 0/0 | ? ? | - | 2632 | 2683 | 2488 |
| reduce_arg_min_u16_cell | 16 | 10 | 0/0 | ? ? | - | 868 | 965 | 604 |
| reduce_arg_min_u32_cell | 32 | 10 | 0/0 | ? ? | - | 434 | 483 | 302 |
| reduce_arg_min_u64_cell | 128 | 18 | 0/0 | ? ? | - | 262 | 277 | 195 |
| reduce_arg_min_u8_cell | 8 | 12 | 0/0 | ? ? | - | 2632 | 2683 | 2488 |
| reduce_max_bool_cell | 128 | 5 | 0/0 | ? ? | - | 44 | 99 | 35 |
| reduce_max_f16_cell | 128 | 43 | 0/0 | ? ? | - | 901 | 932 | 549 |
| reduce_max_f32_cell | 128 | 17 | 0/0 | ? ? ? | - | 354 | 292 | 196 |
| reduce_max_f64_cell | 128 | 17 | 0/0 | ? ? ? | - | 354 | 292 | 196 |
| reduce_max_i16_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_max_i32_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_max_i64_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 320 | 161 | 33 |
| reduce_max_i8_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_max_u16_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_max_u32_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_max_u64_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 320 | 161 | 33 |
| reduce_max_u8_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_mean_bool_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_mean_complex128_cell | 128 | 11 | 0/0 | ? ? ? | - | 132 | 132 | 132 |
| reduce_mean_complex64_cell | 128 | 35 | 0/0 | ? ? ? | - | 516 | 260 | 260 |
| reduce_mean_f16_cell | 128 | 849 | 202/0 | 32 ?[64] 16 ? | - | 10314 | 10258 | 8201 |
| reduce_mean_f32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 129 | 161 | 97 |
| reduce_mean_f64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 129 | 161 | 97 |
| reduce_mean_i16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_i32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 132 | 132 | 196 |
| reduce_mean_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 33 | 97 | 33 |
| reduce_mean_i8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_mean_u16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 264 | 265 | 392 |
| reduce_mean_u32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 132 | 132 | 196 |
| reduce_mean_u8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 529 | 530 | 783 |
| reduce_min_bool_cell | 128 | 5 | 0/0 | ? ? | - | 44 | 99 | 35 |
| reduce_min_f16_cell | 128 | 43 | 0/0 | ? ? | - | 756 | 749 | 506 |
| reduce_min_f32_cell | 128 | 17 | 0/0 | ? ? ? | - | 354 | 292 | 196 |
| reduce_min_f64_cell | 128 | 17 | 0/0 | ? ? ? | - | 354 | 292 | 196 |
| reduce_min_i16_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_min_i32_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_min_i64_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 320 | 161 | 33 |
| reduce_min_i8_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_min_u16_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_min_u32_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_min_u64_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 320 | 161 | 33 |
| reduce_min_u8_cell | 512 | 11 | 0/0 | ? ? ? ? ? | - | 65 | 97 | 33 |
| reduce_product_complex128_cell | 128 | 43 | 0/0 | ? ? | - | 389 | 350 | 262 |
| reduce_product_complex64_cell | 128 | 111 | 0/0 | ? ? | - | 1146 | 1028 | 858 |
| reduce_product_f16_cell | 128 | 849 | 202/0 | 32 ?[64] 16 ? | - | 10314 | 10257 | 7369 |
| reduce_product_f32_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 129 | 193 | 97 |
| reduce_product_f64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 129 | 193 | 97 |
| reduce_product_i16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 1930 | 1931 | 393 |
| reduce_product_i32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 965 | 965 | 196 |
| reduce_product_i64_cell | 512 | 11 | 0/0 | ? ? ? ? | - | 704 | 545 | 97 |
| reduce_product_i8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 3860 | 3862 | 786 |
| reduce_product_u16_cell | 64 | 11 | 0/0 | ? ? ? ? | - | 1930 | 1931 | 393 |
| reduce_product_u32_cell | 128 | 11 | 0/0 | ? ? ? ? | - | 965 | 965 | 196 |
| reduce_product_u8_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 3860 | 3862 | 786 |
| reduce_sum_f16_cell | 128 | 849 | 202/0 | 32 ?[64] 16 ? | - | 10314 | 10258 | 8201 |
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
| **geomean** | | | | | | **606** | **617** | **379** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | skylake |
|---|---:|---:|---:|---|---|---:|
| reduce_all_bool_cell | 32 | 11 | 0/0 | ? ? | - | 396 |
| reduce_any_bool_cell | 32 | 11 | 0/0 | ? ? | - | 396 |
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
| reduce_max_bool_cell | 32 | 7 | 0/0 | ? ? | - | 180 |
| reduce_max_f16_cell | 32 | 110 | 19/0 | ? ? | - | 3690 |
| reduce_max_f32_cell | 32 | 27 | 0/0 | ? ? | - | 1229 |
| reduce_max_f64_cell | 64 | 51 | 0/0 | ? ? | - | 945 |
| reduce_max_i16_cell | 32 | 7 | 0/0 | ? ? ? ? | - | 180 |
| reduce_max_i32_cell | 32 | 15 | 0/0 | ? ? ? | - | 463 |
| reduce_max_i64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_max_i8_cell | 32 | 15 | 0/0 | ? ? ? ? | - | 463 |
| reduce_max_u16_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 394 |
| reduce_max_u32_cell | 32 | 21 | 0/0 | ? ? ? | - | 698 |
| reduce_max_u64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_max_u8_cell | 32 | 7 | 0/0 | ? ? ? ? | - | 180 |
| reduce_mean_bool_cell | 8 | 31 | 3/0 | ? ? ? | - | 4009 |
| reduce_mean_complex128_cell | 128 | 27 | 4/0 | ? ? | - | 196 |
| reduce_mean_complex64_cell | 64 | 27 | 4/0 | ? ? | - | 391 |
| reduce_mean_f16_cell | 32 | 4280 | 8/54 | 16 ?[16] ? | - | 173075 |
| reduce_mean_f32_cell | 32 | 7 | 0/0 | ? ? | - | 524 |
| reduce_mean_f64_cell | 64 | 11 | 0/0 | ? ? | - | 262 |
| reduce_mean_i16_cell | 16 | 31 | 0/0 | ? ? ? | - | 2071 |
| reduce_mean_i32_cell | 32 | 23 | 0/0 | ? ? ? | - | 525 |
| reduce_mean_i64_cell | 64 | 11 | 0/0 | ? ? ? | - | 134 |
| reduce_mean_i8_cell | 8 | 51 | 4/0 | ? ? ? | - | 7726 |
| reduce_mean_u16_cell | 16 | 19 | 0/0 | ? ? ? | - | 2071 |
| reduce_mean_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 524 |
| reduce_mean_u8_cell | 8 | 31 | 3/0 | ? ? ? | - | 4009 |
| reduce_min_bool_cell | 32 | 7 | 0/0 | ? ? | - | 180 |
| reduce_min_f16_cell | 32 | 103 | 16/0 | ? ? | - | 3558 |
| reduce_min_f32_cell | 32 | 27 | 0/0 | ? ? | - | 1229 |
| reduce_min_f64_cell | 64 | 51 | 0/0 | ? ? | - | 945 |
| reduce_min_i16_cell | 32 | 7 | 0/0 | ? ? ? ? | - | 180 |
| reduce_min_i32_cell | 32 | 17 | 0/0 | ? ? ? | - | 526 |
| reduce_min_i64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_min_i8_cell | 32 | 17 | 0/0 | ? ? ? ? | - | 526 |
| reduce_min_u16_cell | 32 | 11 | 0/0 | ? ? ? ? | - | 394 |
| reduce_min_u32_cell | 32 | 23 | 0/0 | ? ? ? | - | 782 |
| reduce_min_u64_cell | 64 | 78 | 6/0 | ? ? ? | - | 1376 |
| reduce_min_u8_cell | 32 | 7 | 0/0 | ? ? ? ? | - | 180 |
| reduce_product_complex128_cell | 128 | 84 | 20/0 | ? ? | - | 648 |
| reduce_product_complex64_cell | 64 | 111 | 37/0 | ? ? | - | 2252 |
| reduce_product_f16_cell | 32 | 4280 | 8/54 | 16 ?[16] ? | - | 173075 |
| reduce_product_f32_cell | 32 | 7 | 0/0 | ? ? | - | 524 |
| reduce_product_f64_cell | 64 | 11 | 0/0 | ? ? | - | 262 |
| reduce_product_i16_cell | 16 | 120 | 0/54 | ? ? | - | 6464 |
| reduce_product_i32_cell | 32 | 109 | 0/41 | ? ? | - | 3098 |
| reduce_product_i64_cell | 64 | 96 | 0/39 | ? ? ? | - | 1551 |
| reduce_product_i8_cell | 16 | 172 | 40/0 | ? ? ? | - | 10967 |
| reduce_product_u16_cell | 16 | 83 | 0/45 | ? ? | - | 5435 |
| reduce_product_u32_cell | 32 | 78 | 0/40 | ? ? | - | 2703 |
| reduce_product_u8_cell | 16 | 96 | 26/0 | ? ? ? | - | 5937 |
| reduce_sum_f16_cell | 32 | 4280 | 8/54 | 16 ?[16] ? | - | 173075 |
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
| **geomean** | | | | | | **2291** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| reduce_all_bool_cell | 32 | 10 | 0/0 | ? ? | - | 1164 | 781 | 780 |
| reduce_any_bool_cell | 32 | 10 | 0/0 | ? ? | - | 1164 | 781 | 780 |
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
| reduce_max_bool_cell | 32 | 5 | 0/0 | ? ? | - | 396 | 396 | 268 |
| reduce_max_f16_cell | 32 | 66 | 0/0 | ? ? | - | 5132 | 4236 | 3340 |
| reduce_max_f32_cell | 32 | 16 | 0/0 | ? ? | - | 1801 | 1032 | 780 |
| reduce_max_f64_cell | 64 | 29 | 0/0 | ? ? | - | 1542 | 774 | 454 |
| reduce_max_i16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_max_i32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_max_i64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_max_i8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_max_u16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_max_u32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_max_u64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_max_u8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_mean_bool_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 5207 | 3640 | 1894 |
| reduce_mean_complex128_cell | 128 | 15 | 0/0 | ? ? ? | - | 418 | 291 | 131 |
| reduce_mean_complex64_cell | 64 | 15 | 0/0 | ? ? ? | - | 454 | 580 | 219 |
| reduce_mean_f16_cell | 32 | 3419 | 3/0 | 16 ?[16] ? | - | 172841 | 147747 | 74166 |
| reduce_mean_f32_cell | 32 | 6 | 0/0 | ? ? ? | - | 652 | 396 | 268 |
| reduce_mean_f64_cell | 64 | 9 | 0/0 | ? ? ? | - | 454 | 326 | 166 |
| reduce_mean_i16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 1306 | 1810 | 668 |
| reduce_mean_i32_cell | 32 | 9 | 0/0 | ? ? ? ? | - | 522 | 649 | 268 |
| reduce_mean_i64_cell | 64 | 9 | 0/0 | ? ? | - | 326 | 326 | 166 |
| reduce_mean_i8_cell | 8 | 11 | 0/0 | ? ? ? ? | - | 6216 | 4163 | 2120 |
| reduce_mean_u16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 1306 | 1810 | 668 |
| reduce_mean_u32_cell | 32 | 9 | 0/0 | ? ? ? ? | - | 522 | 649 | 268 |
| reduce_mean_u8_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 5207 | 3640 | 1894 |
| reduce_min_bool_cell | 32 | 5 | 0/0 | ? ? | - | 396 | 396 | 268 |
| reduce_min_f16_cell | 32 | 66 | 0/0 | ? ? | - | 5262 | 4236 | 3340 |
| reduce_min_f32_cell | 32 | 16 | 0/0 | ? ? | - | 1801 | 1032 | 780 |
| reduce_min_f64_cell | 64 | 29 | 0/0 | ? ? | - | 1542 | 774 | 454 |
| reduce_min_i16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_min_i32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_min_i64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_min_i8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_min_u16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_min_u32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_min_u64_cell | 64 | 13 | 0/0 | ? ? ? | - | 581 | 454 | 262 |
| reduce_min_u8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 396 | 396 | 268 |
| reduce_product_complex128_cell | 128 | 30 | 0/0 | ? ? | - | 901 | 522 | 261 |
| reduce_product_complex64_cell | 64 | 30 | 0/0 | ? ? | - | 941 | 1033 | 524 |
| reduce_product_f16_cell | 32 | 3419 | 3/0 | 16 ?[16] ? | - | 172841 | 147749 | 74170 |
| reduce_product_f32_cell | 32 | 6 | 0/0 | ? ? | - | 652 | 397 | 396 |
| reduce_product_f64_cell | 64 | 9 | 0/0 | ? ? | - | 454 | 326 | 198 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? ? | - | 2076 | 6164 | 1044 |
| reduce_product_i32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1293 | 3078 | 906 |
| reduce_product_i64_cell | 64 | 15 | 0/0 | ? ? ? | - | 519 | 1541 | 261 |
| reduce_product_i8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 | 12334 | 2094 |
| reduce_product_u16_cell | 16 | 18 | 0/0 | ? ? ? | - | 2076 | 6164 | 1044 |
| reduce_product_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1038 | 3078 | 522 |
| reduce_product_u8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 | 12334 | 2094 |
| reduce_sum_f16_cell | 32 | 3419 | 3/0 | 16 ?[16] ? | - | 172841 | 147747 | 74166 |
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
| **geomean** | | | | | | **2956** | **2105** | **1158** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | stack | loops | calls | apple-m1 |
|---|---:|---:|---:|---|---|---:|
| reduce_all_bool_cell | 32 | 10 | 0/0 | ? ? | - | 905 |
| reduce_any_bool_cell | 32 | 10 | 0/0 | ? ? | - | 905 |
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
| reduce_max_bool_cell | 128 | 16 | 0/0 | ? ? ? | - | 109 |
| reduce_max_f16_cell | 32 | 66 | 0/0 | ? ? | - | 3602 |
| reduce_max_f32_cell | 32 | 16 | 0/0 | ? ? | - | 906 |
| reduce_max_f64_cell | 64 | 29 | 0/0 | ? ? | - | 520 |
| reduce_max_i16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_max_i32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_max_i64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_max_i8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_max_u16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_max_u32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_max_u64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_max_u8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_mean_bool_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 3604 |
| reduce_mean_complex128_cell | 128 | 15 | 0/0 | ? ? ? | - | 163 |
| reduce_mean_complex64_cell | 64 | 15 | 0/0 | ? ? ? | - | 262 |
| reduce_mean_f16_cell | 32 | 98 | 0/0 | ? ? | - | 3089 |
| reduce_mean_f32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 521 |
| reduce_mean_f64_cell | 64 | 9 | 0/0 | ? ? ? | - | 325 |
| reduce_mean_i16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 709 |
| reduce_mean_i32_cell | 128 | 28 | 0/0 | ? ? ? ? ? | - | 258 |
| reduce_mean_i64_cell | 256 | 28 | 0/0 | ? ? ? | - | 98 |
| reduce_mean_i8_cell | 8 | 11 | 0/0 | ? ? ? ? | - | 3599 |
| reduce_mean_u16_cell | 16 | 13 | 0/0 | ? ? ? ? | - | 709 |
| reduce_mean_u32_cell | 128 | 28 | 0/0 | ? ? ? ? ? | - | 258 |
| reduce_mean_u8_cell | 8 | 15 | 0/0 | ? ? ? ? | - | 3604 |
| reduce_min_bool_cell | 128 | 16 | 0/0 | ? ? ? | - | 109 |
| reduce_min_f16_cell | 32 | 66 | 0/0 | ? ? | - | 3602 |
| reduce_min_f32_cell | 32 | 16 | 0/0 | ? ? | - | 906 |
| reduce_min_f64_cell | 64 | 29 | 0/0 | ? ? | - | 520 |
| reduce_min_i16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_min_i32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_min_i64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_min_i8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_min_u16_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_min_u32_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_min_u64_cell | 64 | 12 | 0/0 | ? ? ? | - | 325 |
| reduce_min_u8_cell | 32 | 6 | 0/0 | ? ? ? ? | - | 393 |
| reduce_product_complex128_cell | 128 | 30 | 0/0 | ? ? | - | 434 |
| reduce_product_complex64_cell | 64 | 30 | 0/0 | ? ? | - | 714 |
| reduce_product_f16_cell | 32 | 98 | 0/0 | ? ? | - | 3091 |
| reduce_product_f32_cell | 32 | 6 | 0/0 | ? ? | - | 265 |
| reduce_product_f64_cell | 64 | 9 | 0/0 | ? ? | - | 325 |
| reduce_product_i16_cell | 16 | 19 | 0/0 | ? ? ? | - | 2076 |
| reduce_product_i32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1038 |
| reduce_product_i64_cell | 64 | 15 | 0/0 | ? ? ? | - | 519 |
| reduce_product_i8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 |
| reduce_product_u16_cell | 16 | 18 | 0/0 | ? ? ? | - | 2076 |
| reduce_product_u32_cell | 32 | 15 | 0/0 | ? ? ? | - | 1038 |
| reduce_product_u8_cell | 8 | 19 | 0/0 | ? ? ? | - | 4157 |
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
| **geomean** | | | | | | **1265** |
