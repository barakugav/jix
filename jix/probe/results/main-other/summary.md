# llvm-mca summary: `main-other`

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
- `loops`: asm loop tree with trip counts; `?` = runtime trip count.
- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include. Values of such
  kernels (and geomeans including them) are flagged `*`: the real cost is higher.

## Out-of-line Rust calls

- i686/add_f16: `_RNvNtNtCs3Bjx3dgswfe_4half8binary164arch16add_f16_fallback`
- i686/approx_eq_f16_bool: `_RNCINvNtNtCsh6Id2leklyS_3jix4util7arr_ext20array_from_fn_inlinebKj1_NCINvXNvXs_NtNtB8_3ops3op2INtB1f_3Op2pppENtNtNtB8_7storage10core_trait12ArrayStorage28read_as_elementwise_pipelineINtB1a_11Op2PipelineINtNtB1N_20elementwise_pipeline12OperandTypedNtNtCs3Bjx3dgswfe_4half8binary163f16EB3e_INtNtB1h_3cmp14ApproxEqKernelB3X_EB3X_B3X_EINtB3h_23ElementwisePipelineImplbE9read_bulkKB11_Kb0_E0E0CsextcoAILMwY_3jix`
- i686/div_f16: `_RNvXsy_NtCs3Bjx3dgswfe_4half8binary16NtB5_3f16NtNtNtCs2TSF2CumT7P_4core3ops5arith3Div3div`
- i686/mul_f16: `_RNvNtNtCs3Bjx3dgswfe_4half8binary164arch21multiply_f16_fallback`
- i686/square_f16: `_RNvNtNtCs3Bjx3dgswfe_4half8binary164arch21multiply_f16_fallback`
- i686/sub_f16: `_RNvXsm_NtCs3Bjx3dgswfe_4half8binary16NtB5_3f16NtNtNtCs2TSF2CumT7P_4core3ops5arith3Sub3sub`
- aarch64/add_f16: `_RNvNtNtCs9fgWqoz3mDR_4half8binary164arch16add_f16_fallback`
- aarch64/approx_eq_f16_bool: `_RNCINvNtNtCs9vrYeryiWyG_3jix4util7arr_ext20array_from_fn_inlinebKj1_NCINvXNvXs_NtNtB8_3ops3op2INtB1f_3Op2pppENtNtNtB8_7storage10core_trait12ArrayStorage28read_as_elementwise_pipelineINtB1a_11Op2PipelineINtNtB1N_20elementwise_pipeline12OperandTypedNtNtCs9fgWqoz3mDR_4half8binary163f16EB3e_INtNtB1h_3cmp14ApproxEqKernelB3X_EB3X_B3X_EINtB3h_23ElementwisePipelineImplbE9read_bulkKB11_Kb0_E0E0CslPYrSSPMJVr_3jix`
- aarch64/div_f16: `_RNvXsy_NtCs9fgWqoz3mDR_4half8binary16NtB5_3f16NtNtNtCscldGzOC5D0D_4core3ops5arith3Div3div`
- aarch64/mul_f16: `_RNvNtNtCs9fgWqoz3mDR_4half8binary164arch21multiply_f16_fallback`
- aarch64/square_f16: `_RNvNtNtCs9fgWqoz3mDR_4half8binary164arch21multiply_f16_fallback`
- aarch64/sub_f16: `_RNvXsm_NtCs9fgWqoz3mDR_4half8binary16NtB5_3f16NtNtNtCscldGzOC5D0D_4core3ops5arith3Sub3sub`

## Overview

| kernel | i686 skylake | aarch64 cortex-a72 | aarch64 neoverse-n1 | aarch64 neoverse-v2 | aarch64-apple apple-m1 |
|---|---:|---:|---:|---:|---:|
| abs_complex128_f64 | 3595* | 3757* | 4713* | 1696* | 2268* |
| abs_complex64_f32 | 7195* | 7050* | 9158* | 3566* | 4008* |
| abs_f16 | 62571* | 68238 | 62628 | 31111 | 5606 |
| abs_f32 | 4994* | 579 | 610 | 227 | 196 |
| abs_f64 | 292 | 579 | 610 | 227 | 196 |
| abs_i16 | 7843* | 13475 | 12482 | 6050 | 7781 |
| abs_i32 | 5122* | 579 | 642 | 227 | 196 |
| abs_i64 | 363 | 579 | 642 | 227 | 196 |
| abs_i8 | 14499* | 29825 | 25793 | 12706 | 15461 |
| acos_f16 | 109116* | 125483* | 123114* | 56924* | 16556* |
| acos_f32 | 4170* | 4394* | 7558* | 2736* | 3367* |
| acos_f64 | 2575* | 2752* | 3753* | 1288* | 1641* |
| add_complex128 | 260 | 898 | 802 | 292 | 260 |
| add_complex64 | 260 | 898 | 802 | 292 | 260 |
| add_f16 | 6309* | 20615* | 17350* | 6734* | 4484 |
| add_f32 | 260 | 898 | 802 | 292 | 260 |
| add_f64 | 260 | 898 | 802 | 292 | 260 |
| add_i16 | 293 | 739 | 770 | 323 | 260 |
| add_i32 | 259 | 771 | 770 | 323 | 259 |
| add_i64 | 259 | 771 | 770 | 323 | 259 |
| add_i8 | 293 | 771 | 802 | 323 | 261 |
| and_i16 | 293 | 739 | 770 | 323 | 260 |
| and_i32 | 259 | 771 | 770 | 323 | 259 |
| and_i64 | 259 | 771 | 770 | 323 | 259 |
| and_i8 | 293 | 771 | 802 | 323 | 261 |
| approx_eq_complex128_bool | 64036* | 84806* | 97028* | 45573* | 42567* |
| approx_eq_complex64_bool | 66985* | 84806* | 97028* | 45573* | 42534* |
| approx_eq_f16_bool | 923430* | 886955 | 817797 | 410412 | 127785 |
| approx_eq_f32_bool | 33358* | 47879* | 60068* | 26437* | 27110* |
| approx_eq_f64_bool | 33316* | 47879* | 60068* | 26437* | 27110* |
| asin_f16 | 109116* | 125483* | 123114* | 56924* | 16556* |
| asin_f32 | 4170* | 4394* | 7558* | 2736* | 3367* |
| asin_f64 | 2575* | 2752* | 3753* | 1288* | 1641* |
| atan_f16 | 109116* | 125483* | 123114* | 56924* | 16556* |
| atan_f32 | 4170* | 4394* | 7558* | 2736* | 3367* |
| atan_f64 | 2575* | 2752* | 3753* | 1288* | 1641* |
| bitwise_rotate_left_i16_u32 | 6079* | 2052 | 1634 | 740 | 688 |
| bitwise_rotate_left_i32_u32 | 2820 | 1572 | 1315 | 580 | 603 |
| bitwise_rotate_left_i64_u32 | 2108 | 1026 | 1057 | 482 | 805 |
| bitwise_rotate_left_i8_u32 | 10175* | 2788 | 2050 | 900 | 688 |
| bitwise_rotate_right_i16_u32 | 6079* | 1956 | 1602 | 708 | 688 |
| bitwise_rotate_right_i32_u32 | 2820 | 1508 | 1314 | 548 | 602 |
| bitwise_rotate_right_i64_u32 | 2108 | 866 | 801 | 386 | 389 |
| bitwise_rotate_right_i8_u32 | 10175* | 2821 | 2082 | 900 | 688 |
| bitwise_shift_left_i16 | 1449 | 963 | 897 | 387 | 262 |
| bitwise_shift_left_i32 | 2788 | 995 | 898 | 355 | 261 |
| bitwise_shift_left_i64 | 1028 | 995 | 898 | 355 | 261 |
| bitwise_shift_left_i8 | 2086 | 899 | 834 | 355 | 263 |
| bitwise_shift_right_i16 | 2660 | 1091 | 962 | 419 | 308 |
| bitwise_shift_right_i32 | 3076 | 1123 | 1026 | 419 | 292 |
| bitwise_shift_right_i64 | 1796 | 1123 | 1026 | 419 | 292 |
| bitwise_shift_right_i8 | 5758* | 1027 | 930 | 355 | 293 |
| bitwise_shift_right_u16 | 2660 | 1091 | 962 | 419 | 308 |
| bitwise_shift_right_u32 | 3076 | 1123 | 1026 | 419 | 292 |
| bitwise_shift_right_u64 | 1028 | 1123 | 1026 | 419 | 292 |
| bitwise_shift_right_u8 | 2203 | 1027 | 930 | 355 | 293 |
| cast_bool_complex128 | 517 | 516 | 705 | 291 | 483 |
| cast_bool_complex64 | 1315 | 1158 | 1697 | 675 | 1091 |
| cast_bool_f16 | 36101* | 19665 | 15913 | 7720 | 6023 |
| cast_bool_f32 | 518 | 870 | 995 | 390 | 347 |
| cast_bool_f64 | 803 | 936 | 836 | 389 | 390 |
| cast_bool_i16 | 7842* | 13250 | 11266 | 5474 | 6756 |
| cast_bool_i32 | 517 | 676 | 801 | 323 | 261 |
| cast_bool_i64 | 867 | 678 | 740 | 356 | 310 |
| cast_bool_i8 | 11363* | 25762 | 25761 | 12674 | 13412 |
| cast_complex128 | 259 | 515 | 483 | 163 | 195 |
| cast_complex128_bool | 18565* | 35459* | 39426* | 18147* | 18629* |
| cast_complex128_complex64 | 772 | 1251 | 898 | 324 | 261 |
| cast_complex64 | 259 | 515 | 483 | 163 | 195 |
| cast_complex64_bool | 16484* | 35459* | 39426* | 18147* | 18629* |
| cast_complex64_complex128 | 260 | 644 | 610 | 196 | 196 |
| cast_f16 | 74316* | 79311 | 66695 | 33162 | 5126 |
| cast_f16_bool | 45351* | 21233 | 18471 | 8841 | 10854 |
| cast_f16_complex128 | 3420 | 3629 | 3874 | 1922 | 388 |
| cast_f16_complex64 | 8004* | 3045 | 2756 | 1437 | 770 |
| cast_f16_f32 | 14180* | 5653 | 5300 | 2628 | 804 |
| cast_f16_f64 | 8547* | 3174 | 2756 | 1372 | 431 |
| cast_f16_i16 | 29975* | 11985 | 11177 | 5386 | 7720 |
| cast_f16_i32 | 16549* | 5911 | 5621 | 2773 | 3095 |
| cast_f16_i64 | 11620* | 3430 | 2885 | 1437 | 645 |
| cast_f16_i8 | 57429* | 27184 | 24807 | 11977 | 14136 |
| cast_f16_u16 | 29610* | 11985 | 11177 | 5386 | 6728 |
| cast_f16_u32 | 19187* | 5911 | 5621 | 2773 | 3095 |
| cast_f16_u64 | 12481* | 3430 | 2885 | 1437 | 645 |
| cast_f16_u8 | 56585* | 26096 | 24072 | 11465 | 12936 |
| cast_f32 | 1955 | 515 | 483 | 163 | 195 |
| cast_f32_bool | 14404* | 29346* | 31330* | 15107* | 16486* |
| cast_f32_complex128 | 580 | 515 | 801 | 291 | 356 |
| cast_f32_complex64 | 1316 | 1026 | 1313 | 515 | 642 |
| cast_f32_f16 | 36602* | 41582 | 36388 | 17573 | 6597 |
| cast_f32_f64 | 260 | 644 | 609 | 196 | 195 |
| cast_f32_i16 | 9287* | 15910 | 15971 | 7108 | 9607 |
| cast_f32_i32 | 6690* | 739 | 771 | 292 | 196 |
| cast_f32_i64 | 4393 | 899 | 739 | 292 | 212 |
| cast_f32_i8 | 16455* | 35269* | 35363* | 16132* | 19654* |
| cast_f32_u16 | 8871* | 14885 | 15907 | 6660 | 8551 |
| cast_f32_u32 | 8729* | 739 | 771 | 292 | 196 |
| cast_f32_u64 | 5061 | 899 | 739 | 292 | 212 |
| cast_f32_u8 | 15526* | 31364* | 35298* | 15108* | 17606* |
| cast_f64 | 804 | 516 | 451 | 163 | 196 |
| cast_f64_bool | 14404* | 29346* | 31330* | 15107* | 16486* |
| cast_f64_complex128 | 579 | 514 | 673 | 259 | 323 |
| cast_f64_complex64 | 1315 | 1027 | 1505 | 547 | 770 |
| cast_f64_f16 | 38820* | 45896* | 39527* | 20293* | 8614* |
| cast_f64_f32 | 5219* | 1251 | 898 | 324 | 260 |
| cast_f64_i16 | 9287* | 16902* | 16963* | 8331* | 11367* |
| cast_f64_i32 | 6242* | 1573 | 2114 | 1027 | 1028 |
| cast_f64_i64 | 4393 | 739 | 610 | 228 | 196 |
| cast_f64_i8 | 16455* | 35269* | 35363* | 16132* | 19654* |
| cast_f64_u16 | 8871* | 15941* | 16931* | 7876* | 10279* |
| cast_f64_u32 | 7522* | 1573 | 2114 | 1027 | 1028 |
| cast_f64_u64 | 5061 | 739 | 610 | 228 | 196 |
| cast_f64_u8 | 15526* | 31364* | 35298* | 15108* | 17606* |
| cast_i16 | 263* | 515 | 483 | 163 | 195 |
| cast_i16_bool | 804 | 1251 | 1089 | 451 | 347 |
| cast_i16_complex128 | 675 | 517 | 833 | 291 | 420 |
| cast_i16_complex64 | 1315 | 1031 | 1601 | 548 | 770 |
| cast_i16_f16 | 41733* | 47851 | 37031 | 18311 | 7429 |
| cast_i16_f32 | 5570* | 709 | 834 | 293 | 212 |
| cast_i16_f64 | 516 | 1029 | 1026 | 516 | 602 |
| cast_i16_i32 | 259 | 516 | 642 | 195 | 196 |
| cast_i16_i64 | 515 | 516 | 547 | 228 | 189 |
| cast_i16_i8 | 804 | 899 | 802 | 323 | 260 |
| cast_i32 | 263* | - | - | - | - |
| cast_i32_bool | 2055 | 2244 | 1794 | 772 | 688 |
| cast_i32_complex128 | 580 | 517 | 802 | 307 | 388 |
| cast_i32_complex64 | 1317 | 1028 | 1505 | 643 | 706 |
| cast_i32_f16 | 40816* | 48228 | 38438 | 18951 | 7205 |
| cast_i32_f32 | 5442* | 740 | 771 | 292 | 196 |
| cast_i32_f64 | 260 | 709 | 737 | 228 | 212 |
| cast_i32_i16 | 516 | 899 | 834 | 323 | 259 |
| cast_i32_i64 | 292 | 515 | 641 | 227 | 243 |
| cast_i32_i8 | 2053 | 1669 | 1186 | 517 | 387 |
| cast_i64 | 263* | - | - | - | - |
| cast_i64_bool | 9022* | 4581 | 3299 | 1476 | 1542 |
| cast_i64_complex128 | 834 | 548 | 801 | 291 | 356 |
| cast_i64_complex64 | 1826 | 1029 | 1794 | 563 | 962 |
| cast_i64_f16 | 36707* | 48616* | 39496* | 20251* | 9544* |
| cast_i64_f32 | 5443* | 1574 | 2083 | 1028 | 1028 |
| cast_i64_f64 | 1283 | 740 | 610 | 228 | 196 |
| cast_i64_i16 | 772 | 1667 | 1282 | 547 | 399 |
| cast_i64_i32 | 259 | 899 | 770 | 323 | 260 |
| cast_i64_i8 | 6706 | 3688 | 2690 | 1252 | 773 |
| cast_i8 | 263* | 515 | 483 | 163 | 195 |
| cast_i8_bool | 292 | 644 | 642 | 259 | 211 |
| cast_i8_complex128 | 675 | 516 | 833 | 292 | 581 |
| cast_i8_complex64 | 1315 | 1031 | 1601 | 548 | 835 |
| cast_i8_f16 | 39066* | 47036 | 37864 | 18758 | 7430 |
| cast_i8_f32 | 5538* | 933 | 705 | 348 | 213 |
| cast_i8_f64 | 1028 | 1029 | 1026 | 516 | 646 |
| cast_i8_i16 | 6467* | 13251 | 13281 | 5826 | 7269 |
| cast_i8_i32 | 515 | 612 | 706 | 292 | 213 |
| cast_i8_i64 | 1027 | 710 | 1028 | 548 | 347 |
| cast_u16_complex128 | 675 | 518 | 769 | 292 | 387 |
| cast_u16_complex64 | 1315 | 1031 | 1409 | 676 | 770 |
| cast_u16_f16 | 36037* | 38025 | 36004 | 17798 | 7109 |
| cast_u16_f32 | 5570* | 709 | 834 | 293 | 212 |
| cast_u16_f64 | 516 | 1029 | 929 | 292 | 387 |
| cast_u16_i32 | 291 | 516 | 642 | 195 | 196 |
| cast_u16_i64 | 515 | 516 | 547 | 228 | 189 |
| cast_u32_complex128 | 613 | 548 | 801 | 291 | 356 |
| cast_u32_complex64 | 1350 | 1028 | 1505 | 643 | 706 |
| cast_u32_f16 | 37985* | 40007 | 36388 | 18278 | 7205 |
| cast_u32_f32 | 5730* | 740 | 771 | 292 | 196 |
| cast_u32_f64 | 315 | 709 | 737 | 228 | 212 |
| cast_u32_i64 | 291 | 516 | 642 | 195 | 196 |
| cast_u64_complex128 | 552 | 548 | 801 | 291 | 356 |
| cast_u64_complex64 | 2563 | 1029 | 1794 | 563 | 962 |
| cast_u64_f16 | 36678* | 44079* | 37511* | 19253* | 9544* |
| cast_u64_f32 | 6788* | 1574 | 2083 | 1028 | 1028 |
| cast_u64_f64 | 664 | 740 | 610 | 228 | 196 |
| cast_u8_complex128 | 675 | 517 | 833 | 292 | 580 |
| cast_u8_complex64 | 1315 | 1031 | 1601 | 676 | 771 |
| cast_u8_f16 | 36101* | 42632 | 35812 | 18086 | 7077 |
| cast_u8_f32 | 5538* | 805 | 899 | 353 | 261 |
| cast_u8_f64 | 868 | 1029 | 1058 | 292 | 388 |
| cast_u8_i16 | 6467* | 11234 | 11265 | 5250 | 6756 |
| cast_u8_i32 | 515 | 612 | 706 | 292 | 213 |
| cast_u8_i64 | 899 | 614 | 708 | 356 | 262 |
| ceil_f16 | 108970* | 76366 | 58505 | 29068 | 5606 |
| ceil_f32 | 4106* | 739 | 771 | 292 | 196 |
| ceil_f64 | 2571* | 739 | 610 | 228 | 196 |
| clamp_op_descriptor_bool | 14277* | 21634 | 21665 | 9955 | 19557 |
| clamp_op_descriptor_f16 | 24386* | 38116 | 36098 | 17635 | 25733 |
| clamp_op_descriptor_f32 | 5186* | 994 | 769 | 292 | 219 |
| clamp_op_descriptor_f64 | 356 | 994 | 769 | 292 | 219 |
| clamp_op_descriptor_i16 | 7747* | 15490 | 13505 | 6530 | 8293 |
| clamp_op_descriptor_i32 | 5027* | 579 | 642 | 227 | 196 |
| clamp_op_descriptor_i64 | 1183 | 802 | 705 | 260 | 219 |
| clamp_op_descriptor_i8 | 13475* | 29826 | 27841 | 13730 | 16485 |
| clamp_op_descriptor_u16 | 8259* | 15490 | 13505 | 6530 | 8293 |
| clamp_op_descriptor_u32 | 5282* | 579 | 642 | 227 | 196 |
| clamp_op_descriptor_u64 | 1183 | 802 | 705 | 260 | 219 |
| clamp_op_descriptor_u8 | 14468* | 29826 | 27841 | 13730 | 16485 |
| cos_f16 | 108970* | 125483* | 123051* | 56813* | 16556* |
| cos_f32 | 9425* | 4394* | 7590* | 2736* | 3367* |
| cos_f64 | 2571* | 2752* | 3753* | 1288* | 1641* |
| count_ones_i16_u32 | 8834* | 8134 | 7684 | 3909 | 3888 |
| count_ones_i32_u32 | 8322* | 8326 | 7876 | 3973 | 4037 |
| count_ones_i64_u32 | 8386* | 7717 | 8740 | 3941 | 3782 |
| count_ones_i8_u32 | 5762* | 7045 | 7075 | 3396 | 3591 |
| count_zeros_i16_u32 | 9106* | 8167 | 8196 | 4101 | 4103 |
| count_zeros_i32_u32 | 8578* | 8358 | 8388 | 4165 | 4239 |
| count_zeros_i64_u32 | 9730* | 8742 | 8772 | 4293 | 4519 |
| count_zeros_i8_u32 | 6274* | 8037 | 7107 | 3652 | 3806 |
| div_complex128 | 1031 | 16390 | 3145 | 1992 | 3273 |
| div_complex64 | 1033 | 8777 | 1864 | 1287 | 732 |
| div_f16 | 6309* | 20615* | 17350* | 6734* | 17420 |
| div_f32 | 2178 | 8708 | 1797 | 1284 | 259 |
| div_f64 | 1030 | 16388 | 1798 | 1796 | 259 |
| div_i16 | 106590* | 39406 | 11909 | 24583 | 20491 |
| div_i32 | 47268* | 19854 | 6535 | 12295 | 10251 |
| div_i64 | 10368* | 18436 | 3620 | 10242 | 6660 |
| div_i8 | 40972* | 78222 | 24165 | 49159 | 40969 |
| div_u16 | 35452* | 39441 | 11621 | 24584 | 20491 |
| div_u32 | 20123* | 19921 | 6501 | 12296 | 10251 |
| div_u64 | 7597* | 17928 | 2571 | 10244 | 6663 |
| div_u8 | 40988* | 78289 | 21862 | 49160 | 40970 |
| equal_complex128_bool | 23774* | 16043* | 12068* | 6033* | 5736* |
| equal_complex64_bool | 23165* | 9707* | 7779* | 4048* | 4327* |
| equal_f16_bool | 4895* | 7339 | 6309 | 2886 | 2983 |
| equal_f32_bool | 2410 | 3332 | 2529 | 1188 | 774 |
| equal_f64_bool | 7424* | 6885 | 5089 | 2276 | 1457 |
| equal_i16_bool | 838 | 1539 | 1441 | 644 | 432 |
| equal_i32_bool | 2314 | 2853 | 2498 | 1092 | 774 |
| equal_i64_bool | 9183* | 5894 | 5089 | 2243 | 1457 |
| equal_i8_bool | 304 | 835 | 833 | 355 | 262 |
| exp_f16 | 108970* | 125483* | 123051* | 56813* | 16556* |
| exp_f32 | 4106* | 4394* | 7590* | 2736* | 3367* |
| exp_f64 | 2571* | 2752* | 3753* | 1288* | 1641* |
| floor_f16 | 108970* | 76366 | 58505 | 29068 | 5606 |
| floor_f32 | 4106* | 739 | 771 | 292 | 196 |
| floor_f64 | 2571* | 739 | 610 | 228 | 196 |
| greater_bool | 357 | 835 | 833 | 355 | 262 |
| greater_equal_bool | 357 | 835 | 833 | 355 | 262 |
| greater_equal_f16_bool | 7198* | 10316 | 7880 | 3658 | 4167 |
| greater_equal_f32_bool | 2410 | 3332 | 2529 | 1188 | 773 |
| greater_equal_f64_bool | 7424* | 6885 | 5089 | 2276 | 1457 |
| greater_equal_i16_bool | 838 | 1539 | 1441 | 644 | 432 |
| greater_equal_i32_bool | 2568 | 2853 | 2498 | 1092 | 774 |
| greater_equal_i64_bool | 13600* | 5894 | 5089 | 2243 | 1457 |
| greater_equal_i8_bool | 304 | 835 | 833 | 355 | 262 |
| greater_equal_u16_bool | 901 | 1539 | 1441 | 644 | 432 |
| greater_equal_u32_bool | 4800* | 2853 | 2498 | 1092 | 774 |
| greater_equal_u64_bool | 13600* | 5894 | 5089 | 2243 | 1457 |
| greater_equal_u8_bool | 357 | 835 | 833 | 355 | 262 |
| greater_f16_bool | 7647* | 47460 | 41315 | 19940 | 32871 |
| greater_f32_bool | 2410 | 3332 | 2529 | 1188 | 774 |
| greater_f64_bool | 7424* | 6885 | 5089 | 2276 | 1457 |
| greater_i16_bool | 838 | 1539 | 1441 | 644 | 432 |
| greater_i32_bool | 2314 | 2853 | 2498 | 1092 | 774 |
| greater_i64_bool | 13391* | 5894 | 5089 | 2243 | 1457 |
| greater_i8_bool | 304 | 835 | 833 | 355 | 262 |
| greater_u16_bool | 1030 | 1539 | 1441 | 644 | 432 |
| greater_u32_bool | 2794 | 2853 | 2498 | 1092 | 774 |
| greater_u64_bool | 13391* | 5894 | 5089 | 2243 | 1457 |
| greater_u8_bool | 357 | 835 | 833 | 355 | 262 |
| imaginary_complex128_f64 | 804 | 515 | 738 | 259 | 387 |
| imaginary_complex64_f32 | 1955 | 1027 | 1187 | 515 | 771 |
| is_finite_f16_bool | 14403* | 30210 | 28225 | 13858 | 16645 |
| is_finite_f32_bool | 15395* | 23236* | 25219* | 13060* | 14374* |
| is_finite_f64_bool | 15651* | 23236* | 25219* | 13060* | 14374* |
| is_infinite_f16_bool | 13476* | 30210 | 26177 | 12834 | 16645 |
| is_infinite_f32_bool | 15395* | 23108* | 23234* | 12100* | 14279* |
| is_infinite_f64_bool | 16612* | 23108* | 23234* | 12100* | 14279* |
| is_nan_f16_bool | 13476* | 30211 | 30241 | 13858 | 17701 |
| is_nan_f32_bool | 13363* | 29346* | 31330* | 15107* | 16549* |
| is_nan_f64_bool | 13363* | 29346* | 31330* | 15107* | 16549* |
| leading_zeros_i16_u32 | 5570* | 580 | 705 | 228 | 212 |
| leading_zeros_i32_u32 | 5218* | 579 | 610 | 227 | 196 |
| leading_zeros_i64_u32 | 11427* | 1058 | 1090 | 546 | 548 |
| leading_zeros_i8_u32 | 5226* | 772 | 866 | 366 | 347 |
| less_bool | 357 | 835 | 833 | 355 | 262 |
| less_equal_bool | 357 | 835 | 833 | 355 | 262 |
| less_equal_f16_bool | 7871* | 47493 | 43363 | 21284 | 32870 |
| less_equal_f32_bool | 2410 | 3332 | 2529 | 1188 | 773 |
| less_equal_f64_bool | 7424* | 6885 | 5089 | 2276 | 1457 |
| less_equal_i16_bool | 838 | 1539 | 1441 | 644 | 432 |
| less_equal_i32_bool | 2536 | 2853 | 2498 | 1092 | 774 |
| less_equal_i64_bool | 13600* | 5894 | 5089 | 2243 | 1457 |
| less_equal_i8_bool | 304 | 835 | 833 | 355 | 262 |
| less_equal_u16_bool | 901 | 1539 | 1441 | 644 | 432 |
| less_equal_u32_bool | 4800* | 2853 | 2498 | 1092 | 774 |
| less_equal_u64_bool | 13600* | 5894 | 5089 | 2243 | 1457 |
| less_equal_u8_bool | 357 | 835 | 833 | 355 | 262 |
| less_f16_bool | 7390* | 10348 | 7976 | 3659 | 4231 |
| less_f32_bool | 2410 | 3332 | 2529 | 1188 | 774 |
| less_f64_bool | 7424* | 6885 | 5089 | 2276 | 1457 |
| less_i16_bool | 838 | 1539 | 1441 | 644 | 432 |
| less_i32_bool | 2314 | 2853 | 2498 | 1092 | 774 |
| less_i64_bool | 13391* | 5894 | 5089 | 2243 | 1457 |
| less_i8_bool | 304 | 835 | 833 | 355 | 262 |
| less_u16_bool | 1030 | 1539 | 1441 | 644 | 432 |
| less_u32_bool | 2794 | 2853 | 2498 | 1092 | 774 |
| less_u64_bool | 13391* | 5894 | 5089 | 2243 | 1457 |
| less_u8_bool | 357 | 835 | 833 | 355 | 262 |
| ln_f16 | 108970* | 125483* | 123051* | 56813* | 16556* |
| ln_f32 | 4106* | 4394* | 7590* | 2736* | 3367* |
| ln_f64 | 2571* | 2752* | 3753* | 1288* | 1641* |
| maximum_bool | 293 | 771 | 802 | 323 | 261 |
| maximum_f16 | 3569 | 6157 | 5735 | 2381 | 3178 |
| maximum_f32 | 5826* | 2275 | 1346 | 580 | 603 |
| maximum_f64 | 1626 | 2275 | 1346 | 580 | 603 |
| maximum_i16 | 293 | 739 | 770 | 323 | 259 |
| maximum_i32 | 431 | 771 | 770 | 323 | 260 |
| maximum_i64 | 1285 | 963 | 897 | 356 | 261 |
| maximum_i8 | 442 | 771 | 802 | 323 | 262 |
| maximum_u16 | 294 | 739 | 770 | 323 | 259 |
| maximum_u32 | 687 | 771 | 770 | 323 | 260 |
| maximum_u64 | 1285 | 963 | 897 | 356 | 261 |
| maximum_u8 | 293 | 771 | 802 | 323 | 262 |
| minimum_bool | 293 | 771 | 802 | 323 | 261 |
| minimum_f16 | 3516 | 6029 | 5671 | 2317 | 3004 |
| minimum_f32 | 5826* | 2275 | 1346 | 580 | 603 |
| minimum_f64 | 1626 | 2275 | 1346 | 580 | 603 |
| minimum_i16 | 293 | 739 | 770 | 323 | 259 |
| minimum_i32 | 431 | 771 | 770 | 323 | 260 |
| minimum_i64 | 1285 | 963 | 897 | 356 | 261 |
| minimum_i8 | 452 | 771 | 802 | 323 | 262 |
| minimum_u16 | 357 | 739 | 770 | 323 | 259 |
| minimum_u32 | 687 | 771 | 770 | 323 | 260 |
| minimum_u64 | 1285 | 963 | 897 | 356 | 261 |
| minimum_u8 | 293 | 771 | 802 | 323 | 262 |
| mul_complex128 | 1030 | 2019 | 1377 | 548 | 533 |
| mul_complex64 | 2318 | 2020 | 1475 | 644 | 774 |
| mul_f16 | 6309* | 20615* | 17350* | 6734* | 4517 |
| mul_f32 | 260 | 898 | 802 | 292 | 259 |
| mul_f64 | 260 | 898 | 802 | 292 | 260 |
| mul_i16 | 294 | 867 | 866 | 324 | 260 |
| mul_i32 | 1284 | 899 | 898 | 292 | 260 |
| mul_i64 | 858 | 835 | 1539 | 386 | 518 |
| mul_i8 | 1285 | 803 | 738 | 292 | 260 |
| neg_complex128 | 292 | 579 | 610 | 227 | 196 |
| neg_complex64 | 292 | 579 | 610 | 227 | 196 |
| neg_f16 | 292 | 579 | 642 | 227 | 196 |
| neg_f32 | 292 | 579 | 610 | 227 | 196 |
| neg_f64 | 292 | 579 | 610 | 227 | 196 |
| neg_i16 | 7304* | 13442 | 13473 | 5890 | 8516 |
| neg_i32 | 291 | 579 | 642 | 227 | 196 |
| neg_i64 | 291 | 579 | 642 | 227 | 196 |
| neg_i8 | 13448* | 29826 | 25793 | 12706 | 16630 |
| not_bool | 13368* | 29858 | 27841 | 13730 | 15461 |
| not_equal_complex128_bool | 24029* | 16555* | 12356* | 6257* | 5896* |
| not_equal_complex64_bool | 24733* | 9932* | 7812* | 4113* | 4392* |
| not_equal_f16_bool | 4991* | 7371 | 6310 | 2918 | 3047 |
| not_equal_f32_bool | 2410 | 3332 | 2529 | 1188 | 774 |
| not_equal_f64_bool | 7424* | 6855 | 5090 | 2308 | 1543 |
| not_equal_i16_bool | 838 | 1539 | 1441 | 644 | 432 |
| not_equal_i32_bool | 2536 | 2853 | 2498 | 1092 | 774 |
| not_equal_i64_bool | 9471* | 5797 | 4962 | 2276 | 1543 |
| not_equal_i8_bool | 304 | 835 | 833 | 355 | 262 |
| not_i16 | 292 | 579 | 642 | 227 | 196 |
| not_i32 | 259 | 579 | 642 | 227 | 195 |
| not_i64 | 259 | 579 | 642 | 227 | 195 |
| not_i8 | 292 | 548 | 610 | 227 | 197 |
| or_i16 | 293 | 739 | 770 | 323 | 260 |
| or_i32 | 259 | 771 | 770 | 323 | 259 |
| or_i64 | 259 | 771 | 770 | 323 | 259 |
| or_i8 | 293 | 771 | 802 | 323 | 261 |
| pow_f32 | 7193* | 14279* | 14853* | 4611* | 6665* |
| pow_f32_i32 | 6923* | 7178* | 8550* | 3630* | 4121* |
| pow_f64 | 3595* | 3693* | 4873* | 1760* | 2278* |
| pow_f64_i32 | 3436* | 3628* | 4201* | 1728* | 2246* |
| real_complex128_f64 | 804 | 515 | 738 | 259 | 387 |
| real_complex64_f32 | 1955 | 1027 | 1187 | 515 | 771 |
| reverse_bits_i16 | 1898 | 707 | 737 | 259 | 212 |
| reverse_bits_i32 | 2310 | 707 | 737 | 259 | 207 |
| reverse_bits_i64 | 2567 | 675 | 705 | 291 | 244 |
| reverse_bits_i8 | 1546 | 548 | 578 | 211 | 197 |
| round_f16 | 108970* | 76366 | 58505 | 29068 | 5606 |
| round_f32 | 4106* | 739 | 771 | 292 | 196 |
| round_f64 | 2571* | 739 | 610 | 228 | 196 |
| sign_f16 | 80523* | 72271 | 59561 | 29483 | 6663 |
| sign_f32 | 6883* | 6275 | 6339 | 2788 | 2853 |
| sign_f64 | 1348 | 1250 | 833 | 355 | 272 |
| sign_i16 | 8827* | 13475 | 13505 | 6531 | 8293 |
| sign_i32 | 5602* | 7331 | 7361 | 3459 | 4197 |
| sign_i64 | 1049 | 771 | 769 | 323 | 260 |
| sign_i8 | 14499* | 29827 | 29857 | 13730 | 16485 |
| sign_u16 | 7427* | 15492 | 13506 | 6563 | 8301 |
| sign_u32 | 4898* | 707 | 705 | 259 | 208 |
| sign_u64 | 388 | 707 | 705 | 259 | 208 |
| sign_u8 | 12483* | 31146 | 29857 | 14723 | 16493 |
| sin_f16 | 108970* | 125483* | 123051* | 56813* | 16556* |
| sin_f32 | 4106* | 4394* | 7590* | 2736* | 3367* |
| sin_f64 | 2571* | 2752* | 3753* | 1288* | 1641* |
| sqrt_f16 | 77581* | 83406 | 58507 | 29070 | 17421 |
| sqrt_f32 | 774 | 8708 | 1797 | 1156 | 196 |
| sqrt_f64 | 1543 | 16388 | 1799 | 1924 | 196 |
| square_complex128 | 806 | 1028 | 1249 | 452 | 554 |
| square_complex64 | 1286 | 1860 | 1316 | 643 | 773 |
| square_f16 | 6124* | 17672* | 16839* | 6484* | 7121 |
| square_f32 | 5091* | 738 | 610 | 228 | 195 |
| square_f64 | 260 | 738 | 610 | 228 | 197 |
| square_i16 | 261 | 740 | 771 | 228 | 197 |
| square_i32 | 1029 | 739 | 739 | 228 | 197 |
| square_i64 | 517 | 611 | 1539 | 291 | 517 |
| square_i8 | 805 | 708 | 578 | 212 | 198 |
| sub_complex128 | 260 | 898 | 802 | 292 | 260 |
| sub_complex64 | 260 | 898 | 802 | 292 | 260 |
| sub_f16 | 6309* | 20615* | 17350* | 6734* | 4484 |
| sub_f32 | 260 | 898 | 802 | 292 | 260 |
| sub_f64 | 260 | 898 | 802 | 292 | 260 |
| sub_i16 | 293 | 739 | 770 | 323 | 260 |
| sub_i32 | 259 | 771 | 770 | 323 | 259 |
| sub_i64 | 259 | 771 | 770 | 323 | 259 |
| sub_i8 | 293 | 771 | 802 | 323 | 261 |
| swap_bytes_i16 | 346 | 579 | 610 | 227 | 196 |
| swap_bytes_i32 | 1796 | 611 | 610 | 227 | 195 |
| swap_bytes_i64 | 2308 | 611 | 610 | 227 | 195 |
| tan_f16 | 109116* | 125483* | 123114* | 56924* | 16556* |
| tan_f32 | 4170* | 4394* | 7558* | 2736* | 3367* |
| tan_f64 | 2575* | 2752* | 3753* | 1288* | 1641* |
| trailing_zeros_i16_u32 | 5058* | 772 | 897 | 325 | 347 |
| trailing_zeros_i32_u32 | 4962* | 772 | 801 | 323 | 261 |
| trailing_zeros_i64_u32 | 13506* | 1314 | 1569 | 722 | 719 |
| trailing_zeros_i8_u32 | 4963* | 934 | 1060 | 505 | 518 |
| xor_i16 | 293 | 739 | 770 | 323 | 260 |
| xor_i32 | 259 | 771 | 770 | 323 | 259 |
| xor_i64 | 259 | 771 | 770 | 323 | 259 |
| xor_i8 | 293 | 771 | 802 | 323 | 261 |
| **geomean** | **2679*** | **2824*** | **2672*** | **1160*** | **980*** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| abs_complex128_f64 | 128 | 193 | ? ? | `hypot@PLT` | 3595* |
| abs_complex64_f32 | 128 | 383 | ? ? | `hypotf@PLT` | 7195* |
| abs_f16 | 128 | 6367 | ?[64] ? ? | `memcpy@PLT` | 62571* |
| abs_f32 | 128 | 413 | ?[32] ? ? | `memcpy@PLT` | 4994* |
| abs_f64 | 128 | 30 | ? ? ? | - | 292 |
| abs_i16 | 128 | 992 | ?[64] ? ? ? | `memcpy@PLT` | 7843* |
| abs_i32 | 128 | 509 | ?[32] ? ? | `memcpy@PLT` | 5122* |
| abs_i64 | 128 | 53 | ? ? ? | - | 363 |
| abs_i8 | 128 | 2080 | ?[128] ? ? ? | `memcpy@PLT` | 14499* |
| acos_f16 | 128 | 7330 | ?[64] ? | `acosf@PLT`, `memcpy@PLT` | 109116* |
| acos_f32 | 128 | 257 | ? ? | `acosf@PLT` | 4170* |
| acos_f64 | 128 | 129 | ? ? | `acos@PLT` | 2575* |
| add_complex128 | 128 | 36 | ? ? ? | - | 260 |
| add_complex64 | 128 | 35 | ? ? ? | - | 260 |
| add_f16 | 128 | 422 | ?[64] ? | `_RNvNtNtCs3Bjx3dgswfe_4half8binary164arch16add_f16_fallback`, `memcpy@PLT` | 6309* |
| add_f32 | 128 | 35 | ? ? ? | - | 260 |
| add_f64 | 128 | 35 | ? ? ? | - | 260 |
| add_i16 | 128 | 37 | ? ? ? | - | 293 |
| add_i32 | 128 | 35 | ? ? ? | - | 259 |
| add_i64 | 128 | 35 | ? ? ? | - | 259 |
| add_i8 | 128 | 37 | ? ? ? ? | - | 293 |
| and_i16 | 128 | 37 | ? ? ? | - | 293 |
| and_i32 | 128 | 35 | ? ? ? | - | 259 |
| and_i64 | 128 | 35 | ? ? ? | - | 259 |
| and_i8 | 128 | 37 | ? ? ? ? | - | 293 |
| approx_eq_complex128_bool | 128 | 9521 | ?[128] ? | `memcpy@PLT` | 64036* |
| approx_eq_complex64_bool | 128 | 7848 | ?[128] ? | `memcpy@PLT` | 66985* |
| approx_eq_f16_bool | 128 | 96938 | ?[128] ? | `memcpy@PLT` | 923430* |
| approx_eq_f32_bool | 128 | 4264 | ?[128] ? | `memcpy@PLT` | 33358* |
| approx_eq_f64_bool | 128 | 5033 | ?[128] ? | `memcpy@PLT` | 33316* |
| asin_f16 | 128 | 7330 | ?[64] ? | `asinf@PLT`, `memcpy@PLT` | 109116* |
| asin_f32 | 128 | 257 | ? ? | `asinf@PLT` | 4170* |
| asin_f64 | 128 | 129 | ? ? | `asin@PLT` | 2575* |
| atan_f16 | 128 | 7330 | ?[64] ? | `atanf@PLT`, `memcpy@PLT` | 109116* |
| atan_f32 | 128 | 257 | ? ? | `atanf@PLT` | 4170* |
| atan_f64 | 128 | 129 | ? ? | `atan@PLT` | 2575* |
| bitwise_rotate_left_i16_u32 | 128 | 421 | ?[64] ? | `memcpy@PLT` | 6079* |
| bitwise_rotate_left_i32_u32 | 128 | 213 | ? ? | - | 2820 |
| bitwise_rotate_left_i64_u32 | 128 | 267 | ? ? ? | - | 2108 |
| bitwise_rotate_left_i8_u32 | 128 | 805 | ?[128] ? | `memcpy@PLT` | 10175* |
| bitwise_rotate_right_i16_u32 | 128 | 421 | ?[64] ? | `memcpy@PLT` | 6079* |
| bitwise_rotate_right_i32_u32 | 128 | 213 | ? ? | - | 2820 |
| bitwise_rotate_right_i64_u32 | 128 | 267 | ? ? ? | - | 2108 |
| bitwise_rotate_right_i8_u32 | 128 | 805 | ?[128] ? | `memcpy@PLT` | 10175* |
| bitwise_shift_left_i16 | 128 | 176 | ? ? ? | - | 1449 |
| bitwise_shift_left_i32 | 128 | 213 | ? ? ? | - | 2788 |
| bitwise_shift_left_i64 | 128 | 75 | ? ? ? | - | 1028 |
| bitwise_shift_left_i8 | 128 | 258 | ? ? ? ? | - | 2086 |
| bitwise_shift_right_i16 | 128 | 284 | ? ? ? | - | 2660 |
| bitwise_shift_right_i32 | 128 | 155 | ? ? ? | - | 3076 |
| bitwise_shift_right_i64 | 128 | 131 | ? ? ? | - | 1796 |
| bitwise_shift_right_i8 | 128 | 518 | ?[8] ? ? ? | `memcpy@PLT` | 5758* |
| bitwise_shift_right_u16 | 128 | 284 | ? ? ? | - | 2660 |
| bitwise_shift_right_u32 | 128 | 155 | ? ? ? | - | 3076 |
| bitwise_shift_right_u64 | 128 | 75 | ? ? ? | - | 1028 |
| bitwise_shift_right_u8 | 128 | 272 | ? ? ? ? | - | 2203 |
| cast_bool_complex128 | 128 | 59 | ? ? | - | 517 |
| cast_bool_complex64 | 128 | 89 | ? ? ? | - | 1315 |
| cast_bool_f16 | 128 | 3815 | ?[64] ? ? | `memcpy@PLT` | 36101* |
| cast_bool_f32 | 128 | 56 | ? ? ? | - | 518 |
| cast_bool_f64 | 128 | 77 | ? ? | - | 803 |
| cast_bool_i16 | 128 | 936 | ?[64] ? ? ? | `memcpy@PLT` | 7842* |
| cast_bool_i32 | 128 | 48 | ? ? ? | - | 517 |
| cast_bool_i64 | 128 | 64 | ? ? ? | - | 867 |
| cast_bool_i8 | 128 | 1567 | ?[128] | `memcpy@PLT` | 11363* |
| cast_complex128 | 128 | 20 | ? ? ? | - | 259 |
| cast_complex128_bool | 128 | 2721 | ?[128] ? ? | `memcpy@PLT` | 18565* |
| cast_complex128_complex64 | 128 | 54 | ? ? ? | - | 772 |
| cast_complex64 | 128 | 19 | ? ? ? | - | 259 |
| cast_complex64_bool | 128 | 2336 | ?[128] ? ? | `memcpy@PLT` | 16484* |
| cast_complex64_complex128 | 128 | 20 | ? ? ? | - | 260 |
| cast_f16 | 128 | 6752 | ?[64] ? | `memcpy@PLT` | 74316* |
| cast_f16_bool | 128 | 4897 | ?[128] ? ? | `memcpy@PLT` | 45351* |
| cast_f16_complex128 | 128 | 444 | ? ? | - | 3420 |
| cast_f16_complex64 | 128 | 766 | ?[16] ? ? | `memcpy@PLT` | 8004* |
| cast_f16_f32 | 128 | 1478 | ?[32] ? ? | `memcpy@PLT` | 14180* |
| cast_f16_f64 | 128 | 782 | ?[16] ? | `memcpy@PLT` | 8547* |
| cast_f16_i16 | 128 | 3362 | ?[64] ? ? | `memcpy@PLT` | 29975* |
| cast_f16_i32 | 128 | 1737 | ?[32] ? ? | `memcpy@PLT` | 16549* |
| cast_f16_i64 | 128 | 1155 | ?[16] ? | `memcpy@PLT` | 11620* |
| cast_f16_i8 | 128 | 6561 | ?[128] ? ? | `memcpy@PLT` | 57429* |
| cast_f16_u16 | 128 | 3361 | ?[64] ? ? | `memcpy@PLT` | 29610* |
| cast_f16_u32 | 128 | 1931 | ?[32] ? ? | `memcpy@PLT` | 19187* |
| cast_f16_u64 | 128 | 1249 | ?[16] ? | `memcpy@PLT` | 12481* |
| cast_f16_u8 | 128 | 6560 | ?[128] ? ? | `memcpy@PLT` | 56585* |
| cast_f32 | 128 | 126 | ? | - | 1955 |
| cast_f32_bool | 128 | 1952 | ?[128] ? ? | `memcpy@PLT` | 14404* |
| cast_f32_complex128 | 128 | 44 | ? ? ? | - | 580 |
| cast_f32_complex64 | 128 | 69 | ? ? ? | - | 1316 |
| cast_f32_f16 | 128 | 4065 | ?[64] ? ? | `memcpy@PLT` | 36602* |
| cast_f32_f64 | 128 | 19 | ? ? ? | - | 260 |
| cast_f32_i16 | 128 | 1057 | ?[64] ? ? | `memcpy@PLT` | 9287* |
| cast_f32_i32 | 128 | 576 | ?[32] ? ? | `memcpy@PLT` | 6690* |
| cast_f32_i64 | 128 | 484 | ? ? | - | 4393 |
| cast_f32_i8 | 128 | 2081 | ?[128] ? ? | `memcpy@PLT` | 16455* |
| cast_f32_u16 | 128 | 1056 | ?[64] ? ? | `memcpy@PLT` | 8871* |
| cast_f32_u32 | 128 | 834 | ?[32] ? ? | `memcpy@PLT` | 8729* |
| cast_f32_u64 | 128 | 584 | ? ? | - | 5061 |
| cast_f32_u8 | 128 | 2080 | ?[128] ? ? | `memcpy@PLT` | 15526* |
| cast_f64 | 128 | 53 | ? | - | 804 |
| cast_f64_bool | 128 | 1952 | ?[128] ? ? | `memcpy@PLT` | 14404* |
| cast_f64_complex128 | 128 | 32 | ? ? ? | - | 579 |
| cast_f64_complex64 | 128 | 91 | ? ? ? | - | 1315 |
| cast_f64_f16 | 128 | 4193 | ?[64] ? | `memcpy@PLT` | 38820* |
| cast_f64_f32 | 128 | 412 | ?[32] ? ? | `memcpy@PLT` | 5219* |
| cast_f64_i16 | 128 | 1057 | ?[64] ? | `memcpy@PLT` | 9287* |
| cast_f64_i32 | 128 | 576 | ?[32] ? | `memcpy@PLT` | 6242* |
| cast_f64_i64 | 128 | 484 | ? ? | - | 4393 |
| cast_f64_i8 | 128 | 2081 | ?[128] ? | `memcpy@PLT` | 16455* |
| cast_f64_u16 | 128 | 1056 | ?[64] ? | `memcpy@PLT` | 8871* |
| cast_f64_u32 | 128 | 704 | ?[32] ? | `memcpy@PLT` | 7522* |
| cast_f64_u64 | 128 | 584 | ? ? | - | 5061 |
| cast_f64_u8 | 128 | 2080 | ?[128] ? | `memcpy@PLT` | 15526* |
| cast_i16 | 128 | 12 | ? | `memcpy@PLT` | 263* |
| cast_i16_bool | 128 | 105 | ? ? ? ? | - | 804 |
| cast_i16_complex128 | 128 | 51 | ? ? ? | - | 675 |
| cast_i16_complex64 | 128 | 89 | ? ? ? | - | 1315 |
| cast_i16_f16 | 128 | 4257 | ?[64] ? ? | `memcpy@PLT` | 41733* |
| cast_i16_f32 | 128 | 486 | ?[32] ? ? | `memcpy@PLT` | 5570* |
| cast_i16_f64 | 128 | 43 | ? ? ? | - | 516 |
| cast_i16_i32 | 128 | 35 | ? ? ? | - | 259 |
| cast_i16_i64 | 128 | 59 | ? ? ? | - | 515 |
| cast_i16_i8 | 128 | 86 | ? ? ? ? | - | 804 |
| cast_i32 | 128 | 12 | ? | `memcpy@PLT` | 263* |
| cast_i32_bool | 128 | 285 | ? ? ? | - | 2055 |
| cast_i32_complex128 | 128 | 40 | ? ? ? | - | 580 |
| cast_i32_complex64 | 128 | 80 | ? ? ? | - | 1317 |
| cast_i32_f16 | 128 | 4193 | ?[64] ? ? | `memcpy@PLT` | 40816* |
| cast_i32_f32 | 128 | 445 | ?[32] ? ? | `memcpy@PLT` | 5442* |
| cast_i32_f64 | 128 | 27 | ? ? ? | - | 260 |
| cast_i32_i16 | 128 | 69 | ? ? ? | - | 516 |
| cast_i32_i64 | 128 | 45 | ? ? ? | - | 292 |
| cast_i32_i8 | 128 | 213 | ? ? ? | - | 2053 |
| cast_i64 | 128 | 12 | ? | `memcpy@PLT` | 263* |
| cast_i64_bool | 128 | 799 | ?[32] ? ? | `memcpy@PLT` | 9022* |
| cast_i64_complex128 | 128 | 48 | ? ? ? | - | 834 |
| cast_i64_complex64 | 128 | 101 | ? ? ? | - | 1826 |
| cast_i64_f16 | 128 | 4193 | ?[64] ? | `memcpy@PLT` | 36707* |
| cast_i64_f32 | 128 | 477 | ?[32] ? ? | `memcpy@PLT` | 5443* |
| cast_i64_f64 | 128 | 91 | ? ? ? | - | 1283 |
| cast_i64_i16 | 128 | 103 | ? ? ? | - | 772 |
| cast_i64_i32 | 128 | 35 | ? ? ? | - | 259 |
| cast_i64_i8 | 128 | 512 | ? ? | - | 6706 |
| cast_i8 | 128 | 12 | ? | `memcpy@PLT` | 263* |
| cast_i8_bool | 128 | 40 | ? ? ? ? | - | 292 |
| cast_i8_complex128 | 128 | 51 | ? ? ? | - | 675 |
| cast_i8_complex64 | 128 | 89 | ? ? ? | - | 1315 |
| cast_i8_f16 | 128 | 4266 | ?[64] ? ? | `memcpy@PLT` | 39066* |
| cast_i8_f32 | 128 | 478 | ?[32] ? ? | `memcpy@PLT` | 5538* |
| cast_i8_f64 | 128 | 59 | ? ? ? | - | 1028 |
| cast_i8_i16 | 128 | 808 | ?[64] ? ? ? | `memcpy@PLT` | 6467* |
| cast_i8_i32 | 128 | 43 | ? ? ? | - | 515 |
| cast_i8_i64 | 128 | 75 | ? ? ? | - | 1027 |
| cast_u16_complex128 | 128 | 51 | ? ? ? | - | 675 |
| cast_u16_complex64 | 128 | 89 | ? ? ? | - | 1315 |
| cast_u16_f16 | 128 | 3808 | ?[64] ? ? | `memcpy@PLT` | 36037* |
| cast_u16_f32 | 128 | 486 | ?[32] ? ? | `memcpy@PLT` | 5570* |
| cast_u16_f64 | 128 | 38 | ? ? ? | - | 516 |
| cast_u16_i32 | 128 | 30 | ? ? ? | - | 291 |
| cast_u16_i64 | 128 | 39 | ? ? ? | - | 515 |
| cast_u32_complex128 | 128 | 54 | ? ? ? | - | 613 |
| cast_u32_complex64 | 128 | 119 | ? ? ? | - | 1350 |
| cast_u32_f16 | 128 | 3871 | ?[64] ? ? | `memcpy@PLT` | 37985* |
| cast_u32_f32 | 128 | 510 | ?[32] ? ? | `memcpy@PLT` | 5730* |
| cast_u32_f64 | 128 | 48 | ? ? ? | - | 315 |
| cast_u32_i64 | 128 | 30 | ? ? ? | - | 291 |
| cast_u64_complex128 | 128 | 74 | ? ? ? | - | 552 |
| cast_u64_complex64 | 128 | 199 | ? ? ? | - | 2563 |
| cast_u64_f16 | 128 | 4130 | ?[64] ? | `memcpy@PLT` | 36678* |
| cast_u64_f32 | 128 | 637 | ?[32] ? ? | `memcpy@PLT` | 6788* |
| cast_u64_f64 | 128 | 89 | ? ? ? | - | 664 |
| cast_u8_complex128 | 128 | 51 | ? ? ? | - | 675 |
| cast_u8_complex64 | 128 | 89 | ? ? ? | - | 1315 |
| cast_u8_f16 | 128 | 3815 | ?[64] ? ? | `memcpy@PLT` | 36101* |
| cast_u8_f32 | 128 | 478 | ?[32] ? ? | `memcpy@PLT` | 5538* |
| cast_u8_f64 | 128 | 58 | ? ? ? | - | 868 |
| cast_u8_i16 | 128 | 808 | ?[64] ? ? ? | `memcpy@PLT` | 6467* |
| cast_u8_i32 | 128 | 39 | ? ? ? | - | 515 |
| cast_u8_i64 | 128 | 58 | ? ? ? | - | 899 |
| ceil_f16 | 128 | 7329 | ?[64] ? | `ceilf@PLT`, `memcpy@PLT` | 108970* |
| ceil_f32 | 128 | 256 | ? ? ? | `ceilf@PLT` | 4106* |
| ceil_f64 | 128 | 128 | ? ? ? | `ceil@PLT` | 2571* |
| clamp_op_descriptor_bool | 128 | 1696 | ?[128] ? ? ? | `memcpy@PLT` | 14277* |
| clamp_op_descriptor_f16 | 128 | 3114 | ?[64 64 64] ? ? ? ? ? | `memcpy@PLT` | 24386* |
| clamp_op_descriptor_f32 | 128 | 445 | ?[32] ? ? | `memcpy@PLT` | 5186* |
| clamp_op_descriptor_f64 | 128 | 40 | ? ? ? | - | 356 |
| clamp_op_descriptor_i16 | 128 | 928 | ?[64] ? ? ? | `memcpy@PLT` | 7747* |
| clamp_op_descriptor_i32 | 128 | 479 | ?[32] ? ? | `memcpy@PLT` | 5027* |
| clamp_op_descriptor_i64 | 128 | 139 | ? ? ? | - | 1183 |
| clamp_op_descriptor_i8 | 128 | 1824 | ?[128] ? ? ? | `memcpy@PLT` | 13475* |
| clamp_op_descriptor_u16 | 128 | 928 | ?[64] ? ? ? | `memcpy@PLT` | 8259* |
| clamp_op_descriptor_u32 | 128 | 479 | ?[32] ? ? | `memcpy@PLT` | 5282* |
| clamp_op_descriptor_u64 | 128 | 139 | ? ? ? | - | 1183 |
| clamp_op_descriptor_u8 | 128 | 1824 | ?[128] ? ? ? | `memcpy@PLT` | 14468* |
| cos_f16 | 128 | 7329 | ?[64] ? | `cosf@PLT`, `memcpy@PLT` | 108970* |
| cos_f32 | 128 | 537 | ?[32] ? ? | `cosf@PLT`, `memcpy@PLT` | 9425* |
| cos_f64 | 128 | 128 | ? ? ? | `cos@PLT` | 2571* |
| count_ones_i16_u32 | 128 | 966 | ?[32] ? ? | `memcpy@PLT` | 8834* |
| count_ones_i32_u32 | 128 | 893 | ?[32] ? ? | `memcpy@PLT` | 8322* |
| count_ones_i64_u32 | 128 | 897 | ?[32] ? ? | `memcpy@PLT` | 8386* |
| count_ones_i8_u32 | 128 | 574 | ?[32] ? ? | `memcpy@PLT` | 5762* |
| count_zeros_i16_u32 | 128 | 998 | ?[32] ? ? | `memcpy@PLT` | 9106* |
| count_zeros_i32_u32 | 128 | 925 | ?[32] ? ? | `memcpy@PLT` | 8578* |
| count_zeros_i64_u32 | 128 | 1089 | ?[32] ? ? | `memcpy@PLT` | 9730* |
| count_zeros_i8_u32 | 128 | 638 | ?[32] ? ? | `memcpy@PLT` | 6274* |
| div_complex128 | 256 | 239 | ? ? ? ? | - | 1031 |
| div_complex64 | 128 | 127 | ? ? ? | - | 1033 |
| div_f16 | 128 | 422 | ?[64] ? | `_RNvXsy_NtCs3Bjx3dgswfe_4half8binary16NtB5_3f16NtNtNtCs2TSF2CumT7P_4core3ops5arith3Div3div`, `memcpy@PLT` | 6309* |
| div_f32 | 512 | 785 | ? ? ? ? | - | 2178 |
| div_f64 | 128 | 35 | ? ? ? | - | 1030 |
| div_i16 | 128 | 1062 | ?[64] ? | `memcpy@PLT` | 106590* |
| div_i32 | 128 | 477 | ?[32] ? | `memcpy@PLT` | 47268* |
| div_i64 | 128 | 531 | ? ? | `__divdi3@PLT` | 10368* |
| div_i8 | 128 | 1957 | ?[128] ? | `memcpy@PLT` | 40972* |
| div_u16 | 128 | 614 | ?[64] ? | `memcpy@PLT` | 35452* |
| div_u32 | 128 | 323 | ?[32] ? | `memcpy@PLT` | 20123* |
| div_u64 | 128 | 397 | ? ? | `__udivdi3@PLT` | 7597* |
| div_u8 | 128 | 1061 | ?[128] ? | `memcpy@PLT` | 40988* |
| equal_complex128_bool | 128 | 2154 | ?[64] ? ? | `memcpy@PLT` | 23774* |
| equal_complex64_bool | 128 | 1736 | ?[32] ? ? | `memcpy@PLT` | 23165* |
| equal_f16_bool | 128 | 424 | ?[16] ? ? | `memcpy@PLT` | 4895* |
| equal_f32_bool | 128 | 278 | ? ? ? | - | 2410 |
| equal_f64_bool | 128 | 742 | ?[64] ? ? | `memcpy@PLT` | 7424* |
| equal_i16_bool | 128 | 118 | ? ? ? | - | 838 |
| equal_i32_bool | 128 | 278 | ? ? ? | - | 2314 |
| equal_i64_bool | 128 | 870 | ?[64] ? ? | `memcpy@PLT` | 9183* |
| equal_i8_bool | 128 | 46 | ? ? ? ? | - | 304 |
| exp_f16 | 128 | 7329 | ?[64] ? | `expf@PLT`, `memcpy@PLT` | 108970* |
| exp_f32 | 128 | 256 | ? ? ? | `expf@PLT` | 4106* |
| exp_f64 | 128 | 128 | ? ? ? | `exp@PLT` | 2571* |
| floor_f16 | 128 | 7329 | ?[64] ? | `floorf@PLT`, `memcpy@PLT` | 108970* |
| floor_f32 | 128 | 256 | ? ? ? | `floorf@PLT` | 4106* |
| floor_f64 | 128 | 128 | ? ? ? | `floor@PLT` | 2571* |
| greater_bool | 128 | 55 | ? ? ? ? | - | 357 |
| greater_equal_bool | 128 | 55 | ? ? ? ? | - | 357 |
| greater_equal_f16_bool | 128 | 694 | ?[16] ? ? | `memcpy@PLT` | 7198* |
| greater_equal_f32_bool | 128 | 278 | ? ? ? | - | 2410 |
| greater_equal_f64_bool | 128 | 742 | ?[64] ? ? | `memcpy@PLT` | 7424* |
| greater_equal_i16_bool | 128 | 118 | ? ? ? | - | 838 |
| greater_equal_i32_bool | 128 | 313 | ? ? ? | - | 2568 |
| greater_equal_i64_bool | 128 | 1384 | ?[64] ? ? | `memcpy@PLT` | 13600* |
| greater_equal_i8_bool | 128 | 46 | ? ? ? ? | - | 304 |
| greater_equal_u16_bool | 128 | 137 | ? ? ? | - | 901 |
| greater_equal_u32_bool | 128 | 424 | ?[32] ? ? | `memcpy@PLT` | 4800* |
| greater_equal_u64_bool | 128 | 1384 | ?[64] ? ? | `memcpy@PLT` | 13600* |
| greater_equal_u8_bool | 128 | 55 | ? ? ? ? | - | 357 |
| greater_f16_bool | 128 | 757 | ?[16] ? ? | `memcpy@PLT` | 7647* |
| greater_f32_bool | 128 | 278 | ? ? ? | - | 2410 |
| greater_f64_bool | 128 | 742 | ?[64] ? ? | `memcpy@PLT` | 7424* |
| greater_i16_bool | 128 | 118 | ? ? ? | - | 838 |
| greater_i32_bool | 128 | 278 | ? ? ? | - | 2314 |
| greater_i64_bool | 128 | 1319 | ?[64] ? ? | `memcpy@PLT` | 13391* |
| greater_i8_bool | 128 | 46 | ? ? ? ? | - | 304 |
| greater_u16_bool | 128 | 155 | ? ? ? | - | 1030 |
| greater_u32_bool | 128 | 347 | ? ? ? | - | 2794 |
| greater_u64_bool | 128 | 1319 | ?[64] ? ? | `memcpy@PLT` | 13391* |
| greater_u8_bool | 128 | 55 | ? ? ? ? | - | 357 |
| imaginary_complex128_f64 | 128 | 54 | ? ? ? | - | 804 |
| imaginary_complex64_f32 | 128 | 126 | ? ? ? | - | 1955 |
| is_finite_f16_bool | 128 | 1951 | ?[128] ? ? ? | `memcpy@PLT` | 14403* |
| is_finite_f32_bool | 128 | 1822 | ?[128] ? ? | `memcpy@PLT` | 15395* |
| is_finite_f64_bool | 128 | 2078 | ?[128] ? ? | `memcpy@PLT` | 15651* |
| is_infinite_f16_bool | 128 | 1951 | ?[128] ? ? ? | `memcpy@PLT` | 13476* |
| is_infinite_f32_bool | 128 | 1822 | ?[128] ? ? | `memcpy@PLT` | 15395* |
| is_infinite_f64_bool | 128 | 2079 | ?[128] ? ? | `memcpy@PLT` | 16612* |
| is_nan_f16_bool | 128 | 1951 | ?[128] ? ? ? | `memcpy@PLT` | 13476* |
| is_nan_f32_bool | 128 | 1823 | ?[128] ? ? | `memcpy@PLT` | 13363* |
| is_nan_f64_bool | 128 | 1823 | ?[128] ? ? | `memcpy@PLT` | 13363* |
| leading_zeros_i16_u32 | 128 | 519 | ?[32] ? ? | `memcpy@PLT` | 5570* |
| leading_zeros_i32_u32 | 128 | 479 | ?[32] ? ? | `memcpy@PLT` | 5218* |
| leading_zeros_i64_u32 | 128 | 671 | ?[32] ? ? | `memcpy@PLT` | 11427* |
| leading_zeros_i8_u32 | 128 | 511 | ?[32] ? ? | `memcpy@PLT` | 5226* |
| less_bool | 128 | 55 | ? ? ? ? | - | 357 |
| less_equal_bool | 128 | 55 | ? ? ? ? | - | 357 |
| less_equal_f16_bool | 128 | 757 | ?[16] ? ? | `memcpy@PLT` | 7871* |
| less_equal_f32_bool | 128 | 278 | ? ? ? | - | 2410 |
| less_equal_f64_bool | 128 | 742 | ?[64] ? ? | `memcpy@PLT` | 7424* |
| less_equal_i16_bool | 128 | 118 | ? ? ? | - | 838 |
| less_equal_i32_bool | 128 | 313 | ? ? ? | - | 2536 |
| less_equal_i64_bool | 128 | 1384 | ?[64] ? ? | `memcpy@PLT` | 13600* |
| less_equal_i8_bool | 128 | 46 | ? ? ? ? | - | 304 |
| less_equal_u16_bool | 128 | 137 | ? ? ? | - | 901 |
| less_equal_u32_bool | 128 | 424 | ?[32] ? ? | `memcpy@PLT` | 4800* |
| less_equal_u64_bool | 128 | 1384 | ?[64] ? ? | `memcpy@PLT` | 13600* |
| less_equal_u8_bool | 128 | 55 | ? ? ? ? | - | 357 |
| less_f16_bool | 128 | 725 | ?[16] ? ? | `memcpy@PLT` | 7390* |
| less_f32_bool | 128 | 278 | ? ? ? | - | 2410 |
| less_f64_bool | 128 | 742 | ?[64] ? ? | `memcpy@PLT` | 7424* |
| less_i16_bool | 128 | 118 | ? ? ? | - | 838 |
| less_i32_bool | 128 | 278 | ? ? ? | - | 2314 |
| less_i64_bool | 128 | 1319 | ?[64] ? ? | `memcpy@PLT` | 13391* |
| less_i8_bool | 128 | 46 | ? ? ? ? | - | 304 |
| less_u16_bool | 128 | 155 | ? ? ? | - | 1030 |
| less_u32_bool | 128 | 347 | ? ? ? | - | 2794 |
| less_u64_bool | 128 | 1319 | ?[64] ? ? | `memcpy@PLT` | 13391* |
| less_u8_bool | 128 | 55 | ? ? ? ? | - | 357 |
| ln_f16 | 128 | 7329 | ?[64] ? | `logf@PLT`, `memcpy@PLT` | 108970* |
| ln_f32 | 128 | 256 | ? ? ? | `logf@PLT` | 4106* |
| ln_f64 | 128 | 128 | ? ? ? | `log@PLT` | 2571* |
| maximum_bool | 128 | 37 | ? ? ? ? | - | 293 |
| maximum_f16 | 128 | 445 | ? ? ? | - | 3569 |
| maximum_f32 | 128 | 549 | ?[32] ? ? | `memcpy@PLT` | 5826* |
| maximum_f64 | 128 | 256 | ? ? | - | 1626 |
| maximum_i16 | 128 | 37 | ? ? ? | - | 293 |
| maximum_i32 | 128 | 67 | ? ? ? | - | 431 |
| maximum_i64 | 128 | 147 | ? ? ? | - | 1285 |
| maximum_i8 | 128 | 71 | ? ? ? ? | - | 442 |
| maximum_u16 | 128 | 45 | ? ? ? | - | 294 |
| maximum_u32 | 128 | 91 | ? ? ? | - | 687 |
| maximum_u64 | 128 | 147 | ? ? ? | - | 1285 |
| maximum_u8 | 128 | 37 | ? ? ? ? | - | 293 |
| minimum_bool | 128 | 37 | ? ? ? ? | - | 293 |
| minimum_f16 | 128 | 417 | ? ? ? | - | 3516 |
| minimum_f32 | 128 | 549 | ?[32] ? ? | `memcpy@PLT` | 5826* |
| minimum_f64 | 128 | 256 | ? ? | - | 1626 |
| minimum_i16 | 128 | 37 | ? ? ? | - | 293 |
| minimum_i32 | 128 | 67 | ? ? ? | - | 431 |
| minimum_i64 | 128 | 147 | ? ? ? | - | 1285 |
| minimum_i8 | 128 | 72 | ? ? ? ? | - | 452 |
| minimum_u16 | 128 | 55 | ? ? ? | - | 357 |
| minimum_u32 | 128 | 91 | ? ? ? | - | 687 |
| minimum_u64 | 128 | 147 | ? ? ? | - | 1285 |
| minimum_u8 | 128 | 37 | ? ? ? ? | - | 293 |
| mul_complex128 | 128 | 123 | ? ? ? | - | 1030 |
| mul_complex64 | 512 | 941 | ? ? ? ? | - | 2318 |
| mul_f16 | 128 | 422 | ?[64] ? | `_RNvNtNtCs3Bjx3dgswfe_4half8binary164arch21multiply_f16_fallback`, `memcpy@PLT` | 6309* |
| mul_f32 | 128 | 35 | ? ? ? | - | 260 |
| mul_f64 | 128 | 35 | ? ? ? | - | 260 |
| mul_i16 | 128 | 37 | ? ? ? | - | 294 |
| mul_i32 | 128 | 83 | ? ? ? | - | 1284 |
| mul_i64 | 128 | 107 | ? ? ? | - | 858 |
| mul_i8 | 128 | 128 | ? ? ? ? | - | 1285 |
| neg_complex128 | 128 | 31 | ? ? ? | - | 292 |
| neg_complex64 | 128 | 30 | ? ? ? | - | 292 |
| neg_f16 | 128 | 30 | ? ? ? ? | - | 292 |
| neg_f32 | 128 | 30 | ? ? ? | - | 292 |
| neg_f64 | 128 | 30 | ? ? ? | - | 292 |
| neg_i16 | 128 | 799 | ?[64] ? ? ? | `memcpy@PLT` | 7304* |
| neg_i32 | 128 | 36 | ? ? ? | - | 291 |
| neg_i64 | 128 | 36 | ? ? ? | - | 291 |
| neg_i8 | 128 | 1567 | ?[128] ? ? ? | `memcpy@PLT` | 13448* |
| not_bool | 128 | 1823 | ?[128] ? ? ? | `memcpy@PLT` | 13368* |
| not_equal_complex128_bool | 128 | 2219 | ?[64] ? ? | `memcpy@PLT` | 24029* |
| not_equal_complex64_bool | 128 | 1930 | ?[32] ? ? | `memcpy@PLT` | 24733* |
| not_equal_f16_bool | 128 | 440 | ?[16] ? ? | `memcpy@PLT` | 4991* |
| not_equal_f32_bool | 128 | 278 | ? ? ? | - | 2410 |
| not_equal_f64_bool | 128 | 742 | ?[64] ? ? | `memcpy@PLT` | 7424* |
| not_equal_i16_bool | 128 | 118 | ? ? ? | - | 838 |
| not_equal_i32_bool | 128 | 313 | ? ? ? | - | 2536 |
| not_equal_i64_bool | 128 | 935 | ?[64] ? ? | `memcpy@PLT` | 9471* |
| not_equal_i8_bool | 128 | 46 | ? ? ? ? | - | 304 |
| not_i16 | 128 | 29 | ? ? ? ? | - | 292 |
| not_i32 | 128 | 27 | ? ? ? | - | 259 |
| not_i64 | 128 | 27 | ? ? ? | - | 259 |
| not_i8 | 128 | 29 | ? ? ? ? | - | 292 |
| or_i16 | 128 | 37 | ? ? ? | - | 293 |
| or_i32 | 128 | 35 | ? ? ? | - | 259 |
| or_i64 | 128 | 35 | ? ? ? | - | 259 |
| or_i8 | 128 | 37 | ? ? ? ? | - | 293 |
| pow_f32 | 128 | 383 | ? ? ? | `powf@PLT` | 7193* |
| pow_f32_i32 | 128 | 383 | ? ? | `__powisf2@PLT` | 6923* |
| pow_f64 | 128 | 191 | ? ? ? | `pow@PLT` | 3595* |
| pow_f64_i32 | 128 | 191 | ? ? | `__powidf2@PLT` | 3436* |
| real_complex128_f64 | 128 | 54 | ? ? ? | - | 804 |
| real_complex64_f32 | 128 | 126 | ? ? ? | - | 1955 |
| reverse_bits_i16 | 128 | 204 | ? ? ? | - | 1898 |
| reverse_bits_i32 | 128 | 249 | ? ? ? | - | 2310 |
| reverse_bits_i64 | 128 | 265 | ? ? ? | - | 2567 |
| reverse_bits_i8 | 128 | 190 | ? ? ? ? | - | 1546 |
| round_f16 | 128 | 7329 | ?[64] ? | `memcpy@PLT`, `roundf@PLT` | 108970* |
| round_f32 | 128 | 256 | ? ? ? | `roundf@PLT` | 4106* |
| round_f64 | 128 | 128 | ? ? ? | `round@PLT` | 2571* |
| sign_f16 | 128 | 7201 | ?[64] ? ? | `memcpy@PLT` | 80523* |
| sign_f32 | 128 | 703 | ?[32] ? ? | `memcpy@PLT` | 6883* |
| sign_f64 | 128 | 205 | ? ? | - | 1348 |
| sign_i16 | 128 | 1055 | ?[64] ? ? ? | `memcpy@PLT` | 8827* |
| sign_i32 | 128 | 541 | ?[32] ? ? | `memcpy@PLT` | 5602* |
| sign_i64 | 128 | 134 | ? ? ? | - | 1049 |
| sign_i8 | 128 | 1954 | ?[128] ? ? ? | `memcpy@PLT` | 14499* |
| sign_u16 | 128 | 927 | ?[64] ? ? ? | `memcpy@PLT` | 7427* |
| sign_u32 | 128 | 445 | ?[32] ? ? | `memcpy@PLT` | 4898* |
| sign_u64 | 128 | 59 | ? ? ? | - | 388 |
| sign_u8 | 128 | 1695 | ?[128] ? ? ? | `memcpy@PLT` | 12483* |
| sin_f16 | 128 | 7329 | ?[64] ? | `memcpy@PLT`, `sinf@PLT` | 108970* |
| sin_f32 | 128 | 256 | ? ? ? | `sinf@PLT` | 4106* |
| sin_f64 | 128 | 128 | ? ? ? | `sin@PLT` | 2571* |
| sqrt_f16 | 128 | 6945 | ?[64] ? | `memcpy@PLT` | 77581* |
| sqrt_f32 | 128 | 27 | ? ? ? | - | 774 |
| sqrt_f64 | 128 | 27 | ? ? ? | - | 1543 |
| square_complex128 | 128 | 102 | ? ? ? | - | 806 |
| square_complex64 | 128 | 119 | ? ? ? | - | 1286 |
| square_f16 | 128 | 413 | ?[64] ? | `_RNvNtNtCs3Bjx3dgswfe_4half8binary164arch21multiply_f16_fallback`, `memcpy@PLT` | 6124* |
| square_f32 | 128 | 412 | ?[32] ? ? | `memcpy@PLT` | 5091* |
| square_f64 | 128 | 27 | ? ? ? | - | 260 |
| square_i16 | 128 | 27 | ? ? ? ? | - | 261 |
| square_i32 | 128 | 67 | ? ? ? | - | 1029 |
| square_i64 | 128 | 67 | ? ? ? | - | 517 |
| square_i8 | 128 | 91 | ? ? ? ? | - | 805 |
| sub_complex128 | 128 | 36 | ? ? ? | - | 260 |
| sub_complex64 | 128 | 35 | ? ? ? | - | 260 |
| sub_f16 | 128 | 422 | ?[64] ? | `_RNvXsm_NtCs3Bjx3dgswfe_4half8binary16NtB5_3f16NtNtNtCs2TSF2CumT7P_4core3ops5arith3Sub3sub`, `memcpy@PLT` | 6309* |
| sub_f32 | 128 | 35 | ? ? ? | - | 260 |
| sub_f64 | 128 | 35 | ? ? ? | - | 260 |
| sub_i16 | 128 | 37 | ? ? ? | - | 293 |
| sub_i32 | 128 | 35 | ? ? ? | - | 259 |
| sub_i64 | 128 | 35 | ? ? ? | - | 259 |
| sub_i8 | 128 | 37 | ? ? ? ? | - | 293 |
| swap_bytes_i16 | 128 | 53 | ? ? ? ? | - | 346 |
| swap_bytes_i32 | 128 | 83 | ? ? ? | - | 1796 |
| swap_bytes_i64 | 128 | 99 | ? ? ? | - | 2308 |
| tan_f16 | 128 | 7330 | ?[64] ? | `memcpy@PLT`, `tanf@PLT` | 109116* |
| tan_f32 | 128 | 257 | ? ? | `tanf@PLT` | 4170* |
| tan_f64 | 128 | 129 | ? ? | `tan@PLT` | 2575* |
| trailing_zeros_i16_u32 | 128 | 486 | ?[32] ? ? | `memcpy@PLT` | 5058* |
| trailing_zeros_i32_u32 | 128 | 447 | ?[32] ? ? | `memcpy@PLT` | 4962* |
| trailing_zeros_i64_u32 | 128 | 542 | ?[32] ? | `memcpy@PLT` | 13506* |
| trailing_zeros_i8_u32 | 128 | 478 | ?[32] ? ? | `memcpy@PLT` | 4963* |
| xor_i16 | 128 | 37 | ? ? ? | - | 293 |
| xor_i32 | 128 | 35 | ? ? ? | - | 259 |
| xor_i64 | 128 | 35 | ? ? ? | - | 259 |
| xor_i8 | 128 | 37 | ? ? ? ? | - | 293 |
| **geomean** | | | | | **2679*** |

Warnings:

- pow_i16_u16: hot loop with an inner loop of unknown trip count
- pow_i32_u32: hot loop with an inner loop of unknown trip count
- pow_i64_u32: hot loop with an inner loop of unknown trip count
- pow_i8_u8: hot loop with an inner loop of unknown trip count

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| abs_complex128_f64 | 128 | 100 | ? ? | `hypot` | 3757* | 4713* | 1696* |
| abs_complex64_f32 | 128 | 214 | ? ? | `hypotf` | 7050* | 9158* | 3566* |
| abs_f16 | 128 | 5723 | ?[64] ? ? | - | 68238 | 62628 | 31111 |
| abs_f32 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| abs_f64 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| abs_i16 | 128 | 925 | ?[64] ? ? ? | - | 13475 | 12482 | 6050 |
| abs_i32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| abs_i64 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| abs_i8 | 128 | 2077 | ?[128] ? ? ? | - | 29825 | 25793 | 12706 |
| acos_f16 | 128 | 5601 | ?[64] ? | `acosf` | 125483* | 123114* | 56924* |
| acos_f32 | 128 | 170 | ? ? | `acosf` | 4394* | 7558* | 2736* |
| acos_f64 | 128 | 81 | ? ? | `acos` | 2752* | 3753* | 1288* |
| add_complex128 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| add_complex64 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| add_f16 | 128 | 492 | ?[64] ? | `_RNvNtNtCs9fgWqoz3mDR_4half8binary164arch16add_f16_fallback` | 20615* | 17350* | 6734* |
| add_f32 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| add_f64 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| add_i16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| add_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| add_i64 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| add_i8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| and_i16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| and_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| and_i64 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| and_i8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| approx_eq_complex128_bool | 128 | 7206 | ?[128] ? | `memcpy` | 84806* | 97028* | 45573* |
| approx_eq_complex64_bool | 128 | 7206 | ?[128] ? | `memcpy` | 84806* | 97028* | 45573* |
| approx_eq_f16_bool | 128 | 76365 | ?[128] ? | - | 886955 | 817797 | 410412 |
| approx_eq_f32_bool | 128 | 4132 | ?[128] ? | `memcpy` | 47879* | 60068* | 26437* |
| approx_eq_f64_bool | 128 | 4132 | ?[128] ? | `memcpy` | 47879* | 60068* | 26437* |
| asin_f16 | 128 | 5601 | ?[64] ? | `asinf` | 125483* | 123114* | 56924* |
| asin_f32 | 128 | 170 | ? ? | `asinf` | 4394* | 7558* | 2736* |
| asin_f64 | 128 | 81 | ? ? | `asin` | 2752* | 3753* | 1288* |
| atan_f16 | 128 | 5601 | ?[64] ? | `atanf` | 125483* | 123114* | 56924* |
| atan_f32 | 128 | 170 | ? ? | `atanf` | 4394* | 7558* | 2736* |
| atan_f64 | 128 | 81 | ? ? | `atan` | 2752* | 3753* | 1288* |
| bitwise_rotate_left_i16_u32 | 128 | 85 | ? ? ? ? | - | 2052 | 1634 | 740 |
| bitwise_rotate_left_i32_u32 | 128 | 74 | ? ? ? | - | 1572 | 1315 | 580 |
| bitwise_rotate_left_i64_u32 | 128 | 62 | ? ? ? | - | 1026 | 1057 | 482 |
| bitwise_rotate_left_i8_u32 | 128 | 94 | ? ? ? ? | - | 2788 | 2050 | 900 |
| bitwise_rotate_right_i16_u32 | 128 | 86 | ? ? ? ? | - | 1956 | 1602 | 708 |
| bitwise_rotate_right_i32_u32 | 128 | 74 | ? ? ? | - | 1508 | 1314 | 548 |
| bitwise_rotate_right_i64_u32 | 128 | 46 | ? ? ? | - | 866 | 801 | 386 |
| bitwise_rotate_right_i8_u32 | 128 | 94 | ? ? ? ? | - | 2821 | 2082 | 900 |
| bitwise_shift_left_i16 | 128 | 34 | ? ? ? ? | - | 963 | 897 | 387 |
| bitwise_shift_left_i32 | 128 | 34 | ? ? ? | - | 995 | 898 | 355 |
| bitwise_shift_left_i64 | 128 | 34 | ? ? ? | - | 995 | 898 | 355 |
| bitwise_shift_left_i8 | 128 | 34 | ? ? ? ? | - | 899 | 834 | 355 |
| bitwise_shift_right_i16 | 128 | 41 | ? ? ? ? | - | 1091 | 962 | 419 |
| bitwise_shift_right_i32 | 128 | 42 | ? ? ? | - | 1123 | 1026 | 419 |
| bitwise_shift_right_i64 | 128 | 42 | ? ? ? | - | 1123 | 1026 | 419 |
| bitwise_shift_right_i8 | 128 | 42 | ? ? ? ? | - | 1027 | 930 | 355 |
| bitwise_shift_right_u16 | 128 | 41 | ? ? ? ? | - | 1091 | 962 | 419 |
| bitwise_shift_right_u32 | 128 | 42 | ? ? ? | - | 1123 | 1026 | 419 |
| bitwise_shift_right_u64 | 128 | 42 | ? ? ? | - | 1123 | 1026 | 419 |
| bitwise_shift_right_u8 | 128 | 42 | ? ? ? ? | - | 1027 | 930 | 355 |
| cast_bool_complex128 | 128 | 37 | ? ? ? ? | - | 516 | 705 | 291 |
| cast_bool_complex64 | 128 | 85 | ? ? ? ? | - | 1158 | 1697 | 675 |
| cast_bool_f16 | 128 | 999 | ?[4] ? ? ? | - | 19665 | 15913 | 7720 |
| cast_bool_f32 | 128 | 45 | ? ? ? ? | - | 870 | 995 | 390 |
| cast_bool_f64 | 128 | 47 | ? ? ? ? | - | 936 | 836 | 389 |
| cast_bool_i16 | 128 | 857 | ?[64] ? ? ? | - | 13250 | 11266 | 5474 |
| cast_bool_i32 | 128 | 37 | ? ? ? ? | - | 676 | 801 | 323 |
| cast_bool_i64 | 128 | 39 | ? ? ? ? | - | 678 | 740 | 356 |
| cast_bool_i8 | 128 | 1949 | ?[128] | - | 25762 | 25761 | 12674 |
| cast_complex128 | 128 | 13 | ? ? ? | - | 515 | 483 | 163 |
| cast_complex128_bool | 128 | 2584 | ?[128] ? ? ? | `memcpy` | 35459* | 39426* | 18147* |
| cast_complex128_complex64 | 128 | 37 | ? ? ? | - | 1251 | 898 | 324 |
| cast_complex64 | 128 | 13 | ? ? ? | - | 515 | 483 | 163 |
| cast_complex64_bool | 128 | 2584 | ?[128] ? ? ? | `memcpy` | 35459* | 39426* | 18147* |
| cast_complex64_complex128 | 128 | 21 | ? ? ? | - | 644 | 610 | 196 |
| cast_f16 | 128 | 6043 | ?[64] ? ? | - | 79311 | 66695 | 33162 |
| cast_f16_bool | 128 | 1139 | ?[8] ? ? ? | - | 21233 | 18471 | 8841 |
| cast_f16_complex128 | 128 | 340 | ? ? ? | - | 3629 | 3874 | 1922 |
| cast_f16_complex64 | 128 | 178 | ? ? ? | - | 3045 | 2756 | 1437 |
| cast_f16_f32 | 128 | 339 | ? ? ? | - | 5653 | 5300 | 2628 |
| cast_f16_f64 | 128 | 174 | ? ? ? | - | 3174 | 2756 | 1372 |
| cast_f16_i16 | 128 | 690 | ?[8] ? ? | - | 11985 | 11177 | 5386 |
| cast_f16_i32 | 128 | 349 | ? ? ? | - | 5911 | 5621 | 2773 |
| cast_f16_i64 | 128 | 182 | ? ? ? | - | 3430 | 2885 | 1437 |
| cast_f16_i8 | 128 | 1523 | ?[8] ? ? ? | - | 27184 | 24807 | 11977 |
| cast_f16_u16 | 128 | 690 | ?[8] ? ? | - | 11985 | 11177 | 5386 |
| cast_f16_u32 | 128 | 349 | ? ? ? | - | 5911 | 5621 | 2773 |
| cast_f16_u64 | 128 | 182 | ? ? ? | - | 3430 | 2885 | 1437 |
| cast_f16_u8 | 128 | 1475 | ?[8] ? ? ? | - | 26096 | 24072 | 11465 |
| cast_f32 | 128 | 13 | ? | - | 515 | 483 | 163 |
| cast_f32_bool | 128 | 2200 | ?[128] ? ? ? | `memcpy` | 29346* | 31330* | 15107* |
| cast_f32_complex128 | 128 | 33 | ? ? ? | - | 515 | 801 | 291 |
| cast_f32_complex64 | 128 | 45 | ? ? ? | - | 1026 | 1313 | 515 |
| cast_f32_f16 | 128 | 3171 | ?[64] ? ? | - | 41582 | 36388 | 17573 |
| cast_f32_f64 | 128 | 21 | ? ? ? | - | 644 | 609 | 196 |
| cast_f32_i16 | 128 | 1125 | ?[64] ? ? ? | - | 15910 | 15971 | 7108 |
| cast_f32_i32 | 128 | 21 | ? ? ? | - | 739 | 771 | 292 |
| cast_f32_i64 | 128 | 29 | ? ? ? | - | 899 | 739 | 292 |
| cast_f32_i8 | 128 | 2460 | ?[128] ? ? ? | `memcpy` | 35269* | 35363* | 16132* |
| cast_f32_u16 | 128 | 997 | ?[64] ? ? ? | - | 14885 | 15907 | 6660 |
| cast_f32_u32 | 128 | 21 | ? ? ? | - | 739 | 771 | 292 |
| cast_f32_u64 | 128 | 29 | ? ? ? | - | 899 | 739 | 292 |
| cast_f32_u8 | 128 | 2200 | ?[128] ? ? ? | `memcpy` | 31364* | 35298* | 15108* |
| cast_f64 | 128 | 14 | ? | - | 516 | 451 | 163 |
| cast_f64_bool | 128 | 2200 | ?[128] ? ? ? | `memcpy` | 29346* | 31330* | 15107* |
| cast_f64_complex128 | 128 | 25 | ? ? ? | - | 514 | 673 | 259 |
| cast_f64_complex64 | 128 | 61 | ? ? ? | - | 1027 | 1505 | 547 |
| cast_f64_f16 | 128 | 3292 | ?[64] ? ? | `memcpy` | 45896* | 39527* | 20293* |
| cast_f64_f32 | 128 | 37 | ? ? ? | - | 1251 | 898 | 324 |
| cast_f64_i16 | 128 | 1116 | ?[64] ? ? | `memcpy` | 16902* | 16963* | 8331* |
| cast_f64_i32 | 128 | 69 | ? ? ? | - | 1573 | 2114 | 1027 |
| cast_f64_i64 | 128 | 21 | ? ? ? | - | 739 | 610 | 228 |
| cast_f64_i8 | 128 | 2460 | ?[128] ? ? ? | `memcpy` | 35269* | 35363* | 16132* |
| cast_f64_u16 | 128 | 984 | ?[64] ? ? | `memcpy` | 15941* | 16931* | 7876* |
| cast_f64_u32 | 128 | 69 | ? ? ? | - | 1573 | 2114 | 1027 |
| cast_f64_u64 | 128 | 21 | ? ? ? | - | 739 | 610 | 228 |
| cast_f64_u8 | 128 | 2200 | ?[128] ? ? ? | `memcpy` | 31364* | 35298* | 15108* |
| cast_i16 | 128 | 13 | ? | - | 515 | 483 | 163 |
| cast_i16_bool | 128 | 48 | ? ? ? ? | - | 1251 | 1089 | 451 |
| cast_i16_complex128 | 128 | 37 | ? ? ? | - | 517 | 833 | 291 |
| cast_i16_complex64 | 128 | 69 | ? ? ? ? | - | 1031 | 1601 | 548 |
| cast_i16_f16 | 128 | 3227 | ?[64] ? ? | - | 47851 | 37031 | 18311 |
| cast_i16_f32 | 128 | 29 | ? ? ? ? | - | 709 | 834 | 293 |
| cast_i16_f64 | 128 | 45 | ? ? ? | - | 1029 | 1026 | 516 |
| cast_i16_i32 | 128 | 21 | ? ? ? ? | - | 516 | 642 | 195 |
| cast_i16_i64 | 128 | 22 | ? ? ? ? | - | 516 | 547 | 228 |
| cast_i16_i8 | 128 | 25 | ? ? ? ? | - | 899 | 802 | 323 |
| cast_i32_bool | 128 | 89 | ? ? ? ? | - | 2244 | 1794 | 772 |
| cast_i32_complex128 | 128 | 33 | ? ? ? | - | 517 | 802 | 307 |
| cast_i32_complex64 | 128 | 61 | ? ? ? | - | 1028 | 1505 | 643 |
| cast_i32_f16 | 128 | 3299 | ?[64] ? ? | - | 48228 | 38438 | 18951 |
| cast_i32_f32 | 128 | 21 | ? ? ? | - | 740 | 771 | 292 |
| cast_i32_f64 | 128 | 29 | ? ? ? | - | 709 | 737 | 228 |
| cast_i32_i16 | 128 | 25 | ? ? ? ? | - | 899 | 834 | 323 |
| cast_i32_i64 | 128 | 24 | ? ? ? | - | 515 | 641 | 227 |
| cast_i32_i8 | 128 | 33 | ? ? ? ? | - | 1669 | 1186 | 517 |
| cast_i64_bool | 128 | 185 | ? ? ? ? | - | 4581 | 3299 | 1476 |
| cast_i64_complex128 | 128 | 33 | ? ? ? | - | 548 | 801 | 291 |
| cast_i64_complex64 | 128 | 61 | ? ? ? | - | 1029 | 1794 | 563 |
| cast_i64_f16 | 128 | 3292 | ?[64] ? ? | `memcpy` | 48616* | 39496* | 20251* |
| cast_i64_f32 | 128 | 69 | ? ? ? | - | 1574 | 2083 | 1028 |
| cast_i64_f64 | 128 | 21 | ? ? ? | - | 740 | 610 | 228 |
| cast_i64_i16 | 128 | 49 | ? ? ? ? | - | 1667 | 1282 | 547 |
| cast_i64_i32 | 128 | 25 | ? ? ? | - | 899 | 770 | 323 |
| cast_i64_i8 | 128 | 78 | ? ? ? ? | - | 3688 | 2690 | 1252 |
| cast_i8 | 128 | 13 | ? | - | 515 | 483 | 163 |
| cast_i8_bool | 128 | 29 | ? ? ? ? | - | 644 | 642 | 259 |
| cast_i8_complex128 | 128 | 37 | ? ? ? ? | - | 516 | 833 | 292 |
| cast_i8_complex64 | 128 | 69 | ? ? ? ? | - | 1031 | 1601 | 548 |
| cast_i8_f16 | 128 | 3287 | ?[64] ? ? ? | - | 47036 | 37864 | 18758 |
| cast_i8_f32 | 128 | 29 | ? ? ? ? | - | 933 | 705 | 348 |
| cast_i8_f64 | 128 | 45 | ? ? ? ? | - | 1029 | 1026 | 516 |
| cast_i8_i16 | 128 | 921 | ?[64] ? ? ? | - | 13251 | 13281 | 5826 |
| cast_i8_i32 | 128 | 29 | ? ? ? ? | - | 612 | 706 | 292 |
| cast_i8_i64 | 128 | 43 | ? ? ? ? | - | 710 | 1028 | 548 |
| cast_u16_complex128 | 128 | 37 | ? ? ? | - | 518 | 769 | 292 |
| cast_u16_complex64 | 128 | 69 | ? ? ? ? | - | 1031 | 1409 | 676 |
| cast_u16_f16 | 128 | 3099 | ?[64] ? ? | - | 38025 | 36004 | 17798 |
| cast_u16_f32 | 128 | 29 | ? ? ? ? | - | 709 | 834 | 293 |
| cast_u16_f64 | 128 | 45 | ? ? ? | - | 1029 | 929 | 292 |
| cast_u16_i32 | 128 | 21 | ? ? ? ? | - | 516 | 642 | 195 |
| cast_u16_i64 | 128 | 22 | ? ? ? ? | - | 516 | 547 | 228 |
| cast_u32_complex128 | 128 | 33 | ? ? ? | - | 548 | 801 | 291 |
| cast_u32_complex64 | 128 | 61 | ? ? ? | - | 1028 | 1505 | 643 |
| cast_u32_f16 | 128 | 3171 | ?[64] ? ? | - | 40007 | 36388 | 18278 |
| cast_u32_f32 | 128 | 21 | ? ? ? | - | 740 | 771 | 292 |
| cast_u32_f64 | 128 | 29 | ? ? ? | - | 709 | 737 | 228 |
| cast_u32_i64 | 128 | 21 | ? ? ? | - | 516 | 642 | 195 |
| cast_u64_complex128 | 128 | 33 | ? ? ? | - | 548 | 801 | 291 |
| cast_u64_complex64 | 128 | 61 | ? ? ? | - | 1029 | 1794 | 563 |
| cast_u64_f16 | 128 | 3164 | ?[64] ? ? | `memcpy` | 44079* | 37511* | 19253* |
| cast_u64_f32 | 128 | 69 | ? ? ? | - | 1574 | 2083 | 1028 |
| cast_u64_f64 | 128 | 21 | ? ? ? | - | 740 | 610 | 228 |
| cast_u8_complex128 | 128 | 37 | ? ? ? ? | - | 517 | 833 | 292 |
| cast_u8_complex64 | 128 | 69 | ? ? ? ? | - | 1031 | 1601 | 676 |
| cast_u8_f16 | 128 | 3159 | ?[64] ? ? ? | - | 42632 | 35812 | 18086 |
| cast_u8_f32 | 128 | 37 | ? ? ? ? | - | 805 | 899 | 353 |
| cast_u8_f64 | 128 | 45 | ? ? ? ? | - | 1029 | 1058 | 292 |
| cast_u8_i16 | 128 | 793 | ?[64] ? ? ? | - | 11234 | 11265 | 5250 |
| cast_u8_i32 | 128 | 29 | ? ? ? ? | - | 612 | 706 | 292 |
| cast_u8_i64 | 128 | 35 | ? ? ? ? | - | 614 | 708 | 356 |
| ceil_f16 | 128 | 5275 | ?[64] ? ? | - | 76366 | 58505 | 29068 |
| ceil_f32 | 128 | 21 | ? ? ? | - | 739 | 771 | 292 |
| ceil_f64 | 128 | 21 | ? ? ? | - | 739 | 610 | 228 |
| clamp_op_descriptor_bool | 128 | 1692 | ?[128] ? ? ? | - | 21634 | 21665 | 9955 |
| clamp_op_descriptor_f16 | 128 | 2914 | ?[64 64 64] ? ? ? ? | - | 38116 | 36098 | 17635 |
| clamp_op_descriptor_f32 | 128 | 32 | ? ? ? | - | 994 | 769 | 292 |
| clamp_op_descriptor_f64 | 128 | 32 | ? ? ? | - | 994 | 769 | 292 |
| clamp_op_descriptor_i16 | 128 | 989 | ?[64] ? ? ? | - | 15490 | 13505 | 6530 |
| clamp_op_descriptor_i32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| clamp_op_descriptor_i64 | 128 | 30 | ? ? ? | - | 802 | 705 | 260 |
| clamp_op_descriptor_i8 | 128 | 2205 | ?[128] ? ? ? | - | 29826 | 27841 | 13730 |
| clamp_op_descriptor_u16 | 128 | 989 | ?[64] ? ? ? | - | 15490 | 13505 | 6530 |
| clamp_op_descriptor_u32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| clamp_op_descriptor_u64 | 128 | 30 | ? ? ? | - | 802 | 705 | 260 |
| clamp_op_descriptor_u8 | 128 | 2205 | ?[128] ? ? ? | - | 29826 | 27841 | 13730 |
| cos_f16 | 128 | 5602 | ?[64] ? ? | `cosf` | 125483* | 123051* | 56813* |
| cos_f32 | 128 | 170 | ? ? ? | `cosf` | 4394* | 7590* | 2736* |
| cos_f64 | 128 | 81 | ? ? ? | `cos` | 2752* | 3753* | 1288* |
| count_ones_i16_u32 | 128 | 537 | ?[32] ? ? ? | - | 8134 | 7684 | 3909 |
| count_ones_i32_u32 | 128 | 541 | ?[32] ? ? | - | 8326 | 7876 | 3973 |
| count_ones_i64_u32 | 128 | 517 | ?[32] ? ? | - | 7717 | 8740 | 3941 |
| count_ones_i8_u32 | 128 | 503 | ?[32] ? ? ? | - | 7045 | 7075 | 3396 |
| count_zeros_i16_u32 | 128 | 569 | ?[32] ? ? ? | - | 8167 | 8196 | 4101 |
| count_zeros_i32_u32 | 128 | 573 | ?[32] ? ? | - | 8358 | 8388 | 4165 |
| count_zeros_i64_u32 | 128 | 581 | ?[32] ? ? | - | 8742 | 8772 | 4293 |
| count_zeros_i8_u32 | 128 | 535 | ?[32] ? ? ? | - | 8037 | 7107 | 3652 |
| div_complex128 | 128 | 124 | ? ? ? | - | 16390 | 3145 | 1992 |
| div_complex64 | 128 | 71 | ? ? ? | - | 8777 | 1864 | 1287 |
| div_f16 | 128 | 492 | ?[64] ? | `_RNvXsy_NtCs9fgWqoz3mDR_4half8binary16NtB5_3f16NtNtNtCscldGzOC5D0D_4core3ops5arith3Div3div` | 20615* | 17350* | 6734* |
| div_f32 | 128 | 26 | ? ? ? | - | 8708 | 1797 | 1284 |
| div_f64 | 128 | 26 | ? ? ? | - | 16388 | 1798 | 1796 |
| div_i16 | 128 | 937 | ?[64] ? | - | 39406 | 11909 | 24583 |
| div_i32 | 128 | 425 | ?[32] ? | - | 19854 | 6535 | 12295 |
| div_i64 | 128 | 182 | ? ? | - | 18436 | 3620 | 10242 |
| div_i8 | 128 | 1960 | ?[128] ? | - | 78222 | 24165 | 49159 |
| div_u16 | 128 | 552 | ?[64] ? | - | 39441 | 11621 | 24584 |
| div_u32 | 128 | 296 | ?[32] ? | - | 19921 | 6501 | 12296 |
| div_u64 | 128 | 71 | ? ? | - | 17928 | 2571 | 10244 |
| div_u8 | 128 | 1191 | ?[128] ? | - | 78289 | 21862 | 49160 |
| equal_complex128_bool | 128 | 485 | ?[8] ? ? ? | `memcpy` | 16043* | 12068* | 6033* |
| equal_complex64_bool | 128 | 293 | ?[8] ? ? ? | `memcpy` | 9707* | 7779* | 4048* |
| equal_f16_bool | 128 | 330 | ?[8] ? ? ? | - | 7339 | 6309 | 2886 |
| equal_f32_bool | 128 | 105 | ? ? ? ? | - | 3332 | 2529 | 1188 |
| equal_f64_bool | 128 | 216 | ? ? ? ? | - | 6885 | 5089 | 2276 |
| equal_i16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| equal_i32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| equal_i64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| equal_i8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| exp_f16 | 128 | 5602 | ?[64] ? ? | `expf` | 125483* | 123051* | 56813* |
| exp_f32 | 128 | 170 | ? ? ? | `expf` | 4394* | 7590* | 2736* |
| exp_f64 | 128 | 81 | ? ? ? | `exp` | 2752* | 3753* | 1288* |
| floor_f16 | 128 | 5275 | ?[64] ? ? | - | 76366 | 58505 | 29068 |
| floor_f32 | 128 | 21 | ? ? ? | - | 739 | 771 | 292 |
| floor_f64 | 128 | 21 | ? ? ? | - | 739 | 610 | 228 |
| greater_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| greater_equal_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| greater_equal_f16_bool | 128 | 458 | ?[8] ? ? ? | - | 10316 | 7880 | 3658 |
| greater_equal_f32_bool | 128 | 105 | ? ? ? ? | - | 3332 | 2529 | 1188 |
| greater_equal_f64_bool | 128 | 216 | ? ? ? ? | - | 6885 | 5089 | 2276 |
| greater_equal_i16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| greater_equal_i32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| greater_equal_i64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| greater_equal_i8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| greater_equal_u16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| greater_equal_u32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| greater_equal_u64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| greater_equal_u8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| greater_f16_bool | 128 | 3513 | ?[128] ? ? ? | - | 47460 | 41315 | 19940 |
| greater_f32_bool | 128 | 105 | ? ? ? ? | - | 3332 | 2529 | 1188 |
| greater_f64_bool | 128 | 216 | ? ? ? ? | - | 6885 | 5089 | 2276 |
| greater_i16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| greater_i32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| greater_i64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| greater_i8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| greater_u16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| greater_u32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| greater_u64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| greater_u8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| imaginary_complex128_f64 | 128 | 29 | ? ? ? | - | 515 | 738 | 259 |
| imaginary_complex64_f32 | 128 | 53 | ? ? ? | - | 1027 | 1187 | 515 |
| is_finite_f16_bool | 128 | 2213 | ?[128] ? ? ? | - | 30210 | 28225 | 13858 |
| is_finite_f32_bool | 128 | 1815 | ?[128] ? ? ? | `memcpy` | 23236* | 25219* | 13060* |
| is_finite_f64_bool | 128 | 1815 | ?[128] ? ? ? | `memcpy` | 23236* | 25219* | 13060* |
| is_infinite_f16_bool | 128 | 2085 | ?[128] ? ? ? | - | 30210 | 26177 | 12834 |
| is_infinite_f32_bool | 128 | 1687 | ?[128] ? ? ? | `memcpy` | 23108* | 23234* | 12100* |
| is_infinite_f64_bool | 128 | 1687 | ?[128] ? ? ? | `memcpy` | 23108* | 23234* | 12100* |
| is_nan_f16_bool | 128 | 2213 | ?[128] ? ? ? | - | 30211 | 30241 | 13858 |
| is_nan_f32_bool | 128 | 2200 | ?[128] ? ? ? | `memcpy` | 29346* | 31330* | 15107* |
| is_nan_f64_bool | 128 | 2200 | ?[128] ? ? ? | `memcpy` | 29346* | 31330* | 15107* |
| leading_zeros_i16_u32 | 128 | 29 | ? ? ? ? | - | 580 | 705 | 228 |
| leading_zeros_i32_u32 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| leading_zeros_i64_u32 | 128 | 69 | ? ? ? | - | 1058 | 1090 | 546 |
| leading_zeros_i8_u32 | 128 | 45 | ? ? ? ? | - | 772 | 866 | 366 |
| less_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| less_equal_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| less_equal_f16_bool | 128 | 3641 | ?[128] ? ? ? | - | 47493 | 43363 | 21284 |
| less_equal_f32_bool | 128 | 105 | ? ? ? ? | - | 3332 | 2529 | 1188 |
| less_equal_f64_bool | 128 | 216 | ? ? ? ? | - | 6885 | 5089 | 2276 |
| less_equal_i16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| less_equal_i32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| less_equal_i64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| less_equal_i8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| less_equal_u16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| less_equal_u32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| less_equal_u64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| less_equal_u8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| less_f16_bool | 128 | 466 | ?[8] ? ? ? | - | 10348 | 7976 | 3659 |
| less_f32_bool | 128 | 105 | ? ? ? ? | - | 3332 | 2529 | 1188 |
| less_f64_bool | 128 | 216 | ? ? ? ? | - | 6885 | 5089 | 2276 |
| less_i16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| less_i32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| less_i64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| less_i8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| less_u16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| less_u32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| less_u64_bool | 128 | 211 | ? ? ? ? | - | 5894 | 5089 | 2243 |
| less_u8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| ln_f16 | 128 | 5602 | ?[64] ? ? | `logf` | 125483* | 123051* | 56813* |
| ln_f32 | 128 | 170 | ? ? ? | `logf` | 4394* | 7590* | 2736* |
| ln_f64 | 128 | 81 | ? ? ? | `log` | 2752* | 3753* | 1288* |
| maximum_bool | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| maximum_f16 | 128 | 344 | ?[8] ? ? | - | 6157 | 5735 | 2381 |
| maximum_f32 | 128 | 77 | ? ? ? | - | 2275 | 1346 | 580 |
| maximum_f64 | 128 | 77 | ? ? ? | - | 2275 | 1346 | 580 |
| maximum_i16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| maximum_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| maximum_i64 | 128 | 34 | ? ? ? | - | 963 | 897 | 356 |
| maximum_i8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| maximum_u16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| maximum_u32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| maximum_u64 | 128 | 34 | ? ? ? | - | 963 | 897 | 356 |
| maximum_u8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| minimum_bool | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| minimum_f16 | 128 | 336 | ?[8] ? ? | - | 6029 | 5671 | 2317 |
| minimum_f32 | 128 | 77 | ? ? ? | - | 2275 | 1346 | 580 |
| minimum_f64 | 128 | 77 | ? ? ? | - | 2275 | 1346 | 580 |
| minimum_i16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| minimum_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| minimum_i64 | 128 | 34 | ? ? ? | - | 963 | 897 | 356 |
| minimum_i8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| minimum_u16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| minimum_u32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| minimum_u64 | 128 | 34 | ? ? ? | - | 963 | 897 | 356 |
| minimum_u8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| mul_complex128 | 128 | 70 | ? ? ? | - | 2019 | 1377 | 548 |
| mul_complex64 | 128 | 89 | ? ? ? | - | 2020 | 1475 | 644 |
| mul_f16 | 128 | 492 | ?[64] ? | `_RNvNtNtCs9fgWqoz3mDR_4half8binary164arch21multiply_f16_fallback` | 20615* | 17350* | 6734* |
| mul_f32 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| mul_f64 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| mul_i16 | 128 | 26 | ? ? ? ? | - | 867 | 866 | 324 |
| mul_i32 | 128 | 26 | ? ? ? | - | 899 | 898 | 292 |
| mul_i64 | 128 | 46 | ? ? ? | - | 835 | 1539 | 386 |
| mul_i8 | 128 | 26 | ? ? ? ? | - | 803 | 738 | 292 |
| neg_complex128 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| neg_complex64 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| neg_f16 | 128 | 21 | ? ? ? ? | - | 579 | 642 | 227 |
| neg_f32 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| neg_f64 | 128 | 21 | ? ? ? | - | 579 | 610 | 227 |
| neg_i16 | 128 | 925 | ?[64] ? ? ? | - | 13442 | 13473 | 5890 |
| neg_i32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| neg_i64 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| neg_i8 | 128 | 2077 | ?[128] ? ? ? | - | 29826 | 25793 | 12706 |
| not_bool | 128 | 2205 | ?[128] ? ? ? | - | 29858 | 27841 | 13730 |
| not_equal_complex128_bool | 128 | 517 | ?[8] ? ? ? | `memcpy` | 16555* | 12356* | 6257* |
| not_equal_complex64_bool | 128 | 309 | ?[8] ? ? ? | `memcpy` | 9932* | 7812* | 4113* |
| not_equal_f16_bool | 128 | 338 | ?[8] ? ? ? | - | 7371 | 6310 | 2918 |
| not_equal_f32_bool | 128 | 105 | ? ? ? ? | - | 3332 | 2529 | 1188 |
| not_equal_f64_bool | 128 | 226 | ? ? ? ? | - | 6855 | 5090 | 2308 |
| not_equal_i16_bool | 128 | 58 | ? ? ? ? | - | 1539 | 1441 | 644 |
| not_equal_i32_bool | 128 | 106 | ? ? ? ? | - | 2853 | 2498 | 1092 |
| not_equal_i64_bool | 128 | 223 | ? ? ? ? | - | 5797 | 4962 | 2276 |
| not_equal_i8_bool | 128 | 34 | ? ? ? ? | - | 835 | 833 | 355 |
| not_i16 | 128 | 21 | ? ? ? ? | - | 579 | 642 | 227 |
| not_i32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| not_i64 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| not_i8 | 128 | 21 | ? ? ? ? | - | 548 | 610 | 227 |
| or_i16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| or_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| or_i64 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| or_i8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| pow_f32 | 128 | 284 | ?[16] ? ? | `powf` | 14279* | 14853* | 4611* |
| pow_f32_i32 | 128 | 240 | ? ? | `__powisf2` | 7178* | 8550* | 3630* |
| pow_f64 | 128 | 114 | ? ? ? | `pow` | 3693* | 4873* | 1760* |
| pow_f64_i32 | 128 | 112 | ? ? | `__powidf2` | 3628* | 4201* | 1728* |
| real_complex128_f64 | 128 | 29 | ? ? ? | - | 515 | 738 | 259 |
| real_complex64_f32 | 128 | 53 | ? ? ? | - | 1027 | 1187 | 515 |
| reverse_bits_i16 | 128 | 29 | ? ? ? ? | - | 707 | 737 | 259 |
| reverse_bits_i32 | 128 | 29 | ? ? ? | - | 707 | 737 | 259 |
| reverse_bits_i64 | 128 | 33 | ? ? ? | - | 675 | 705 | 291 |
| reverse_bits_i8 | 128 | 21 | ? ? ? ? | - | 548 | 578 | 211 |
| round_f16 | 128 | 5275 | ?[64] ? ? | - | 76366 | 58505 | 29068 |
| round_f32 | 128 | 21 | ? ? ? | - | 739 | 771 | 292 |
| round_f64 | 128 | 21 | ? ? ? | - | 739 | 610 | 228 |
| sign_f16 | 128 | 5403 | ?[64] ? ? | - | 72271 | 59561 | 29483 |
| sign_f32 | 128 | 444 | ?[32] ? ? | - | 6275 | 6339 | 2788 |
| sign_f64 | 128 | 40 | ? ? ? | - | 1250 | 833 | 355 |
| sign_i16 | 128 | 989 | ?[64] ? ? ? | - | 13475 | 13505 | 6531 |
| sign_i32 | 128 | 509 | ?[32] ? ? | - | 7331 | 7361 | 3459 |
| sign_i64 | 128 | 37 | ? ? ? | - | 771 | 769 | 323 |
| sign_i8 | 128 | 2205 | ?[128] ? ? ? | - | 29827 | 29857 | 13730 |
| sign_u16 | 128 | 1053 | ?[64] ? ? ? | - | 15492 | 13506 | 6563 |
| sign_u32 | 128 | 29 | ? ? ? | - | 707 | 705 | 259 |
| sign_u64 | 128 | 29 | ? ? ? | - | 707 | 705 | 259 |
| sign_u8 | 128 | 2333 | ?[128] ? ? ? | - | 31146 | 29857 | 14723 |
| sin_f16 | 128 | 5602 | ?[64] ? ? | `sinf` | 125483* | 123051* | 56813* |
| sin_f32 | 128 | 170 | ? ? ? | `sinf` | 4394* | 7590* | 2736* |
| sin_f64 | 128 | 81 | ? ? ? | `sin` | 2752* | 3753* | 1288* |
| sqrt_f16 | 128 | 5275 | ?[64] ? ? | - | 83406 | 58507 | 29070 |
| sqrt_f32 | 128 | 21 | ? ? ? | - | 8708 | 1797 | 1156 |
| sqrt_f64 | 128 | 21 | ? ? ? | - | 16388 | 1799 | 1924 |
| square_complex128 | 128 | 65 | ? ? ? | - | 1028 | 1249 | 452 |
| square_complex64 | 128 | 85 | ? ? ? | - | 1860 | 1316 | 643 |
| square_f16 | 128 | 474 | ?[64] ? | `_RNvNtNtCs9fgWqoz3mDR_4half8binary164arch21multiply_f16_fallback` | 17672* | 16839* | 6484* |
| square_f32 | 128 | 21 | ? ? ? | - | 738 | 610 | 228 |
| square_f64 | 128 | 21 | ? ? ? | - | 738 | 610 | 228 |
| square_i16 | 128 | 21 | ? ? ? ? | - | 740 | 771 | 228 |
| square_i32 | 128 | 21 | ? ? ? | - | 739 | 739 | 228 |
| square_i64 | 128 | 37 | ? ? ? | - | 611 | 1539 | 291 |
| square_i8 | 128 | 21 | ? ? ? ? | - | 708 | 578 | 212 |
| sub_complex128 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| sub_complex64 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| sub_f16 | 128 | 492 | ?[64] ? | `_RNvXsm_NtCs9fgWqoz3mDR_4half8binary16NtB5_3f16NtNtNtCscldGzOC5D0D_4core3ops5arith3Sub3sub` | 20615* | 17350* | 6734* |
| sub_f32 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| sub_f64 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| sub_i16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| sub_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| sub_i64 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| sub_i8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| swap_bytes_i16 | 128 | 21 | ? ? ? ? | - | 579 | 610 | 227 |
| swap_bytes_i32 | 128 | 21 | ? ? ? | - | 611 | 610 | 227 |
| swap_bytes_i64 | 128 | 21 | ? ? ? | - | 611 | 610 | 227 |
| tan_f16 | 128 | 5601 | ?[64] ? | `tanf` | 125483* | 123114* | 56924* |
| tan_f32 | 128 | 170 | ? ? | `tanf` | 4394* | 7558* | 2736* |
| tan_f64 | 128 | 81 | ? ? | `tan` | 2752* | 3753* | 1288* |
| trailing_zeros_i16_u32 | 128 | 45 | ? ? ? ? | - | 772 | 897 | 325 |
| trailing_zeros_i32_u32 | 128 | 37 | ? ? ? | - | 772 | 801 | 323 |
| trailing_zeros_i64_u32 | 128 | 101 | ? ? ? | - | 1314 | 1569 | 722 |
| trailing_zeros_i8_u32 | 128 | 61 | ? ? ? ? | - | 934 | 1060 | 505 |
| xor_i16 | 128 | 26 | ? ? ? ? | - | 739 | 770 | 323 |
| xor_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| xor_i64 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| xor_i8 | 128 | 26 | ? ? ? ? | - | 771 | 802 | 323 |
| **geomean** | | | | | **2824*** | **2672*** | **1160*** |

Warnings:

- pow_i16_u16: hot loop with an inner loop of unknown trip count
- pow_i32_u32: hot loop with an inner loop of unknown trip count
- pow_i64_u32: hot loop with an inner loop of unknown trip count
- pow_i8_u8: hot loop with an inner loop of unknown trip count

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| abs_complex128_f64 | 128 | 98 | ? ? | `_hypot` | 2268* |
| abs_complex64_f32 | 128 | 212 | ? ? | `_hypotf` | 4008* |
| abs_f16 | 128 | 860 | ?[64] ? | - | 5606 |
| abs_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| abs_f64 | 128 | 21 | ? ? ? | - | 196 |
| abs_i16 | 128 | 989 | ?[64] ? ? ? | - | 7781 |
| abs_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| abs_i64 | 128 | 21 | ? ? ? | - | 196 |
| abs_i8 | 128 | 1949 | ?[128] ? ? ? | - | 15461 |
| acos_f16 | 128 | 864 | ?[64] ? | `_acosf` | 16556* |
| acos_f32 | 128 | 170 | ? ? | `_acosf` | 3367* |
| acos_f64 | 128 | 81 | ? ? | `_acos` | 1641* |
| add_complex128 | 128 | 26 | ? ? ? | - | 260 |
| add_complex64 | 128 | 26 | ? ? ? | - | 260 |
| add_f16 | 128 | 488 | ?[64] ? | - | 4484 |
| add_f32 | 128 | 26 | ? ? ? ? | - | 260 |
| add_f64 | 128 | 26 | ? ? ? | - | 260 |
| add_i16 | 128 | 26 | ? ? ? ? | - | 260 |
| add_i32 | 128 | 26 | ? ? ? ? | - | 259 |
| add_i64 | 128 | 26 | ? ? ? | - | 259 |
| add_i8 | 128 | 26 | ? ? ? ? | - | 261 |
| and_i16 | 128 | 26 | ? ? ? ? | - | 260 |
| and_i32 | 128 | 26 | ? ? ? ? | - | 259 |
| and_i64 | 128 | 26 | ? ? ? | - | 259 |
| and_i8 | 128 | 26 | ? ? ? ? | - | 261 |
| approx_eq_complex128_bool | 128 | 7079 | ?[128] ? | `_memcpy` | 42567* |
| approx_eq_complex64_bool | 128 | 7079 | ?[128] ? | `_memcpy` | 42534* |
| approx_eq_f16_bool | 128 | 14009 | ?[128] ? | - | 127785 |
| approx_eq_f32_bool | 128 | 4004 | ?[128] ? | `_memcpy` | 27110* |
| approx_eq_f64_bool | 128 | 4004 | ?[128] ? | `_memcpy` | 27110* |
| asin_f16 | 128 | 864 | ?[64] ? | `_asinf` | 16556* |
| asin_f32 | 128 | 170 | ? ? | `_asinf` | 3367* |
| asin_f64 | 128 | 81 | ? ? | `_asin` | 1641* |
| atan_f16 | 128 | 864 | ?[64] ? | `_atanf` | 16556* |
| atan_f32 | 128 | 170 | ? ? | `_atanf` | 3367* |
| atan_f64 | 128 | 81 | ? ? | `_atan` | 1641* |
| bitwise_rotate_left_i16_u32 | 128 | 84 | ? ? ? ? | - | 688 |
| bitwise_rotate_left_i32_u32 | 128 | 74 | ? ? ? ? | - | 603 |
| bitwise_rotate_left_i64_u32 | 128 | 62 | ? ? ? | - | 805 |
| bitwise_rotate_left_i8_u32 | 128 | 95 | ? ? ? ? | - | 688 |
| bitwise_rotate_right_i16_u32 | 128 | 84 | ? ? ? ? | - | 688 |
| bitwise_rotate_right_i32_u32 | 128 | 74 | ? ? ? ? | - | 602 |
| bitwise_rotate_right_i64_u32 | 128 | 46 | ? ? ? | - | 389 |
| bitwise_rotate_right_i8_u32 | 128 | 95 | ? ? ? ? | - | 688 |
| bitwise_shift_left_i16 | 128 | 34 | ? ? ? ? | - | 262 |
| bitwise_shift_left_i32 | 128 | 34 | ? ? ? ? | - | 261 |
| bitwise_shift_left_i64 | 128 | 34 | ? ? ? | - | 261 |
| bitwise_shift_left_i8 | 128 | 34 | ? ? ? ? | - | 263 |
| bitwise_shift_right_i16 | 128 | 42 | ? ? ? ? | - | 308 |
| bitwise_shift_right_i32 | 128 | 42 | ? ? ? ? | - | 292 |
| bitwise_shift_right_i64 | 128 | 42 | ? ? ? | - | 292 |
| bitwise_shift_right_i8 | 128 | 42 | ? ? ? ? | - | 293 |
| bitwise_shift_right_u16 | 128 | 42 | ? ? ? ? | - | 308 |
| bitwise_shift_right_u32 | 128 | 42 | ? ? ? ? | - | 292 |
| bitwise_shift_right_u64 | 128 | 42 | ? ? ? | - | 292 |
| bitwise_shift_right_u8 | 128 | 42 | ? ? ? ? | - | 293 |
| cast_bool_complex128 | 128 | 37 | ? ? ? ? | - | 483 |
| cast_bool_complex64 | 128 | 85 | ? ? ? ? | - | 1091 |
| cast_bool_f16 | 128 | 856 | ?[64] ? | - | 6023 |
| cast_bool_f32 | 128 | 45 | ? ? ? ? | - | 347 |
| cast_bool_f64 | 128 | 47 | ? ? ? ? | - | 390 |
| cast_bool_i16 | 128 | 857 | ?[64] ? ? ? | - | 6756 |
| cast_bool_i32 | 128 | 37 | ? ? ? ? | - | 261 |
| cast_bool_i64 | 128 | 39 | ? ? ? ? | - | 310 |
| cast_bool_i8 | 128 | 1693 | ?[128] | - | 13412 |
| cast_complex128 | 128 | 13 | ? ? ? | - | 195 |
| cast_complex128_bool | 128 | 2328 | ?[128] ? ? ? | `_memcpy` | 18629* |
| cast_complex128_complex64 | 128 | 33 | ? ? ? | - | 261 |
| cast_complex64 | 128 | 13 | ? ? ? | - | 195 |
| cast_complex64_bool | 128 | 2328 | ?[128] ? ? ? | `_memcpy` | 18629* |
| cast_complex64_complex128 | 128 | 21 | ? ? ? | - | 196 |
| cast_f16 | 128 | 796 | ?[64] ? | - | 5126 |
| cast_f16_bool | 128 | 1700 | ?[128] ? | - | 10854 |
| cast_f16_complex128 | 128 | 45 | ? ? | - | 388 |
| cast_f16_complex64 | 128 | 69 | ? ? | - | 770 |
| cast_f16_f32 | 128 | 87 | ? ? | - | 804 |
| cast_f16_f64 | 128 | 61 | ? ? | - | 431 |
| cast_f16_i16 | 128 | 1052 | ?[64] ? | - | 7720 |
| cast_f16_i32 | 128 | 408 | ?[32] ? | - | 3095 |
| cast_f16_i64 | 128 | 61 | ? ? | - | 645 |
| cast_f16_i8 | 128 | 2084 | ?[128] ? | - | 14136 |
| cast_f16_u16 | 128 | 924 | ?[64] ? | - | 6728 |
| cast_f16_u32 | 128 | 408 | ?[32] ? | - | 3095 |
| cast_f16_u64 | 128 | 61 | ? ? | - | 645 |
| cast_f16_u8 | 128 | 1828 | ?[128] ? | - | 12936 |
| cast_f32 | 128 | 13 | ? | - | 195 |
| cast_f32_bool | 128 | 1944 | ?[128] ? ? ? | `_memcpy` | 16486* |
| cast_f32_complex128 | 128 | 33 | ? ? ? ? | - | 356 |
| cast_f32_complex64 | 128 | 45 | ? ? ? ? | - | 642 |
| cast_f32_f16 | 128 | 933 | ?[64] ? | - | 6597 |
| cast_f32_f64 | 128 | 21 | ? ? ? ? | - | 195 |
| cast_f32_i16 | 128 | 1189 | ?[64] ? ? ? | - | 9607 |
| cast_f32_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| cast_f32_i64 | 128 | 29 | ? ? ? ? | - | 212 |
| cast_f32_i8 | 128 | 2332 | ?[128] ? ? ? | `_memcpy` | 19654* |
| cast_f32_u16 | 128 | 1061 | ?[64] ? ? ? | - | 8551 |
| cast_f32_u32 | 128 | 21 | ? ? ? ? | - | 196 |
| cast_f32_u64 | 128 | 29 | ? ? ? ? | - | 212 |
| cast_f32_u8 | 128 | 2072 | ?[128] ? ? ? | `_memcpy` | 17606* |
| cast_f64 | 128 | 14 | ? | - | 196 |
| cast_f64_bool | 128 | 1944 | ?[128] ? ? ? | `_memcpy` | 16486* |
| cast_f64_complex128 | 128 | 25 | ? ? ? | - | 323 |
| cast_f64_complex64 | 128 | 61 | ? ? ? ? | - | 770 |
| cast_f64_f16 | 128 | 984 | ?[64] ? | `_memcpy` | 8614* |
| cast_f64_f32 | 128 | 33 | ? ? ? ? | - | 260 |
| cast_f64_i16 | 128 | 1180 | ?[64] ? ? | `_memcpy` | 11367* |
| cast_f64_i32 | 128 | 69 | ? ? ? ? | - | 1028 |
| cast_f64_i64 | 128 | 21 | ? ? ? | - | 196 |
| cast_f64_i8 | 128 | 2332 | ?[128] ? ? ? | `_memcpy` | 19654* |
| cast_f64_u16 | 128 | 1048 | ?[64] ? ? | `_memcpy` | 10279* |
| cast_f64_u32 | 128 | 69 | ? ? ? ? | - | 1028 |
| cast_f64_u64 | 128 | 21 | ? ? ? | - | 196 |
| cast_f64_u8 | 128 | 2072 | ?[128] ? ? ? | `_memcpy` | 17606* |
| cast_i16 | 128 | 13 | ? | - | 195 |
| cast_i16_bool | 128 | 49 | ? ? ? ? | - | 347 |
| cast_i16_complex128 | 128 | 53 | ? ? ? | - | 420 |
| cast_i16_complex64 | 128 | 85 | ? ? ? ? | - | 770 |
| cast_i16_f16 | 128 | 1117 | ?[64] ? | - | 7429 |
| cast_i16_f32 | 128 | 29 | ? ? ? ? | - | 212 |
| cast_i16_f64 | 128 | 77 | ? ? ? | - | 602 |
| cast_i16_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| cast_i16_i64 | 128 | 22 | ? ? ? ? | - | 189 |
| cast_i16_i8 | 128 | 25 | ? ? ? ? | - | 260 |
| cast_i32_bool | 128 | 89 | ? ? ? ? | - | 688 |
| cast_i32_complex128 | 128 | 41 | ? ? ? ? | - | 388 |
| cast_i32_complex64 | 128 | 61 | ? ? ? ? | - | 706 |
| cast_i32_f16 | 128 | 1061 | ?[64] ? | - | 7205 |
| cast_i32_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| cast_i32_f64 | 128 | 29 | ? ? ? ? | - | 212 |
| cast_i32_i16 | 128 | 25 | ? ? ? ? | - | 259 |
| cast_i32_i64 | 128 | 24 | ? ? ? ? | - | 243 |
| cast_i32_i8 | 128 | 33 | ? ? ? ? | - | 387 |
| cast_i64_bool | 128 | 185 | ? ? ? ? | - | 1542 |
| cast_i64_complex128 | 128 | 33 | ? ? ? | - | 356 |
| cast_i64_complex64 | 128 | 61 | ? ? ? | - | 962 |
| cast_i64_f16 | 128 | 1048 | ?[64] ? | `_memcpy` | 9544* |
| cast_i64_f32 | 128 | 69 | ? ? ? | - | 1028 |
| cast_i64_f64 | 128 | 21 | ? ? ? | - | 196 |
| cast_i64_i16 | 128 | 49 | ? ? ? ? | - | 399 |
| cast_i64_i32 | 128 | 25 | ? ? ? ? | - | 260 |
| cast_i64_i8 | 128 | 65 | ? ? ? ? | - | 773 |
| cast_i8 | 128 | 13 | ? | - | 195 |
| cast_i8_bool | 128 | 29 | ? ? ? ? | - | 211 |
| cast_i8_complex128 | 128 | 37 | ? ? ? ? | - | 581 |
| cast_i8_complex64 | 128 | 101 | ? ? ? ? | - | 835 |
| cast_i8_f16 | 128 | 1177 | ?[64] ? | - | 7430 |
| cast_i8_f32 | 128 | 29 | ? ? ? ? | - | 213 |
| cast_i8_f64 | 128 | 45 | ? ? ? ? | - | 646 |
| cast_i8_i16 | 128 | 921 | ?[64] ? ? ? | - | 7269 |
| cast_i8_i32 | 128 | 29 | ? ? ? ? | - | 213 |
| cast_i8_i64 | 128 | 43 | ? ? ? ? | - | 347 |
| cast_u16_complex128 | 128 | 37 | ? ? ? | - | 387 |
| cast_u16_complex64 | 128 | 69 | ? ? ? ? | - | 770 |
| cast_u16_f16 | 128 | 1053 | ?[64] ? | - | 7109 |
| cast_u16_f32 | 128 | 29 | ? ? ? ? | - | 212 |
| cast_u16_f64 | 128 | 45 | ? ? ? | - | 387 |
| cast_u16_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| cast_u16_i64 | 128 | 22 | ? ? ? ? | - | 189 |
| cast_u32_complex128 | 128 | 33 | ? ? ? ? | - | 356 |
| cast_u32_complex64 | 128 | 61 | ? ? ? ? | - | 706 |
| cast_u32_f16 | 128 | 1061 | ?[64] ? | - | 7205 |
| cast_u32_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| cast_u32_f64 | 128 | 29 | ? ? ? ? | - | 212 |
| cast_u32_i64 | 128 | 21 | ? ? ? ? | - | 196 |
| cast_u64_complex128 | 128 | 33 | ? ? ? | - | 356 |
| cast_u64_complex64 | 128 | 61 | ? ? ? | - | 962 |
| cast_u64_f16 | 128 | 1048 | ?[64] ? | `_memcpy` | 9544* |
| cast_u64_f32 | 128 | 69 | ? ? ? | - | 1028 |
| cast_u64_f64 | 128 | 21 | ? ? ? | - | 196 |
| cast_u8_complex128 | 128 | 37 | ? ? ? ? | - | 580 |
| cast_u8_complex64 | 128 | 69 | ? ? ? ? | - | 771 |
| cast_u8_f16 | 128 | 1049 | ?[64] ? | - | 7077 |
| cast_u8_f32 | 128 | 37 | ? ? ? ? | - | 261 |
| cast_u8_f64 | 128 | 45 | ? ? ? ? | - | 388 |
| cast_u8_i16 | 128 | 857 | ?[64] ? ? ? | - | 6756 |
| cast_u8_i32 | 128 | 29 | ? ? ? ? | - | 213 |
| cast_u8_i64 | 128 | 35 | ? ? ? ? | - | 262 |
| ceil_f16 | 128 | 860 | ?[64] ? | - | 5606 |
| ceil_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| ceil_f64 | 128 | 21 | ? ? ? | - | 196 |
| clamp_op_descriptor_bool | 128 | 2205 | ?[128] ? ? ? | - | 19557 |
| clamp_op_descriptor_f16 | 128 | 2851 | ?[64 64 64] ? ? ? ? | - | 25733 |
| clamp_op_descriptor_f32 | 128 | 30 | ? ? ? ? | - | 219 |
| clamp_op_descriptor_f64 | 128 | 30 | ? ? ? | - | 219 |
| clamp_op_descriptor_i16 | 128 | 989 | ?[64] ? ? ? | - | 8293 |
| clamp_op_descriptor_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| clamp_op_descriptor_i64 | 128 | 30 | ? ? ? | - | 219 |
| clamp_op_descriptor_i8 | 128 | 1949 | ?[128] ? ? ? | - | 16485 |
| clamp_op_descriptor_u16 | 128 | 989 | ?[64] ? ? ? | - | 8293 |
| clamp_op_descriptor_u32 | 128 | 21 | ? ? ? ? | - | 196 |
| clamp_op_descriptor_u64 | 128 | 30 | ? ? ? | - | 219 |
| clamp_op_descriptor_u8 | 128 | 1949 | ?[128] ? ? ? | - | 16485 |
| cos_f16 | 128 | 864 | ?[64] ? | `_cosf` | 16556* |
| cos_f32 | 128 | 170 | ? ? ? | `_cosf` | 3367* |
| cos_f64 | 128 | 81 | ? ? ? | `_cos` | 1641* |
| count_ones_i16_u32 | 128 | 569 | ?[32] ? ? ? | - | 3888 |
| count_ones_i32_u32 | 128 | 573 | ?[32] ? ? ? | - | 4037 |
| count_ones_i64_u32 | 128 | 549 | ?[32] ? ? ? | - | 3782 |
| count_ones_i8_u32 | 128 | 535 | ?[32] ? ? ? | - | 3591 |
| count_zeros_i16_u32 | 128 | 601 | ?[32] ? ? ? | - | 4103 |
| count_zeros_i32_u32 | 128 | 605 | ?[32] ? ? ? | - | 4239 |
| count_zeros_i64_u32 | 128 | 613 | ?[32] ? ? ? | - | 4519 |
| count_zeros_i8_u32 | 128 | 567 | ?[32] ? ? ? | - | 3806 |
| div_complex128 | 128 | 124 | ? ? ? | - | 3273 |
| div_complex64 | 128 | 71 | ? ? ? | - | 732 |
| div_f16 | 128 | 488 | ?[64] ? | - | 17420 |
| div_f32 | 128 | 26 | ? ? ? ? | - | 259 |
| div_f64 | 128 | 26 | ? ? ? | - | 259 |
| div_i16 | 128 | 872 | ?[64] ? | - | 20491 |
| div_i32 | 128 | 392 | ?[32] ? | - | 10251 |
| div_i64 | 128 | 183 | ? ? | - | 6660 |
| div_i8 | 128 | 1832 | ?[128] ? | - | 40969 |
| div_u16 | 128 | 552 | ?[64] ? | - | 20491 |
| div_u32 | 128 | 296 | ?[32] ? | - | 10251 |
| div_u64 | 128 | 70 | ? ? | - | 6663 |
| div_u8 | 128 | 1063 | ?[128] ? | - | 40970 |
| equal_complex128_bool | 128 | 493 | ?[8] ? ? ? | `_memcpy` | 5736* |
| equal_complex64_bool | 128 | 293 | ?[8] ? ? ? | `_memcpy` | 4327* |
| equal_f16_bool | 128 | 298 | ?[8] ? ? ? | - | 2983 |
| equal_f32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| equal_f64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| equal_i16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| equal_i32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| equal_i64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| equal_i8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| exp_f16 | 128 | 864 | ?[64] ? | `_expf` | 16556* |
| exp_f32 | 128 | 170 | ? ? ? | `_expf` | 3367* |
| exp_f64 | 128 | 81 | ? ? ? | `_exp` | 1641* |
| floor_f16 | 128 | 860 | ?[64] ? | - | 5606 |
| floor_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| floor_f64 | 128 | 21 | ? ? ? | - | 196 |
| greater_bool | 128 | 34 | ? ? ? ? | - | 262 |
| greater_equal_bool | 128 | 34 | ? ? ? ? | - | 262 |
| greater_equal_f16_bool | 128 | 418 | ?[8] ? ? ? | - | 4167 |
| greater_equal_f32_bool | 128 | 107 | ? ? ? ? | - | 773 |
| greater_equal_f64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| greater_equal_i16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| greater_equal_i32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| greater_equal_i64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| greater_equal_i8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| greater_equal_u16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| greater_equal_u32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| greater_equal_u64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| greater_equal_u8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| greater_f16_bool | 128 | 3385 | ?[128] ? ? ? | - | 32871 |
| greater_f32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| greater_f64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| greater_i16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| greater_i32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| greater_i64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| greater_i8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| greater_u16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| greater_u32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| greater_u64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| greater_u8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| imaginary_complex128_f64 | 128 | 29 | ? ? ? | - | 387 |
| imaginary_complex64_f32 | 128 | 53 | ? ? ? ? | - | 771 |
| is_finite_f16_bool | 128 | 1957 | ?[128] ? ? ? | - | 16645 |
| is_finite_f32_bool | 128 | 1687 | ?[128] ? ? ? | `_memcpy` | 14374* |
| is_finite_f64_bool | 128 | 1687 | ?[128] ? ? ? | `_memcpy` | 14374* |
| is_infinite_f16_bool | 128 | 1957 | ?[128] ? ? ? | - | 16645 |
| is_infinite_f32_bool | 128 | 1559 | ?[128] ? ? ? | `_memcpy` | 14279* |
| is_infinite_f64_bool | 128 | 1559 | ?[128] ? ? ? | `_memcpy` | 14279* |
| is_nan_f16_bool | 128 | 2085 | ?[128] ? ? ? | - | 17701 |
| is_nan_f32_bool | 128 | 1944 | ?[128] ? ? ? | `_memcpy` | 16549* |
| is_nan_f64_bool | 128 | 1944 | ?[128] ? ? ? | `_memcpy` | 16549* |
| leading_zeros_i16_u32 | 128 | 29 | ? ? ? ? | - | 212 |
| leading_zeros_i32_u32 | 128 | 21 | ? ? ? ? | - | 196 |
| leading_zeros_i64_u32 | 128 | 69 | ? ? ? | - | 548 |
| leading_zeros_i8_u32 | 128 | 45 | ? ? ? ? | - | 347 |
| less_bool | 128 | 34 | ? ? ? ? | - | 262 |
| less_equal_bool | 128 | 34 | ? ? ? ? | - | 262 |
| less_equal_f16_bool | 128 | 3513 | ?[128] ? ? ? | - | 32870 |
| less_equal_f32_bool | 128 | 107 | ? ? ? ? | - | 773 |
| less_equal_f64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| less_equal_i16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| less_equal_i32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| less_equal_i64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| less_equal_i8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| less_equal_u16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| less_equal_u32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| less_equal_u64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| less_equal_u8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| less_f16_bool | 128 | 426 | ?[8] ? ? ? | - | 4231 |
| less_f32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| less_f64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| less_i16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| less_i32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| less_i64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| less_i8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| less_u16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| less_u32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| less_u64_bool | 128 | 203 | ? ? ? ? | - | 1457 |
| less_u8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| ln_f16 | 128 | 864 | ?[64] ? | `_logf` | 16556* |
| ln_f32 | 128 | 170 | ? ? ? | `_logf` | 3367* |
| ln_f64 | 128 | 81 | ? ? ? | `_log` | 1641* |
| maximum_bool | 128 | 26 | ? ? ? ? | - | 261 |
| maximum_f16 | 128 | 344 | ?[8] ? ? | - | 3178 |
| maximum_f32 | 128 | 74 | ? ? ? | - | 603 |
| maximum_f64 | 128 | 74 | ? ? ? | - | 603 |
| maximum_i16 | 128 | 26 | ? ? ? ? | - | 259 |
| maximum_i32 | 128 | 26 | ? ? ? ? | - | 260 |
| maximum_i64 | 128 | 34 | ? ? ? | - | 261 |
| maximum_i8 | 128 | 26 | ? ? ? ? | - | 262 |
| maximum_u16 | 128 | 26 | ? ? ? ? | - | 259 |
| maximum_u32 | 128 | 26 | ? ? ? ? | - | 260 |
| maximum_u64 | 128 | 34 | ? ? ? | - | 261 |
| maximum_u8 | 128 | 26 | ? ? ? ? | - | 262 |
| minimum_bool | 128 | 26 | ? ? ? ? | - | 261 |
| minimum_f16 | 128 | 336 | ?[8] ? ? | - | 3004 |
| minimum_f32 | 128 | 74 | ? ? ? | - | 603 |
| minimum_f64 | 128 | 74 | ? ? ? | - | 603 |
| minimum_i16 | 128 | 26 | ? ? ? ? | - | 259 |
| minimum_i32 | 128 | 26 | ? ? ? ? | - | 260 |
| minimum_i64 | 128 | 34 | ? ? ? | - | 261 |
| minimum_i8 | 128 | 26 | ? ? ? ? | - | 262 |
| minimum_u16 | 128 | 26 | ? ? ? ? | - | 259 |
| minimum_u32 | 128 | 26 | ? ? ? ? | - | 260 |
| minimum_u64 | 128 | 34 | ? ? ? | - | 261 |
| minimum_u8 | 128 | 26 | ? ? ? ? | - | 262 |
| mul_complex128 | 128 | 70 | ? ? ? | - | 533 |
| mul_complex64 | 128 | 90 | ? ? ? | - | 774 |
| mul_f16 | 128 | 488 | ?[64] ? | - | 4517 |
| mul_f32 | 128 | 26 | ? ? ? ? | - | 259 |
| mul_f64 | 128 | 26 | ? ? ? | - | 260 |
| mul_i16 | 128 | 26 | ? ? ? ? | - | 260 |
| mul_i32 | 128 | 26 | ? ? ? ? | - | 260 |
| mul_i64 | 128 | 46 | ? ? ? | - | 518 |
| mul_i8 | 128 | 26 | ? ? ? ? | - | 260 |
| neg_complex128 | 128 | 21 | ? ? ? | - | 196 |
| neg_complex64 | 128 | 21 | ? ? ? | - | 196 |
| neg_f16 | 128 | 21 | ? ? ? ? | - | 196 |
| neg_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| neg_f64 | 128 | 21 | ? ? ? | - | 196 |
| neg_i16 | 128 | 925 | ?[64] ? ? ? | - | 8516 |
| neg_i32 | 128 | 21 | ? ? ? ? | - | 196 |
| neg_i64 | 128 | 21 | ? ? ? | - | 196 |
| neg_i8 | 128 | 1821 | ?[128] ? ? ? | - | 16630 |
| not_bool | 128 | 1949 | ?[128] ? ? ? | - | 15461 |
| not_equal_complex128_bool | 128 | 525 | ?[8] ? ? ? | `_memcpy` | 5896* |
| not_equal_complex64_bool | 128 | 309 | ?[8] ? ? ? | `_memcpy` | 4392* |
| not_equal_f16_bool | 128 | 306 | ?[8] ? ? ? | - | 3047 |
| not_equal_f32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| not_equal_f64_bool | 128 | 218 | ? ? ? ? | - | 1543 |
| not_equal_i16_bool | 128 | 59 | ? ? ? ? | - | 432 |
| not_equal_i32_bool | 128 | 107 | ? ? ? ? | - | 774 |
| not_equal_i64_bool | 128 | 218 | ? ? ? ? | - | 1543 |
| not_equal_i8_bool | 128 | 34 | ? ? ? ? | - | 262 |
| not_i16 | 128 | 21 | ? ? ? ? | - | 196 |
| not_i32 | 128 | 21 | ? ? ? ? | - | 195 |
| not_i64 | 128 | 21 | ? ? ? | - | 195 |
| not_i8 | 128 | 21 | ? ? ? ? | - | 197 |
| or_i16 | 128 | 26 | ? ? ? ? | - | 260 |
| or_i32 | 128 | 26 | ? ? ? ? | - | 259 |
| or_i64 | 128 | 26 | ? ? ? | - | 259 |
| or_i8 | 128 | 26 | ? ? ? ? | - | 261 |
| pow_f32 | 128 | 284 | ?[16] ? ? | `_powf` | 6665* |
| pow_f32_i32 | 128 | 240 | ? ? | `___powisf2` | 4121* |
| pow_f64 | 128 | 114 | ? ? ? | `_pow` | 2278* |
| pow_f64_i32 | 128 | 113 | ? ? | `___powidf2` | 2246* |
| real_complex128_f64 | 128 | 29 | ? ? ? | - | 387 |
| real_complex64_f32 | 128 | 53 | ? ? ? ? | - | 771 |
| reverse_bits_i16 | 128 | 29 | ? ? ? ? | - | 212 |
| reverse_bits_i32 | 128 | 29 | ? ? ? ? | - | 207 |
| reverse_bits_i64 | 128 | 33 | ? ? ? | - | 244 |
| reverse_bits_i8 | 128 | 21 | ? ? ? ? | - | 197 |
| round_f16 | 128 | 860 | ?[64] ? | - | 5606 |
| round_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| round_f64 | 128 | 21 | ? ? ? | - | 196 |
| sign_f16 | 128 | 1052 | ?[64] ? | - | 6663 |
| sign_f32 | 128 | 444 | ?[32] ? ? | - | 2853 |
| sign_f64 | 128 | 38 | ? ? ? | - | 272 |
| sign_i16 | 128 | 1053 | ?[64] ? ? ? | - | 8293 |
| sign_i32 | 128 | 541 | ?[32] ? ? ? | - | 4197 |
| sign_i64 | 128 | 37 | ? ? ? | - | 260 |
| sign_i8 | 128 | 2077 | ?[128] ? ? ? | - | 16485 |
| sign_u16 | 128 | 1053 | ?[64] ? ? ? | - | 8301 |
| sign_u32 | 128 | 29 | ? ? ? ? | - | 208 |
| sign_u64 | 128 | 29 | ? ? ? | - | 208 |
| sign_u8 | 128 | 2077 | ?[128] ? ? ? | - | 16493 |
| sin_f16 | 128 | 864 | ?[64] ? | `_sinf` | 16556* |
| sin_f32 | 128 | 170 | ? ? ? | `_sinf` | 3367* |
| sin_f64 | 128 | 81 | ? ? ? | `_sin` | 1641* |
| sqrt_f16 | 128 | 860 | ?[64] ? | - | 17421 |
| sqrt_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| sqrt_f64 | 128 | 21 | ? ? ? | - | 196 |
| square_complex128 | 128 | 65 | ? ? ? | - | 554 |
| square_complex64 | 128 | 85 | ? ? ? | - | 773 |
| square_f16 | 128 | 1053 | ?[64] ? | - | 7121 |
| square_f32 | 128 | 21 | ? ? ? ? | - | 195 |
| square_f64 | 128 | 21 | ? ? ? | - | 197 |
| square_i16 | 128 | 21 | ? ? ? ? | - | 197 |
| square_i32 | 128 | 21 | ? ? ? ? | - | 197 |
| square_i64 | 128 | 37 | ? ? ? | - | 517 |
| square_i8 | 128 | 21 | ? ? ? ? | - | 198 |
| sub_complex128 | 128 | 26 | ? ? ? | - | 260 |
| sub_complex64 | 128 | 26 | ? ? ? | - | 260 |
| sub_f16 | 128 | 488 | ?[64] ? | - | 4484 |
| sub_f32 | 128 | 26 | ? ? ? ? | - | 260 |
| sub_f64 | 128 | 26 | ? ? ? | - | 260 |
| sub_i16 | 128 | 26 | ? ? ? ? | - | 260 |
| sub_i32 | 128 | 26 | ? ? ? ? | - | 259 |
| sub_i64 | 128 | 26 | ? ? ? | - | 259 |
| sub_i8 | 128 | 26 | ? ? ? ? | - | 261 |
| swap_bytes_i16 | 128 | 21 | ? ? ? ? | - | 196 |
| swap_bytes_i32 | 128 | 21 | ? ? ? ? | - | 195 |
| swap_bytes_i64 | 128 | 21 | ? ? ? | - | 195 |
| tan_f16 | 128 | 864 | ?[64] ? | `_tanf` | 16556* |
| tan_f32 | 128 | 170 | ? ? | `_tanf` | 3367* |
| tan_f64 | 128 | 81 | ? ? | `_tan` | 1641* |
| trailing_zeros_i16_u32 | 128 | 45 | ? ? ? ? | - | 347 |
| trailing_zeros_i32_u32 | 128 | 37 | ? ? ? ? | - | 261 |
| trailing_zeros_i64_u32 | 128 | 101 | ? ? ? | - | 719 |
| trailing_zeros_i8_u32 | 128 | 61 | ? ? ? ? | - | 518 |
| xor_i16 | 128 | 26 | ? ? ? ? | - | 260 |
| xor_i32 | 128 | 26 | ? ? ? ? | - | 259 |
| xor_i64 | 128 | 26 | ? ? ? | - | 259 |
| xor_i8 | 128 | 26 | ? ? ? ? | - | 261 |
| **geomean** | | | | | **980*** |

Warnings:

- pow_i16_u16: hot loop with an inner loop of unknown trip count
- pow_i32_u32: hot loop with an inner loop of unknown trip count
- pow_i64_u32: hot loop with an inner loop of unknown trip count
- pow_i8_u8: hot loop with an inner loop of unknown trip count
