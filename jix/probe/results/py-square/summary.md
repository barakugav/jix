# llvm-mca summary: `py-square`

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`
- source: `py` (`jix-py`)
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
| square_complex128 | 776 | 647 | 551 | 678 | 1445 | 387 | 261 | 259 | 182 | 149 | 162 | 556 | 964 | 1217 | 436 | 552 |
| square_complex64 | 1850 | 1514 | 1538 | 805 | 1076 | 515 | 421 | 379 | 181 | 149 | 162 | 1411 | 1828 | 1316 | 643 | 774 |
| square_f16 | 13231* | 11138* | 6776* | 13231* | 26826* | 31094 | 32629 | 26966 | 31890 | 35770 | 27819 | 6913* | 17672* | 16839* | 6484* | 3333 |
| square_f32 | 260 | 260 | 260 | 260 | 515 | 130 | 95 | 130 | 65 | 66 | 65 | 264 | 739 | 610 | 228 | 196 |
| square_f64 | 260 | 260 | 260 | 260 | 548 | 130 | 95 | 130 | 65 | 66 | 65 | 264 | 739 | 610 | 228 | 197 |
| square_i16 | 260 | 260 | 260 | 260 | 436 | 130 | 95 | 130 | 60 | 66 | 65 | 265 | 740 | 771 | 228 | 197 |
| square_i32 | 581 | 1027 | 517 | 260 | 802 | 131 | 133 | 130 | 71 | 130 | 65 | 1031 | 740 | 771 | 228 | 197 |
| square_i64 | 660 | 644 | 654 | 660 | 2053 | 259 | 260 | 171 | 194 | 195 | 65 | 524 | 611 | 1539 | 291 | 517 |
| square_i8 | 784 | 773 | 457 | 773 | 1332 | 258 | 260 | 195 | 129 | 194 | 130 | 778 | 708 | 578 | 212 | 198 |
| square_u8 | 1028 | 602 | 406 | 821 | 1332 | 258 | 260 | 195 | 129 | 194 | 129 | 619 | 708 | 578 | 212 | 197 |
| **geomean** | **797*** | **754*** | **603*** | **652*** | **1311*** | **354** | **306** | **295** | **190** | **216** | **164** | **694*** | **1110*** | **1119*** | **381*** | **366** |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| square_complex128 | 128 | 87 | ? ? ? | - | 776 | 647 | 551 |
| square_complex64 | 512 | 649 | ? ? ? ? | - | 1850 | 1514 | 1538 |
| square_f16 | 128 | 556 | ?[64] ? | `_RNvNtNtCseSBabXzVpFX_4half8binary164arch21multiply_f16_fallback` | 13231* | 11138* | 6776* |
| square_f32 | 128 | 27 | ? ? ? ? | - | 260 | 260 | 260 |
| square_f64 | 128 | 27 | ? ? ? ? | - | 260 | 260 | 260 |
| square_i16 | 128 | 27 | ? ? ? ? ? | - | 260 | 260 | 260 |
| square_i32 | 128 | 67 | ? ? ? ? | - | 581 | 1027 | 517 |
| square_i64 | 128 | 60 | ? ? ? | - | 660 | 644 | 654 |
| square_i8 | 128 | 83 | ? ? ? ? ? | - | 784 | 773 | 457 |
| square_u8 | 128 | 75 | ? ? ? ? ? | - | 1028 | 602 | 406 |
| **geomean** | | | | | **797*** | **754*** | **603*** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| square_complex128 | 128 | 84 | ? ? ? | - | 678 | 1445 |
| square_complex64 | 128 | 67 | ? ? ? | - | 805 | 1076 |
| square_f16 | 128 | 556 | ?[64] ? | `_RNvNtNtCseSBabXzVpFX_4half8binary164arch21multiply_f16_fallback` | 13231* | 26826* |
| square_f32 | 128 | 27 | ? ? ? ? | - | 260 | 515 |
| square_f64 | 128 | 27 | ? ? ? ? | - | 260 | 548 |
| square_i16 | 128 | 27 | ? ? ? ? ? | - | 260 | 436 |
| square_i32 | 128 | 27 | ? ? ? ? | - | 260 | 802 |
| square_i64 | 128 | 60 | ? ? ? ? | - | 660 | 2053 |
| square_i8 | 128 | 83 | ? ? ? ? ? | - | 773 | 1332 |
| square_u8 | 128 | 83 | ? ? ? ? ? | - | 821 | 1332 |
| **geomean** | | | | | **652*** | **1311*** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| square_complex128 | 256 | 68 | ? ? ? | - | 387 | 261 | 259 |
| square_complex64 | 256 | 71 | ? ? ? | - | 515 | 421 | 379 |
| square_f16 | 256 | 876 | ?[64] ? | - | 31094 | 32629 | 26966 |
| square_f32 | 256 | 27 | ? ? ? ? ? | - | 130 | 95 | 130 |
| square_f64 | 256 | 27 | ? ? ? ? ? | - | 130 | 95 | 130 |
| square_i16 | 256 | 27 | ? ? ? ? ? | - | 130 | 95 | 130 |
| square_i32 | 256 | 27 | ? ? ? ? ? | - | 131 | 133 | 130 |
| square_i64 | 256 | 59 | ? ? ? ? ? | - | 259 | 260 | 171 |
| square_i8 | 256 | 67 | ? ? ? ? ? | - | 258 | 260 | 195 |
| square_u8 | 256 | 67 | ? ? ? ? ? | - | 258 | 260 | 195 |
| **geomean** | | | | | **354** | **306** | **295** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| square_complex128 | 512 | 64 | ? ? ? | - | 182 | 149 | 162 |
| square_complex64 | 512 | 63 | ? ? ? ? | - | 181 | 149 | 162 |
| square_f16 | 512 | 1708 | ?[128] ? | - | 31890 | 35770 | 27819 |
| square_f32 | 512 | 27 | ? ? ? ? | - | 65 | 66 | 65 |
| square_f64 | 512 | 27 | ? ? ? ? | - | 65 | 66 | 65 |
| square_i16 | 512 | 27 | ? ? ? ? | - | 60 | 66 | 65 |
| square_i32 | 512 | 27 | ? ? ? ? | - | 71 | 130 | 65 |
| square_i64 | 512 | 27 | ? ? ? ? | - | 194 | 195 | 65 |
| square_i8 | 512 | 51 | ? ? ? ? | - | 129 | 194 | 130 |
| square_u8 | 512 | 51 | ? ? ? ? | - | 129 | 194 | 129 |
| **geomean** | | | | | **190** | **216** | **164** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| square_complex128 | 128 | 66 | ? ? ? ? | - | 556 |
| square_complex64 | 256 | 306 | ? ? ? ? | - | 1411 |
| square_f16 | 64 | 256 | ? ? | `_RNvNtNtCs3Bjx3dgswfe_4half8binary164arch21multiply_f16_fallback` | 6913* |
| square_f32 | 64 | 15 | ? ? ? | - | 264 |
| square_f64 | 64 | 15 | ? ? ? | - | 264 |
| square_i16 | 64 | 15 | ? ? ? ? | - | 265 |
| square_i32 | 64 | 35 | ? ? ? | - | 1031 |
| square_i64 | 64 | 35 | ? ? ? | - | 524 |
| square_i8 | 64 | 43 | ? ? ? ? | - | 778 |
| square_u8 | 64 | 39 | ? ? ? ? | - | 619 |
| **geomean** | | | | | **694*** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| square_complex128 | 128 | 63 | ? ? ? | - | 964 | 1217 | 436 |
| square_complex64 | 128 | 85 | ? ? ? | - | 1828 | 1316 | 643 |
| square_f16 | 128 | 474 | ?[64] ? | `_RNvNtNtCs9fgWqoz3mDR_4half8binary164arch21multiply_f16_fallback` | 17672* | 16839* | 6484* |
| square_f32 | 128 | 21 | ? ? ? | - | 739 | 610 | 228 |
| square_f64 | 128 | 21 | ? ? ? | - | 739 | 610 | 228 |
| square_i16 | 128 | 21 | ? ? ? ? | - | 740 | 771 | 228 |
| square_i32 | 128 | 21 | ? ? ? | - | 740 | 771 | 228 |
| square_i64 | 128 | 37 | ? ? ? | - | 611 | 1539 | 291 |
| square_i8 | 128 | 21 | ? ? ? ? | - | 708 | 578 | 212 |
| square_u8 | 128 | 21 | ? ? ? ? | - | 708 | 578 | 212 |
| **geomean** | | | | | **1110*** | **1119*** | **381*** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| square_complex128 | 128 | 63 | ? ? ? | - | 552 |
| square_complex64 | 128 | 85 | ? ? ? | - | 774 |
| square_f16 | 128 | 410 | ?[64] ? | - | 3333 |
| square_f32 | 128 | 21 | ? ? ? ? | - | 196 |
| square_f64 | 128 | 21 | ? ? ? | - | 197 |
| square_i16 | 128 | 21 | ? ? ? ? | - | 197 |
| square_i32 | 128 | 21 | ? ? ? ? | - | 197 |
| square_i64 | 128 | 37 | ? ? ? | - | 517 |
| square_i8 | 128 | 21 | ? ? ? ? | - | 198 |
| square_u8 | 128 | 21 | ? ? ? ? | - | 197 |
| **geomean** | | | | | **366** |
