# llvm-mca summary: `py-approx`

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
| approx_eq_complex128_bool | 91812 | 67062 | 54431 | 91812 | 148588 | 64837 | 73027 | 48325 | 25051 | 21189 | 12604 | 76052 | 75361 | 83517 | 40269 | 37996 |
| approx_eq_complex64_bool | 85844 | 65334 | 48832 | 85844 | 132129 | 65816 | 73507 | 45985 | 13873 | 12161 | 6242 | 63877 | 90158 | 102431 | 46633 | 42038 |
| approx_eq_f16_bool | 1090122 | 868750 | 815821 | 1090122 | 1705480 | 176392 | 130747 | 97767 | 172204 | 130627 | 95039 | 938790 | 882558 | 817802 | 410422 | 127762 |
| approx_eq_f32_bool | 45715 | 30885 | 26346 | 45715 | 72401 | 30764 | 36215 | 25055 | 5126 | 3575 | 2057 | 32156 | 49304 | 61455 | 26007 | 25627 |
| approx_eq_f64_bool | 45642 | 31288 | 27195 | 45642 | 74788 | 30552 | 37422 | 25504 | 6913 | 6968 | 4278 | 40586 | 52787 | 65055 | 27310 | 26424 |
| **geomean** | **112384** | **81871** | **68908** | **112384** | **178517** | **58877** | **62467** | **42511** | **18420** | **15300** | **9197** | **90143** | **109311** | **122840** | **55932** | **42471** |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| approx_eq_complex128_bool | 8 | 577 | ? ? | - | 91812 | 67062 | 54431 |
| approx_eq_complex64_bool | 16 | 1032 | ?[16] ? | - | 85844 | 65334 | 48832 |
| approx_eq_f16_bool | 64 | 46322 | ?[64] ? | - | 1090122 | 868750 | 815821 |
| approx_eq_f32_bool | 32 | 1131 | ?[32] ? | - | 45715 | 30885 | 26346 |
| approx_eq_f64_bool | 16 | 584 | ?[16] ? | - | 45642 | 31288 | 27195 |
| **geomean** | | | | | **112384** | **81871** | **68908** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| approx_eq_complex128_bool | 8 | 577 | ? ? | - | 91812 | 148588 |
| approx_eq_complex64_bool | 16 | 1032 | ?[16] ? | - | 85844 | 132129 |
| approx_eq_f16_bool | 64 | 46322 | ?[64] ? | - | 1090122 | 1705480 |
| approx_eq_f32_bool | 32 | 1131 | ?[32] ? | - | 45715 | 72401 |
| approx_eq_f64_bool | 16 | 584 | ?[16] ? | - | 45642 | 74788 |
| **geomean** | | | | | **112384** | **178517** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| approx_eq_complex128_bool | 16 | 1019 | ?[16] ? | - | 64837 | 73027 | 48325 |
| approx_eq_complex64_bool | 32 | 1960 | ?[32] ? | - | 65816 | 73507 | 45985 |
| approx_eq_f16_bool | 128 | 17201 | ?[128] ? | - | 176392 | 130747 | 97767 |
| approx_eq_f32_bool | 64 | 2155 | ?[64] ? | - | 30764 | 36215 | 25055 |
| approx_eq_f64_bool | 32 | 1096 | ?[32] ? | - | 30552 | 37422 | 25504 |
| **geomean** | | | | | **58877** | **62467** | **42511** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| approx_eq_complex128_bool | 32 | 318 | ?[4] ? | - | 25051 | 21189 | 12604 |
| approx_eq_complex64_bool | 64 | 311 | ?[4] ? | - | 13873 | 12161 | 6242 |
| approx_eq_f16_bool | 256 | 34353 | ?[256] ? | - | 172204 | 130627 | 95039 |
| approx_eq_f32_bool | 128 | 227 | ? ? ? ? | - | 5126 | 3575 | 2057 |
| approx_eq_f64_bool | 64 | 227 | ? ? ? | - | 6913 | 6968 | 4278 |
| **geomean** | | | | | **18420** | **15300** | **9197** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| approx_eq_complex128_bool | 4 | 373 | ? ? | - | 76052 |
| approx_eq_complex64_bool | 8 | 529 | ?[8] ? | - | 63877 |
| approx_eq_f16_bool | 32 | 23894 | ?[32] ? | - | 938790 |
| approx_eq_f32_bool | 16 | 573 | ?[16] ? | - | 32156 |
| approx_eq_f64_bool | 8 | 398 | ? ? | - | 40586 |
| **geomean** | | | | | **90143** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| approx_eq_complex128_bool | 8 | 413 | ? ? | - | 75361 | 83517 | 40269 |
| approx_eq_complex64_bool | 16 | 925 | ?[16] ? | - | 90158 | 102431 | 46633 |
| approx_eq_f16_bool | 64 | 38185 | ?[64] ? | - | 882558 | 817802 | 410422 |
| approx_eq_f32_bool | 32 | 1052 | ?[32] ? | - | 49304 | 61455 | 26007 |
| approx_eq_f64_bool | 16 | 539 | ?[16] ? | - | 52787 | 65055 | 27310 |
| **geomean** | | | | | **109311** | **122840** | **55932** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| approx_eq_complex128_bool | 8 | 413 | ? ? | - | 37996 |
| approx_eq_complex64_bool | 16 | 909 | ?[16] ? | - | 42038 |
| approx_eq_f16_bool | 64 | 7015 | ?[64] ? | - | 127762 |
| approx_eq_f32_bool | 32 | 1020 | ?[32] ? | - | 25627 |
| approx_eq_f64_bool | 16 | 523 | ?[16] ? | - | 26424 |
| **geomean** | | | | | **42471** |
