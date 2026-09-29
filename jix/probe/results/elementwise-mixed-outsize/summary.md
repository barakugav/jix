# llvm-mca summary: `elementwise-mixed-outsize` (elementwise kernels)

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`
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

| platform | cpu | geomean | narrow_i32 | widen_i64 | narrow_f32 | widen_f64 |
|---|---|---:|---:|---:|---:|---:|
| x86_64 | sandybridge | **592** | 643 | 475 | 1034 | 389 |
| x86_64 | skylake | **451** | 516 | 365 | 778 | 283 |
| x86_64 | znver3 | **517** | 643 | 388 | 741 | 388 |
| x86_64-v2 | sandybridge | **562** | 643 | 387 | 1034 | 389 |
| x86_64-v2 | btver2 | **847** | 1076 | 564 | 1383 | 613 |
| x86_64-v3 | skylake | **201** | 260 | 141 | 315 | 141 |
| x86_64-v3 | alderlake | **192** | 229 | 138 | 310 | 138 |
| x86_64-v3 | znver3 | **294** | 322 | 226 | 452 | 226 |
| x86_64-v4 | icelake-server | **120** | 158 | 71 | 257 | 71 |
| x86_64-v4 | sapphirerapids | **122** | 135 | 69 | 195 | 122 |
| x86_64-v4 | znver4 | **208** | 195 | 129 | 385 | 193 |
| i686 | skylake | **470** | 520 | 391 | 781 | 308 |
| aarch64 | cortex-a72 | **1169** | 1380 | 675 | 1893 | 1058 |
| aarch64 | neoverse-n1 | **1070** | 1314 | 770 | 1443 | 898 |
| aarch64 | neoverse-v2 | **410** | 548 | 275 | 580 | 323 |
| aarch64-apple | apple-m1 | **330** | 388 | 259 | 453 | 261 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 128 | 67 | ? ? ? ? | - | 643 | 516 | 643 |
| widen_i64 | 128 | 59 | ? ? ? ? | - | 475 | 365 | 388 |
| narrow_f32 | 128 | 86 | ? ? ? | - | 1034 | 778 | 741 |
| widen_f64 | 128 | 35 | ? ? ? | - | 389 | 283 | 388 |
| **geomean** | | | | | **592** | **451** | **517** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| narrow_i32 | 128 | 67 | ? ? ? ? | - | 643 | 1076 |
| widen_i64 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| narrow_f32 | 128 | 86 | ? ? ? | - | 1034 | 1383 |
| widen_f64 | 128 | 35 | ? ? ? | - | 389 | 613 |
| **geomean** | | | | | **562** | **847** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 256 | 59 | ? ? ? ? ? | - | 260 | 229 | 322 |
| widen_i64 | 256 | 27 | ? ? ? ? ? | - | 141 | 138 | 226 |
| narrow_f32 | 256 | 67 | ? ? ? ? | - | 315 | 310 | 452 |
| widen_f64 | 256 | 27 | ? ? ? ? | - | 141 | 138 | 226 |
| **geomean** | | | | | **201** | **192** | **294** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 512 | 51 | ? ? ? ? | - | 158 | 135 | 195 |
| widen_i64 | 512 | 27 | ? ? ? ? | - | 71 | 69 | 129 |
| narrow_f32 | 512 | 67 | ? ? ? ? | - | 257 | 195 | 385 |
| widen_f64 | 512 | 27 | ? ? ? ? | - | 71 | 122 | 193 |
| **geomean** | | | | | **120** | **122** | **208** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| narrow_i32 | 64 | 35 | ? ? ? | - | 520 |
| widen_i64 | 64 | 31 | ? ? ? | - | 391 |
| narrow_f32 | 64 | 45 | ? ? ? | - | 781 |
| widen_f64 | 64 | 19 | ? ? ? | - | 308 |
| **geomean** | | | | | **470** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 128 | 50 | ? ? ? | - | 1380 | 1314 | 548 |
| widen_i64 | 128 | 26 | ? ? ? | - | 675 | 770 | 275 |
| narrow_f32 | 128 | 58 | ? ? ? | - | 1893 | 1443 | 580 |
| widen_f64 | 128 | 34 | ? ? ? | - | 1058 | 898 | 323 |
| **geomean** | | | | | **1169** | **1070** | **410** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| narrow_i32 | 128 | 50 | ? ? ? ? | - | 388 |
| widen_i64 | 128 | 26 | ? ? ? ? | - | 259 |
| narrow_f32 | 128 | 58 | ? ? ? ? | - | 453 |
| widen_f64 | 128 | 34 | ? ? ? ? | - | 261 |
| **geomean** | | | | | **330** |
