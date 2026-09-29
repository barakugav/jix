# llvm-mca summary: `elementwise-mixed` (elementwise kernels)

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
| x86_64 | sandybridge | **593** | 646 | 475 | 1037 | 389 |
| x86_64 | skylake | **452** | 519 | 365 | 781 | 283 |
| x86_64 | znver3 | **501** | 647 | 388 | 650 | 388 |
| x86_64-v2 | sandybridge | **563** | 646 | 387 | 1037 | 389 |
| x86_64-v2 | btver2 | **857** | 1127 | 564 | 1385 | 613 |
| x86_64-v3 | skylake | **203** | 261 | 141 | 328 | 141 |
| x86_64-v3 | alderlake | **179** | 169 | 138 | 320 | 138 |
| x86_64-v3 | znver3 | **254** | 196 | 226 | 419 | 226 |
| x86_64-v4 | icelake-server | **121** | 163 | 71 | 259 | 71 |
| x86_64-v4 | sapphirerapids | **124** | 140 | 69 | 197 | 122 |
| x86_64-v4 | znver4 | **208** | 196 | 129 | 386 | 193 |
| i686 | skylake | **474** | 526 | 391 | 792 | 308 |
| aarch64 | cortex-a72 | **1218** | 1544 | 675 | 1995 | 1058 |
| aarch64 | neoverse-n1 | **1084** | 1349 | 770 | 1479 | 898 |
| aarch64 | neoverse-v2 | **440** | 648 | 275 | 650 | 323 |
| aarch64-apple | apple-m1 | **337** | 414 | 259 | 460 | 261 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 64 | 35 | ? ? ? ? | - | 646 | 519 | 647 |
| widen_i64 | 128 | 59 | ? ? ? ? | - | 475 | 365 | 388 |
| narrow_f32 | 64 | 43 | ? ? ? | - | 1037 | 781 | 650 |
| widen_f64 | 128 | 35 | ? ? ? | - | 389 | 283 | 388 |
| **geomean** | | | | | **593** | **452** | **501** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| narrow_i32 | 64 | 35 | ? ? ? ? | - | 646 | 1127 |
| widen_i64 | 128 | 35 | ? ? ? ? | - | 387 | 564 |
| narrow_f32 | 64 | 43 | ? ? ? | - | 1037 | 1385 |
| widen_f64 | 128 | 35 | ? ? ? | - | 389 | 613 |
| **geomean** | | | | | **563** | **857** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 128 | 23 | ? ? ? ? ? | - | 261 | 169 | 196 |
| widen_i64 | 256 | 27 | ? ? ? ? ? | - | 141 | 138 | 226 |
| narrow_f32 | 128 | 35 | ? ? ? ? | - | 328 | 320 | 419 |
| widen_f64 | 256 | 27 | ? ? ? ? | - | 141 | 138 | 226 |
| **geomean** | | | | | **203** | **179** | **254** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 256 | 27 | ? ? ? ? ? | - | 163 | 140 | 196 |
| widen_i64 | 512 | 27 | ? ? ? ? | - | 71 | 69 | 129 |
| narrow_f32 | 256 | 35 | ? ? ? ? | - | 259 | 197 | 386 |
| widen_f64 | 512 | 27 | ? ? ? ? | - | 71 | 122 | 193 |
| **geomean** | | | | | **121** | **124** | **208** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| narrow_i32 | 32 | 19 | ? ? ? | - | 526 |
| widen_i64 | 64 | 31 | ? ? ? | - | 391 |
| narrow_f32 | 32 | 23 | ? ? ? | - | 792 |
| widen_f64 | 64 | 19 | ? ? ? | - | 308 |
| **geomean** | | | | | **474** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| narrow_i32 | 64 | 26 | ? ? ? | - | 1544 | 1349 | 648 |
| widen_i64 | 128 | 26 | ? ? ? | - | 675 | 770 | 275 |
| narrow_f32 | 64 | 31 | ? ? ? | - | 1995 | 1479 | 650 |
| widen_f64 | 128 | 34 | ? ? ? | - | 1058 | 898 | 323 |
| **geomean** | | | | | **1218** | **1084** | **440** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| narrow_i32 | 64 | 26 | ? ? ? ? | - | 414 |
| widen_i64 | 128 | 26 | ? ? ? ? | - | 259 |
| narrow_f32 | 64 | 31 | ? ? ? ? | - | 460 |
| widen_f64 | 128 | 34 | ? ? ? ? | - | 261 |
| **geomean** | | | | | **337** |
