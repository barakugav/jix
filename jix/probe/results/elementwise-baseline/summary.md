# llvm-mca summary: `elementwise-baseline` (elementwise kernels)

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`
- kernels: the pipeline's inner loop for contiguous operands (`inner_loop::<T, LANES, true, true, _>`); the bytes are output bytes

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

| platform | cpu | geomean | neg_f32 | add_f32 | chain_f32 | neg_f64 | add_f64 | chain_f64 | neg_i32 | add_i32 | chain_i32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x86_64 | sandybridge | **433** | 291 | 387 | 645 | 291 | 387 | 645 | 283 | 387 | 934 |
| x86_64 | skylake | **362** | 259 | 260 | 518 | 259 | 260 | 518 | 259 | 259 | 1285 |
| x86_64 | znver3 | **402** | 260 | 387 | 645 | 260 | 387 | 645 | 260 | 387 | 647 |
| x86_64-v2 | sandybridge | **416** | 291 | 387 | 645 | 291 | 387 | 645 | 283 | 387 | 644 |
| x86_64-v2 | btver2 | **676** | 435 | 564 | 1076 | 435 | 564 | 1077 | 563 | 563 | 1348 |
| x86_64-v3 | skylake | **167** | 132 | 132 | 262 | 132 | 132 | 262 | 132 | 132 | 279 |
| x86_64-v3 | alderlake | **151** | 103 | 136 | 236 | 103 | 136 | 236 | 103 | 136 | 268 |
| x86_64-v3 | znver3 | **203** | 132 | 195 | 324 | 132 | 195 | 324 | 132 | 195 | 324 |
| x86_64-v4 | icelake-server | **1125** | 961 | 1481 | 2567 | 362 | 574 | 1086 | 953 | 1469 | 2515 |
| x86_64-v4 | sapphirerapids | **3059** | 2476 | 4583 | 6429 | 456 | 1923 | 5635 | 2243 | 4443 | 6513 |
| x86_64-v4 | znver4 | **226** | 198 | 265 | 457 | 88 | 160 | 288 | 169 | 232 | 401 |
| i686 | skylake | **362** | 259 | 260 | 518 | 259 | 260 | 518 | 259 | 259 | 1285 |
| aarch64 | cortex-a72 | **925** | 611 | 898 | 1635 | 611 | 898 | 1635 | 579 | 771 | 1380 |
| aarch64 | neoverse-n1 | **878** | 610 | 802 | 1378 | 610 | 802 | 1378 | 642 | 770 | 1380 |
| aarch64 | neoverse-v2 | **339** | 227 | 292 | 579 | 227 | 292 | 579 | 227 | 323 | 548 |
| aarch64-apple | apple-m1 | **277** | 195 | 260 | 422 | 195 | 260 | 423 | 195 | 259 | 420 |

## x86_64 (`x86_64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | sandybridge | skylake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 27 | ? ? ? | - | 291 | 259 | 260 |
| add_f32 | 128 | 35 | ? ? ? | - | 387 | 260 | 387 |
| chain_f32 | 128 | 67 | ? ? ? | - | 645 | 518 | 645 |
| neg_f64 | 128 | 27 | ? ? ? | - | 291 | 259 | 260 |
| add_f64 | 128 | 35 | ? ? ? | - | 387 | 260 | 387 |
| chain_f64 | 128 | 67 | ? ? ? | - | 645 | 518 | 645 |
| neg_i32 | 128 | 35 | ? ? ? | - | 283 | 259 | 260 |
| add_i32 | 128 | 35 | ? ? ? | - | 387 | 259 | 387 |
| chain_i32 | 128 | 115 | ? ? ? | - | 934 | 1285 | 647 |
| **geomean** | | | | | **433** | **362** | **402** |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| neg_f32 | 128 | 27 | ? ? ? | - | 291 | 435 |
| add_f32 | 128 | 35 | ? ? ? | - | 387 | 564 |
| chain_f32 | 128 | 67 | ? ? ? | - | 645 | 1076 |
| neg_f64 | 128 | 27 | ? ? ? | - | 291 | 435 |
| add_f64 | 128 | 35 | ? ? ? | - | 387 | 564 |
| chain_f64 | 128 | 67 | ? ? ? | - | 645 | 1077 |
| neg_i32 | 128 | 35 | ? ? ? | - | 283 | 563 |
| add_i32 | 128 | 35 | ? ? ? | - | 387 | 563 |
| chain_i32 | 128 | 67 | ? ? ? | - | 644 | 1348 |
| **geomean** | | | | | **416** | **676** |

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 11 | ? ? ? ? | - | 132 | 103 | 132 |
| add_f32 | 128 | 15 | ? ? ? ? | - | 132 | 136 | 195 |
| chain_f32 | 128 | 23 | ? ? ? ? | - | 262 | 236 | 324 |
| neg_f64 | 128 | 11 | ? ? ? ? | - | 132 | 103 | 132 |
| add_f64 | 128 | 15 | ? ? ? ? | - | 132 | 136 | 195 |
| chain_f64 | 128 | 23 | ? ? ? ? | - | 262 | 236 | 324 |
| neg_i32 | 128 | 11 | ? ? ? ? | - | 132 | 103 | 132 |
| add_i32 | 128 | 15 | ? ? ? ? | - | 132 | 136 | 195 |
| chain_i32 | 128 | 23 | ? ? ? ? | - | 279 | 268 | 324 |
| **geomean** | | | | | **167** | **151** | **203** |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 2048 | 425 | ? ? ? ? ? ? ? | - | 961 | 2476 | 198 |
| add_f32 | 2048 | 653 | ? ? ? ? ? ? ? | - | 1481 | 4583 | 265 |
| chain_f32 | 2048 | 1182 | ? ? ? ? ? ? | - | 2567 | 6429 | 457 |
| neg_f64 | 1024 | 99 | ? ? ? ? ? ? | - | 362 | 456 | 88 |
| add_f64 | 1024 | 147 | ? ? ? ? ? ? | - | 574 | 1923 | 160 |
| chain_f64 | 1024 | 275 | ? ? ? ? ? | - | 1086 | 5635 | 288 |
| neg_i32 | 2048 | 425 | ? ? ? ? ? ? ? | - | 953 | 2243 | 169 |
| add_i32 | 2048 | 653 | ? ? ? ? ? ? ? | - | 1469 | 4443 | 232 |
| chain_i32 | 2048 | 1182 | ? ? ? ? ? ? | - | 2515 | 6513 | 401 |
| **geomean** | | | | | **1125** | **3059** | **226** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| neg_f32 | 128 | 27 | ? ? ? | - | 259 |
| add_f32 | 128 | 35 | ? ? ? | - | 260 |
| chain_f32 | 128 | 67 | ? ? ? | - | 518 |
| neg_f64 | 128 | 27 | ? ? ? | - | 259 |
| add_f64 | 128 | 35 | ? ? ? | - | 260 |
| chain_f64 | 128 | 67 | ? ? ? | - | 518 |
| neg_i32 | 128 | 35 | ? ? ? | - | 259 |
| add_i32 | 128 | 35 | ? ? ? | - | 259 |
| chain_i32 | 128 | 115 | ? ? ? | - | 1285 |
| **geomean** | | | | | **362** |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| neg_f32 | 128 | 21 | ? ? ? | - | 611 | 610 | 227 |
| add_f32 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| chain_f32 | 128 | 52 | ? ? ? | - | 1635 | 1378 | 579 |
| neg_f64 | 128 | 21 | ? ? ? | - | 611 | 610 | 227 |
| add_f64 | 128 | 26 | ? ? ? | - | 898 | 802 | 292 |
| chain_f64 | 128 | 52 | ? ? ? | - | 1635 | 1378 | 579 |
| neg_i32 | 128 | 21 | ? ? ? | - | 579 | 642 | 227 |
| add_i32 | 128 | 26 | ? ? ? | - | 771 | 770 | 323 |
| chain_i32 | 128 | 52 | ? ? ? | - | 1380 | 1380 | 548 |
| **geomean** | | | | | **925** | **878** | **339** |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| neg_f32 | 128 | 21 | ? ? ? ? | - | 195 |
| add_f32 | 128 | 26 | ? ? ? ? | - | 260 |
| chain_f32 | 128 | 52 | ? ? ? ? | - | 422 |
| neg_f64 | 128 | 21 | ? ? ? | - | 195 |
| add_f64 | 128 | 26 | ? ? ? | - | 260 |
| chain_f64 | 128 | 52 | ? ? ? | - | 423 |
| neg_i32 | 128 | 21 | ? ? ? ? | - | 195 |
| add_i32 | 128 | 26 | ? ? ? ? | - | 259 |
| chain_i32 | 128 | 52 | ? ? ? ? | - | 420 |
| **geomean** | | | | | **277** |
