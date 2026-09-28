# Byte-shuffle llvm-mca summary: `baseline`

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`

Fixed-itemsize kernels: llvm-mca steady-state **cycles per 128 bytes** (one outer-loop
iteration, inner loops flattened by their trip counts). Lower is better. `geomean` is the
per-CPU geometric mean over the 8 kernels, the single number to optimize.

- `instrs`: instructions per 128 bytes (flattened trace).
- `loops`: asm loop tree with trip counts; `?` = runtime trip count (the outer loop).
- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include (flagged `*`).

## Overview

| platform | cpu | geomean | encode | decode | worst kernel |
|---|---|---:|---:|---:|---|
| x86_64-v1 | x86-64 | **94.3** | 123.3 | 72.1 | decode_16 (136.1) |
| x86_64-v1 | skylake | **85.9** | 113.2 | 65.2 | decode_16 (136.1) |
| x86_64-v2 | sandybridge | **59.0** | 113.6 | 30.6 | decode_16 (136.1) |
| x86_64-v2 | btver2 | **87.4** | 122.7 | 62.3 | decode_16 (204.1) |
| x86_64-v3 | skylake | **67.2** | 98.1 | 46.1 | decode_16 (180.2) |
| x86_64-v3 | alderlake | **39.2** | 56.4 | 27.2 | decode_16 (107.3) |
| x86_64-v3 | znver3 | **60.1** | 107.3 | 33.7 | decode_8 (128.1) |
| x86_64-v4 | icelake-server | **17.9** | 16.1 | 19.8 | encode_8 (49.7) |
| x86_64-v4 | sapphirerapids | **19.8** | 17.3 | 22.5 | decode_16 (60.2) |
| x86_64-v4 | znver4 | **15.3** | 13.7 | 17.1 | encode_8 (36.2) |
| i686 | skylake | **105.7** | 153.5 | 72.8 | encode_16 (264.1) |
| aarch64 | cortex-a72 | **93.9** | 131.9 | 66.9 | decode_16 (203.2) |
| aarch64 | neoverse-n1 | **89.8** | 144.7 | 55.8 | encode_16 (189.1) |
| aarch64 | neoverse-v2 | **44.2** | 87.7 | 22.3 | encode_16 (95.2) |
| aarch64-apple | apple-m1 | **45.6** | 119.2 | 17.4 | encode_2 (128.1) |
| armv7 | cortex-a9 | **524.6** | 507.2 | 542.7 | decode_8 (624.0) |
| armv7 | cortex-a57 | **231.8** | 232.6 | 231.0 | encode_2 (256.1) |
| ppc64le | pwr8 | **80.0** | 114.9 | 55.7 | encode_16 (126.1) |
| ppc64le | pwr9 | **68.2** | 99.1 | 46.9 | decode_16 (127.0) |

## x86_64-v1 (`x86_64-unknown-linux-gnu`)

| kernel | instrs | loops | calls | x86-64 | skylake |
|---|---:|---|---:|---:|---:|
| encode_2 | 260 | ? | 0 | 128.1 | 128.1 |
| encode_4 | 260 | ? | 0 | 128.1 | 128.1 |
| encode_8 | 260 | ? | 0 | 128.1 | 128.1 |
| encode_16 | 440 | ?[16] | 0 | 110.2 | 78.1 |
| decode_2 | 35 | ? | 0 | 12.1 | 8.1 |
| decode_4 | 260 | ? | 0 | 128.1 | 128.1 |
| decode_8 | 260 | ? | 0 | 128.1 | 128.1 |
| decode_16 | 408 | ?[16] | 0 | 136.1 | 136.1 |
| **geomean** | | | | **94.3** | **85.9** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | x86-64 | skylake |
|---|---:|---|---:|---:|---:|
| encode_generic | 8 | ?[? ? ?] | 0 | 2.1 | 2.1 |
| decode_generic | 19 | ?[? ?] | 0 | 4.9 | 4.1 |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b`

| kernel | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---|---:|---:|---:|
| encode_2 | 260 | ? | 0 | 128.1 | 130.1 |
| encode_4 | 260 | ? | 0 | 128.1 | 130.1 |
| encode_8 | 260 | ? | 0 | 128.1 | 130.1 |
| encode_16 | 203 | ? | 0 | 79.2 | 103.2 |
| decode_2 | 35 | ? | 0 | 12.1 | 17.6 |
| decode_4 | 68 | ? | 0 | 21.2 | 58.1 |
| decode_8 | 75 | ? | 0 | 25.3 | 72.1 |
| decode_16 | 408 | ?[16] | 0 | 136.1 | 204.1 |
| **geomean** | | | | **59.0** | **87.4** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | sandybridge | btver2 |
|---|---:|---|---:|---:|---:|
| encode_generic | 8 | ?[? ? ?] | 0 | 2.1 | 4.1 |
| decode_generic | 54 | ?[?[? ?] ?] | 0 | 17.8 | 64.1 |

Warnings:

- decode_generic: asm and IR loop trees differ, trip counts unknown

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave`

| kernel | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---|---:|---:|---:|---:|
| encode_2 | 260 | ? | 0 | 128.1 | 64.7 | 128.1 |
| encode_4 | 260 | ? | 0 | 128.1 | 64.7 | 128.1 |
| encode_8 | 260 | ? | 0 | 128.1 | 64.7 | 128.1 |
| encode_16 | 116 | ? | 0 | 44.1 | 37.3 | 63.2 |
| decode_2 | 27 | ? | 0 | 12.1 | 8.3 | 10.2 |
| decode_4 | 44 | ? | 0 | 16.1 | 9.6 | 12.1 |
| decode_8 | 260 | ? | 0 | 128.1 | 64.7 | 128.1 |
| decode_16 | 273 | ? | 0 | 180.2 | 107.3 | 81.3 |
| **geomean** | | | | **67.2** | **39.2** | **60.1** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---|---:|---:|---:|---:|
| encode_generic | 12 | ?[? ? ?] | 0 | 4.1 | 2.9 | 4.1 |
| decode_generic | 104 | ?[? ? ?] | 0 | 47.6 | 65.4 | 48.5 |

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl`

| kernel | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---|---:|---:|---:|---:|
| encode_2 | 15 | ? | 0 | 6.1 | 6.3 | 4.2 |
| encode_4 | 27 | ? | 0 | 12.1 | 12.3 | 8.2 |
| encode_8 | 123 | ? | 0 | 49.7 | 48.3 | 36.2 |
| encode_16 | 47 | ? | 0 | 18.3 | 24.2 | 28.1 |
| decode_2 | 27 | ? | 0 | 12.2 | 12.3 | 12.1 |
| decode_4 | 36 | ? | 0 | 13.0 | 12.3 | 11.1 |
| decode_8 | 55 | ? | 0 | 30.1 | 28.3 | 26.2 |
| decode_16 | 41 | ? | 0 | 32.3 | 60.2 | 24.2 |
| **geomean** | | | | **17.9** | **19.8** | **15.3** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---|---:|---:|---:|---:|
| encode_generic | 12 | ?[? ? ?] | 0 | 3.4 | 2.9 | 4.1 |
| decode_generic | 179 | ?[? ? ?] | 0 | 94.3 | 88.8 | 158.8 |

## i686 (`i686-unknown-linux-gnu`)

| kernel | instrs | loops | calls | skylake |
|---|---:|---|---:|---:|
| encode_2 | 260 | ? | 0 | 128.1 |
| encode_4 | 261 | ? | 0 | 128.1 |
| encode_8 | 266 | ? | 0 | 128.1 |
| encode_16 | 474 | ? | 0 | 264.1 |
| decode_2 | 35 | ? | 0 | 8.1 |
| decode_4 | 261 | ? | 0 | 128.1 |
| decode_8 | 320 | ? | 0 | 128.1 |
| decode_16 | 479 | ?[16] | 1 | 211.5* |
| **geomean** | | | | **105.7** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake |
|---|---:|---|---:|---:|
| encode_generic | 9 | ?[? ? ?] | 0 | 2.1 |
| decode_generic | 8 | ?[?] | 0 | 2.1 |

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---|---:|---:|---:|---:|
| encode_2 | 262 | ? | 0 | 128.1 | 131.1 | 85.4 |
| encode_4 | 264 | ? | 0 | 128.1 | 132.1 | 85.4 |
| encode_8 | 268 | ? | 0 | 128.1 | 134.1 | 85.4 |
| encode_16 | 329 | ? | 0 | 144.2 | 189.1 | 95.2 |
| decode_2 | 26 | ? | 0 | 25.1 | 25.0 | 7.6 |
| decode_4 | 67 | ? | 0 | 52.1 | 50.0 | 19.3 |
| decode_8 | 72 | ? | 0 | 75.3 | 50.1 | 20.3 |
| decode_16 | 175 | ? | 0 | 203.2 | 154.1 | 83.2 |
| **geomean** | | | | **93.9** | **89.8** | **44.2** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---|---:|---:|---:|---:|
| encode_generic | 8 | ?[? ? ?] | 0 | 5.1 | 6.1 | 2.1 |
| decode_generic | 19 | ?[? ?] | 0 | 8.1 | 12.1 | 4.1 |

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | instrs | loops | calls | apple-m1 |
|---|---:|---|---:|---:|
| encode_2 | 262 | ? | 0 | 128.1 |
| encode_4 | 264 | ? | 0 | 128.1 |
| encode_8 | 268 | ? | 0 | 128.1 |
| encode_16 | 340 | ? | 0 | 96.2 |
| decode_2 | 26 | ? | 0 | 8.1 |
| decode_4 | 67 | ? | 0 | 18.7 |
| decode_8 | 70 | ? | 0 | 18.3 |
| decode_16 | 112 | ? | 0 | 33.2 |
| **geomean** | | | | **45.6** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | apple-m1 |
|---|---:|---|---:|---:|
| encode_generic | 9 | ?[? ? ?] | 0 | 3.1 |
| decode_generic | 19 | ?[? ?] | 0 | 4.2 |

## armv7 (`armv7-unknown-linux-gnueabihf`)

| kernel | instrs | loops | calls | cortex-a9 | cortex-a57 |
|---|---:|---|---:|---:|---:|
| encode_2 | 505 | ? | 0 | 568.0 | 256.1 |
| encode_4 | 510 | ? | 0 | 575.0 | 254.1 |
| encode_8 | 511 | ? | 0 | 577.0 | 255.1 |
| encode_16 | 348 | ?[16] | 1 | 351.1* | 176.2* |
| decode_2 | 499 | ? | 0 | 623.0 | 251.1 |
| decode_4 | 509 | ? | 0 | 617.0 | 253.2 |
| decode_8 | 511 | ? | 0 | 624.0 | 255.2 |
| decode_16 | 346 | ?[16] | 1 | 361.5* | 175.4* |
| **geomean** | | | | **524.6** | **231.8** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | cortex-a9 | cortex-a57 |
|---|---:|---|---:|---:|---:|
| encode_generic | 5 | ?[?] | 0 | 7.0 | 2.1 |
| decode_generic | 6 | ?[?] | 0 | 7.1 | 2.6 |

## ppc64le (`powerpc64le-unknown-linux-gnu`)

| kernel | instrs | loops | calls | pwr8 | pwr9 |
|---|---:|---|---:|---:|---:|
| encode_2 | 258 | ? | 0 | 111.6 | 97.1 |
| encode_4 | 260 | ? | 0 | 111.3 | 95.7 |
| encode_8 | 264 | ? | 0 | 111.5 | 97.1 |
| encode_16 | 297 | ?[8] | 0 | 126.1 | 107.1 |
| decode_2 | 45 | ? | 0 | 32.1 | 15.1 |
| decode_4 | 93 | ? | 0 | 44.0 | 32.1 |
| decode_8 | 228 | ? | 0 | 87.1 | 79.0 |
| decode_16 | 268 | ? | 0 | 78.0 | 127.0 |
| **geomean** | | | | **80.0** | **68.2** |

Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | pwr8 | pwr9 |
|---|---:|---|---:|---:|---:|
| encode_generic | 21 | ?[? ? ?] | 0 | 5.3 | 5.1 |
| decode_generic | 34 | ?[? ? ?] | 0 | 12.6 | 13.1 |

Warnings:

- encode_16: llvm-mca -mcpu=pwr8 skipped (no sched info): mtctr 25
