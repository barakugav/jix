# llvm-mca summary: `bit-encode-fearless` (bit-encode kernels)

- rustc: `rustc 1.101.0-nightly (75a75c3e0 2026-09-26)`
- llvm-mca: `LLVM version 23.1.1`

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

| platform | cpu | geomean | vs bit-encode-baseline | trans_bit_byte_2 | trans_bit_byte_4 | trans_bit_byte_8 | trans_bit_byte_16 |
|---|---|---:|---:|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | **6951** | 1.00x | 6951 | 6951 | 6951 | 6951 |
| x86_64-v2 | btver2 | **15045** | 1.00x | 15045 | 15045 | 15045 | 15045 |
| x86_64-v3 | skylake | **9732** | 1.00x | 9732 | 9732 | 9732 | 9732 |
| x86_64-v3 | alderlake | **6633** | 1.00x | 6633 | 6633 | 6633 | 6633 |
| x86_64-v3 | znver3 | **9102** | 1.00x | 9102 | 9102 | 9102 | 9102 |
| x86_64-v4 | icelake-server | **483** | 5.33x | 483 | 483 | 483 | 483 |
| x86_64-v4 | sapphirerapids | **363** | 7.74x | 363 | 363 | 363 | 363 |
| x86_64-v4 | znver4 | **357** | 5.63x | 357 | 357 | 357 | 357 |
| i686 | skylake | **11736** | 1.00x | 11736 | 11736 | 11736 | 11736 |
| aarch64 | cortex-a72 | **12743** | 1.00x | 12743 | 12743 | 12743 | 12743 |
| aarch64 | neoverse-n1 | **11972** | 1.00x | 11972 | 11972 | 11972 | 11972 |
| aarch64 | neoverse-v2 | **5523** | 1.00x | 5523 | 5523 | 5523 | 5523 |
| aarch64-apple | apple-m1 | **6726** | 1.00x | 6726 | 6726 | 6726 | 6726 |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| trans_bit_byte_2 | ~128 | 714 | ?[? ?] | - | 6951 | 15045 |
| trans_bit_byte_4 | ~128 | 714 | ?[? ?] | - | 6951 | 15045 |
| trans_bit_byte_8 | ~128 | 714 | ?[? ?] | - | 6951 | 15045 |
| trans_bit_byte_16 | ~128 | 714 | ?[? ?] | - | 6951 | 15045 |
| **geomean** | | | | | **6951** | **15045** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | sandybridge | btver2 |
|---|---:|---|---|---:|---:|
| trans_bitrow_eight_2 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 16.3 | 28.6 |
| trans_bitrow_eight_4 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 16.3 | 28.6 |
| trans_bitrow_eight_8 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 16.3 | 28.6 |
| trans_bitrow_eight_16 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 16.3 | 28.6 |

Warnings:

- trans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| trans_bit_byte_2 | ~128 | 447 | ?[? ?] | - | 9732 | 6633 | 9102 |
| trans_bit_byte_4 | ~128 | 447 | ?[? ?] | - | 9732 | 6633 | 9102 |
| trans_bit_byte_8 | ~128 | 447 | ?[? ?] | - | 9732 | 6633 | 9102 |
| trans_bit_byte_16 | ~128 | 447 | ?[? ?] | - | 9732 | 6633 | 9102 |
| **geomean** | | | | | **9732** | **6633** | **9102** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---|---|---:|---:|---:|
| trans_bitrow_eight_2 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 13.5 | 6.5 | 9.4 |
| trans_bitrow_eight_4 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 13.5 | 6.5 | 9.4 |
| trans_bitrow_eight_8 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 13.5 | 6.5 | 9.4 |
| trans_bitrow_eight_16 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 13.5 | 6.5 | 9.4 |

Warnings:

- trans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| trans_bit_byte_2 | ~256 | 132 | ?[? ?] | - | 483 | 363 | 357 |
| trans_bit_byte_4 | ~256 | 132 | ?[? ?] | - | 483 | 363 | 357 |
| trans_bit_byte_8 | ~256 | 132 | ?[? ?] | - | 483 | 363 | 357 |
| trans_bit_byte_16 | ~256 | 132 | ?[? ?] | - | 483 | 363 | 357 |
| **geomean** | | | | | **483** | **363** | **357** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---|---|---:|---:|---:|
| trans_bitrow_eight_2 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 8.9 | 6.5 | 7.7 |
| trans_bitrow_eight_4 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 8.9 | 6.5 | 7.7 |
| trans_bitrow_eight_8 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 8.9 | 6.5 | 7.7 |
| trans_bitrow_eight_16 | 21 | ? ? ? ? ? ? ? ? | `*memcpy@GOTPCREL(%rip)` | 8.9 | 6.5 | 7.7 |

Warnings:

- trans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| trans_bit_byte_2 | ~128 | 1194 | ?[? ?] | - | 11736 |
| trans_bit_byte_4 | ~128 | 1194 | ?[? ?] | - | 11736 |
| trans_bit_byte_8 | ~128 | 1194 | ?[? ?] | - | 11736 |
| trans_bit_byte_16 | ~128 | 1194 | ?[? ?] | - | 11736 |
| **geomean** | | | | | **11736** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake |
|---|---:|---|---|---:|
| trans_bitrow_eight_2 | 32 | ? ? ? ? ? ? ? ? | `memcpy@PLT` | 18.3 |
| trans_bitrow_eight_4 | 32 | ? ? ? ? ? ? ? ? | `memcpy@PLT` | 18.3 |
| trans_bitrow_eight_8 | 32 | ? ? ? ? ? ? ? ? | `memcpy@PLT` | 18.3 |
| trans_bitrow_eight_16 | 32 | ? ? ? ? ? ? ? ? | `memcpy@PLT` | 18.3 |

Warnings:

- trans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| trans_bit_byte_2 | ~128 | 699 | ?[? ?] | - | 12743 | 11972 | 5523 |
| trans_bit_byte_4 | ~128 | 699 | ?[? ?] | - | 12743 | 11972 | 5523 |
| trans_bit_byte_8 | ~128 | 699 | ?[? ?] | - | 12743 | 11972 | 5523 |
| trans_bit_byte_16 | ~128 | 699 | ?[? ?] | - | 12743 | 11972 | 5523 |
| **geomean** | | | | | **12743** | **11972** | **5523** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---|---|---:|---:|---:|
| trans_bitrow_eight_2 | 21 | ? ? ? ? ? ? ? ? | `memcpy` | 17.9 | 17.9 | 7.7 |
| trans_bitrow_eight_4 | 21 | ? ? ? ? ? ? ? ? | `memcpy` | 17.9 | 17.9 | 7.7 |
| trans_bitrow_eight_8 | 21 | ? ? ? ? ? ? ? ? | `memcpy` | 17.9 | 17.9 | 7.7 |
| trans_bitrow_eight_16 | 21 | ? ? ? ? ? ? ? ? | `memcpy` | 17.9 | 17.9 | 7.7 |

Warnings:

- trans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| trans_bit_byte_2 | ~128 | 681 | ?[? ?] | - | 6726 |
| trans_bit_byte_4 | ~128 | 681 | ?[? ?] | - | 6726 |
| trans_bit_byte_8 | ~128 | 681 | ?[? ?] | - | 6726 |
| trans_bit_byte_16 | ~128 | 681 | ?[? ?] | - | 6726 |
| **geomean** | | | | | **6726** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | apple-m1 |
|---|---:|---|---|---:|
| trans_bitrow_eight_2 | 19 | ? ? ? ? ? ? ? ? | `_memcpy` | 11.8 |
| trans_bitrow_eight_4 | 19 | ? ? ? ? ? ? ? ? | `_memcpy` | 11.8 |
| trans_bitrow_eight_8 | 19 | ? ? ? ? ? ? ? ? | `_memcpy` | 11.8 |
| trans_bitrow_eight_16 | 19 | ? ? ? ? ? ? ? ? | `_memcpy` | 11.8 |

Warnings:

- trans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- trans_bitrow_eight_16: no loop whose iteration count is linear in the input length
