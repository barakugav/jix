# llvm-mca summary: `bit-decode-fearless` (bit-decode kernels)

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

| platform | cpu | geomean | vs bit-decode-baseline | untrans_bit_byte_2 | untrans_bit_byte_4 | untrans_bit_byte_8 | untrans_bit_byte_16 |
|---|---|---:|---:|---:|---:|---:|---:|
| x86_64-v2 | sandybridge | **4774** | 1.00x | 4774 | 4774 | 4774 | 4774 |
| x86_64-v2 | btver2 | **7990** | 1.00x | 7990 | 7990 | 7990 | 7990 |
| x86_64-v3 | skylake | **1829** | 1.00x | 1829 | 1829 | 1829 | 1829 |
| x86_64-v3 | alderlake | **1815** | 1.00x | 1815 | 1815 | 1815 | 1815 |
| x86_64-v3 | znver3 | **2527** | 1.00x | 2527 | 2527 | 2527 | 2527 |
| x86_64-v4 | icelake-server | **849** | 1.00x | 849 | 849 | 849 | 849 |
| x86_64-v4 | sapphirerapids | **1030** | 1.00x | 1030 | 1030 | 1030 | 1030 |
| x86_64-v4 | znver4 | **1076** | 1.00x | 1076 | 1076 | 1076 | 1076 |
| i686 | skylake | **8540** | 1.00x | 8540 | 8540 | 8540 | 8540 |
| aarch64 | cortex-a72 | **809** | 1.00x | 809 | 809 | 809 | 809 |
| aarch64 | neoverse-n1 | **725** | 1.00x | 725 | 725 | 725 | 725 |
| aarch64 | neoverse-v2 | **365** | 1.00x | 365 | 365 | 365 | 365 |
| aarch64-apple | apple-m1 | **1400** | 1.00x | 1400 | 1400 | 1400 | 1400 |

## x86_64-v2 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+fxsr,+sse4.2,+cmpxchg16b,+popcnt`

| kernel | B/iter | instrs | loops | calls | sandybridge | btver2 |
|---|---:|---:|---|---|---:|---:|
| untrans_bit_byte_2 | ~16 | 49 | ?[? ?] | - | 4774 | 7990 |
| untrans_bit_byte_4 | ~16 | 49 | ?[? ?] | - | 4774 | 7990 |
| untrans_bit_byte_8 | ~16 | 49 | ?[? ?] | - | 4774 | 7990 |
| untrans_bit_byte_16 | ~16 | 49 | ?[? ?] | - | 4774 | 7990 |
| **geomean** | | | | | **4774** | **7990** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | sandybridge | btver2 |
|---|---:|---|---|---:|---:|
| untrans_bitrow_eight_2 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 139.9 | 279.6 |
| untrans_bitrow_eight_4 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 139.9 | 279.6 |
| untrans_bitrow_eight_8 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 139.9 | 279.6 |
| untrans_bitrow_eight_16 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 139.9 | 279.6 |

Warnings:

- untrans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## x86_64-v3 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave`

| kernel | B/iter | instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---:|---|---|---:|---:|---:|
| untrans_bit_byte_2 | ~32 | 46 | ?[? ?] | - | 1829 | 1815 | 2527 |
| untrans_bit_byte_4 | ~32 | 46 | ?[? ?] | - | 1829 | 1815 | 2527 |
| untrans_bit_byte_8 | ~32 | 46 | ?[? ?] | - | 1829 | 1815 | 2527 |
| untrans_bit_byte_16 | ~32 | 46 | ?[? ?] | - | 1829 | 1815 | 2527 |
| **geomean** | | | | | **1829** | **1815** | **2527** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake | alderlake | znver3 |
|---|---:|---|---|---:|---:|---:|
| untrans_bitrow_eight_2 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 105.6 | 49.6 | 94.1 |
| untrans_bitrow_eight_4 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 105.6 | 49.6 | 94.1 |
| untrans_bitrow_eight_8 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 105.6 | 49.6 | 94.1 |
| untrans_bitrow_eight_16 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 105.6 | 49.6 | 94.1 |

Warnings:

- untrans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## x86_64-v4 (`x86_64-unknown-linux-gnu`)

`RUSTFLAGS=-Ctarget-feature=+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves`

| kernel | B/iter | instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---:|---|---|---:|---:|---:|
| untrans_bit_byte_2 | ~64 | 42 | ?[? ?] | - | 849 | 1030 | 1076 |
| untrans_bit_byte_4 | ~64 | 42 | ?[? ?] | - | 849 | 1030 | 1076 |
| untrans_bit_byte_8 | ~64 | 42 | ?[? ?] | - | 849 | 1030 | 1076 |
| untrans_bit_byte_16 | ~64 | 42 | ?[? ?] | - | 849 | 1030 | 1076 |
| **geomean** | | | | | **849** | **1030** | **1076** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | icelake-server | sapphirerapids | znver4 |
|---|---:|---|---|---:|---:|---:|
| untrans_bitrow_eight_2 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 70.7 | 49.6 | 94.1 |
| untrans_bitrow_eight_4 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 70.7 | 49.6 | 94.1 |
| untrans_bitrow_eight_8 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 70.7 | 49.6 | 94.1 |
| untrans_bitrow_eight_16 | 184 | ? | `*memcpy@GOTPCREL(%rip)` | 70.7 | 49.6 | 94.1 |

Warnings:

- untrans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | skylake |
|---|---:|---:|---|---|---:|
| untrans_bit_byte_2 | ~16 | 87 | ?[? ?] | - | 8540 |
| untrans_bit_byte_4 | ~16 | 87 | ?[? ?] | - | 8540 |
| untrans_bit_byte_8 | ~16 | 87 | ?[? ?] | - | 8540 |
| untrans_bit_byte_16 | ~16 | 87 | ?[? ?] | - | 8540 |
| **geomean** | | | | | **8540** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | skylake |
|---|---:|---|---|---:|
| untrans_bitrow_eight_2 | 223 | ? | `memcpy@PLT` | 133.5 |
| untrans_bitrow_eight_4 | 223 | ? | `memcpy@PLT` | 133.5 |
| untrans_bitrow_eight_8 | 223 | ? | `memcpy@PLT` | 133.5 |
| untrans_bitrow_eight_16 | 223 | ? | `memcpy@PLT` | 133.5 |

Warnings:

- untrans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## aarch64 (`aarch64-unknown-linux-gnu`)

| kernel | B/iter | instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---:|---|---|---:|---:|---:|
| untrans_bit_byte_2 | ~128 | 66 | ?[? ?] | - | 809 | 725 | 365 |
| untrans_bit_byte_4 | ~128 | 66 | ?[? ?] | - | 809 | 725 | 365 |
| untrans_bit_byte_8 | ~128 | 66 | ?[? ?] | - | 809 | 725 | 365 |
| untrans_bit_byte_16 | ~128 | 66 | ?[? ?] | - | 809 | 725 | 365 |
| **geomean** | | | | | **809** | **725** | **365** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | cortex-a72 | neoverse-n1 | neoverse-v2 |
|---|---:|---|---|---:|---:|---:|
| untrans_bitrow_eight_2 | 163 | ? | `memcpy` | 141.5 | 141.5 | 59.2 |
| untrans_bitrow_eight_4 | 163 | ? | `memcpy` | 141.5 | 141.5 | 59.2 |
| untrans_bitrow_eight_8 | 163 | ? | `memcpy` | 141.5 | 141.5 | 59.2 |
| untrans_bitrow_eight_16 | 163 | ? | `memcpy` | 141.5 | 141.5 | 59.2 |

Warnings:

- untrans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_16: no loop whose iteration count is linear in the input length

## aarch64-apple (`aarch64-apple-darwin`)

| kernel | B/iter | instrs | loops | calls | apple-m1 |
|---|---:|---:|---|---|---:|
| untrans_bit_byte_2 | ~128 | 69 | ?[? ?] | - | 1400 |
| untrans_bit_byte_4 | ~128 | 69 | ?[? ?] | - | 1400 |
| untrans_bit_byte_8 | ~128 | 69 | ?[? ?] | - | 1400 |
| untrans_bit_byte_16 | ~128 | 69 | ?[? ?] | - | 1400 |
| **geomean** | | | | | **1400** |

Kernels without a flattenable hot loop, cycles per iteration of the largest innermost loop (informational):

| kernel | loop instrs | loops | calls | apple-m1 |
|---|---:|---|---|---:|
| untrans_bitrow_eight_2 | 163 | ? | `_memcpy` | 96.3 |
| untrans_bitrow_eight_4 | 163 | ? | `_memcpy` | 96.3 |
| untrans_bitrow_eight_8 | 163 | ? | `_memcpy` | 96.3 |
| untrans_bitrow_eight_16 | 163 | ? | `_memcpy` | 96.3 |

Warnings:

- untrans_bitrow_eight_2: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_4: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_8: no loop whose iteration count is linear in the input length
- untrans_bitrow_eight_16: no loop whose iteration count is linear in the input length
