# llvm-mca summary: `rb-i686`

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

None.

## Overview

| kernel | i686 skylake |
|---|---:|
| reduce_arg_max_bool_cell | 12887 |
| reduce_arg_max_f16_cell | 30774 |
| reduce_arg_max_f32_cell | 2449 |
| reduce_arg_max_f64_cell | 1197 |
| reduce_arg_max_i16_cell | 2972 |
| reduce_arg_max_i32_cell | 1558 |
| reduce_arg_max_i64_cell | 1612 |
| reduce_arg_max_i8_cell | 17510 |
| reduce_arg_max_u16_cell | 2944 |
| reduce_arg_max_u32_cell | 1692 |
| reduce_arg_max_u64_cell | 6275 |
| reduce_arg_max_u8_cell | 17516 |
| reduce_arg_min_bool_cell | 14428 |
| reduce_arg_min_f16_cell | 34852 |
| reduce_arg_min_f32_cell | 2255 |
| reduce_arg_min_f64_cell | 1225 |
| reduce_arg_min_i16_cell | 3110 |
| reduce_arg_min_i32_cell | 1559 |
| reduce_arg_min_i64_cell | 1613 |
| reduce_arg_min_i8_cell | 17510 |
| reduce_arg_min_u16_cell | 2862 |
| reduce_arg_min_u32_cell | 1691 |
| reduce_arg_min_u64_cell | 5956 |
| reduce_arg_min_u8_cell | 17521 |
| **geomean** | **4598** |

## i686 (`i686-unknown-linux-gnu`)

| kernel | B/iter | instrs | stack | loops | calls | skylake |
|---|---:|---:|---:|---|---|---:|
| reduce_arg_max_bool_cell | 8 | 123 | 21/22 | ? ? | - | 12887 |
| reduce_arg_max_f16_cell | 16 | 483 | 0/216 | ? ? | - | 30774 |
| reduce_arg_max_f32_cell | 32 | 63 | 25/0 | ? ? | - | 2449 |
| reduce_arg_max_f64_cell | 64 | 83 | 31/0 | ? ? | - | 1197 |
| reduce_arg_max_i16_cell | 16 | 49 | 16/0 | ? ? | - | 2972 |
| reduce_arg_max_i32_cell | 32 | 56 | 23/0 | ? ? | - | 1558 |
| reduce_arg_max_i64_cell | 64 | 110 | 35/0 | ? ? | - | 1612 |
| reduce_arg_max_i8_cell | 8 | 161 | 25/45 | ? ? | - | 17510 |
| reduce_arg_max_u16_cell | 16 | 52 | 18/0 | ? ? | - | 2944 |
| reduce_arg_max_u32_cell | 32 | 62 | 22/0 | ? ? | - | 1692 |
| reduce_arg_max_u64_cell | 64 | 178 | 0/117 | ? ? | - | 6275 |
| reduce_arg_max_u8_cell | 8 | 161 | 25/45 | ? ? | - | 17516 |
| reduce_arg_min_bool_cell | 8 | 124 | 21/22 | ? ? | - | 14428 |
| reduce_arg_min_f16_cell | 16 | 498 | 0/219 | ? ? | - | 34852 |
| reduce_arg_min_f32_cell | 32 | 63 | 25/0 | ? ? | - | 2255 |
| reduce_arg_min_f64_cell | 64 | 87 | 35/0 | ? ? | - | 1225 |
| reduce_arg_min_i16_cell | 16 | 51 | 18/0 | ? ? | - | 3110 |
| reduce_arg_min_i32_cell | 32 | 58 | 23/0 | ? ? | - | 1559 |
| reduce_arg_min_i64_cell | 64 | 110 | 35/0 | ? ? | - | 1613 |
| reduce_arg_min_i8_cell | 8 | 161 | 25/45 | ? ? | - | 17510 |
| reduce_arg_min_u16_cell | 16 | 50 | 18/0 | ? ? | - | 2862 |
| reduce_arg_min_u32_cell | 32 | 61 | 21/0 | ? ? | - | 1691 |
| reduce_arg_min_u64_cell | 64 | 168 | 0/107 | ? ? | - | 5956 |
| reduce_arg_min_u8_cell | 8 | 161 | 25/45 | ? ? | - | 17521 |
| **geomean** | | | | | | **4598** |
