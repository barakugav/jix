# Known issues

Issues already reported, accepted, or deliberately left as-is. Hunters skip these; the dedup
stage drops findings that match one. Add an entry when you triage a finding (fixed, won't fix,
intended) so later runs do not report it again.

Format: one entry per issue.

```
## K<n>: <one-line title>
- Status: open | fixed in <commit/PR> | wont-fix | intended
- Location: <file:line>
- Notes: <what exactly is known; which triggers are covered>
```

## K1: Short zstd block decode only debug_asserted; release reads of a malformed .jix return uninit/stale destination bytes
- Status: open (found in run 2026-10-21-pilot as F001)
- Location: jix/src/storage/block.rs:193
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F001.md; repro in that run's repros/.

## K2: Pull-mode buffer size product(shape)*itemsize wraps after an itemsize-widening op (cast/Op2/Map skip check_shape_overflow)
- Status: open (found in run 2026-10-21-pilot as F002)
- Location: jix/src/storage/buf.rs:312
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F002.md; repro in that run's repros/.

## K3: Padded struct dtypes: NdCopier copies elements as [u32;2]/[u64;2], reading uninit padding
- Status: open (found in run 2026-10-21-pilot as F003)
- Location: jix/src/util/nd_copy.rs:31
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F003.md; repro in that run's repros/.

## K4: read_from_reader hands uninitialized buffers to the caller's safe Read impl (value_from_io, read_slice, Owned::read_section)
- Status: open (found in run 2026-10-21-pilot as F004)
- Location: jix/src/util/mod.rs:577
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F004.md; repro in that run's repros/.

## K5: mmap reader bounds sections by the caller's len, not the mapping: truncated file + explicit len segfaults
- Status: open (found in run 2026-10-21-pilot as F005)
- Location: jix/src/archive/block.rs:487
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F005.md; repro in that run's repros/.

## K6: ArrayParams::tune divides by zero / overflows on an invalid fixed block dim when another dim is auto-scaled
- Status: open (found in run 2026-10-21-pilot as F006)
- Location: jix/src/storage/params.rs:427
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F006.md; repro in that run's repros/.

## K7: mmap reader rejects archives whose block-locations section is not 8-aligned in the file
- Status: open (found in run 2026-10-21-pilot as F007)
- Location: jix/src/archive/block.rs:489
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F007.md; repro in that run's repros/.

## K8: i686 Linux: alignment hard-coded to 8 for f64/i64/u64/c128 while the ABI says 4
- Status: open (found in run 2026-10-21-pilot as F008)
- Location: jix-py/src/dtype.rs:80, jix/src/dtype.rs:967
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F008.md; repro in that run's repros/.

## K9: RuntimeAlign::new accepts align 0 (latent)
- Status: open (found in run 2026-10-21-pilot as F009)
- Location: jix/src/util/aligned_vec/mod.rs:28
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F009.md; repro in that run's repros/.

## K10: RawAlignedBytes::shrink_to reallocs to size 0, no null check, leaks at capacity 0 (latent)
- Status: open (found in run 2026-10-21-pilot as F010)
- Location: jix/src/util/aligned_vec/raw.rs:155
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F010.md; repro in that run's repros/.

## K11: ArrayVec::extend_from_iter double-drops ZST elements; reachable via Dimension::vec
- Status: open (found in run 2026-10-21-pilot as F011)
- Location: jix/src/util/arrayvec.rs:416
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F011.md; repro in that run's repros/.

## K12: ArrayParams::block_shape / block_shape_fixed_dims panic on more than 8 entries
- Status: open (found in run 2026-10-21-pilot as F012)
- Location: jix/src/storage/params.rs:121
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F012.md; repro in that run's repros/.

## K13: A leading 0 dim hides an overflowing shape product in try_product
- Status: open (found in run 2026-10-21-pilot as F013)
- Location: jix/src/util/mod.rs:145
- Notes: bug-hunt/runs/2026-10-21-pilot/findings/F013.md; repro in that run's repros/.
