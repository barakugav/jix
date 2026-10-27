// F009 repro (run 2026-10-27-batch5), crate-internal: NdIterExtBlockOffsetSize and the NdIter
// builder are pub(crate), so this module is appended to a SCRATCH COPY of
// jix/src/util/iter/block.rs (never the shared tree):
//
//   S=bug-hunt/runs/2026-10-27-batch5/scratch/crate-F009
//   cat bug-hunt/runs/2026-10-27-batch5/repros/F009_internal.rs >> $S/src/util/iter/block.rs
//   CARGO_TARGET_DIR=bug-hunt/runs/2026-10-27-batch5/scratch/target-F009 \
//     cargo test --manifest-path $S/Cargo.toml --all-features --lib f009_repro -- --nocapture
//
// The tests assert the CORRECT behavior: walking the block grid with the bounds that the type's
// doc prescribes (block.rs:11-12: `begin / b` .. `end.div_ceil(b)`) must visit exactly the blocks
// the element range touches, i.e. the same items as the `calc_block_end` bounds every real caller
// uses. They fail while the doc names `end.div_ceil(b)`, because for an empty range whose begin is
// not block-aligned the documented bounds visit a phantom block of size 0.

#[cfg(test)]
mod f009_repro {
    use crate::dimension::DimDyn;
    use crate::util::calc_block_end;
    use crate::util::iter::NdIter;
    use crate::SliceExt;

    type Item = (Vec<u64>, Vec<u64>, Vec<u64>);

    fn walk(
        block_begin: &[u64],
        block_end: &[u64],
        begin: &[u64],
        end: &[u64],
        block: &[u64],
    ) -> Vec<Item> {
        NdIter::<DimDyn, ()>::builder_with_begin(
            block_begin.to_dim_vec::<DimDyn>(),
            block_end.to_dim_vec::<DimDyn>(),
        )
        .with_block_offset_size_ext(
            &begin.to_dim_vec::<DimDyn>(),
            &end.to_dim_vec::<DimDyn>(),
            block.to_dim_vec::<DimDyn>(),
        )
        .build()
        .map(|(idx, (off, size))| {
            (
                idx.as_ref().to_vec(),
                off.as_ref().to_vec(),
                size.as_ref().to_vec(),
            )
        })
        .collect()
    }

    /// Bounds exactly as the doc of `NdIterExtBlockOffsetSize` (block.rs:11-12) prescribes.
    fn walk_documented(begin: &[u64], end: &[u64], block: &[u64]) -> Vec<Item> {
        let bb: Vec<u64> = begin.iter().zip(block).map(|(&b, &c)| b / c).collect();
        let be: Vec<u64> = end.iter().zip(block).map(|(&e, &c)| e.div_ceil(c)).collect();
        walk(&bb, &be, begin, end, block)
    }

    /// Bounds as every real caller computes them (util/mod.rs:625 `calc_block_end`).
    fn walk_calc_block_end(begin: &[u64], end: &[u64], block: &[u64]) -> Vec<Item> {
        let bb: Vec<u64> = begin.iter().zip(block).map(|(&b, &c)| b / c).collect();
        let be: Vec<u64> = (0..begin.len())
            .map(|d| calc_block_end(begin[d], end[d], block[d]))
            .collect();
        walk(&bb, &be, begin, end, block)
    }

    #[test]
    fn f009_documented_bounds_empty_unaligned_1d() {
        // [5, 5) touches no element, so no block may be visited.
        let got = walk_documented(&[5], &[5], &[4]);
        println!("documented bounds, [5,5) b=4: {got:?}");
        println!(
            "calc_block_end bounds, [5,5) b=4: {:?}",
            walk_calc_block_end(&[5], &[5], &[4])
        );
        assert_eq!(got, Vec::<Item>::new());
    }

    #[test]
    fn f009_documented_bounds_empty_unaligned_2d() {
        // Dim 0 is the empty range [5, 5); dim 1 covers two full blocks. The region is empty.
        let got = walk_documented(&[5, 0], &[5, 8], &[4, 4]);
        println!("documented bounds, [5,5)x[0,8) b=[4,4]: {got:?}");
        assert_eq!(got, Vec::<Item>::new());
    }

    #[test]
    fn f009_documented_bounds_match_calc_block_end_exhaustive_1d() {
        // Exhaustive 1-d sweep: the documented bounds should give the same walk as the bounds the
        // callers use. Print every disagreement so the write-up can name the exact condition.
        let mut mismatches = Vec::new();
        for b in 1u64..=5 {
            for begin in 0u64..=12 {
                for end in begin..=12 {
                    let doc = walk_documented(&[begin], &[end], &[b]);
                    let real = walk_calc_block_end(&[begin], &[end], &[b]);
                    if doc != real {
                        mismatches.push((begin, end, b, doc));
                    }
                }
            }
        }
        let n = mismatches.len();
        let all_empty_unaligned = mismatches
            .iter()
            .all(|&(begin, end, b, _)| begin == end && begin % b != 0);
        println!(
            "{n} mismatches; all of them begin == end && begin % b != 0: {all_empty_unaligned}"
        );
        for m in mismatches.iter().take(6) {
            println!("  begin={} end={} b={} documented walk={:?}", m.0, m.1, m.2, m.3);
        }
        assert_eq!(n, 0, "documented block_end disagrees with calc_block_end");
    }
}
