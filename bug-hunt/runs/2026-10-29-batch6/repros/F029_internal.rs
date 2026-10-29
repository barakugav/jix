// F029 (run 2026-10-29-batch6) - crate-internal repro, appended to a scratch COPY of
// jix/src/util/test_util.rs (ScalarStrategy is pub(crate), unreachable from jix/tests/).
// Asserts the correct behavior for the test infrastructure: every bool8 strategy must be able to
// draw a byte other than 0/1 (since #89 any non-zero byte is a valid `true`). Fails while the
// gap exists (test_util.rs:348-356 builds bool8 only from `any::<bool>()` / `Just(TRUE)`).
//
// Run (scratch copy):
//   CARGO_TARGET_DIR=bug-hunt/runs/2026-10-29-batch6/scratch/target-f029 cargo test \
//     --manifest-path bug-hunt/runs/2026-10-29-batch6/scratch/crate-f029/Cargo.toml \
//     --all-features --lib f029_
#[cfg(test)]
mod f029_repro {
    use super::ScalarStrategy;
    use crate::scalar::bool8;
    use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
    use proptest::test_runner::TestRunner;

    fn draws_noncanonical(s: BoxedStrategy<bool8>) -> bool {
        let mut runner = TestRunner::deterministic();
        (0..5000).any(|_| s.new_tree(&mut runner).unwrap().current().0 > 1)
    }

    #[test]
    fn f029_bool8_strategies_draw_noncanonical_bytes() {
        let strategies: [(&str, BoxedStrategy<bool8>); 6] = [
            ("any_strategy", <bool8 as ScalarStrategy>::any_strategy()),
            ("logical_op_strategy", <bool8 as ScalarStrategy>::logical_op_strategy()),
            ("op_safe_strategy", <bool8 as ScalarStrategy>::op_safe_strategy()),
            ("op_safe_non_zero_strategy", <bool8 as ScalarStrategy>::op_safe_non_zero_strategy()),
            ("comparable_strategy", <bool8 as ScalarStrategy>::comparable_strategy()),
            ("maybe_non_finite_strategy", <bool8 as ScalarStrategy>::maybe_non_finite_strategy()),
        ];
        let gaps: Vec<&str> = strategies
            .into_iter()
            .filter(|(_, s)| !draws_noncanonical(s.clone()))
            .map(|(name, _)| name)
            .collect();
        assert!(
            gaps.is_empty(),
            "bool8 strategies that never draw a byte other than 0/1 in 5000 draws: {gaps:?}"
        );
    }

    #[test]
    fn f029_bool8_array_strategy_draws_noncanonical_bytes() {
        // array_strategy_from_shape::<bool8> (used e.g. by where_op's property tests) inherits it.
        let s = crate::util::array_strategy_from_shape::<bool8>(
            proptest::strategy::Just(vec![64usize]),
            <bool8 as ScalarStrategy>::any_strategy(),
        );
        let mut runner = TestRunner::deterministic();
        let found = (0..500).any(|_| {
            let (nd, _arr) = s.new_tree(&mut runner).unwrap().current();
            nd.iter().any(|b| b.0 > 1)
        });
        assert!(found, "array_strategy_from_shape::<bool8> never draws a byte other than 0/1");
    }
}
