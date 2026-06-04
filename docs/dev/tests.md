# Testing Guide

## Overview

Tests live in `#[cfg(test)]` modules inline with the source code. The gold standard for
element-wise op tests is `ops/op1.rs`; match its style when adding tests to other modules.

## Test crates

| Crate | Purpose |
|---|---|
| `proptest` | Property-based testing with automatic shrinking. |
| `paste` | Expands `[<ident_$var>]` tokens inside `paste::paste! { }` - required to generate named test functions from macros. |
| `ndarray` | Already a regular dependency. Used in tests to construct reference arrays and compare results. |
| `tempfile` | Used by archive I/O tests that write/read actual files. |

## Shared test utilities (`jix/src/util/test_util.rs`)

All helpers are `pub(crate)` and re-exported from `util/mod.rs` under `#[cfg(test)]`.

### `arr_params`

```rust
pub(crate) fn arr_params(block_shape: &[usize]) -> ArrayParams
```

Creates `ArrayParams` with the given block shape and defaults elsewhere.
Always use this rather than constructing `ArrayParams` by hand in tests.

### `ScalarStrategy`

```rust
pub(crate) trait ScalarStrategy: Dtyped + Default + Debug + Clone + 'static {
    fn any_strategy()          -> BoxedStrategy<Self>;
    fn op_safe_strategy()      -> BoxedStrategy<Self> { Self::any_strategy() }
    fn unit_strategy()         -> BoxedStrategy<Self> { Self::op_safe_strategy() }
    fn shift_safe_strategy()   -> BoxedStrategy<Self> { Self::any_strategy() }
    fn comparable_strategy()   -> BoxedStrategy<Self> { Self::any_strategy() }
    fn maybe_non_finite_strategy() -> BoxedStrategy<Self> { Self::any_strategy() }
}
```

Implemented for every scalar dtype (including feature-gated `f16` and `Complex`).

| Method | Integer types | Float types | When to use |
|---|---|---|---|
| `any_strategy()` | full `u8`/`i32`/... range | arbitrary bits (rarely NaN/inf) | Codec roundtrip tests, logical/bitwise ops. |
| `op_safe_strategy()` | small positive range (e.g. `1..=100`) | `1.0..=100.0` | Arithmetic ops where values combine (`a + b`, `a * b`) - prevents overflow. |
| `unit_strategy()` | (falls back to `op_safe_strategy`) | `[-1.0, 1.0]` | Domain-restricted ops (`asin`, `acos`) - values outside the domain produce NaN which breaks `==`. |
| `shift_safe_strategy()` | `0..bit_width` (e.g. `0u8..8`) | n/a | Shift amounts for `<<`/`>>` - prevents debug-mode panic on out-of-range shifts. |
| `comparable_strategy()` | `{1, 2, 3}` | `{1.0, 2.0, NaN}` | Equality/comparison ops (`equal`, `not_equal`) - small set ensures ~33 % of pairs are equal and NaN semantics are exercised. |
| `maybe_non_finite_strategy()` | (falls back to `any_strategy`) | 7/8 `any_strategy` + 1/8 `{NaN, inf, -inf}` | Ops whose *bool* output is well-defined for NaN inputs (`greater`, `less`, `is_nan`, `is_finite`) - tests edge-case branches without breaking `assert_eq`. |

**Key rule:** `assert_array_matches` uses `PartialEq`. For ops whose *float output* may be NaN
(e.g. `maximum`, `sqrt` with negative input), use `op_safe_strategy` to keep outputs finite.
For ops whose output is `bool`, `maybe_non_finite_strategy` is safe even with NaN inputs
because the reference closure computes the same `false`/`true` and `bool: Eq`.

### Array strategies

```rust
// Random ndarray with shape from shape_strategy() and elements from T::any_strategy().
pub(crate) fn ndarray_strategy<T: ScalarStrategy>()
    -> impl Strategy<Value = ndarray::ArrayD<T>>

// Same, but caller supplies the element strategy.
pub(crate) fn ndarray_strategy_generic<T>(
    shape: impl Strategy<Value = Vec<usize>>,
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = ndarray::ArrayD<T>>

// Pairs (ndarray, compact Array) sharing the same data. Uses any_strategy().
pub(crate) fn carray_strategy_any<T: ScalarStrategy>()
    -> impl Strategy<Value = (ndarray::ArrayD<T>, Array<Compact>)>

// Same, but caller supplies the element strategy - used in op1 tests.
pub(crate) fn carray_strategy_from_shape<T: ScalarStrategy>(
    element: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = (ndarray::ArrayD<T>, Array<Compact>)>

// Two independent same-shape pairs - used in op2 tests.
pub(crate) fn carrays2_strategy<T: ScalarStrategy>()
    -> impl Strategy<Value = (
        (ndarray::ArrayD<T>, Array<Compact>),
        (ndarray::ArrayD<T>, Array<Compact>),
    )>
```

All compact strategies randomize both the array values and the block shape.

### `sub_range_strategy`

```rust
pub(crate) fn sub_range_strategy(shape: &[u64]) -> BoxedStrategy<Vec<Range<u64>>>
```

Generates a random per-dimension sub-range compatible with `to_ndarray_sub`.
Each dimension independently samples `start..end` with `0 <= start <= end <= shape[i]`.
Empty ranges and the full range are both in the sample space.

### `assert_array_matches`

```rust
pub(crate) fn assert_array_matches<S, T>(actual: &Array<S>, expected: &ndarray::ArrayD<T>)
```

The standard comparison helper for element-wise op tests:

1. Reads `actual` in full with `to_ndarray` and `assert_eq!`s against `expected`.
2. Runs an internal `TestRunner` with **16 cases**, each generating a random sub-range via
   `sub_range_strategy`, reading that region with `to_ndarray_sub`, slicing `expected` the
   same way, and comparing. This exercises block-boundary handling that a single full read
   would not catch.

Panics on failure (caught correctly by an enclosing `proptest!` runner).

---

## The op1 pattern - gold standard

`ops/op1.rs` is the reference implementation. Mirror it when writing tests for any
element-wise unary operation.

### Macro structure

```rust
// One proptest function per (op, dtype). $strategy is a ScalarStrategy method name.
macro_rules! test_op1_dtype {
    // Shorthand: same input and output dtype.
    ($op_method:ident, |$arg:ident| $body:expr, $dtype:ident, $strategy:ident) => {
        test_op1_dtype!($op_method, |$arg| $body, $dtype => $dtype, $strategy);
    };
    // General form: input dtype may differ from output dtype (e.g. complex abs -> real).
    ($op_method:ident, |$arg:ident| $body:expr, $in_dtype:ident => $out_dtype:ident, $strategy:ident) => {
        paste::paste! {
            proptest::proptest! {
                #[test]
                fn [<$op_method _ $in_dtype>](
                    (nd, za) in crate::util::carray_strategy_from_shape::<$in_dtype>(
                        <$in_dtype as crate::util::ScalarStrategy>::$strategy()
                    )
                ) {
                    #[allow(unused_imports)] use std::ops::Neg;
                    let result = za.$op_method();
                    let expected = nd.mapv(|$arg| $body);
                    crate::util::assert_array_matches(&result, &expected);
                }
            }
        }
    };
}

// Creates one test per dtype. Optional trailing groups add feature-gated dtypes.
macro_rules! test_op1 {
    (
        $op_method:ident, |$arg:ident| $body:expr,
        [$($dtype:ident),+], $strategy:ident
        $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+])*
    ) => {
        $(test_op1_dtype!($op_method, |$arg| $body, $dtype, $strategy);)+
        $($(
            #[cfg($cfg)]
            test_op1_dtype!($op_method, |$arg| $body, $cfg_dtype, $strategy);
        )+)*
    };
}
```

### Calling the macros

```rust
#[cfg(test)]
mod tests {
    // Bring feature-gated type aliases into scope.
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    // ... macro definitions ...

    // Standard same-dtype op:
    test_op1!(floor, |a| a.floor(), [f32, f64], op_safe_strategy);

    // Op with overflow risk: use op_safe_strategy for integer types.
    test_op1!(neg, |a| -a, [i8, i16, i32, i64, f32, f64], op_safe_strategy,
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );

    // Op with restricted domain: use unit_strategy to avoid NaN comparison failures.
    test_op1!(asin, |a| a.asin(), [f32, f64], unit_strategy);

    // Op with a different output dtype (e.g. complex abs -> real):
    // use the $in_dtype => $out_dtype form in a separate sub-module.
    #[cfg(feature = "num-complex")]
    mod abs_complex {
        use super::{complex_f32, complex_f64};
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f32 => f32, op_safe_strategy);
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f64 => f64, op_safe_strategy);
    }
}
```

### Key rules

- **`#[allow(unused_imports)] use std::ops::Neg;`** must appear in the test body so that
  `za.neg()` resolves when `$op_method = neg`. It is harmless for all other ops.
- The **scalar closure** (`|a| $body`) is applied with `ndarray::ArrayD::mapv` to produce
  the reference result. It must express exactly the same computation as the kernel.
- Use **`assert_array_matches`** rather than `to_ndarray` + `prop_assert_eq!`. It verifies
  both full reads and random sub-range reads in one call.
- The strategy shape is fully random (from `shape_strategy()`), covering 1D through 8D,
  small and large sizes, and zero-length dimensions. No need to add separate 1D/2D/multi-block
  tests; `carray_strategy_from_shape` handles this via random block shapes.

---

## Strategy selection reference

| Op kind | Strategy | Reason |
|---|---|---|
| Codec roundtrip | `any_strategy()` | Full domain is valid input. |
| Logical / bitwise ops (bool output) | `any_strategy()` | No overflow; any input is valid. |
| Arithmetic / unary with overflow risk | `op_safe_strategy()` | Prevents wrap on integer types. |
| Float transcendentals (floor, exp, ln, sin...) | `op_safe_strategy()` | Gives finite positive values; both sides compute identically so inf/NaN comparisons don't arise. |
| Domain-restricted (`asin`, `acos`) | `unit_strategy()` | Inputs outside `[-1, 1]` produce NaN; `NaN != NaN` under `PartialEq`. |
| Arithmetic binary ops (op2) | `op_safe_strategy()` | Same overflow avoidance. |
| Bit shifts | `shift_safe_strategy()` for the shift amount | Keeps shift in `[0, bit_width)` to avoid debug-mode panic. |
| Equality / comparison with bool output | `comparable_strategy()` | Ensures ~33 % equal pairs so both `true`/`false` branches are hit; float variant includes NaN for IEEE edge cases. |
| Ordering / logical ops with bool output and float input | `maybe_non_finite_strategy()` | Exercises NaN -> `false` paths without breaking `assert_array_matches` (output is `bool`, not float). |
| Float ops where NaN/inf propagates to the *output* (`maximum`, `minimum`, `sqrt`) | `op_safe_strategy()` | Avoids NaN in the output, which would break `assert_eq` via `PartialEq`. |

### Deciding what strategy to use

1. **Will the op ever output NaN/inf given these inputs?**
   - Yes (float arithmetic output): use `op_safe_strategy` or `unit_strategy`.
   - No (bool output, or integer output): you can use `any_strategy` or stronger.

2. **Do you need to test NaN/inf edge cases?**
   - Inputs produce NaN in the output -> doc-test it explicitly; proptest with `op_safe_strategy`.
   - Bool output with NaN inputs -> `maybe_non_finite_strategy` is safe.

3. **Do you need to test the equality branch?**
   - Equality / `not_equal` -> `comparable_strategy` ensures ~33 % hits.
   - For integers with `any_strategy`, collisions are astronomically rare (e.g. 1/2^32 for `i32`).

4. **Can values overflow when combined?**
   - Arithmetic (`+`, `*`, `-`) on integers -> `op_safe_strategy`.
   - Bitwise / logical -> no overflow, `any_strategy` is fine.

---

## Binary op tests (op2 style)

`ops/op2.rs` uses a similar but distinct macro structure because it needs two arrays of the
same shape and because it tests chaining (`(a op b) op c`). It still uses proptest and
`op_safe_strategy`, but builds arrays from fixed sizes rather than `carray_strategy_any`.
That module predates the op1 refactor; future binary op tests should be considered for
migration to the `carrays2_strategy` + `assert_array_matches` pattern.

---

## Explicit-value tests

Use plain `#[test]` functions with hardcoded `i32` arrays when:

- The test verifies a **specific computed shape** (reshape, slice, broadcast).
- The property holds only for a **particular input** (e.g. keepdims semantics, error paths).
- The input domain is tiny and exhaustive testing is trivial.

```rust
fn make(vals: Vec<i32>, shape: &[usize]) -> Array<Compact> {
    let nd = ndarray::ArrayD::from_shape_vec(shape.to_vec(), vals).unwrap();
    Array::compact_array(&nd).unwrap()
}

#[test]
fn shape_after_op() {
    assert_eq!(make(vec![1, 2, 3, 4, 5, 6], &[2, 3]).my_op().shape(), &[3]);
}
```

Prefer `i32` as the default dtype in explicit-value tests - it is exact, readable, and
not feature-gated.

---

## Feature-gated types

Wrap `f16` and complex tests in the appropriate `#[cfg]`:

```rust
#[cfg(feature = "half")]
use crate::scalar::f16;

#[cfg(feature = "num-complex")]
#[allow(non_camel_case_types)]
type complex_f32 = crate::scalar::Complex<f32>;
```

Pass them as `#[cfg(feature = "...")] [dtype, ...]` trailing groups in `test_op1!`.

---

## How to test a new element-wise unary op

1. Add `#[cfg(test)] mod tests` at the bottom of the op's source file.
2. Copy the `test_op1_dtype!` / `test_op1!` macro definitions from `ops/op1.rs`.
3. For each dtype the op supports, add a `test_op1!` invocation:
   - Choose `op_safe_strategy` unless the op has a restricted domain.
   - If output dtype differs from input dtype, use the `$in => $out` form in a sub-module.
4. Run `cargo test <module_path>` and confirm one test per dtype appears.
