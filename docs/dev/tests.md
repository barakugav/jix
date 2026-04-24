# Testing Guide

## Overview

Tests live in `#[cfg(test)]` modules inline with the source code. There is no
separate `tests/` directory for unit tests. Integration tests (archive roundtrip,
file I/O) are also inline rather than in a top-level `tests/` crate.

## Test crates

| Crate | Purpose |
|---|---|
| `proptest` | Property-based testing with automatic shrinking. Used for codec filters, arithmetic ops, and type casts. |
| `paste` | Procedural macro that expands `[<ident_$var>]` tokens inside `paste::paste! { }`. Required to generate named test functions inside `proptest! { }` blocks from outer macros. |
| `ndarray` | Already a regular dependency. Used in tests to construct reference arrays and compare results. |
| `tempfile` | Used by archive I/O tests that write/read actual files. |
| `static_assertions` | Compile-time assertions (e.g., size/alignment checks). |

## Shared test utilities (`zix/src/util/test_util.rs`)

All helpers are declared in `test_util.rs` and re-exported unconditionally under
`#[cfg(test)]` from `util/mod.rs`:

```rust
#[cfg(test)]
mod test_util;
#[cfg(test)]
pub(crate) use test_util::*;
```

Any test module can therefore write `use crate::util::arr_params;` (or just rely
on the glob) without duplicating anything.

### `arr_params`

```rust
pub(crate) fn arr_params(block_shape: &[usize]) -> ArrayParams
```

Creates `ArrayParams` with the given block shape and all other fields defaulted.
Always use this instead of constructing `ArrayParams` by hand in tests.

### `gen_data_bytes_from_slice`

```rust
pub(crate) fn gen_data_bytes_from_slice<T: Dtyped>(items: &[T]) -> AlignedBytes
```

Casts a typed slice to correctly-aligned raw bytes. Used by the codec filter
roundtrip helper to feed proptest-generated values into `FilterImpl::encode`.

### `ScalarStrategy`

```rust
pub(crate) trait ScalarStrategy: Dtyped + Debug + Clone + 'static {
    fn any_strategy() -> BoxedStrategy<Self>;
    fn op_safe_strategy() -> BoxedStrategy<Self> { Self::any_strategy() }
}
```

Implemented for every scalar dtype (including feature-gated `f16` and `Complex`).

| Method | When to use |
|---|---|
| `any_strategy()` | Codec roundtrip tests — the full value domain is valid input. |
| `op_safe_strategy()` | Arithmetic op tests — restricted range prevents integer overflow. |

The safe ranges are sized so that the heaviest test pattern (`a = a_extra + b + c`,
then `(a * b) * c`) fits inside the type without wrapping:

| Type | `op_safe_strategy()` range | Derivation |
|---|---|---|
| i8 / u8 | 1..=4 | `3 * 4³ = 192 ≤ 127` (signed) |
| i16 | 1..=22 | `3r³ ≤ 32767 → r ≤ 22` |
| u16 | 1..=27 | `3r³ ≤ 65535 → r ≤ 27` |
| i32 / u32 / i64 / u64 | 1..=30 or 1..=100 | Safe for the test shapes used |
| f32 / f64 | 1..=100 (integer-valued) | Floats don't overflow; bounded for readability |
| f16 | 1..=15 | Stays inside f16 exact range |
| Complex | re=1..=15, im=0 | Avoids complex overflow; im=0 keeps expected values simple |

## When to use proptest

Use proptest when the correctness property holds for **all** inputs (or all inputs
in a defined safe range) and a failure case would be hard to construct by hand.
This covers:

- **Codec filter roundtrips** — encode then decode must recover the original bytes
  for any sequence of values of any dtype.
- **Arithmetic ops** — `Array(a) op Array(b)` must equal the elementwise ndarray
  result for any pair of arrays.
- **Type casts** — `Array.astype(D)` must produce the same values as a scalar cast
  on each element.

Do **not** use proptest when:

- The test verifies a specific computed value (shape transformations, reductions
  with a known result). Use explicit inputs and `assert_eq!` instead.
- The input domain is tiny and exhaustive testing is trivial (e.g., `bool`-only
  tests).

## Proptest style

Use the idiomatic function-level form throughout:

```rust
proptest::proptest! {
    #[test]
    fn my_test(x in some_strategy()) {
        // prop_assert_eq! instead of assert_eq!
        proptest::prop_assert_eq!(computed, expected);
    }
}
```

Do **not** use the closure form (`proptest!(|(x in ...)| { ... })` inside a plain
`#[test]`) unless you need conditional early-return logic that cannot be expressed
as a strategy filter. The closure form disables test naming and makes CI output
harder to read.

When generating named test functions inside a macro, wrap both `paste!` and
`proptest!` together:

```rust
macro_rules! test_something {
    ($dtype:ident) => {
        paste::paste! {
            proptest::proptest! {
                #[test]
                fn [<test_ $dtype>](val in <$dtype as crate::util::ScalarStrategy>::any_strategy()) {
                    // ...
                }
            }
        }
    };
}
```

`paste!` is a proc-macro that expands `[<...>]` identifiers before `proptest!`
(a declarative macro) sees the token stream, so the nesting works correctly.

## When to use macros

Macros are appropriate when you need the same test repeated for many dtypes or
many ops. The two standard patterns are:

### `test_op!` + `test_op_dtype!` (arithmetic ops, `ops/op2.rs`)

`test_op_dtype!($op, $dtype)` emits several proptest functions (1D, 2D, single
block, multi-block, three-array chaining) for one `(op, dtype)` pair.
`test_op!($mod_name, $op, [...dtypes...])` creates a module and calls
`test_op_dtype!` for each dtype, including feature-gated ones.

### `test_cast_pair!` (type casts, `ops/astype.rs`)

One invocation per `(src, dst)` type pair. Emits 4 proptest functions covering
1D / 2D × single-block / multi-block combinations.

### Codec filter macro

`test_roundtrip!($ty, $fn_name)` emits one proptest function that generates a
`Vec<$ty>` and calls the shared `test_roundtrip` helper. One invocation per dtype.

### Reduction macro

`test_reduce_axis0!` is a helper macro for explicit-value axis-0 reduction tests.
It is not dtype-parametrized; each invocation hardcodes input and expected values.

### When to skip macros

Shape op tests (`slice`, `reshape`, `broadcast`, etc.) and archive roundtrip tests
use plain `#[test]` functions with hardcoded `i32` arrays. Introducing a macro
there would add indirection without removing duplication — each test exercises a
distinct semantic (shape, padding, stride, multi-block read) that does not repeat
across dtypes.

## How to test a new op

1. **Decide on a test category.** If the op is correct for all values (or all
   values in a safe range), use proptest. If it produces specific computed values
   best verified against hand-written expectations, use explicit tests.

2. **Add proptest tests.** Mirror the structure of `ops/op2.rs`:
   - Write a `test_op_dtype!`-style macro that generates tests for one dtype.
   - Call it inside a `test_op!`-style macro for each supported dtype.
   - Use `op_safe_strategy()` if the op can overflow, `any_strategy()` otherwise.
   - Cover at least: single-block 1D, multi-block 1D, single-block 2D, multi-block 2D.

3. **Add explicit-value tests** for edge cases that proptest will not naturally
   find: empty arrays, shapes with padding, keepdims semantics, error paths
   (e.g., incompatible dtypes).

4. **Test multi-block layouts.** A common source of bugs is mishandling block
   boundaries. Always include at least one test where the array spans multiple
   blocks (e.g., `arr_params(&[2])` for a length-6 array).

5. **Test feature-gated dtypes separately.** Wrap `f16` and `Complex` tests in
   `#[cfg(feature = "half")]` / `#[cfg(feature = "num-complex")]`.

### Minimal proptest op example

```rust
#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::util::{arr_params, ScalarStrategy};
    use ndarray::ArrayD;

    macro_rules! test_my_op_dtype {
        ($dtype:ident) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<test_ $dtype _1d>](
                        vals in proptest::collection::vec(
                            <$dtype as ScalarStrategy>::op_safe_strategy(), 4usize,
                        )
                    ) {
                        let a = Array::from_ndarray(
                            &ArrayD::from_shape_vec(vec![4], vals.clone()).unwrap(),
                            arr_params(&[4]),
                        ).unwrap();
                        let actual = a.my_op().data().to_ndarray::<$dtype>().unwrap();
                        let expected = ArrayD::from_shape_vec(
                            vec![4],
                            vals.iter().map(|&x| my_op_scalar(x)).collect(),
                        ).unwrap();
                        proptest::prop_assert_eq!(actual, expected);
                    }
                }
            }
        };
    }

    macro_rules! test_my_op {
        ($($dtype:ident),+) => {
            $(test_my_op_dtype!($dtype);)+
        };
    }

    test_my_op!(i32, i64, f32, f64);
}
```

## Explicit-value test style

For correctness tests with known inputs and outputs, use the pattern established
in `ops/reduction.rs` and `ops/shape_ops/`:

```rust
fn make(vals: Vec<i32>, shape: &[usize]) -> Array<crate::storage::Compact> {
    let nd = ndarray::ArrayD::from_shape_vec(shape.to_vec(), vals).unwrap();
    Array::from_ndarray(&nd, arr_params(shape)).unwrap()
}

fn seq(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

#[test]
fn shape_after_op() {
    assert_eq!(make(seq(12), &[3, 4]).my_op(&[0]).shape(), &[4]);
}

#[test]
fn values_after_op() {
    let got: ndarray::ArrayD<i32> = make(seq(12), &[3, 4])
        .my_op(&[0])
        .data()
        .to_ndarray()
        .unwrap();
    assert_eq!(
        got,
        ndarray::ArrayD::from_shape_vec(vec![4], vec![8, 9, 10, 11]).unwrap()
    );
}
```

Prefer `i32` as the default dtype in explicit-value tests — it is exact, readable,
and not feature-gated. Add dtype-specific tests only when the op behaviour differs
by dtype (e.g., float `sum` vs integer `sum`).
