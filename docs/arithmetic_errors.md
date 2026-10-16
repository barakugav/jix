# Arithmetic errors in element-wise kernels

Kernels never panic on bad inputs. A kernel that hits an arithmetic error produces a value for
the element and sets the error's bit in an `ArithmeticOpStatus`. When the read finishes, the
`ReadContext` handles each set bit according to its mode: ignore it, record a warning, or fail
the read. In Python the modes come from `jix.errstate()` / `jix.geterr()`, as in numpy.

The error kinds follow numpy: `divide` (`DivideByZero`), `over` (`Overflow`), `under`
(`Underflow`) and `invalid` (`InvalidOperation`). NaN inputs propagate silently, as in numpy.

## Integers

| Kernel | Value | Status | State |
|---|---|---|---|
| `Add` `Sub` `Mul` `Square` | wraps | - | todo |
| `Neg` `Abs` of `MIN` | `MIN` | - | todo |
| `Pow`, unsigned exponent | wraps | - | done |
| `Pow`, signed exponent `< 0` | `0` | `invalid` | done |
| `Div` by `0` | `0` | `divide` | done |
| `Div` `MIN / -1` | `MIN` | `over` | done |
| `Shl` `Shr` with count `>= width` or `< 0` | undecided | - | open |
| `Sum` `Product` `Mean` | wraps | - | todo |

numpy raises `ValueError` for a negative integer exponent, whatever `errstate` says. jix reports
it as `invalid` instead, so it follows the mode like the other errors.

## Floats

Rust float arithmetic never panics: results are the IEEE 754 values (`inf`, `NaN`), so the kernels
detect the errors from their inputs and outputs. The values stay the IEEE ones.

| Kernel | Error | Status | State |
|---|---|---|---|
| `Div` | `x / 0` with finite `x != 0` | `divide` | done |
| | `0 / 0`, `inf / inf` | `invalid` | done |
| | finite `/` finite gives `inf` | `over` | done |
| `Sqrt` | `x < 0` | `invalid` | done |
| `Ln` | `x == 0` / `x < 0` | `divide` / `invalid` | todo |
| `Exp` | finite `x` gives `inf` | `over` | todo |
| `Asin` `Acos` | `\|x\| > 1` | `invalid` | todo |
| `Sin` `Cos` `Tan` | `x == +-inf` | `invalid` | todo |
| `Pow` | negative base with a non-integer exponent / `0 ** negative` / finite gives `inf` | `invalid` / `divide` / `over` | todo |
| `Add` `Sub` `Mul` | no checks: too costly next to a one-instruction kernel | - | - |
| `Sum` `Product` `Var` `Std` | no checks for now | - | - |

Underflow is not detected (numpy ignores it by default). Complex `Div` does not report errors yet.

## Casts

Deferred.

## Implementation

- **Scalar path**: a rare error goes to an out-of-line `#[cold]` function that sets the status
  (e.g. integer `Div`). The opaque call also keeps LLVM from if-converting a loop that has no
  SIMD instruction (integer division) into scalarized vector code. A check that is cheap
  and branch-free, like `x < 0` for `Sqrt`, uses `ArithmeticOpStatus::set_if` directly, which
  LLVM can still vectorize.
- **SIMD path**: each vector ORs a cheap predicate into a mask; once per bulk, a non-empty mask
  calls an out-of-line function that classifies the bulk's elements exactly. For `Sqrt`
  the predicate is exact, so the mask sets the status directly.
- The status lives in a register during a loop and is checked once per read, in
  `to_buf_type_erased(_dyn)`.
