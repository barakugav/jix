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
| `Add` `Sub` `Mul` `Square` | wraps | - | done |
| `Neg` `Abs` of `MIN` | `MIN` | - | todo |
| `Pow`, unsigned exponent | wraps | - | done |
| `Pow`, signed exponent `< 0` | `0` | `invalid` | done |
| `Div` `DivideTrunc` `DivideFloor` `DivideEuclid` by `0` | `0` | `divide` | done |
| the same, `MIN / -1` | `MIN` | `over` | done |
| `RemainderTrunc` `RemainderFloor` `RemainderEuclid` by `0` | `0` | `divide` | done |
| the same, `MIN % -1` | `0` | - | done |
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
| `DivideTrunc` `DivideFloor` `DivideEuclid` | `x / 0` / `0 / 0`, `inf / x` / finite gives `inf` | `divide` / `invalid` / `over` | todo |
| `RemainderTrunc` `RemainderFloor` `RemainderEuclid` | `x % 0`, `inf % x` | `invalid` | todo |
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

The rules below come from reading the generated assembly and A/B benchmarks against a version
without the checks.

- A kernel that can hit an error is declared as `f(status)` in `define_scalar_op1_trait!` /
  `define_scalar_op2_trait!` and `define_op1!` / `define_op2!`, and implemented with
  `|a, b, status|` closures in `impl_scalar_op1!` / `impl_scalar_op2!`, which map the SIMD
  closure with `map_vectors_status` / `map_vectors2_status`. The integer quotients and
  remainders come from `checked_div` & co.: `int_quotient` / `int_remainder` give numpy's
  result where they are `None`.
- The status of a loop must stay in a register. Never pass `&mut status` (or a bulk's input
  arrays) to an out-of-line function: the pointer forces the status, or the arrays, into memory
  on every iteration. Out-of-line functions return the status bits by value instead.
- **Scalar path.** A rare error branches to an out-of-line `#[cold]` function. For integer
  `Div` the opaque call is required: without it LLVM if-converts the loop into vector code with
  a scalarized division per lane, which measured up to 2x slower. A call that returns the status
  by value is still opaque enough. A check that is cheap and
  branch-free, like `x < 0` for `Sqrt` or the negative exponent of `Pow`, uses
  `ArithmeticOpStatus::set_if` directly, which LLVM can still vectorize.
- **SIMD path.** When the predicate is exact (`Sqrt`), set the status from it per vector
  (`set_if` of `any_true`): it measured the same as one mask test per bulk, even at AVX2 and
  SSE4.2. When it is a superset (float `Div`), test it per vector and classify the rare vector
  that has a hit out of line, from a copy of its lanes. Order the tests so the common
  case is a single test: `!is_finite(r)` first, and only then the NaN-operand filter, so NaN
  inputs do not reach the out-of-line classification.
- The status is checked once per read, in `to_buf_type_erased(_dyn)`.
