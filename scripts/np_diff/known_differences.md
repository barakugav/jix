# Known differences

Differences between jix and numpy/torch that are known and accepted as is, one per line. Run
`np_diff.py`: a diff in its report that no line here explains is new, and needs fixing or adding
here. Differences that are known but still need fixing are in the repo's `todo.md`, not here.

## API

- Names follow torch: `dim`/`keepdim` (numpy `axis`/`keepdims`), `clamp`, `divide_floor`, `remainder_trunc`/`remainder_floor`, `is_nan`/`is_infinite`/`is_finite`, `permute_dims`, `swapdims`, `asin`/`acos`/`atan`.
- jix-only ops, no numpy/torch equivalent: `divide_trunc` (torch: `div(rounding_mode="trunc")`), `divide_euclid`, `remainder_euclid`, `bitwise_rotate_left/right`, `count_zeros`, `leading_zeros`, `trailing_zeros`, `reverse_bits`.
- Ops are lazy: errors, FP errors included, are raised when the result is read, under the `jix.errstate` in effect then.
- Ops take no `out=` or `dtype=` arguments.

## Supported dtypes

- `sign`, `round` reject complex (numpy accepts; torch rejects).
- `floor`, `ceil`, `round` reject bool (numpy: bool, bool, float16; torch rejects).
- `max`, `min`, `argmax`, `argmin` reject complex (numpy orders lexicographically; torch rejects).
- `maximum`, `minimum`, `clamp` reject complex (numpy orders lexicographically; torch rejects).
- `greater`, `greater_equal`, `less`, `less_equal` reject complex (numpy orders lexicographically; torch rejects).
- `cast` from complex to a real dtype raises TypeError (numpy, torch drop the imaginary part).
- `square` rejects bool (numpy int8, torch int64).
- `product` rejects bool (numpy, torch int64).
- `count_ones` rejects bool (numpy `bitwise_count` gives uint8).
- `swap_bytes` rejects bool, float and complex (numpy `byteswap` takes any dtype).
- `imag` rejects non-complex dtypes (numpy: zeros; torch rejects).
- `negative(bool)` is int8, True -> -1 (numpy, torch raise).

## Type promotion

- A Python float with a bool/8-bit/16-bit int array is float32 (numpy float64, torch float32), so `int8 + 1e300` is `inf`.
- A Python complex with a bool/8-bit/16-bit int array is complex64 (numpy complex128, torch complex64).
- `bool + bool`, `bool * bool` are uint8 (numpy, torch bool).
- `bool - bool` is int8, `True - False` is 1, `False - True` is -1 (numpy, torch raise).
- `divide_floor`, `divide_trunc`, `remainder_floor`, `remainder_trunc`, `power`, shifts of two bools are uint8 (numpy int8; torch raises).
- Arithmetic of a bool array with a Python int is uint8, int8 for `subtract` (numpy, torch int64), so `bool + -1` and `bool + 1000` raise OverflowError.
- A Python int outside the int64 range raises OverflowError, `2**63` included (numpy takes up to `2**64 - 1` with a uint64 array, any with a float or complex array, in its dtype, and with an int array in `divide`, comparisons, `clamp`, `isclose`; torch raises from `2**64`).
- `sqrt`, `exp`, `log`, trig of bool/int8/uint8 are float32 (numpy float16, torch float32).
- `divide` of two bool/8-bit/16-bit ints, or of one with a Python int, is float32 (numpy float64, torch float32).
- `power` of a signed int by an unsigned int is the signed int of the wider width, `int8 ** uint8` is int8 and `int8 ** uint16` is int16 (numpy one size up: int16, int32).
- `power` of a float by an int keeps the float dtype: `float16 ** int16/uint16/int32` is float16, `float32 ** int32` is float32 (numpy float32/float64; torch like jix).
- `power` with a uint64 operand (uint ** uint64, Python int ** uint64, uint64 ** uint64) is float64 (numpy uint64).
- `negative` of an unsigned int is the signed int of the same width, `-uint8(200)` is `int8(56)` (numpy `uint8(56)`; torch like numpy for uint8).
- `argmax`, `argmin` return uint64 (numpy, torch int64).
- `sum` of bool is uint64 (numpy, torch int64).
- A Python scalar passed as the array argument stays weak: `clamp(1, min=uint8 array)` is uint8 (numpy makes it a strong int64/float64 array; torch like jix).

## Values

- `round` rounds half away from zero: `round(0.5)` is 1, `round(-2.5)` is -3 (numpy, torch: half to even).
- `cast` from float to int saturates and maps NaN to 0 (numpy, torch: CPU-dependent, e.g. INT_MIN or wrapped).
- Complex `divide` is the textbook `(a * conj(b)) / |b|^2`: a zero or infinite divisor gives `nan+nanj`, and `|b|` above ~1e154 (complex128) or ~1.8e19 (complex64), or tiny, overflows/underflows to nan/inf (numpy, torch: Smith's algorithm).
- Integer `power` with a negative exponent is 0, bases 1 and -1 included, and reports `invalid` (numpy raises ValueError; torch truncates, 1 and +-1 for bases 1 and -1).
- `mean` doesn't overflow in the sum: `mean([f32 max, f32 max])` is f32 max (numpy, torch inf).
- `var`, `std` of `[max, max]` are 0 (numpy inf; torch 0).
- `divide_trunc(+-inf, finite)` is nan, like numpy's `floor_divide` (torch `div(rounding_mode="trunc")`: +-inf).

## FP errors

- The default errstate is `raise` for `divide` and `invalid`, `warn` for `over` and `under` (numpy: warn, warn, ignore under).
- `add`, `multiply`, `square`, `sum`, `product` report no `over` or `under` (numpy does).
- `under` is reported by `exp`, `sin`, `divide`, `power` only. numpy also reports it from intermediate results: in `floor_divide` (jix `divide_floor`), `isclose`, and float32 `exp`, `sin`, `cos` (e.g. `exp` of a subnormal, `sin(1e-30)`).
- `add`, `subtract`, `multiply`, `sum`, `product` report no `invalid` when they produce NaN, e.g. `inf - inf`, `0 * inf` (numpy does).
- `cast` reports no FP errors (numpy: `invalid` for nan/inf/out-of-range float -> int, `over` for int -> float16).
- `mean`, `var`, `std` report no FP errors (numpy: `over`, `invalid`, `divide`, `under` on inf/max inputs).
- Complex `divide`, `mean`, `var`, `std` report no FP errors (numpy: `divide`, `invalid`).
- `power` with uint64 operands computes in float64, so overflow reports `over` (numpy wraps in uint64 silently).

## Shapes and indexing

- At most 8 dims (numpy 64).
- `argmax`, `argmin` without `dim` need a 1-d array (numpy, torch: index into the flattened array).
- `roll` without `dim` needs a 1-d array (numpy, torch roll the flattened array).
- `repeat` without `dim` needs a 1-d array (numpy `repeat`, torch `repeat_interleave` flatten it).
- `tile` repeats one dim by one int, and needs a 1-d array without `dim` (numpy, torch take a reps tuple over the last dims).
- `broadcast` accepts -1 in the target shape to keep a dim (numpy `broadcast_to` raises; torch like jix).
- `mean`, `var`, `std` of an empty array raise RuntimeError (numpy nan with a RuntimeWarning; torch nan).
- Indexing bounds are strict: a slice end past the dim raises IndexError (numpy, torch clamp it).
- Indexing doesn't take `None`/`np.newaxis` (TypeError).
- `where` needs a bool condition (numpy casts it; torch raises like jix).

## torch only (jix matches numpy)

- torch has no arithmetic for uint16/uint32/uint64 and doesn't promote them with other dtypes.
- torch 0-d tensors are weak in promotion: uint8 tensor + int64 0-d tensor is uint8.
- torch int + float is the float's dtype: `int32 + float32` is float32 (numpy, jix float64); float16 + complex needs complex32, which torch mostly lacks.
- torch wraps Python ints that don't fit the tensor, `uint8 + 1000` (numpy, jix raise OverflowError).
- torch raises on integer division/remainder by zero (numpy, jix: 0); `torch.div(rounding_mode="trunc")` kills the process on int32/int64 `INT_MIN / -1` (np_diff.py raises `SIGFPE` instead).
- torch complex arithmetic with infinities gives NaN parts: `(1+0j) + (inf+0j)` is `inf+nanj`.
- torch at extremes: `sign(nan)` is 0, `remainder(f64 max, 0.5)` is nan, `std([0, max])` is finite (numpy, jix: nan, 0, inf).
- torch functions take tensors/tuples where numpy and jix take scalars/ints/None: `maximum(t, 1)`, `isclose(t, 1)`, `flip(t)`, `permute(t)`, `reshape(t, -1)`, `unsqueeze(t, (0, 2))`, `concatenate` with a list (np_diff.py passes Python scalars to the first ones as 0-d tensors).
- `torch.squeeze(t, dim)` on a dim whose size isn't 1 is a no-op (numpy, jix raise).
