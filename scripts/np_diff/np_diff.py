"""Map how jix ops differ from numpy and torch.

Every check calls the same op in jix, numpy and torch on the same inputs, and records a `Diff`
whenever jix disagrees with numpy or with torch. The diffs are printed as a markdown report with a
section per check, split into the diffs where jix differs from numpy and the ones where only torch
differs. Rows that differ only in the op are merged, and then rows that differ only in the inputs.

Checks:
- dtypes:    every combination of operand dtypes and Python scalars -> result dtype, or error
- values:    every combination of special values (0, -1, min, max, nan, inf, ...) of each dtype
             -> result value
- fp-errors: the floating point errors (divide, over, under, invalid) reported for those values
- broadcast: pairs of operand shapes -> result shape, or error
- cases:     hand-written calls (reduction dims, shape ops, indexing, ...) -> result, or error
- missing:   numpy ufuncs with no jix counterpart

Usage (from the repo root, with jix, numpy and torch installed):
    python scripts/np_diff/np_diff.py                    # report on stdout
    python scripts/np_diff/np_diff.py --out report.md
    python scripts/np_diff/np_diff.py --op add --op sum  # only these ops
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import operator
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import jix

# ----------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------


@dataclass
class Lib:
    name: str
    asarray: Callable[[np.ndarray], Any]  # numpy array -> this library's array
    errstate: Callable[[], Any]  # context manager under which FP errors are reported as warnings


LIBS = [
    Lib("jix", jix.compact, lambda: jix.errstate(all="warn")),  # jix raises on divide/invalid by default
    Lib("numpy", np.array, lambda: np.errstate(all="warn")),
    Lib("torch", lambda a: torch.from_numpy(a.copy()), contextlib.nullcontext),  # torch reports no FP errors
]

# Words in the RuntimeWarning messages of numpy and jix -> FP error kind
FP_ERROR_WORDS = {"divide": "divide", "overflow": "over", "underflow": "under", "invalid": "invalid"}


@dataclass(frozen=True)
class Outcome:
    """What one library returned for one call: a value or an error, plus the FP errors it reported."""

    value: np.ndarray | None = None
    error: str = ""  # exception type, if the call raised
    fp_errors: str = ""  # e.g. "divide,invalid"


def run(lib: Lib, fn: Callable, args: list, kwargs: dict) -> Outcome:
    """Call `fn` with the numpy arrays in `args` converted to `lib`'s arrays, and read the result."""
    with warnings.catch_warnings(record=True) as caught, lib.errstate():
        warnings.simplefilter("always")
        try:
            res = fn(*[to_lib(lib, a) for a in args], **kwargs)
            res = np.asarray(res.numpy() if hasattr(res, "numpy") else res)  # jix is lazy, this computes it
        except BaseException as e:  # a Rust panic in jix is a BaseException
            if isinstance(e, KeyboardInterrupt):
                raise
            return Outcome(error=type(e).__name__)
    messages = " ".join(str(w.message) for w in caught)
    fp_errors = sorted({kind for word, kind in FP_ERROR_WORDS.items() if word in messages})
    return Outcome(value=res, fp_errors=",".join(fp_errors))


def to_lib(lib: Lib, arg: Any) -> Any:
    """numpy arrays, also inside lists (as for concatenate), become `lib`'s arrays."""
    if isinstance(arg, np.ndarray):
        return lib.asarray(arg)
    if isinstance(arg, list):
        return [to_lib(lib, a) for a in arg]
    return arg


# ----------------------------------------------------------------------------------------------------
# Ops: the same op in the three libraries, all called with jix's arguments
# ----------------------------------------------------------------------------------------------------


@dataclass
class Op:
    name: str
    jix: Callable  # the fields are named after the libraries in LIBS
    numpy: Callable | None  # None: no equivalent
    torch: Callable | None
    arity: int = 1  # number of array operands in the dtypes/values/broadcast checks
    reduction: bool = False  # called as fn(a, dim=None, keepdim=False, ...)


def scalars_as_tensors(fn: Callable) -> Callable:
    """For torch functions that take no Python scalars (e.g. torch.maximum): pass them as 0-d tensors."""

    def call(*args, **kwargs):
        args = [torch.tensor(a) if isinstance(a, (bool, int, float, complex)) else a for a in args]
        return fn(*args, **kwargs)

    return call


class SIGFPE(Exception):
    """The call would have killed the process with a SIGFPE."""


def torch_divide_trunc(a, b):
    """torch.div(a, b, rounding_mode="trunc") kills the process on an int32 or int64 INT_MIN / -1."""
    dtype = torch.result_type(a, b)
    int_min = torch.iinfo(dtype).min if dtype in (torch.int32, torch.int64) else None
    if int_min is not None and ((torch.as_tensor(a) == int_min) & (torch.as_tensor(b) == -1)).any():
        raise SIGFPE()
    return torch.div(a, b, rounding_mode="trunc")


def np_reduction(fn: Callable) -> Callable:
    """numpy names jix's `dim`/`keepdim` `axis`/`keepdims`."""
    return lambda a, dim=None, keepdim=False, **kw: fn(a, axis=dim, keepdims=keepdim, **kw)


def torch_reduction(fn: Callable) -> Callable:
    """Most torch reductions take no `dim=None` together with `keepdim`, so pass all dims."""

    def call(a, dim=None, keepdim=False, **kw):
        return fn(a, dim=tuple(range(a.ndim)) if dim is None else dim, keepdim=keepdim, **kw)

    return call


DTYPES = [
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

UNARY_OPS = [
    Op("negative", jix.negative, np.negative, torch.negative),
    Op("absolute", jix.absolute, np.absolute, torch.abs),
    Op("sign", jix.sign, np.sign, torch.sign),
    Op("square", jix.square, np.square, torch.square),
    Op("sqrt", jix.sqrt, np.sqrt, torch.sqrt),
    Op("exp", jix.exp, np.exp, torch.exp),
    Op("log", jix.log, np.log, torch.log),
    Op("log(base=2)", lambda a: jix.log(a, base=2), np.log2, torch.log2),
    Op("log(base=10)", lambda a: jix.log(a, base=10), np.log10, torch.log10),
    Op("sin", jix.sin, np.sin, torch.sin),
    Op("cos", jix.cos, np.cos, torch.cos),
    Op("tan", jix.tan, np.tan, torch.tan),
    Op("asin", jix.asin, np.asin, torch.asin),
    Op("acos", jix.acos, np.acos, torch.acos),
    Op("atan", jix.atan, np.atan, torch.atan),
    Op("floor", jix.floor, np.floor, torch.floor),
    Op("ceil", jix.ceil, np.ceil, torch.ceil),
    Op("round", jix.round, np.round, torch.round),
    Op("is_nan", jix.is_nan, np.isnan, torch.isnan),
    Op("is_infinite", jix.is_infinite, np.isinf, torch.isinf),
    Op("is_finite", jix.is_finite, np.isfinite, torch.isfinite),
    Op("real", jix.real, np.real, torch.real),
    Op("imag", jix.imag, np.imag, torch.imag),
    Op("logical_not", jix.logical_not, np.logical_not, torch.logical_not),
    Op("bitwise_not", jix.bitwise_not, np.bitwise_not, torch.bitwise_not),
    Op("count_ones", jix.count_ones, np.bitwise_count, None),
    Op("swap_bytes", jix.swap_bytes, lambda a: a.byteswap(), None),
    Op("-a", operator.neg, operator.neg, operator.neg),
    Op("abs(a)", abs, abs, abs),
    Op("~a", operator.invert, operator.invert, operator.invert),
] + [
    Op(
        f"cast({dt})",
        lambda a, dt=dt: jix.cast(a, dt),
        lambda a, dt=dt: a.astype(dt),
        lambda a, dt=dt: a.to(getattr(torch, dt)),
    )
    for dt in DTYPES
]


def binary(name, jix_fn, np_fn, torch_fn) -> Op:
    return Op(name, jix_fn, np_fn, torch_fn, arity=2)


BINARY_OPS = [
    binary("add", jix.add, np.add, torch.add),
    binary("subtract", jix.subtract, np.subtract, torch.subtract),
    binary("multiply", jix.multiply, np.multiply, torch.multiply),
    binary("divide", jix.divide, np.divide, torch.divide),
    binary("divide_floor", jix.divide_floor, np.floor_divide, torch.floor_divide),
    binary("divide_trunc", jix.divide_trunc, None, torch_divide_trunc),
    binary("remainder_floor", jix.remainder_floor, np.remainder, torch.remainder),
    binary("remainder_trunc", jix.remainder_trunc, np.fmod, scalars_as_tensors(torch.fmod)),
    binary("power", jix.power, np.power, torch.pow),
    binary("maximum", jix.maximum, np.maximum, scalars_as_tensors(torch.maximum)),
    binary("minimum", jix.minimum, np.minimum, scalars_as_tensors(torch.minimum)),
    binary("equal", jix.equal, np.equal, scalars_as_tensors(torch.eq)),
    binary("not_equal", jix.not_equal, np.not_equal, scalars_as_tensors(torch.ne)),
    binary("greater", jix.greater, np.greater, scalars_as_tensors(torch.gt)),
    binary("greater_equal", jix.greater_equal, np.greater_equal, scalars_as_tensors(torch.ge)),
    binary("less", jix.less, np.less, scalars_as_tensors(torch.lt)),
    binary("less_equal", jix.less_equal, np.less_equal, scalars_as_tensors(torch.le)),
    binary(
        "isclose",
        lambda a, b: jix.isclose(a, b, rtol=1e-5, atol=1e-8),
        lambda a, b: np.isclose(a, b, rtol=1e-5, atol=1e-8),
        scalars_as_tensors(lambda a, b: torch.isclose(a, b, rtol=1e-5, atol=1e-8)),
    ),
    binary("logical_and", jix.logical_and, np.logical_and, scalars_as_tensors(torch.logical_and)),
    binary("logical_or", jix.logical_or, np.logical_or, scalars_as_tensors(torch.logical_or)),
    binary("logical_xor", jix.logical_xor, np.logical_xor, scalars_as_tensors(torch.logical_xor)),
    binary("bitwise_and", jix.bitwise_and, np.bitwise_and, torch.bitwise_and),
    binary("bitwise_or", jix.bitwise_or, np.bitwise_or, torch.bitwise_or),
    binary("bitwise_xor", jix.bitwise_xor, np.bitwise_xor, torch.bitwise_xor),
    binary("bitwise_left_shift", jix.bitwise_left_shift, np.left_shift, torch.bitwise_left_shift),
    binary("bitwise_right_shift", jix.bitwise_right_shift, np.right_shift, torch.bitwise_right_shift),
    binary(
        "clamp(a, min=b)",
        lambda a, b: jix.clamp(a, min=b),
        lambda a, b: np.clip(a, b, None),
        scalars_as_tensors(lambda a, b: torch.clamp(a, min=b)),
    ),
    binary(
        "clamp(a, max=b)",
        lambda a, b: jix.clamp(a, max=b),
        lambda a, b: np.clip(a, None, b),
        scalars_as_tensors(lambda a, b: torch.clamp(a, max=b)),
    ),
    binary(
        "where(True, a, b)",
        lambda a, b: jix.where(np.array(True), a, b),
        lambda a, b: np.where(np.array(True), a, b),
        lambda a, b: torch.where(torch.tensor(True), a, b),
    ),
] + [
    # The Python operators, which may pick another function than the ones above (or none)
    binary(f"a {symbol} b", fn, fn, fn)
    for symbol, fn in [
        ("+", operator.add),
        ("-", operator.sub),
        ("*", operator.mul),
        ("/", operator.truediv),
        ("//", operator.floordiv),
        ("%", operator.mod),
        ("**", operator.pow),
        ("&", operator.and_),
        ("|", operator.or_),
        ("^", operator.xor),
        ("<<", operator.lshift),
        (">>", operator.rshift),
        ("==", operator.eq),
        ("<", operator.lt),
    ]
]


def reduction(name, jix_fn, np_fn, torch_fn) -> Op:
    return Op(name, jix_fn, np_fn, torch_fn, reduction=True)


def torch_var(fn: Callable) -> Callable:
    """torch names numpy's `ddof` `correction`."""
    return lambda a, ddof=0.0, **kw: torch_reduction(fn)(a, correction=ddof, **kw)


REDUCTION_OPS = [
    reduction("sum", jix.sum, np_reduction(np.sum), torch_reduction(torch.sum)),
    reduction("product", jix.product, np_reduction(np.prod), torch.prod),  # torch.prod takes one dim only
    reduction("mean", jix.mean, np_reduction(np.mean), torch_reduction(torch.mean)),
    reduction("var", jix.var, np_reduction(np.var), torch_var(torch.var)),
    reduction("std", jix.std, np_reduction(np.std), torch_var(torch.std)),
    reduction("max", jix.max, np_reduction(np.max), torch_reduction(torch.amax)),
    reduction("min", jix.min, np_reduction(np.min), torch_reduction(torch.amin)),
    reduction("argmax", jix.argmax, np_reduction(np.argmax), torch.argmax),
    reduction("argmin", jix.argmin, np_reduction(np.argmin), torch.argmin),
    reduction("all", jix.all, np_reduction(np.all), torch_reduction(torch.all)),
    reduction("any", jix.any, np_reduction(np.any), torch_reduction(torch.any)),
]

# Ops only called by the cases check. Arguments that numpy or torch name differently are renamed.
OTHER_OPS = [
    Op("reshape", jix.reshape, np.reshape, torch.reshape),
    Op("broadcast", jix.broadcast, np.broadcast_to, lambda a, shape: torch.broadcast_to(a, shape)),
    Op("flatten", jix.flatten, np.ravel, torch.flatten),
    Op("flip", jix.flip, lambda a, dim=None: np.flip(a, axis=dim), lambda a, dim=None: torch.flip(a, dims=dim)),
    Op(
        "roll",
        jix.roll,
        lambda a, shift, dim=None: np.roll(a, shift, axis=dim),
        lambda a, shift, dim=None: torch.roll(a, shift, dims=dim),
    ),
    Op(
        "repeat",
        jix.repeat,
        lambda a, repeats, dim=None: np.repeat(a, repeats, axis=dim),
        torch.repeat_interleave,  # torch.repeat is numpy's tile
    ),
    Op(
        "tile",
        jix.tile,
        lambda a, repeats: np.tile(a, repeats),  # no dim argument
        lambda a, repeats: torch.tile(a, (repeats,)),
    ),
    Op("squeeze", jix.squeeze, lambda a, dim=None: np.squeeze(a, axis=dim), torch.squeeze),
    Op("unsqueeze", jix.unsqueeze, lambda a, dim: np.expand_dims(a, axis=dim), torch.unsqueeze),
    Op("permute_dims", jix.permute_dims, lambda a, dims=None: np.permute_dims(a, axes=dims), torch.permute),
    Op("swapdims", jix.swapdims, lambda a, dim0, dim1: np.swapaxes(a, dim0, dim1), torch.swapdims),
    Op("concatenate", jix.concatenate, lambda arrays, dim=0: np.concatenate(arrays, axis=dim), torch.concatenate),
    Op("stack", jix.stack, lambda arrays, dim=0: np.stack(arrays, axis=dim), torch.stack),
    Op("where", jix.where, np.where, torch.where),
    Op("slice", jix.slice, operator.getitem, operator.getitem),
    Op("a[index]", operator.getitem, operator.getitem, operator.getitem),
]

ALL_OPS = UNARY_OPS + BINARY_OPS + REDUCTION_OPS + OTHER_OPS
OPS_BY_NAME = {op.name: op for op in ALL_OPS}


def run_all(op: Op, args: list, kwargs: dict | None = None) -> dict[str, Outcome | None]:
    """Run `op` in each library. None for a library without an equivalent."""
    fns = {lib.name: getattr(op, lib.name) for lib in LIBS}
    return {lib.name: run(lib, fns[lib.name], args, kwargs or {}) if fns[lib.name] else None for lib in LIBS}


# ----------------------------------------------------------------------------------------------------
# Comparing outcomes
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Diff:
    """jix disagrees with numpy or torch: what each library gave for the same inputs."""

    check: str
    op: str
    inputs: str
    jix: str
    numpy: str
    torch: str
    numpy_differs: bool  # False: jix agrees with numpy, and only torch differs


def same_values(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise: are `a` and `b` equal? Floats up to a few ulps, NaN equals NaN."""
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        return a.astype(object) == b.astype(object)  # exact, even beyond 2**53
    eps = max(np.finfo(dt).eps if dt.kind in "fc" else 0 for dt in (a.dtype, b.dtype))
    a, b = a.astype(np.complex128), b.astype(np.complex128)
    return np.isclose(a, b, rtol=4 * eps, atol=0, equal_nan=True) | (a == b)


def same_result(a: Outcome, b: Outcome) -> bool:
    """Both raised (any errors), or both returned the same dtype, shape and values."""
    if a.error or b.error:
        return bool(a.error) == bool(b.error)
    if a.value.dtype != b.value.dtype or a.value.shape != b.value.shape:
        return False
    return bool(same_values(a.value, b.value).all())


def show_result(o: Outcome) -> str:
    if o.error:
        return o.error
    text = np.array2string(o.value, threshold=8, separator=", ").replace("\n", "")
    return f"{o.value.dtype}{list(o.value.shape)} {text}"


def diff_if_any(check: str, op: Op, inputs: str, results: dict, same: Callable, show: Callable) -> list[Diff]:
    """A Diff if jix's result is not `same` as numpy's or torch's, else nothing.

    `results` maps each library to its result, None where it has none to compare.
    """
    differs = {name: not same(results["jix"], r) for name, r in results.items() if name != "jix" and r is not None}
    if not any(differs.values()):
        return []
    shown = {name: "-" if r is None else show(r) for name, r in results.items()}
    return [Diff(check, op.name, inputs, shown["jix"], shown["numpy"], shown["torch"], differs.get("numpy", False))]


# ----------------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------------

# Python scalars mixed with arrays in the dtypes check, by their label in the report
SCALARS = {"True": True, "1": 1, "-1": -1, "1000": 1000, "2**64": 2**64, "1.5": 1.5, "1e300": 1e300, "1j": 1j}


def sample_operand(kind: str) -> Any:
    """`kind` is a dtype (a 1-d array), `0-d <dtype>` (a 0-d array), or a key of SCALARS."""
    if kind in SCALARS:
        return SCALARS[kind]
    if kind.startswith("0-d "):
        return np.array(1).astype(kind.removeprefix("0-d "))
    return np.array([1, 2, 3]).astype(kind)


def check_dtypes(op: Op) -> list[Diff]:
    """Result dtype, or error, for every combination of operand dtypes and Python scalars."""
    kinds = DTYPES + [f"0-d {dt}" for dt in DTYPES] + list(SCALARS)

    def same_dtype(a: Outcome, b: Outcome) -> bool:
        if a.error or b.error:
            return bool(a.error) == bool(b.error)
        return a.value.dtype == b.value.dtype

    def show_dtype(o: Outcome) -> str:
        return o.error or str(o.value.dtype)

    diffs = []
    for combo in itertools.product(kinds, repeat=op.arity):
        if all(kind in SCALARS for kind in combo):
            continue  # not an array op
        outcomes = run_all(op, [sample_operand(kind) for kind in combo])
        diffs += diff_if_any("dtypes", op, ", ".join(combo), outcomes, same_dtype, show_dtype)
    return diffs


def special_values(dtype: str) -> np.ndarray:
    """Edge-case values of `dtype`."""
    dt = np.dtype(dtype)
    if dt.kind == "b":
        values = [False, True]
    elif dt.kind == "u":
        values = [0, 1, 3, np.iinfo(dt).max]
    elif dt.kind == "i":
        values = [0, 1, -1, 3, np.iinfo(dt).min, np.iinfo(dt).max]
    elif dt.kind == "f":
        fi = np.finfo(dt)
        values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.0, np.inf, -np.inf, np.nan, fi.max, fi.smallest_subnormal]
    else:
        values = [0, 1, -1j, 1 + 1j, -2.5 + 0.5j, complex(np.inf, 0), complex(0, np.inf), complex(np.nan, 0)]
    return np.array(values, dtype=dt)


def check_values(op: Op) -> list[Diff]:
    """Every combination of the special values of each dtype: one Diff per differing element.

    Operand i holds the values along axis i, so broadcasting computes all combinations in one call.
    A reduction reduces pairs of values, stacked along its last dim.
    """
    diffs = []
    for dtype in DTYPES:
        values = special_values(dtype)
        if op.reduction:
            pairs = np.stack(np.broadcast_arrays(values[:, None], values[None, :]), axis=-1)
            args, kwargs, shape = [pairs], {"dim": -1}, pairs.shape[:-1]
        else:
            n = op.arity
            args, kwargs = [values.reshape([-1 if axis == i else 1 for axis in range(n)]) for i in range(n)], {}
            shape = (len(values),) * n
        outcomes = run_all(op, args, kwargs)

        # Each library's result, element by element. If the call raised (e.g. torch on an integer
        # division by zero), find out which elements make it raise.
        results = {}
        for lib in LIBS:
            o = outcomes[lib.name]
            if o is None or (not o.error and o.value.shape != shape):
                results[lib.name] = None
            elif o.error:
                results[lib.name] = run_per_element(lib, getattr(op, lib.name), args, kwargs, shape)
            else:
                results[lib.name] = o.value
        if results["jix"] is None:
            continue  # jix supports no such dtype, check_dtypes reports it

        for index in np.ndindex(shape):
            elems = {name: None if r is None else r[index] for name, r in results.items()}
            for name, e in elems.items():
                if hasattr(e, "dtype") and hasattr(elems["jix"], "dtype") and e.dtype != elems["jix"].dtype:
                    elems[name] = None  # another dtype gives other values for that reason alone (see check_dtypes)
            inputs = f"{dtype}: " + ", ".join(repr(values[i].item()) for i in index)
            diffs += diff_if_any("values", op, inputs, elems, same_element, show_element)

        jx, np_out = outcomes["jix"], outcomes["numpy"]
        if np_out and not jx.error and not np_out.error and jx.fp_errors != np_out.fp_errors:
            diffs.append(
                Diff("fp-errors", op.name, dtype, jx.fp_errors or "none", np_out.fp_errors or "none", "-", True)
            )
    return diffs


def run_per_element(lib: Lib, fn: Callable, args: list, kwargs: dict, shape: tuple) -> np.ndarray | None:
    """Call `fn` on the inputs of each element of the result alone, to find the ones that raise.

    An array of numpy scalars, holding the error type where the call raised. None if every call
    raised, as for a dtype the library does not support.
    """
    res = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        # each operand at `index`, keeping its broadcast (length 1) and trailing dims
        element_args = [
            a[tuple(slice(i, i + 1) if n > 1 else slice(None) for i, n in zip(index, a.shape))] for a in args
        ]
        o = run(lib, fn, element_args, kwargs)
        res[index] = o.error or o.value.reshape(-1)[0]
    return None if all(isinstance(r, str) for r in res.flat) else res


def same_element(a: Any, b: Any) -> bool:
    """Two elements of results, each a numpy scalar or an error type."""
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str)
    return bool(same_values(a, b))


def show_element(e: Any) -> str:
    return e if isinstance(e, str) else repr(e.item())


BROADCAST_SHAPES = [
    ((3,), (1,)),
    ((2, 3), (3,)),
    ((2, 1), (1, 3)),
    ((), (2, 3)),
    ((0,), (1,)),
    ((0,), (3,)),
    ((2, 3), (2,)),
    ((1,) * 9, (1,)),
]


def check_broadcast(op: Op) -> list[Diff]:
    """Result shape, or error, for pairs of operand shapes."""
    # the first of these dtypes that jix supports for the op
    dtype = next(
        dt for dt in ["float64", "int64", "bool"] if not run(LIBS[0], op.jix, [sample_operand(dt)] * 2, {}).error
    )

    def same_shape(a: Outcome, b: Outcome) -> bool:
        if a.error or b.error:
            return bool(a.error) == bool(b.error)
        return a.value.shape == b.value.shape

    def show_shape(o: Outcome) -> str:
        return o.error or str(o.value.shape)

    diffs = []
    for shapes in BROADCAST_SHAPES:
        outcomes = run_all(op, [np.ones(shape, dtype=dtype) for shape in shapes])
        inputs = f"{dtype}: " + ", ".join(map(str, shapes))
        diffs += diff_if_any("broadcast", op, inputs, outcomes, same_shape, show_shape)
    return diffs


A = np.arange(6, dtype=np.float64).reshape(2, 3)  # a 2-d array for the cases below
V = np.arange(4, dtype=np.int32)  # a 1-d array
NAN = np.array([1.0, np.nan, 3.0])
EMPTY = np.zeros(0)

# op name -> calls (args, kwargs) with jix's argument names
CASES = {
    "sum": [
        ([A], {}),
        ([A], {"dim": 0}),
        ([A], {"dim": -1, "keepdim": True}),
        ([A], {"dim": (0, 1)}),
        ([EMPTY], {}),
        ([np.zeros((0, 3))], {"dim": 0}),
        ([np.full(10**6, 0.1, dtype=np.float32)], {}),  # accumulation precision
        ([np.full(1000, 100, dtype=np.int8)], {}),  # accumulation dtype
    ],
    "product": [([A], {}), ([A], {"dim": 1}), ([EMPTY], {}), ([np.full(100, 2, dtype=np.int8)], {})],
    "mean": [([A], {"dim": 0}), ([EMPTY], {}), ([np.full(1000, 60000, dtype=np.float16)], {})],
    "var": [([A], {"dim": 1}), ([A], {"ddof": 1}), ([np.ones(1)], {"ddof": 1}), ([EMPTY], {})],
    "std": [([A], {"dim": 1}), ([A], {"ddof": 1}), ([np.ones(1)], {"ddof": 1})],
    "max": [([A], {"dim": 0}), ([EMPTY], {}), ([NAN], {}), ([np.zeros((2, 0))], {"dim": 0})],
    "min": [([A], {"dim": (0, 1)}), ([EMPTY], {}), ([NAN], {})],
    "argmax": [([A], {}), ([A], {"dim": 1}), ([A], {"dim": 1, "keepdim": True}), ([EMPTY], {}), ([NAN], {})],
    "argmin": [([np.array([3, 1, 1])], {}), ([NAN], {})],
    "all": [([A], {}), ([EMPTY], {}), ([V], {})],  # V is not bool
    "any": [([A], {"dim": 0}), ([EMPTY], {})],
    "reshape": [([A], {"shape": (3, 2)}), ([A], {"shape": -1}), ([A], {"shape": (-1, 2)}), ([A], {"shape": (4, -1)})],
    "broadcast": [([A[:1]], {"shape": (4, 3)}), ([V], {"shape": (2, 4)}), ([A], {"shape": (2, -1)})],
    "flatten": [([A], {}), ([np.array(5.0)], {})],
    "flip": [([V], {}), ([A], {}), ([A], {"dim": 1}), ([A], {"dim": (0, 1)})],
    "roll": [([V], {"shift": 1}), ([V], {"shift": -5}), ([A], {"shift": 1}), ([A], {"shift": 1, "dim": 1})],
    "repeat": [([V], {"repeats": 2}), ([V], {"repeats": 0}), ([A], {"repeats": 2}), ([A], {"repeats": 2, "dim": 1})],
    "tile": [([V], {"repeats": 2}), ([A], {"repeats": 2})],
    "squeeze": [([A[None, :, :1]], {}), ([A[None]], {"dim": 0}), ([A], {"dim": 0})],
    "unsqueeze": [([A], {"dim": 0}), ([A], {"dim": -1}), ([A], {"dim": (0, 2)})],
    "permute_dims": [([A], {}), ([np.zeros((2, 3, 4))], {"dims": (2, 0, 1)}), ([A], {"dims": (0, 0)})],
    "swapdims": [([A], {"dim0": 0, "dim1": -1})],
    "concatenate": [
        ([[A, A]], {}),
        ([[A, A]], {"dim": -1}),
        ([[A, V]], {}),  # different ndim
        ([[V, V.astype(np.uint8)]], {}),  # different dtypes
        ([[V, [1.5]]], {}),  # a Python list
    ],
    "stack": [([[V, V]], {}), ([[V, V]], {"dim": 1}), ([[V, V.astype(np.float16)]], {}), ([[V, V[:2]]], {})],
    "where": [
        ([V > 1, V, -V], {}),
        ([V, V, -V], {}),  # non-bool condition
        ([V > 1, 1, 2.5], {}),  # only Python scalars
    ],
    "slice": [
        ([A, 0], {}),
        ([A, (slice(None), slice(1, 3))], {}),
        ([A, slice(0, 10)], {}),  # out of bounds
        ([A, slice(None, None, 2)], {}),  # step
        ([A, (Ellipsis, -1)], {}),
    ],
    "a[index]": [
        ([A, (0, 1)], {}),
        ([A, -1], {}),
        ([A, (slice(None), slice(1, 10))], {}),  # out of bounds
        ([A, slice(None, None, -1)], {}),  # negative step
        ([A, 5], {}),  # out of bounds
        ([A, None], {}),  # new axis
        ([A, [0, 1]], {}),  # integer array
        ([A, A > 2], {}),  # boolean mask
        ([A, ()], {}),
    ],
}


def check_cases(op: Op) -> list[Diff]:
    """The hand-written calls in CASES."""
    diffs = []
    for args, kwargs in CASES.get(op.name, []):
        inputs = ", ".join([show_arg(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
        diffs += diff_if_any("cases", op, inputs, run_all(op, args, kwargs), same_result, show_result)
    return diffs


def show_arg(arg: Any) -> str:
    """An argument of a case, arrays by their dtype and shape."""
    if isinstance(arg, np.ndarray):
        return f"{arg.dtype}{list(arg.shape)}"
    if isinstance(arg, list):
        return "[" + ", ".join(show_arg(a) for a in arg) + "]"
    return repr(arg)


def check_missing() -> list[Diff]:
    """numpy ufuncs that jix has neither under the same name nor as an op in the tables above."""
    covered = {getattr(op.numpy, "__name__", "") for op in ALL_OPS}
    ufuncs = sorted(name for name, fn in vars(np).items() if isinstance(fn, np.ufunc) and fn.__name__ == name)
    return [
        Diff("missing", name, "", "-", "yes", "yes" if hasattr(torch, name) else "-", True)
        for name in ufuncs
        if name not in covered and not hasattr(jix, name)
    ]


# ----------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------

CHECK_TITLES = {
    "dtypes": "Result dtype, or error, for operands of each dtype (1-d and 0-d arrays) and Python scalars",
    "values": "Results for special values (operands are `dtype: values`)",
    "fp-errors": "Floating point errors reported for the special values (torch reports none)",
    "broadcast": "Result shape, or error, for operands of these shapes",
    "cases": "Hand-written calls",
    "missing": "numpy ufuncs that jix lacks",
}


def merge_rows(diffs: list[Diff]) -> list[list[str]]:
    """Table rows: diffs that differ only in the op share a row, then rows that differ only in the inputs."""
    ops_by_row: dict[tuple, list[str]] = {}
    for d in diffs:
        ops_by_row.setdefault((d.inputs, d.jix, d.numpy, d.torch), []).append(d.op)
    inputs_by_row: dict[tuple, list[str]] = {}
    for (inputs, *results), ops in ops_by_row.items():
        inputs_by_row.setdefault((", ".join(ops), *results), []).append(inputs)
    return [[ops, "; ".join(inputs), *results] for (ops, *results), inputs in inputs_by_row.items()]


def table(rows: list[list[str]]) -> list[str]:
    lines = ["| ops | inputs | jix | numpy | torch |", "|---|---|---|---|---|"]
    return lines + ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]


def report(diffs: list[Diff]) -> str:
    """Markdown: a section per check, with a table where jix differs from numpy and one where only torch does."""
    lines = [
        "# jix vs numpy vs torch",
        "",
        f"Generated by `scripts/np_diff/np_diff.py` with jix {jix.__version__} ({jix._build_profile} build), numpy"
        + f" {np.__version__} and torch {torch.__version__}.",
        "",
        "| check | jix differs from numpy | jix agrees with numpy, differs from torch |",
        "|---|---|---|",
    ]
    for check in CHECK_TITLES:
        ds = [d for d in diffs if d.check == check]
        lines.append(f"| {check} | {sum(d.numpy_differs for d in ds)} | {sum(not d.numpy_differs for d in ds)} |")
    for check, title in CHECK_TITLES.items():
        lines += ["", f"## {check}", "", title]
        for numpy_differs, heading in [
            (True, "jix differs from numpy"),
            (False, "jix agrees with numpy, differs from torch"),
        ]:
            ds = [d for d in diffs if d.check == check and d.numpy_differs == numpy_differs]
            if ds:
                lines += ["", f"### {heading}", "", *table(merge_rows(ds))]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="write the report to this file instead of stdout")
    parser.add_argument("--op", action="append", choices=OPS_BY_NAME, help="only check this op (repeatable)")
    args = parser.parse_args()

    ops = [OPS_BY_NAME[name] for name in args.op] if args.op else ALL_OPS
    diffs = []
    for op in ops:
        if op in UNARY_OPS + BINARY_OPS + REDUCTION_OPS:
            diffs += check_dtypes(op)
            diffs += check_values(op)
        if op in BINARY_OPS:
            diffs += check_broadcast(op)
        diffs += check_cases(op)
    if not args.op:
        diffs += check_missing()

    text = report(diffs)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
