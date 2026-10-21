# jix vs numpy vs torch

`np_diff.py` calls every jix op and its numpy and torch counterparts on the same inputs, and
reports where jix disagrees with either of them. `report.md` is its output. Regenerate it after
changing jix; it is not kept up to date automatically.

## Running

The script needs jix (built from this branch), numpy and torch in the active venv:

```bash
# A debug build. With full debug info it ran out of memory on a 15 GB machine, hence the env var.
(cd jix-py && CARGO_PROFILE_DEV_DEBUG=line-tables-only maturin develop)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # or just `torch`, with CUDA

python scripts/np_diff/np_diff.py --out scripts/np_diff/report.md   # ~40s on a debug build
python scripts/np_diff/np_diff.py --op add --op sum                 # only these ops, to stdout
```

Run it as a script, not from the repo root with `python -c`/`-m`: there the `jix/` crate directory
shadows the installed package.

## Reading the report

The report has a section per check:

| check | what is compared |
|---|---|
| dtypes | result dtype, or whether it raises, for every combination of operand kinds: a 1-d array or 0-d array of each dtype, or a Python scalar (`True`, `1`, `-1`, `1000`, `2**64`, `1.5`, `1e300`, `1j`) |
| values | result values for every combination of special values (0, -1, min, max, nan, inf, subnormal, ...) of each dtype, all operands of the same dtype; reductions reduce pairs of them |
| fp-errors | the floating point errors (`divide`, `over`, `under`, `invalid`) jix and numpy report for those values |
| broadcast | result shape, or whether it raises, for pairs of operand shapes |
| cases | hand-written calls: reductions with dims, shape ops, indexing, `where`, `concatenate`, ... |
| missing | numpy ufuncs with no jix counterpart |

Each section has two tables: where jix differs from numpy (torch shown for reference), and where
jix agrees with numpy and only torch differs. A cell is a dtype, a value, or the exception type the
call raised; `-` means the library has no such op (e.g. numpy has no truncating division).
Rows that differ only in the op are merged, and then rows that differ only in the inputs, so an
`inputs` cell lists every input for which the libraries gave those results (`;`-separated).

What counts as a difference:
- Two calls that both raise agree, whatever the exception types.
- Floats agree up to 4 ulps of the least precise dtype, NaN equals NaN, and `0.0` equals `-0.0`.
- The values check compares only results of the same dtype as jix's: a result of another dtype
  has other values for that reason alone, and the dtypes check already reports it.

## Extending

- A new op: add an `Op` to `UNARY_OPS`, `BINARY_OPS` or `REDUCTION_OPS` (all checks run on it) or
  to `OTHER_OPS` (only its `CASES` run). The numpy and torch callables take jix's arguments, so
  wrap them in a lambda where the names differ (`axis` vs `dim`).
- A new hand-written call: add `(args, kwargs)` to `CASES[op name]`.
- More edge values: `special_values()`; more operand kinds: `DTYPES` and `SCALARS`.

Two torch quirks are worked around in the op tables, and it is worth knowing about them when
reading the torch column:
- Many torch functions (`torch.maximum`, `torch.isclose`, `torch.logical_and`, `torch.eq` with the
  scalar first, ...) take no Python scalars; `scalars_as_tensors` passes them as 0-d tensors.
- `torch.div(a, b, rounding_mode="trunc")` kills the process (SIGFPE) on an int32/int64
  `INT_MIN / -1`; `torch_divide_trunc` raises a `SIGFPE` exception instead.

## Differences

Every difference in the report should be either known and accepted, with a line in
`known_differences.md`, or known and waiting to be fixed, with a line in the repo's `todo.md`. After
running the script, compare its report against both, and deal with whatever neither explains: fix it,
or (if it is accepted) add a line for it to `known_differences.md`.
