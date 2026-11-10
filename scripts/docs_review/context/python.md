# Context: Python bindings docs

For units that touch `jix-py/`. Read `general.md` first.

## Format

Docstrings are the `///` comments in `jix-py/src/`, rendered by mkdocstrings with the Google style
(`jix-py/mkdocs.yml`). `mkdocs build --strict` fails on a broken cross-reference.

- Sections are `Args:`, `Returns:`, `Raises:`, `Examples:`, with entries indented by 4 spaces and
  continuation lines by 8.
- Cross-references use the mkdocs-autorefs form: `` [`jix.Array`][jix.Array] ``,
  `` [`jix.add()`][jix.add] ``. Rustdoc links (`` [`Array`](crate::Array) ``) do not work in a
  docstring, and the reverse.
- Examples are fenced `python` blocks. An example is self-contained: it imports `jix` and `numpy`
  and checks its claims with `assert`.
- Argument names in `Args:` must match the Python signature: the Rust parameter names, unless a
  `#[pyo3(signature = (...))]` attribute says otherwise. That attribute also holds the defaults
  and the keyword-only marker `*`.
- Dtypes are written the Python way in prose about Python values (`bool`, `int8`, `float32`,
  `complex64`), but the "Supported dtypes" lists and tables use the short Rust names (`i8`, `f32`,
  `Complex<f32>`). Leave whichever a doc uses, as long as it is consistent within the doc.

## How a Python op relates to the Rust op

A Python function is a dispatch table over the runtime dtypes of its operands
(`jix-py/src/ops/common/dispatch.rs`). Each entry is a typed Rust implementation plus a `CastKind`
(`None`, `Safe`, `Unsafe` in `dtype_promote.rs`) that says which input dtypes may be cast to reach
it. The first matching entry wins.

So the Python docs can legitimately differ from the Rust docs of the same op:

- **More dtypes.** Python may accept a dtype the Rust trait does not implement, by casting first
  (`jix.negative` takes unsigned ints and negates them as the signed type of the same width) or by
  a shortcut (`jix.absolute` returns unsigned and bool inputs as is).
- **Type promotion.** Operands of different dtypes are promoted; Rust requires equal types.
- **Broadcasting.** Python operands broadcast; a Python scalar is an operand.
- **Errors.** Rust returns `Error` with an `ErrorKind`; Python raises an exception.

The supported dtypes, the promotion and the output dtype of a Python function are read from its
dispatch table in the code, not from the Rust trait docs.

## Aliases and methods

- `jix-py/python/jix/__init__.py` defines the aliases `pow`, `abs`, `concat`, `clip`, `prod`,
  `sub`, `mul`, `div`, `neg`, `log2`, `log10`.
- Most `Array` methods and operators in `jix-py/src/array.rs` carry a one-line doc pointing to the
  module function. That is intended; do not expand them.
- The `.pyi` stub is generated (`maturin generate-stubs`) and gitignored. Never edit it.
