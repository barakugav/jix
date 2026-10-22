# Bug-hunt primer

Every hunter and verifier reads this file first. It is the shared briefing: what we are looking
for, what we are not, the library contracts that matter most for correctness, how to test a
hypothesis cheaply, and the bar a finding must meet. Per-subsystem detail lives in
`bug-hunt/context/<subsystem>.md`; CLAUDE.md (architecture, layout, commands) is already in your
context and is not repeated here.

## 1. Mission

Find HARD bugs in jix - the kind that survive a test suite. In priority order:

1. **Memory safety / UB.** Out-of-bounds reads or writes, uninitialized reads, invalid values
   (a `bool` byte that is not 0/1, a misaligned typed read), aliasing violations (`&mut` or
   `PtrMutNoalias` over memory something else reads or writes), use-after-free / dangling views,
   data races, and safe APIs that can be driven into any of those (unsoundness).
2. **Silent wrong results.** Wrong values, wrong shape, wrong element order, a region written to
   the wrong offset, a tile skipped or written twice, state carried wrongly across blocks / tiles.
3. **Panics, aborts and hangs on valid input.** Including arithmetic overflow (a panic in debug,
   silent wraparound in release - which is usually case 1 or 2 in disguise), `unwrap` on
   reachable `None`/`Err`, infinite loops, quadratic blow-ups on ordinary inputs.
4. **Resource bugs.** Unbounded or size-overflowed allocation driven by plausible input, notably
   a malformed or adversarial `.jix` file.
5. **Questionable semantics.** jix deliberately deviates from NumPy in places, and those
   deviations are NOT bugs (see each context doc's "Intentional behaviors"). But if a behavior
   looks unreasonable on its own merits - inconsistent between two code paths that should agree
   (SIMD vs scalar tail, contiguous vs strided, Rust vs Python, `read_data` vs
   `read_as_elementwise_pipeline`), inconsistent between dtypes, lossy without warning, or
   surprising enough that a user would call it a bug - report it with category
   `questionable-semantics` and say why it is unreasonable, not just "NumPy does X".

Out of scope: style, naming, docs typos, missing features, performance (unless pathological),
"NumPy does it differently" with no further argument, and anything listed as intentional.

### Severity

| severity | meaning |
|----------|---------|
| critical | UB / memory corruption reachable from the safe public Rust API or from Python with ordinary input |
| high     | silent wrong results on ordinary input; UB reachable only with unusual-but-valid input (malformed file, odd strides, 32-bit target) |
| medium   | panic / crash on valid input; wrong results only in rare corner cases; resource exhaustion from untrusted input |
| low      | latent hazard (an unsafe precondition that holds today only by accident), questionable semantics |

## 2. Contracts that most bugs violate

Read the relevant context doc for the precise version; this is the short list.

- **`ArrayStorage::read_data(index, ctx, out)`** (`jix/src/storage/core_trait.rs`). `index` has
  one half-open range per dim, in bounds, possibly empty.
  - *pull* (`out=None`): returns any buffer valid for the requested shape at the strides it
    reports - possibly a borrowed view of the storage's own memory, possibly with 0-stride
    (broadcast) dims, possibly a pooled buffer.
  - *push* (`out=Some(dst)`): writes exactly the requested region into `dst` at `dst`'s own
    strides and returns a view of `dst`. `dst` can have any strides (permuted, padded, offset
    into a larger buffer); it must not be broadcast (aliasing strides), see
    `StridedBuf::from_raw_parts_mut`.
- **`StridedBuf`** (`jix/src/storage/buf.rs`): bytes plus one `usize` byte stride per dim. No
  shape, no dtype, no length of its own. Strides are unsigned - there are no negative strides
  (flip and friends must handle direction some other way). NOTHING is aligned: neither the base
  pointer nor the strides are promised to be aligned to the dtype, so typed access must be
  unaligned (or go through `read_maybe_aligned::<REQUIRE_ALIGNED>`, which is unaligned only on
  x86 / x86_64 / aarch64 - on armv7 and others it does an aligned read, so alignment must then be
  proven).
- **`PoolBuf` / `BufferPool`** (`jix/src/util/buf_pool.rs`): a pooled buffer's slice covers the
  whole allocation, which may be LARGER than requested and holds stale bytes from earlier use.
  Any code that uses the slice's length, or reads bytes it did not write, is suspect. The pool is
  single-threaded and uses `UnsafeCell`.
- **`PtrNoalias` / `PtrMutNoalias`** (`jix/src/util/mod.rs`): they carry LLVM `noalias`. Building
  them over overlapping ranges - e.g. an `out=` buffer that views the same memory as an input -
  is UB, not just a wrong result.
- **Element-wise pipeline** (`jix/src/storage/elementwise_pipeline.rs`): fused ops read `N`
  elements at an `offset` from every leaf operand; `CONTIGUOUS` promises
  `inner_stride == itemsize` for every operand; kernels never panic, they write a fallback value
  and OR an `ArithmeticOpStatus` flag, and the `ReadContext` error modes (error / warn / ignore)
  decide what happens next.
- **Typed views of bytes.** Interpreting bytes as `T` is only sound if every bit pattern is a
  valid `T` (false for `bool`; true for ints / floats; struct dtypes inherit from their fields)
  and the dtype check that gates it really proves the layouts match.
- **Integer widths.** Shapes and indices are `u64`; memory offsets are `usize`. x86 (32-bit) and
  armv7 wheels are shipped, so every `as usize` of a `u64`, and every unchecked product of
  shape * stride * itemsize, can truncate or wrap there - and a wrapped size feeding an
  allocation or a `from_raw_parts` is memory corruption.
- **ndim.** At most `NDIM_MAX = 8` array dims (stack `ArrayVec`s), at most 4 inner dtype dims.
  Ops that add dims (insert_dim, stack, reshape, broadcast) must reject overflow cleanly.

Facts that change how much to trust existing tests:

- CI runs the tests only on x86_64 Linux and never runs Miri. The x86, armv7, aarch64 and ppc64le
  wheels are built but never tested, so 32-bit `usize`, `REQUIRE_ALIGNED = true`, and the NEON /
  AVX-512 SIMD paths are unexercised. Reason about them from the code.
- Release builds have overflow checks off: every unchecked `+` / `*` wraps silently in shipped
  wheels.
- `debug_assert!` and `assert_unchecked` guard many invariants. In release, the first vanishes and
  the second becomes an optimizer assumption - a violated one is UB.
- Several mappers found no validation of `bool` byte values anywhere (archive load, numpy input).
  Treat "where can a non-0/1 byte reach a typed `bool` read" as an open question.

## 3. Checklist - apply to every unit

For every function you review, walk these. Most real bugs live in the cross product of an
edge case and a fast path.

- [ ] Every `unsafe` block: what exactly does it assume, and where is that established? Follow the
      callers (grep). "Established by a `debug_assert!`" or "true for all current callers" is a
      latent hazard at best.
- [ ] Zero-length dims, empty ranges, 0-d arrays (shape `[]`), length-1 dims. Look for
      `len - 1`, `/ len`, `% len`, `div_ceil(0)`, `product()` of an empty shape (= 1, not 0).
- [ ] Broadcast (0-stride) dims in inputs; non-packed, permuted, padded, offset strides in `out`.
- [ ] Partial last block / tile; a request that straddles a block, tile, period or concat
      boundary; a step that does not divide the length.
- [ ] Integer edges: overflow in index / size arithmetic; `u64 -> usize` on 32-bit; signed MIN
      and -1; shifts >= bit width; float -> int of NaN / inf / out-of-range.
- [ ] Dtype edges: bool, f16, complex (feature-gated - check both feature states compile the same
      logic), struct dtypes (padding, itemsize != alignment, odd itemsizes like 3 or 12), inner
      shapes.
- [ ] Two paths that must agree: SIMD body vs scalar tail, static `N_OPERANDS` path vs dyn path,
      contiguous vs strided, aligned vs unaligned, pull vs push, `read_data` vs
      `read_as_elementwise_pipeline`, Rust vs Python wrapper, encode vs decode.
- [ ] Error paths: is a partially written `out` or a half-updated state observable after an `Err`?
      Does the error-mode machinery (error / warn / ignore) behave the same in every kernel?
- [ ] Aliasing: can the caller's `out` overlap an input (Python `out=` over a numpy array that a
      jix Plain array also views)?

## 4. Testing a hypothesis

Reading is the main tool; experiments turn a suspicion into evidence. Rules first:

- **Never modify tracked files** (anything under `jix/src`, `jix-py/src`, `jix-macros`, tests,
  Cargo files). Many agents share this checkout and one build cache: editing `src` invalidates
  everyone's build and races with their reads. Only create NEW files at the paths below.
- Do not run the full test suite or `cargo build` of other crates/profiles. Cargo serializes
  concurrent builds on a lock - waiting a bit for "Blocking waiting for file lock" is normal.
- Keep experiments small and fast; you are here to read code, experiments only confirm.

### Rust (preferred for core bugs): an integration test against the public API

Create `jix/tests/<name>.rs` with exactly the file name your prompt gives you (hunters:
`hunt_<run>__<unit>_<n>.rs`, verifiers: `repro_<run>__<finding-id>.rs`; the run prefix keeps
concurrent runs apart and lets the report stage clean up) and run only that target:

```bash
cargo test --manifest-path jix/Cargo.toml --all-features --test <name> -- --nocapture
```

The library is already built, so this rebuilds in a few seconds. It is a **debug build**:
integer overflow panics, `debug_assert!`s are live, and `assert_unchecked_eq!` checks its
condition - so a panic here often marks UB or wraparound that a release build would hit
silently. If release-mode behavior matters, add `--release` (first build takes minutes).

```rust
use jix::Array;
use ndarray::{array, Ix1, Ix2};

#[test]
fn repro() {
    let a = Array::compact_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    // Methods take `self`; use `.view()` to keep `a` around.
    let s = a.view().exp().sum(0);
    let got = s.to_ndarray::<Ix1>().unwrap();
    // ... compare against a reference computed with plain ndarray / loops
    let p = a.permute_dims(&[1, 0]);
    assert_eq!(p.to_ndarray::<Ix2>().unwrap(), array![[1.0f32, 4.0], [2.0, 5.0], [3.0, 6.0]]);
}
```

Crate-internal (`pub(crate)`) items are not reachable from here. That is fine: a bug that no
public path reaches is a latent hazard (severity low); say so and describe what would reach it.
If you must exercise an internal item, do it in a scratch COPY of the crate, never in the shared
tree: `cp -r jix bug-hunt/runs/<run>/scratch/crate-<id>`, add an inline test there, and run it with
`CARGO_TARGET_DIR=bug-hunt/runs/<run>/scratch/target-<id> cargo test --manifest-path
bug-hunt/runs/<run>/scratch/crate-<id>/Cargo.toml --all-features --lib <test>` (a full build, a
few minutes). Save the test as `repros/<ID>_internal.rs`.
Existing inline unit tests can be run by name (`cargo test --manifest-path jix/Cargo.toml
--all-features <test_name>`; the test binary is prebuilt) to see how internals behave.

### Miri (memory-safety claims only, and sparingly)

Miri is the decisive tool for UB that does not crash, but it is VERY slow. Use it at most once
or twice per finding, on a single tiny integration test (a handful of elements, no proptest, no
loops over many shapes), and only when a debug-build run cannot show the problem. Always go
through the script, which sets the right flags:

```bash
python jix/run_miri.py --all-features --test <name>
```

Under Miri zstd is replaced by a memcpy (`cfg(miri)` in `jix/src/codec/mod.rs`), so compact
arrays work. The Miri build of the library is cached; the first run of a new test still takes a
while. To run one test of a file, append `-- <test_fn_name>`.

### 32-bit (i686) - for truncation / overflow / alignment claims that depend on usize = 32 bits

x86 and armv7 wheels ship but are never tested, so a 32-bit claim is worth running for real. The
same integration test builds and runs natively as a static i686-musl binary (`usize` is 32 bits,
`align_of::<u64>()` is 4):

```bash
bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features --test <name> -- --nocapture
```

It uses its own target dir (`target-i686/`), prebuilt by `setup.sh`; a new test rebuilds in tens of
seconds. Huge-size triggers are usually cheap here: a lazy `broadcast` of a 1-element array to a
shape of 2^32+ elements costs no memory until it is read.

### Python

The bindings are installed as a debug build in `.venv` (overflow checks on; a Rust panic surfaces
as `pyo3_runtime.PanicException`). Write the script to `bug-hunt/runs/<run>/scratch/<name>.py`
(verifiers: `bug-hunt/runs/<run>/repros/<finding-id>.py`), `mkdir -p` the directory first, and run it with
`.venv/bin/python <path>`. Always run a script file: `python -c ...` or `python -m ...` from the
repo root puts the root on `sys.path`, where the Rust crate directory `jix/` shadows the installed
package (you get an empty namespace package and `AttributeError`s). Do not run `maturin develop` -
the extension is shared by every agent.

## 5. What a finding must contain

A finding is only useful if someone can act on it without redoing your work:

- **Location**: the file:line where the defect is (not just where it manifests), plus related
  locations (the caller that breaks the precondition, the other path that disagrees).
- **Trigger**: a concrete input or state - shapes, dtypes, index ranges, strides, values, the
  op chain. "Some edge case" is not a trigger.
- **Expected vs actual**: what should happen, what happens instead (wrong value, OOB write of N
  bytes at offset X, panic message...).
- **Reachability**: public Rust API, Python API, or crate-internal only. Before claiming
  "crate-internal", check whether the type leaks through a public associated type or trait
  (e.g. `Dimension::Vec<T>` exposes the crate's `ArrayVec` / `DimArray` to user code with
  user-chosen `T`), a public generic bound, or a public unsafe-free constructor.
- **Evidence**: what you ran and what it printed, if you ran anything.
- **Confidence**: high (you demonstrated it or the code path is unambiguous), medium (the argument
  holds but you could not run it), low (a suspicion worth a second look).

Prefer five well-argued findings to twenty vague ones. Do not pad. If your unit looks clean,
say so in `coverage` - "no findings" from a careful read is a valid and useful result.

### Verdicts (verifiers)

- `confirmed` - a repro demonstrates it (wrong output, panic, Miri error), or the defect is
  unambiguous from the code and trivially reachable.
- `plausible` - the argument survives refutation attempts and the path is reachable, but no repro
  (explain what blocks one: needs a 32-bit target, a 4 GiB array, a race...).
- `refuted` - name the specific reason (precondition established at file:line, unreachable,
  misread code, result is actually correct).
- `intended` - deliberate and documented behavior (cite where), and not unreasonable.

### Repro convention

A repro asserts the CORRECT behavior, so it fails while the bug exists and passes once it is
fixed: an assertion failure, a panic, a Miri error or a crash all count as "fails". Never write a
test that passes because the bug happens (e.g. `assert!(catch_unwind(..).is_err())` for a panic
that should be an `Err`); assert the `Err` instead. Then the repros double as regression tests.
For a repro that crashes the process (segfault, abort), say so at the top of the file; in Python,
run the crashing step in a subprocess and report its exit status. Re-run any hunter claim your
verdict relies on (e.g. "release build segfaults") instead of quoting it.

### Finding write-up (verifiers write one per confirmed / plausible finding)

`bug-hunt/runs/<run>/findings/<ID>.md`:

```markdown
# <ID>: <one-line title>

- Severity: <critical|high|medium|low>  Verdict: <confirmed|plausible>  Category: <category>
- Location: <file:line> (related: <file:line>, ...)
- Reachability: <public Rust API | Python API | crate-internal>
- Reported by: <unit ids>

## What goes wrong
<2-6 sentences: the defect and its consequence>

## Trigger
<concrete input / op chain>

## Repro
<path to repro file, the command, and the relevant output>

## Root cause
<the faulty reasoning in the code, with file:line>

## Suggested fix
<short; a sketch, not a patch>
```

## 6. Known issues

Before reporting, check `bug-hunt/KNOWN.md` - issues already reported or accepted. Do not
re-report them unless you found a new trigger that the known entry does not cover.
