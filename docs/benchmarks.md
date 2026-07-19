# Benchmarks: CI matrix and cross-run comparison

## Running the matrix

Trigger the `Benchmarks` workflow manually (Actions -> Benchmarks -> Run workflow):

- Check the runners to use (ubuntu x86_64, ubuntu aarch64, macos arm64) - all off by default.
- `suites`: run both suites, or just `rust` / `python`.
- `fast`: a quick, low-fidelity run (reduced Criterion sampling and pytest-benchmark timing) for
  short dev cycles - not for authoritative numbers.
- `rust_harness_args` / `python_harness_args`: forwarded verbatim (space-split) to
  `jix/benches/run.py` and `jix-py/python/benches/run_all.py` (for example `--filter reduction`
  or `-k op2`). Do not pass `--out`; the workflow manages it.
- `compare_to`: optional base ref. When set, each runner benches the base ref and the current
  sha back-to-back (same-machine A/B) - the most reliable PR-vs-main signal.

Each runner uploads a `bench-<os>-<arch>-<run_id>` artifact: a top-level `meta.json` plus one
`<sha>/` subdir per benched ref (just the current sha normally; the base sha too under A/B),
each holding the raw Criterion tree, the Python `python.json`, and the PNG/markdown reports.

## Local runs

The same runners work locally for short dev cycles. Everything after `--` is forwarded verbatim
to the harness (`cargo bench --` for the Rust runner, pytest for the Python one):

    python jix/benches/run.py --fast -- reduction
    python jix-py/python/benches/run_all.py --fast -- -k op2

`--fast` trades fidelity for speed; drop it for real measurements.

## Comparing runs

    python scripts/bench/compare.py list -n 20
    python scripts/bench/compare.py compare <A> <B>
    python scripts/bench/compare.py compare --ab <RUN>

`A`/`B`/`RUN` may be a workflow run-id, a branch, a sha, or a local artifact dir. Branch/sha
resolves to the newest successful benchmark run via the `gh` CLI. Output (default
`bench-compare-...`, override with `--out`) holds `report.md`, `report.json`, each run's raw
PNGs under `base/` and `new/`, and generated comparison plots under `plots/` (linked from
`report.md`):

- `change_<platform>.png` - per-bench %-change bars, sorted worst-first, colored by verdict,
  change-CI as error bars (the regression view).
- `heatmap.png` - benches x platforms, cell = %change on a red/green scale (shows whether an
  optimization generalizes across platforms).
- `scatter_<platform>.png` - log-log base-vs-new means, faceted by suite; diagonal = no change.
- `violin_<platform>_<bench>.png` - base-vs-new raw-sample distributions, emitted only for
  benches matching `--bench <substring>`.

`--format human|json|markdown` selects what prints to stdout; `--only jix` restricts the Python
rows to jix; `--fast` lowers the bootstrap resample count for a quick comparison; `--no-plots`
skips plot generation.

Statistics mirror Criterion exactly: a Welch t-statistic, a pooled ("mixed") bootstrap p-value
at nresamples=100000, a change confidence interval at confidence 0.95, and a verdict from
significance 0.05 and noise threshold 0.01 (Improved / Regressed / WithinNoise / NoChange).

## Caveats

- GitHub-hosted runners share vCPUs and are noisy; sub-5-to-10% changes can sit below the
  noise floor. Prefer `compare_to` (same-machine A/B) for PR-vs-main.
- GitHub rotates runner hardware; two same-label runs can land on different CPUs. `meta.json`
  records the CPU model and `compare` warns when base/new CPUs differ.
- macOS/Windows runners carry minute multipliers; keep the checkboxes off unless needed.
