# Producing the benchmark report

Four steps: run the benchmarks on CI, download the results, build the report, commit it.

All commands run from the repo root. Needs the `gh` CLI, authenticated.

## 1. Run the benchmarks

    gh workflow run benchmarks.yaml \
      -f ubuntu_x86=true \
      -f ubuntu_arm64=true \
      -f suites=both

Takes 1-2 hours per runner; they run in parallel. Watch it with:

    gh run watch $(gh run list --workflow=benchmarks.yaml --limit 1 --json databaseId --jq '.[0].databaseId')

Add `-f fast=true` for a quick, low-fidelity run - useful to check the plumbing, not for a report.
Add `--ref <branch>` to bench a branch other than the default one.

## 2. Download the results

    RUN=$(gh run list --workflow=benchmarks.yaml --limit 1 --json databaseId --jq '.[0].databaseId')
    rm -rf scripts/bench/artifacts
    gh run download $RUN --dir scripts/bench/artifacts

That writes one directory per runner (`bench-linux-x86_64-<run>`, `bench-linux-aarch64-<run>`).
Clear the directory first: step 3 reads every run under it, so a leftover run would be mixed in.

## 3. Build the report

    python scripts/bench/build_report.py --artifacts scripts/bench/artifacts

This writes `report/plots/*.png` and `report/report.md` (under `scripts/bench/`), which the
top-level `README.md` links to.

`build_report.py` prints one line per artifact it read - check both platforms are listed before
trusting the output. Output is stamped with the machines it came from, and marked low-fidelity if
the run used `fast`.

## 4. Publish

Edit the prose in `report/report.template.md`, never in `report/report.md` - `report.md` is
generated and any edit to it is overwritten on the next build. Re-run step 3 after editing, then
commit everything under `report/`.

## Running locally

For a single-machine check. Results are noisy and only one platform wide, so they are for
development, not for publishing.

    cd jix-py && maturin develop --release     # the Python benches need the release build

To produce a full local report, run both suites the way CI does and point step 3 at the output,
writing it somewhere other than the published `report/`:

    python scripts/bench/run.py --suites both --fast
    python scripts/bench/build_report.py --artifacts scripts/bench/.results \
      --out /tmp/bench-report/plots --report /tmp/bench-report/report.md

Or run one suite on its own:

    python jix-py/python/benches/run_all.py --out /tmp/bench-python
    python jix/benches/run.py --report

Drop `--fast` for real timings. `--report` restricts the Rust run to `vs_ndarray`, the only target
the report uses; `scripts/bench/run.py` does that by default, and `--all-rust-benches` turns it off.

## Changing the suite

See `DEVELOPER.md` - how the pieces fit, how to add a benchmark or a plot, and the mistakes this
suite has already made.

## Where things live

| file | what it is |
|---|---|
| `run.py` | runs both suites the way CI does |
| `compare.py` | compares two runs - see `CI.md` |
| `CI.md` | the CI matrix, local runs, and comparing two runs for regressions |
| `build_report.py` | builds `report/` from downloaded results |
| `report/report.template.md` | the report's prose, with `<!-- plot:KEY -->` and `<!-- table:KEY -->` markers |
| `report/report.md` | generated - do not edit |
| `../../jix-py/python/benches/report_spec.py` | every plot's cases and libraries; the benches read it too |
| `../../jix-py/python/benches/report_bars.py` | the renderer |
| `DEVELOPER.md` | how the suite is built, and how to add a benchmark |
