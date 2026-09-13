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

## 2. Download the results

    RUN=$(gh run list --workflow=benchmarks.yaml --limit 1 --json databaseId --jq '.[0].databaseId')
    gh run download $RUN --dir bench-artifacts

That writes one directory per runner (`bench-linux-x86_64-<run>`, `bench-linux-aarch64-<run>`).

## 3. Build the report

    python bench-report/build_report.py --artifacts bench-artifacts

Writes `bench-report/plots/*.png` and `bench-report/report.md`. It prints one line per artifact it
read - check both platforms are listed before trusting the output. The report is stamped with the
machines it came from; a draft built from `build_fake_report.py` is stamped as invented instead.

## 4. Publish

Edit the prose in `report.template.md`, never in `report.md` - `report.md` is generated and any
edit to it is overwritten on the next build. Re-run step 3 after editing, then commit
`report.template.md`, `report.md` and `plots/`.

## Running locally

For a single-machine check. Results are noisy and only one platform wide, so they are for
development, not for publishing.

    cd jix-py && maturin develop --release     # the Python benches need the release build

To produce a full local report, run both suites the way CI does and point step 3 at the output:

    python scripts/bench/run.py --suites both --fast
    python bench-report/build_report.py --artifacts scripts/bench/.results

Or run one suite on its own:

    python jix-py/python/benches/run_all.py --out /tmp/bench-python
    python jix/benches/run.py --report

Drop `--fast` for real timings. `--report` restricts the Rust run to `vs_ndarray`, the only target
the report uses; `scripts/bench/run.py` does that by default, and `--all-rust-benches` turns it off.

## Iterating on the plot style

`build_fake_report.py` makes up plausible numbers and drives the same renderer, so the layout and
colors can be changed without waiting for a real run:

    python bench-report/build_fake_report.py

## Where things live

| file | what it is |
|---|---|
| `report.template.md` | the report's prose, with `<!-- plot:KEY -->` and `<!-- table:KEY -->` markers |
| `report.md` | generated - do not edit |
| `../jix-py/python/benches/report_spec.py` | every plot's cases and libraries; the benches read it too |
| `../jix-py/python/benches/report_bars.py` | the renderer |
| `FINDINGS.md` | what we learned about jix, numpy, blosc2 and ndarray while building this |
| `NOTES.md` | design notes and the decision log |
