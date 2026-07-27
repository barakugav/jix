import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()
CRATE_DIR = BENCH_DIR.parent
CRITERION_ROOT = CRATE_DIR / "target" / "criterion"
DEFAULT_OUT = CRITERION_ROOT / "plots"
REPORT_PKG = CRATE_DIR.parent / "jix-py" / "python"  # holds the `benches` package with report.py

sys.path.insert(0, str(REPORT_PKG))
from benches import report  # noqa: E402

# Reduced Criterion sampling for --fast: a quick, low-fidelity run for dev cycles.
FAST_CRITERION_ARGS = ["--warm-up-time", "1", "--measurement-time", "1", "--sample-size", "10"]


def split_harness_args(argv: list[str]):
    """Split argv on the first '--': (own args, harness args passed after `cargo bench --`)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return list(argv), []


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own, harness_args = split_harness_args(argv)
    parser = argparse.ArgumentParser(
        description="Run jix Rust benchmarks and plot the results. Args after `--` go to `cargo bench --`."
    )
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output directory for PNGs and the table")
    parser.add_argument("--fast", action="store_true", help="quick, low-fidelity run for dev cycles")
    parser.add_argument("--no-bench", action="store_true", help="skip cargo bench; plot existing results only")
    args = parser.parse_args(own)

    if not args.no_bench:
        # Args after `--`, then the fast-sampling flags when --fast is set.
        bench_args = list(harness_args) + (FAST_CRITERION_ARGS if args.fast else [])
        subprocess.check_call(
            ["cargo", "bench", *(["--", *bench_args] if bench_args else [])],
            cwd=CRATE_DIR,
        )
    written = report.report_criterion(CRITERION_ROOT, args.out)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
