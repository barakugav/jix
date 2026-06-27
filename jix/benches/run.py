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


def main():
    parser = argparse.ArgumentParser(description="Run jix Rust benchmarks and plot the results.")
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output directory for PNGs and the table")
    parser.add_argument("--filter", default=None, help="substring passed to `cargo bench --` to select benches")
    parser.add_argument("--no-bench", action="store_true", help="skip cargo bench; plot existing results only")
    args = parser.parse_args()

    if not args.no_bench:
        subprocess.run(
            [
                "cargo",
                "bench",
                *(["--", args.filter] if args.filter else []),
            ],
            check=True,
            cwd=CRATE_DIR,
        )
    written = report.report_criterion(CRITERION_ROOT, args.out)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
