import argparse
import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()
DEFAULT_OUT = BENCH_DIR / "results"

sys.path.insert(0, str(BENCH_DIR.parent))  # put python/ on the path

from benches import report  # noqa: E402
from benches.array_impls import CODEC_DESC  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output directory for PNGs and tables")
    parser.add_argument("pytest_args", nargs="*", help="extra args to pass to pytest")
    args = parser.parse_args()
    out_dir: Path = args.out

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "python.json"
    # Benchmarks must run serially. If run_all is invoked from within an xdist worker
    # (e.g. the harness meta-tests under -n auto), the inherited PYTEST_XDIST_* vars would
    # make this nested pytest initialize as an xdist worker, which auto-disables
    # pytest-benchmark and clashes with --benchmark-only. Drop them for a clean run.
    env = dict(os.environ)
    for key in [k for k in env if k.startswith("PYTEST_XDIST")]:
        del env[key]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pytest",
            str(BENCH_DIR),
            "--benchmark-only",
            f"--benchmark-json={json_path}",
            "-q",
            *args.pytest_args,
        ],
        env=env,
    )

    benchmarks = report.load_pytest_json(json_path)
    written = []
    written += report.plot_throughput(benchmarks, out_dir)
    written.append(report.write_ratio_table(benchmarks, out_dir, CODEC_DESC))
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
