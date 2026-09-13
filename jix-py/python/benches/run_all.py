import argparse
import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()
DEFAULT_OUT = BENCH_DIR / "results"

# Reduced pytest-benchmark timing for --fast: a quick, low-fidelity run for dev cycles.
FAST_PYTEST_ARGS = ["--benchmark-max-time=0.1", "--benchmark-min-rounds=1", "--benchmark-warmup=off"]

sys.path.insert(0, str(BENCH_DIR.parent))  # put python/ on the path

from benches import report_bars, report_spec  # noqa: E402


def run_benchmarks(out_dir: Path, pytest_args: list[str]):
    """Run the pytest-benchmark suite, writing python.json into out_dir. Returns the json path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "python.json"
    # Benchmarks must run serially. If invoked from within an xdist worker, inherited
    # PYTEST_XDIST_* vars would make this nested pytest initialize as an xdist worker, which
    # auto-disables pytest-benchmark and clashes with --benchmark-only. Drop them.
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
            *pytest_args,
        ],
        env=env,
    )
    return json_path


def build_reports(json_path: Path, out_dir: Path, platform: str = "local"):
    """Plot the Python half of the report from an existing python.json.

    Returns the list of written paths. Runs no benchmarks. The full report, with both platforms
    and the Rust half, is built by `bench-report/build_report.py`; this is the quick local view.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report_bars.load_python(json_path, platform)
    written = []
    for section in report_spec.SECTIONS:
        if section["key"].startswith("rust_"):
            continue
        png = report_bars.plot_section(rows, section, out_dir)
        if png is not None:
            written.append(png)
    return written


def split_harness_args(argv: list[str]):
    """Split argv on the first '--': (own args, harness args passed through to pytest)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return list(argv), []


def main(out: Path | None = None, argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own, pytest_args = split_harness_args(argv)
    parser = argparse.ArgumentParser(description="Run jix Python benchmarks. Args after `--` go to pytest.")
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output directory for PNGs and tables")
    parser.add_argument("--fast", action="store_true", help="quick, low-fidelity run for dev cycles")
    args = parser.parse_args(own)
    out_dir = Path(out) if out is not None else args.out
    # Prepend the fast-timing args (when --fast) to whatever came after `--`.
    all_pytest_args = (FAST_PYTEST_ARGS if args.fast else []) + pytest_args
    json_path = run_benchmarks(out_dir, all_pytest_args)
    written = build_reports(json_path, out_dir)
    for p in written:
        print(p)
    return written


if __name__ == "__main__":
    main()
