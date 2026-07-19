import importlib.util
from pathlib import Path

# jix/benches/run.py is a standalone script in the jix crate (not part of the benches
# package). Load it by path to unit-test its pure argument builder.
_RUN_PY = Path(__file__).resolve().parents[4] / "jix" / "benches" / "run.py"


def _load_run_py():
    spec = importlib.util.spec_from_file_location("jix_bench_run", _RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_harness_args():
    run = _load_run_py()
    assert run.split_harness_args(["--out", "x", "--", "reduction", "-v"]) == (["--out", "x"], ["reduction", "-v"])
    assert run.split_harness_args(["--fast"]) == (["--fast"], [])
