import subprocess
import sys

from bench import run


def test_repo_root_points_at_repo():
    # REPO is scripts/bench/run.py -> parents[2]; the runner scripts live under it.
    assert (run.REPO / "jix" / "benches" / "run.py").exists()
    assert (run.REPO / "jix-py" / "python" / "benches" / "run_all.py").exists()


def test_cli_help_runs():
    out = subprocess.check_output([sys.executable, str(run.REPO / "scripts" / "bench" / "run.py"), "--help"], text=True)
    assert "--gitsha" in out
    assert "--fast" in out
