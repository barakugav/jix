import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # scripts/bench/run.py -> repo root
RESULTS = Path(__file__).resolve().parent / ".results"  # scripts/bench/.results


def run_suite(
    repo: Path,
    python: str,
    suites: str,
    fast: list[str],
    rust_args: list[str],
    py_args: list[str],
    dst: Path,
    env: dict[str, str] | None = None,
):
    """Run the selected suites in `repo` with interpreter `python`, staging raw output under dst."""
    if dst.exists():
        shutil.rmtree(dst)  # a rerun must start clean (shutil.copytree below refuses an existing dir)
    dst.mkdir(parents=True)
    if suites in ("both", "rust"):
        (dst / "rust").mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [python, "jix/benches/run.py", "--out", str(dst / "rust" / "plots"), *fast, "--", *rust_args],
            cwd=repo,
            env=env,
        )
        shutil.copytree(repo / "jix" / "target" / "criterion", dst / "rust" / "criterion")
    if suites in ("both", "python"):
        subprocess.check_call(
            [python, "jix-py/python/benches/run_all.py", "--out", str(dst / "python"), *fast, "--", *py_args],
            cwd=repo,
            env=env,
        )


def main():
    parser = argparse.ArgumentParser(description="Run the jix benchmark suites for one git sha.")
    parser.add_argument("--suites", choices=["both", "rust", "python"], default="both")
    parser.add_argument("--rust-args", default="", help="extra args forwarded after `--` to jix/benches/run.py")
    parser.add_argument("--py-args", default="", help="extra args forwarded after `--` to run_all.py")
    parser.add_argument("--gitsha", default="", help="bench this sha in an isolated worktree (default: HEAD in place)")
    parser.add_argument("--fast", action="store_true", help="quick, low-fidelity run")
    args = parser.parse_args()

    suites = args.suites
    fast = ["--fast"] if args.fast else []
    rust_args = shlex.split(args.rust_args)
    py_args = shlex.split(args.py_args)
    gitsha = args.gitsha.strip()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not gitsha:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        ref = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True).strip()
        dst = RESULTS / sha[:8]
        run_suite(REPO, sys.executable, suites, fast, rust_args, py_args, dst)
        subprocess.check_call(
            [
                sys.executable,
                "jix-py/python/benches/meta.py",
                "--out",
                str(dst / "meta.json"),
                "--sha",
                sha,
                "--ref",
                ref,
            ],
            cwd=REPO,
        )
        return

    # --gitsha: bench that sha in an isolated worktree + venv; the real repo is never altered.
    subprocess.run(["git", "fetch", "--depth=1", "origin", gitsha], cwd=REPO, check=False)  # get object if remote-only
    worktree = Path(tempfile.mkdtemp(prefix="jix-bench-"))
    try:
        subprocess.check_call(["git", "worktree", "add", "--detach", str(worktree), gitsha], cwd=REPO)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
        dst = RESULTS / sha[:8]
        venv = worktree / ".venv"
        subprocess.check_call(["uv", "venv", "--python", "3.13", str(venv)])
        env = {**os.environ, "VIRTUAL_ENV": str(venv), "PATH": f"{venv / 'bin'}{os.pathsep}{os.environ['PATH']}"}
        subprocess.check_call(
            [
                "uv",
                "pip",
                "install",
                "-r",
                "scripts/dev_requirements.txt",
                "-r",
                "jix-py/python/benches/requirements.txt",
            ],
            cwd=worktree,
            env=env,
        )
        subprocess.check_call(["maturin", "develop", "--release", "--uv"], cwd=worktree / "jix-py", env=env)
        run_suite(worktree, str(venv / "bin" / "python"), suites, fast, rust_args, py_args, dst, env=env)
        subprocess.check_call(
            [
                str(venv / "bin" / "python"),
                "jix-py/python/benches/meta.py",
                "--out",
                str(dst / "meta.json"),
                "--sha",
                sha,
                "--ref",
                gitsha,
            ],
            cwd=worktree,
            env=env,
        )
    finally:
        shutil.rmtree(worktree, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=REPO, check=False)  # best-effort cleanup


if __name__ == "__main__":
    main()
