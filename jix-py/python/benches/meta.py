import argparse
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

_OS_NAMES = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}


def _cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            # /proc/cpuinfo has no model name on ARM; lscpu decodes it from the CPU part id.
            out = subprocess.check_output(["lscpu"], text=True, env={**os.environ, "LC_ALL": "C"})
            for line in out.splitlines():
                if line.startswith("Model name:") and line.split(":", 1)[1].strip() not in ("", "-"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: S110
        pass
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: S110
        pass
    return platform.processor() or "unknown"


def _os_version():
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
        elif system == "Darwin":
            return f"macOS {platform.mac_ver()[0]}"
    except Exception:  # noqa: S110
        pass
    return platform.platform()


def _cpu_cores():
    return os.cpu_count() or 0


def _ram_bytes():
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
        elif system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out) if out else 0
    except Exception:  # noqa: S110
        pass
    return 0


def _rustc():
    try:
        return subprocess.check_output(["rustc", "--version"], text=True).strip()
    except Exception:
        return None


def _version(dist: str):
    try:
        return metadata.version(dist)
    except Exception:
        return None


def collect(sha: str, ref: str | None, fast: bool = False):
    """Return the meta.json dict for this runner and the benched sha.

    Workflow-run identifiers (run id / attempt) are intentionally omitted: they are available
    from the ``gh`` CLI for downloaded runs and are meaningless for local runs.
    """
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha": sha,
        "ref": ref,
        # Reduced sampling: enough to check the plumbing, not to quote.
        "fast": fast,
        "platform": {
            "os": _OS_NAMES.get(platform.system(), platform.system().lower()),
            "arch": platform.machine(),
            "os_version": _os_version(),
            # "github-hosted" / "self-hosted" on Actions, None for a local run.
            "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
            "cpu_model": _cpu_model(),
            "cpu_cores": _cpu_cores(),
            "ram_bytes": _ram_bytes(),
            "rustc": _rustc(),
            "python": platform.python_version(),
            "rustflags": os.environ.get("RUSTFLAGS", ""),
            "profile": "bench",
        },
        "libs": {name: _version(name) for name in ["numpy", "blosc2", "zarr", "jix"]},
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Collect benchmark environment metadata.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha", required=True, type=str)
    parser.add_argument("--ref", default=None, type=str)
    parser.add_argument("--fast", action="store_true", help="record that this was a low-fidelity run")
    args = parser.parse_args(argv)
    result = collect(args.sha, args.ref, args.fast)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
