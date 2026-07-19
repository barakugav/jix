import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

_OS_NAMES = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}


def _cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


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
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(out.stdout.strip()) if out.stdout.strip() else 0
    except Exception:
        pass
    return 0


def _rustc():
    try:
        return subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _version(dist: str):
    try:
        return metadata.version(dist)
    except Exception:
        return None


def collect(sha: str, ref: str | None):
    """Return the meta.json dict for this runner and the benched sha.

    Workflow-run identifiers (run id / attempt) are intentionally omitted: they are available
    from the ``gh`` CLI for downloaded runs and are meaningless for local runs.
    """
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha": sha,
        "ref": ref,
        "platform": {
            "os": _OS_NAMES.get(platform.system(), platform.system().lower()),
            "arch": platform.machine(),
            "runner": os.environ.get("RUNNER_NAME") or os.environ.get("ImageOS"),
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
    args = parser.parse_args(argv)
    result = collect(args.sha, args.ref)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
