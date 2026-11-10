"""Scratch A/B/A/B driver for jix/examples/probe_read.rs (experimental branch).

Runs every case as A, B, A, B - one process per run, two rounds, never more - and prints the best
`min` of each side plus the speedup. A is the baseline build, B the working tree build:

    python scripts/probe_ab.py --a <baseline probe_read binary> [--b <binary>] [--secs 2] [filter...]

Both binaries should be built with RUSTFLAGS="-C llvm-args=-align-loops=64".
Positional filters keep only the cases whose name contains every given substring.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_B = REPO / "jix" / "target" / "release" / "examples" / "probe_read"

BIG_2D = "8192x8192"

# (name, probe args)
CASES = [
    # --- default layout, whole array ---
    ("full i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read full"),
    ("full i32 random", f"--dtype i32 --profile random --shape {BIG_2D} --read full"),
    ("full i32 lowent", f"--dtype i32 --profile lowent --shape {BIG_2D} --read full"),
    ("full i32 zero", f"--dtype i32 --profile zero --shape {BIG_2D} --read full"),
    ("full f32 noisy", f"--dtype f32 --profile noisy --shape {BIG_2D} --read full"),
    ("full f64 smooth", "--dtype f64 --profile smooth --shape 4096x8192 --read full"),
    ("full i32 noisy 1d", "--dtype i32 --profile noisy --shape 67108864 --read full"),
    ("full f32 noisy 3d", "--dtype f32 --profile noisy --shape 256x512x512 --read full"),
    ("nd i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read nd"),
    ("mmap i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read mmap"),
    # --- default layout, other access patterns ---
    ("sub i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read sub"),
    ("col i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read col"),
    ("tiles i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read tiles"),
    ("sum i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read sum"),
    ("neg i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read neg"),
    ("t i32 noisy", f"--dtype i32 --profile noisy --shape {BIG_2D} --read t"),
    ("t f32 noisy 3d", "--dtype f32 --profile noisy --shape 256x512x512 --read t"),
    # --- square-ish blocks ---
    ("full i32 noisy blk256", f"--dtype i32 --profile noisy --shape {BIG_2D} --block 256x256 --read full"),
    ("full i32 random blk256", f"--dtype i32 --profile random --shape {BIG_2D} --block 256x256 --read full"),
    ("sub i32 noisy blk256", f"--dtype i32 --profile noisy --shape {BIG_2D} --block 256x256 --read sub"),
    ("tiles i32 noisy blk256", f"--dtype i32 --profile noisy --shape {BIG_2D} --block 256x256 --read tiles"),
    ("sum i32 noisy blk256", f"--dtype i32 --profile noisy --shape {BIG_2D} --block 256x256 --read sum"),
    ("t i32 noisy blk256", f"--dtype i32 --profile noisy --shape {BIG_2D} --block 256x256 --read t"),
    ("full f32 noisy 3d blk64", "--dtype f32 --profile noisy --shape 256x512x512 --block 64x64x64 --read full"),
]

MIN_RE = re.compile(r"min=\s*([0-9.]+)ms")


def run(binary: Path, case_args: str, secs: float) -> float:
    out = subprocess.run(
        [str(binary), *case_args.split(), "--secs", str(secs)],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        return float("nan")
    return float(MIN_RE.search(out.stdout).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="baseline probe_read binary")
    ap.add_argument("--b", default=DEFAULT_B, type=Path, help="candidate probe_read binary")
    ap.add_argument("--secs", default=2.0, type=float, help="measurement time per run")
    ap.add_argument("filters", nargs="*", help="keep cases whose name contains all of these")
    args = ap.parse_args()

    cases = [c for c in CASES if all(f in c[0] for f in args.filters)]
    print(f"{'case':<28} {'A1':>9} {'B1':>9} {'A2':>9} {'B2':>9}   {'A':>9} {'B':>9}  speedup")
    for name, case_args in cases:
        times = [run(binary, case_args, args.secs) for binary in (args.a, args.b, args.a, args.b)]
        a, b = min(times[0], times[2]), min(times[1], times[3])
        cells = " ".join(f"{t:9.2f}" for t in times)
        print(f"{name:<28} {cells}   {a:9.2f} {b:9.2f}  {a / b:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
