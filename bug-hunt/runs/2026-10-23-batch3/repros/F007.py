# Repro for bug-hunt run 2026-10-23-batch3, finding F007.
#
# CRASHES: the default `jix.read_array(io.BytesIO(data))` (len=None) on a ~106-byte archive whose
# block_data TOC size is patched to 2**44 ABORTS the interpreter (allocation failure, SIGABRT).
# Each case therefore runs in a subprocess and its exit status is reported.
#
# Correct behavior asserted: every malformed case raises an ordinary Exception (no crash), and
# the 1 GiB case does not grow peak RSS by anything close to the declared section size.
# The script exits 1 while the bug exists.
#
# Run: .venv/bin/python bug-hunt/runs/2026-10-23-batch3/repros/F007.py
import os
import subprocess
import sys
import tempfile
import textwrap

CHILD = textwrap.dedent(
    r"""
    import io, resource, struct, sys
    import numpy as np
    import jix

    def read_uleb(b, pos):
        x = 0; s = 0
        while True:
            v = b[pos]; pos += 1
            x |= (v & 0x7F) << s
            if v < 0x80:
                return x, pos
            s += 7

    buf = io.BytesIO()
    jix.write_array(jix.compact(np.arange(8, dtype=np.uint8)), buf)
    data = bytearray(buf.getvalue())
    toc = 4
    for _ in range(3):
        n, toc = read_uleb(data, toc)
        toc += n

    case = sys.argv[1]
    kwargs = {}
    if case == "valid":
        pass
    elif case == "block_data_2_44":
        data[toc + 24:toc + 32] = struct.pack("<Q", 1 << 44)
    elif case == "block_data_2_44_len":
        data[toc + 24:toc + 32] = struct.pack("<Q", 1 << 44)
        kwargs["len"] = len(data)
    elif case == "blocks_loc_12_2_40":
        data[toc + 8:toc + 16] = struct.pack("<Q", 12 << 40)
    elif case == "block_data_2_30":
        data[toc + 24:toc + 32] = struct.pack("<Q", 1 << 30)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        a = jix.read_array(io.BytesIO(bytes(data)), **kwargs)
        out = f"returned shape={a.shape}"
    except Exception as e:
        out = f"raised {type(e).__name__}: {e}"
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"{len(data)} bytes; {out}; peak RSS growth {(rss1 - rss0) // 1024} MiB", flush=True)
    """
)

CASES = [
    # (case, expect_exception, description)
    ("valid", False, "unmodified archive, read_array(BytesIO)"),
    ("block_data_2_44", True, "block_data size 2**44, read_array(BytesIO)  [default len=None]"),
    ("block_data_2_44_len", True, "block_data size 2**44, read_array(BytesIO, len=len(data))"),
    ("blocks_loc_12_2_40", True, "blocks_loc size 12*2**40, read_array(BytesIO)  [default len=None]"),
    ("block_data_2_30", True, "block_data size 2**30, read_array(BytesIO)  [default len=None]"),
]

failures = 0
with tempfile.TemporaryDirectory() as d:
    for case, expect_exc, desc in CASES:
        r = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, case],
            capture_output=True,
            text=True,
            env={**os.environ, "RUST_BACKTRACE": "0"},
            cwd=d,
        )
        lines = (r.stdout + r.stderr).strip().splitlines()
        msg = " | ".join(lines[-2:])
        ok = r.returncode == 0 and (("raised" in r.stdout) == expect_exc)
        if ok and case == "block_data_2_30":
            growth = int(r.stdout.rsplit("growth ", 1)[1].split()[0])
            ok = growth < 64
        failures += not ok
        print(f"[{case}] {desc}\n    exit={r.returncode} :: {msg}\n    -> {'ok' if ok else 'FAIL'}")

print(f"{failures} failing case(s)")
sys.exit(1 if failures else 0)
