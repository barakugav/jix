"""K184 repro: archives written by the published jix 0.0.7 no longer load (and vice versa).

#106 (1169965) reused FileMetadata field 2: `uint64 lib_version_semver` (varint) became
`Semver lib_version` (length-delimited message). prost rejects the wrong wire type, so
the FileMetadata of every pre-#106 archive fails to decode.

JIX_0_0_7 below is the exact file jix 0.0.7 (2b229d2, built in a scratch copy) wrote for
array([[1, 2, 3], [4, 5, 6]], dtype=int32). Asserts the CORRECT behavior, so it fails while
the bug exists.

Optional reverse check: if the scratch jix 0.0.7 CLI exists
(set JIX_0_0_7_CLI to the binary built from bug-hunt/report/repros/K184_oldjix_cli/, whose
README says how), a file written by the current jix is read with it.

Run: .venv/bin/python bug-hunt/report/repros/K184.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

import jix

JIX_0_0_7 = bytes.fromhex(
    "6a6978310608021080801c080a020203120202031a0a0a0a0408031001200428041006180622020a002a020a003a00"
    "4f000000000000000c000000000000005b00000000000000160000000000000000000000000000001600000028b52f"
    "fd20186d000038010203040506000100333002"
)
EXPECTED = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
OLD_CLI = os.environ.get("JIX_0_0_7_CLI", "")

failures = 0


def check(name, fn):
    global failures
    try:
        fn()
        print(f"ok    {name}")
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


with tempfile.TemporaryDirectory() as d:
    old_path = os.path.join(d, "old.jix")
    with open(old_path, "wb") as f:
        f.write(JIX_0_0_7)

    def read_old(mmap):
        a = jix.read_array(old_path, mmap=mmap)
        np.testing.assert_array_equal(np.asarray(a), EXPECTED)

    check("jix.read_array(<0.0.7 file>) loads", lambda: read_old(False))
    check("jix.read_array(<0.0.7 file>, mmap=True) loads", lambda: read_old(True))

    if OLD_CLI and os.path.exists(OLD_CLI):
        new_path = os.path.join(d, "new.jix")
        jix.write_array(jix.asarray(EXPECTED), new_path)

        def old_reads_new():
            r = subprocess.run([OLD_CLI, "read", new_path], capture_output=True, text=True, check=False)
            out = (r.stdout + r.stderr).strip()
            assert r.returncode == 0, f"exit {r.returncode}: {out}"

        check("jix 0.0.7 reads a file written by the current jix", old_reads_new)
    else:
        print("skip  reverse check: JIX_0_0_7_CLI is not set (see K184_oldjix_cli/README.md)")

sys.exit(1 if failures else 0)
