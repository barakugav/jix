# Repro for F006 (run 2026-10-23-batch3): jix.read_array on a tiny file whose first message
# length (uleb128) points past the end of the file must raise a normal exception.
# While the bug exists it CRASHES: 2^44 aborts the interpreter (SIGABRT, 'memory allocation of
# 17592186044416 bytes failed'); 2^63 raises pyo3_runtime.PanicException('capacity overflow').
# Each case runs in a subprocess so the abort does not kill this script. The script exits 1
# while the bug exists and 0 once fixed.
import os
import subprocess
import sys
import tempfile
import textwrap

CHILD = textwrap.dedent(
    r"""
    import sys
    import jix

    def uleb(x):
        out = bytearray()
        while True:
            b = x & 0x7F
            x >>= 7
            if x == 0:
                out.append(b)
                return bytes(out)
            out.append(b | 0x80)

    msg_len, path, use_bytesio = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
    data = b"jix1" + uleb(msg_len)
    with open(path, "wb") as f:
        f.write(data)
    try:
        if use_bytesio:
            import io
            jix.read_array(io.BytesIO(data), len=len(data))
        else:
            jix.read_array(path)  # default: Strict, len = file size
    except (ValueError, OSError, RuntimeError) as e:
        print(f"OK: {type(e).__name__}: {e}")
        sys.exit(0)
    except BaseException as e:
        print(f"BUG: {type(e).__module__}.{type(e).__name__}: {e}")
        sys.exit(2)
    print("BUG: returned normally")
    sys.exit(3)
    """
)

failed = False
with tempfile.TemporaryDirectory() as d:
    for i, (msg_len, bytesio) in enumerate([(1 << 44, False), (1 << 63, False), (1 << 44, True)]):
        path = os.path.join(d, f"f{i}.jix")
        r = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, str(msg_len), path, "1" if bytesio else "0"],
            capture_output=True,
            text=True,
            env={**os.environ, "RUST_BACKTRACE": "0"},
        )
        lines = (r.stdout + r.stderr).strip().splitlines()
        keep = [ln for ln in lines if ln.startswith(("OK:", "BUG:", "memory allocation"))]
        out = " | ".join(keep or lines[-2:])
        src = "BytesIO" if bytesio else "path"
        size = len(b"jix1") + (msg_len.bit_length() + 6) // 7
        print(f"msg_len=2^{msg_len.bit_length() - 1} ({src}, {size}-byte file): exit={r.returncode} :: {out}")
        failed |= r.returncode != 0
sys.exit(1 if failed else 0)
