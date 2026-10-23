# Verifier repro for F014: an 11-byte uleb128 length prefix in a .jix file.
# Expected: jix.read_array raises a normal error (e.g. ValueError / OSError / RuntimeError).
# Actual (debug-built bindings): pyo3_runtime.PanicException 'attempt to shift left with overflow'.
import os
import tempfile

import jix

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "overlong.jix")
    with open(path, "wb") as f:
        f.write(b"jix1" + b"\x80" * 10 + b"\x00")
    try:
        jix.read_array(path)
    except Exception as e:
        print("raised", type(e).__module__, type(e).__name__, e)
        assert type(e).__name__ != "PanicException", "Rust panic leaked instead of an error"
    except BaseException as e:
        print("raised BaseException", type(e).__module__, type(e).__name__, e)
        raise AssertionError("Rust panic leaked instead of an error") from e
    else:
        raise AssertionError("no error for an 11-byte uleb128")
print("ok")
