# Repro for F003 (run 2026-10-23-batch3).
#
# A .jix bool array whose bytes are not 0/1 loads under the default (Strict) validation and the
# bytes reach the Rust kernels as typed `bool` (UB). The file is built by writing a uint8 array and
# patching the one header byte that encodes the scalar kind (U8 = 5 -> Bool = 14).
#
# Each check asserts the CORRECT behavior: the load / read raises, or the values come back as 0/1
# with nonzero-is-true semantics. The script exits 1 while the bug exists.
import os
import sys
import tempfile

import numpy as np

import jix

RAW = np.array([0, 1, 2, 255], dtype=np.uint8)
EXPECT = RAW != 0  # nonzero-is-true: [False, True, True, True]


def patched_bool_file(d):
    pu, pb, pp = (os.path.join(d, n) for n in ("u.jix", "b.jix", "p.jix"))
    jix.write_array(jix.compact(RAW), pu)
    jix.write_array(jix.compact(np.zeros(len(RAW), dtype=np.bool_)), pb)
    u, b = bytearray(open(pu, "rb").read()), open(pb, "rb").read()
    assert len(u) == len(b)
    i = next(k for k in range(len(u)) if u[k] != b[k])
    assert (u[i], b[i]) == (5, 14), (i, u[i], b[i])
    u[i] = b[i]
    open(pp, "wb").write(bytes(u))
    return pp


failures = []


def check(name, fn, expected):
    try:
        got = fn()
    except Exception as e:  # a raised error (e.g. InvalidArchive) is a correct outcome
        print(f"ok   {name}: raised {type(e).__name__}: {e}")
        return
    got = np.asarray(got)
    if got.shape == np.shape(expected) and np.array_equal(got, expected):
        print(f"ok   {name}: {got.tolist()}")
    else:
        print(f"FAIL {name}: jix={got.tolist()} expected={np.asarray(expected).tolist()}")
        failures.append(name)


with tempfile.TemporaryDirectory() as d:
    path = patched_bool_file(d)
    for mmap in (False, True):
        tag = "mmap" if mmap else "read"
        try:
            a = jix.read_array(path, mmap=mmap)  # default validation: Strict
        except Exception as e:
            print(f"ok   {tag}: load raised {type(e).__name__}: {e}")
            continue
        print(f"{tag}: loaded dtype={a.dtype} shape={a.shape}")
        x = np.array([10, 11, 12, 13], dtype=np.int32)
        y = np.array([20, 21, 22, 23], dtype=np.int32)
        check(f"{tag} numpy() bytes", lambda: a.numpy().view(np.uint8), EXPECT.astype(np.uint8))
        check(f"{tag} where(a, x, y)", lambda: jix.where(a, x, y).numpy(), np.where(EXPECT, x, y))
        check(f"{tag} cast(a, int32)", lambda: jix.cast(a, "int32").numpy(), EXPECT.astype(np.int32))
        check(f"{tag} sum(a)", lambda: jix.sum(a).numpy(), np.sum(EXPECT))
        check(f"{tag} any(a[2:3])", lambda: jix.any(a[2:3]).numpy(), np.any(EXPECT[2:3]))

print(f"{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
