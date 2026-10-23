# Repro for F002 (run 2026-10-23-batch3): read_array(params={'block_shape_fixed_dims': [...False...]})
# re-tunes the FILE's block shape, so blocks are decoded with the wrong geometry.
# Asserts the CORRECT behavior (read rejects the params, or returns the original data); every
# check fails while the bug exists. Run with .venv/bin/python.
import io
import os
import tempfile

import numpy as np

import jix

failures = []


def check(name, fn):
    try:
        fn()
        print(f"{name}: OK")
    except AssertionError as e:
        print(f"{name}: FAIL - {e}")
        failures.append(name)


src = np.arange(64 * 64, dtype=np.int32).reshape(64, 64)
d = tempfile.mkdtemp()
path = os.path.join(d, "a.jix")
jix.write_array(jix.compact(src, params={"block_shape": [8, 32]}), path)
bad_params = {"block_shape_fixed_dims": [False, False], "block_size": 1024}


def same_volume():
    try:
        a = jix.read_array(path, params=bad_params)
    except Exception:
        return  # rejecting the params up front is an acceptable fix
    got = a.numpy()
    n = int((got != src).sum())
    assert n == 0, f"{n} of {src.size} elements wrong; got[1,:4]={got[1, :4]} expected {src[1, :4]}"


def same_volume_mmap():
    try:
        a = jix.read_array(path, params=bad_params, mmap=True)
    except Exception:
        return
    got = a.numpy()
    n = int((got != src).sum())
    assert n == 0, f"mmap: {n} of {src.size} elements wrong"


def write_back():
    try:
        a = jix.read_array(path, params=bad_params)
    except Exception:
        return
    buf = io.BytesIO()
    jix.write_array(a, buf)
    buf.seek(0)
    got = jix.read_array(buf, validation="strict").numpy()  # default params
    n = int((got != src).sum())
    assert n == 0, f"re-written file read with default params + strict: {n} of {src.size} elements wrong"


def one_d():
    v = np.arange(100, dtype=np.int32)
    p = os.path.join(d, "b.jix")
    jix.write_array(jix.compact(v, params={"block_shape": [16]}), p)
    try:
        a = jix.read_array(p, params={"block_shape_fixed_dims": [False]})
    except Exception:
        return
    try:
        got = a.numpy()
    except Exception as e:
        raise AssertionError(f"valid file + accepted params, read fails: {type(e).__name__}: {e}")
    assert np.array_equal(got, v), "1-D values wrong"


def larger_grid():
    s = np.arange(64 * 300, dtype=np.int32).reshape(64, 300)
    p = os.path.join(d, "c.jix")
    jix.write_array(jix.compact(s, params={"block_shape": [32, 5]}), p)
    try:
        a = jix.read_array(p, params=bad_params)
    except Exception:
        return
    try:
        got = a.numpy()
    except BaseException as e:  # pyo3 PanicException derives from BaseException
        raise AssertionError(f"read raised {type(e).__name__}: {e}")
    assert np.array_equal(got, s), "values wrong"


check("same_volume", same_volume)
check("same_volume_mmap", same_volume_mmap)
check("write_back", write_back)
check("one_d", one_d)
check("larger_grid", larger_grid)
print("FAILED:", failures)
raise SystemExit(1 if failures else 0)
