"""F006 repro: jix.Array.numpy_dtype holds the per-Array std::sync::Mutex `cache` while
dtype_to_numpy runs Python code (py.import("numpy") -> builtins.__import__ for aligned struct
dtypes). Python code run there that hands the GIL to a thread touching the same Array, or that
touches the same Array itself, hangs the process forever.

Asserts the CORRECT behavior (every scenario finishes), so it FAILS (exit 1) while the bug exists.
Each scenario runs in a child process with a timeout, because the bug is a permanent hang.
Run: .venv/bin/python bug-hunt/runs/2026-10-27-batch5/repros/F006.py
"""
import subprocess
import sys
import textwrap

TIMEOUT = 15

PRELUDE = r'''
import builtins, threading, time, sys
import numpy as np
import jix

dt = np.dtype([('a', 'i4'), ('b', 'i4')], align=True)  # aligned struct, no padding
arr = jix.compact(np.zeros(4, dtype=dt))                  # numpy dtype not cached yet
arr2 = jix.compact(np.zeros(4, dtype=dt))
orig_import = builtins.__import__
'''

SCENARIOS = {
    # A: the import hook gives up the GIL (sleep); a second thread reads the same Array.
    "A_two_threads_hook_sleeps": r'''
started = threading.Event()
def hook(name, *a, **kw):
    if name == "numpy" and threading.current_thread() is threading.main_thread() and not started.is_set():
        started.set()
        time.sleep(0.3)
    return orig_import(name, *a, **kw)
def other():
    started.wait()
    arr[0]
    print("other: done", flush=True)
t = threading.Thread(target=other); t.start()
builtins.__import__ = hook
print("main: requesting dtype", flush=True)
d = arr.dtype
builtins.__import__ = orig_import
t.join()
print("main: done", d, flush=True)
''',
    # B: the hook only runs pure-Python bytecode (no sleep, no I/O): the GIL switch interval
    # alone hands the GIL to the waiting thread.
    "B_two_threads_hook_busy_bytecode": r'''
sys.setswitchinterval(0.001)
started = threading.Event()
def hook(name, *a, **kw):
    if name == "numpy" and threading.current_thread() is threading.main_thread() and not started.is_set():
        started.set()
        x = 0
        for i in range(3_000_000):   # pure bytecode, ~100 ms
            x += i
    return orig_import(name, *a, **kw)
def other():
    started.wait()
    arr.dtype
    print("other: done", flush=True)
t = threading.Thread(target=other); t.start()
builtins.__import__ = hook
print("main: requesting dtype", flush=True)
d = arr.dtype
builtins.__import__ = orig_import
t.join()
print("main: done", d, flush=True)
''',
    # C: single thread, the hook reads the same Array (non-reentrant Mutex re-locked).
    "C_same_thread_reentry": r'''
state = {"in_hook": False}
def hook(name, *a, **kw):
    if name == "numpy" and not state["in_hook"]:
        state["in_hook"] = True
        print("hook: reading arr[0]", flush=True)
        arr[0]
    return orig_import(name, *a, **kw)
builtins.__import__ = hook
d = arr.dtype
builtins.__import__ = orig_import
print("main: done", d, flush=True)
''',
    # Control: same as A, but the second thread reads a different Array (arr2). Must finish.
    "control_other_array": r'''
started = threading.Event()
def hook(name, *a, **kw):
    if name == "numpy" and threading.current_thread() is threading.main_thread() and not started.is_set():
        started.set()
        time.sleep(0.3)
    return orig_import(name, *a, **kw)
def other():
    started.wait()
    arr2[0]
    print("other: done", flush=True)
t = threading.Thread(target=other); t.start()
builtins.__import__ = hook
d = arr.dtype
builtins.__import__ = orig_import
t.join()
print("main: done", d, flush=True)
''',
}


def run(name, body):
    code = PRELUDE + textwrap.dedent(body)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=TIMEOUT, cwd="/tmp")
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        print(f"{name}: HANG (no exit within {TIMEOUT} s); stdout so far: {out!r}")
        return False
    ok = r.returncode == 0 and "main: done" in r.stdout
    print(f"{name}: exit {r.returncode}; stdout: {r.stdout!r}"
          + (f"; stderr tail: {r.stderr[-500:]!r}" if r.returncode else ""))
    return ok


results = {name: run(name, body) for name, body in SCENARIOS.items()}
failed = [n for n, ok in results.items() if not ok]
if failed:
    print("FAIL: scenarios that did not finish:", failed)
    sys.exit(1)
print("PASS: all scenarios finished")
