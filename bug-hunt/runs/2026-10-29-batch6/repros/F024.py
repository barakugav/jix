"""F024: jix.errstate keeps its modes in an OS-thread-local Cell (jix-py/src/error.rs:19-21); each
errstate object snapshots the modes at construction and restores that snapshot on __exit__
(error.rs:113-124, 138-140). Two asyncio tasks on one event loop whose `with jix.errstate(...)`
blocks span an `await` and exit in FIFO order leave the thread in the first task's mode for good,
after every block has exited - divide-by-zero then returns 0 silently instead of raising.
numpy (contextvar-based errstate) ends at its default in the same program.

Asserts the CORRECT behavior, so it fails while the bug exists.
Run: .venv/bin/python bug-hunt/runs/2026-10-29-batch6/repros/F024.py
"""

import asyncio

import numpy as np

import jix

JIX_DEFAULT = jix.geterr()
NP_DEFAULT = np.geterr()
x = jix.compact(np.array([1, 2], dtype=np.int64))
z = jix.compact(np.array([0, 0], dtype=np.int64))


async def worker(errstate, mode, entered, leave):
    with errstate(divide=mode):
        entered.set()
        await leave.wait()


async def main(errstate):
    e1, l1, e2, l2 = (asyncio.Event() for _ in range(4))
    t1 = asyncio.create_task(worker(errstate, "ignore", e1, l1))
    await e1.wait()
    t2 = asyncio.create_task(worker(errstate, "warn", e2, l2))
    await e2.wait()
    l1.set()
    await t1  # the block entered first exits first
    l2.set()
    await t2


asyncio.run(main(np.errstate))
np_after = np.geterr()
print("numpy divide after both tasks:", np_after["divide"], "(default", NP_DEFAULT["divide"] + ")")

asyncio.run(main(jix.errstate))
jix_after = jix.geterr()
print("jix   divide after both tasks:", jix_after["divide"], "(default", JIX_DEFAULT["divide"] + ")")
try:
    r = (x // z).numpy()
    raised = False
    print("x // 0 after both tasks ->", r, "(no exception, no warning)")
except FloatingPointError as e:
    raised = True
    print("x // 0 after both tasks raised FloatingPointError:", e)

# Generator variant, for comparison only (not asserted): a contextvar does not fix it either.
def gen(errstate):
    with errstate(divide="warn"):
        yield


jix.errstate(**JIX_DEFAULT)
with np.errstate(divide="ignore"):
    g = gen(np.errstate)
    next(g)
g.close()
print("generator variant - numpy divide afterwards:", np.geterr()["divide"])
np.seterr(**NP_DEFAULT)

with jix.errstate(divide="ignore"):
    g = gen(jix.errstate)
    next(g)
g.close()
print("generator variant - jix   divide afterwards:", jix.geterr()["divide"])
jix.errstate(**JIX_DEFAULT)

# Cross-talk while both blocks are active (printed only): a task that asked for 'raise' reads
# under another task's 'ignore'.
async def strict(started, go, out):
    with jix.errstate(divide="raise"):
        started.set()
        await go.wait()
        try:
            out.append(("no error", (x // z).numpy()))
        except FloatingPointError as e:
            out.append(("raised", str(e)))


async def lax(started, done):
    with jix.errstate(divide="ignore"):
        started.set()
        await done.wait()


async def crosstalk():
    s1, go, s2, done = (asyncio.Event() for _ in range(4))
    out = []
    t1 = asyncio.create_task(strict(s1, go, out))
    await s1.wait()
    t2 = asyncio.create_task(lax(s2, done))
    await s2.wait()
    go.set()
    await t1
    done.set()
    await t2
    return out


print("task inside `with errstate(divide='raise')`, x // 0 ->", asyncio.run(crosstalk()))
jix.errstate(**JIX_DEFAULT)

assert np_after == NP_DEFAULT  # sanity: numpy is fine
assert jix_after == JIX_DEFAULT, f"jix errstate leaked after all blocks exited: {jix_after}"
assert raised, "x // 0 should raise FloatingPointError with default modes"
