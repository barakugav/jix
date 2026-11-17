"""K190: setting one codec key in `params` resets the source's other codec settings to defaults.

The `jix.compact` docstring says "Unset keys are inherited from the source storage if `array` is a
jix array", and `write_array` says "Unset fields are inherited from the source array". So a key the
caller leaves out (e.g. `filters`) must keep the source's value even when another codec key (e.g.
`compression_level`) is given.

Each check compares the archive bytes of `compact(src, params={<one codec key>})` with those of an
explicitly spelled-out reference that sets every codec key to the documented result (source value
for unset keys). The assertions state the CORRECT behavior, so this script fails while the bug
exists.
"""

import io

import numpy as np

import jix


def archive(a, **kw):
    buf = io.BytesIO()
    jix.write_array(a, buf, **kw)
    return buf.getvalue()


failures = []


def check(name, got, want):
    ok = got == want
    print(f"{name}: got {len(got)} bytes, expected {len(want)} bytes -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(name)


x = np.arange(1 << 16, dtype=np.int32)

# Source: no filters, level 3.
src = jix.compact(x, params={"filters": [], "compression_level": 3})
want = archive(jix.compact(src, params={"filters": [], "compression_level": 3}))
assert archive(jix.compact(src)) == want  # no params: everything inherited
# Only the level is given (and equals the source's): filters=[] must be inherited.
check("compact(src, compression_level=3)", archive(jix.compact(src, params={"compression_level": 3})), want)
# Only codec="zstd" (the sole valid codec, a no-op on its face): filters=[] must be inherited.
check("compact(src, codec='zstd')", archive(jix.compact(src, params={"codec": "zstd"})), want)
# write_array of a lazy view (params are ignored for an already-compact source).
check(
    "write_array(src + 0, compression_level=3)",
    archive(src + 0, params={"compression_level": 3}),
    archive(src + 0, params={"compression_level": 3, "filters": []}),
)

# Source: byte-shuffle (default), level 19.
rng = np.random.default_rng(0)
y = rng.integers(0, 4, 1 << 16).astype(np.int32).cumsum().astype(np.int32)
src19 = jix.compact(y, params={"compression_level": 19})
want19 = archive(jix.compact(src19, params={"filters": ["byte-shuffle"], "compression_level": 19}))
assert archive(jix.compact(src19)) == want19
# Only the filters are given (and equal the source's): level 19 must be inherited.
check(
    "compact(src19, filters=['byte-shuffle'])",
    archive(jix.compact(src19, params={"filters": ["byte-shuffle"]})),
    want19,
)
print("reference sizes: default params on y ->", len(archive(jix.compact(y))))

# The data itself always round-trips.
assert (jix.compact(src, params={"compression_level": 3}).numpy() == x).all()

assert not failures, f"unset codec keys were not inherited from the source: {failures}"
print("all OK")
