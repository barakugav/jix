"""Analyze code-size bloat in a compiled dynamic library (the artifact Python
loads), grouping symbols by generic template and by crate.

This exists because the two obvious tools do not work here:
  * `cargo bloat` cannot target a `cdylib` - it falls back to the crate's only
    `[[bin]]` (`generate_pyi`) and reports that instead.
  * macOS `nm` always prints size 0 for Mach-O symbols.

So we recover per-symbol sizes from *address deltas*: sort defined symbols by
address, and treat (next_addr - addr) as the size of each symbol. Names are
demangled with `rustfilt`, then collapsed to a "template" (every `<...>` becomes
`<_>`, trailing legacy hash dropped) so that all monomorphizations of one
generic item fold into a single row - which is exactly the monomorphization
bloat we want to see.

Usage:
    python scripts/dylib_bloat.py [path-to-dylib] [--top N] [--head-of SUBSTR]

Defaults to jix-py/target/release/libjix.dylib. Build it first with:
    cargo build --release --lib --manifest-path jix-py/Cargo.toml

`--head-of read_data` additionally breaks down every symbol whose template ends
in `::read_data` by the head type of its first `<...>` group (e.g. ReductionOp,
Cast, Add), which is how we localized the reduction-op explosion.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HASH_RE = re.compile(r"::h[0-9a-f]{16}$")
DEFAULT_DYLIB = Path(__file__).parent.parent.resolve() / "jix-py/target/release/libjix.dylib"


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def method_of(collapsed):
    """The method identifier of a collapsed template, ignoring a trailing
    turbofish (`::<_>`) and closure/shim segments (`{closure#0}`,
    `{shim:vtable#0}`). Run on the collapsed form so `::` inside generic args
    can't be mistaken for a path separator.

        <_>::read_data                  -> read_data
        <_>::to_buf::<_>                -> to_buf     (turbofish skipped)
        <_>::call_once::{shim:vtable#0} -> call_once  (shim skipped)
    """
    for seg in reversed(collapsed.split("::")):
        if _IDENT.match(seg):
            return seg
    return ""


def collapse_angles(s, keep_head_for=()):
    """Replace every balanced <...> with <_> so monomorphizations fold together.

    If the symbol's method (see `method_of`) is in `keep_head_for`, the first
    top-level <...> keeps its struct name and becomes <Head<..>> instead of <_>,
    so those rows split per wrapper type (<ReductionOp<..>>::read_data,
    <Compact<..>>::to_buf::<_>, ...) instead of all folding into one row."""
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            if depth == 0:
                out.append("<_>")
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    collapsed = "".join(out)
    if keep_head_for and method_of(collapsed) in keep_head_for and "<_>" in collapsed:
        collapsed = collapsed.replace("<_>", f"<{outer_head(s)}<..>>", 1)
    return collapsed


def crate_of(sym):
    s = sym
    while s.startswith("<"):
        s = s[1:]
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", s)
    if not m:
        return "?"
    first = m.group(1)
    if first in ("impl", "dyn"):
        m2 = re.search(r"\b([a-z_][A-Za-z0-9_]*)::", sym)
        return m2.group(1) if m2 else first
    return first


def outer_head(sym):
    i = sym.find("<")
    if i < 0:
        return sym.split("::")[0]
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_:]*)", sym[i + 1 :])
    return m.group(1).split("::")[-1] if m else "?"


def load_symbols(dylib):
    """Return list of (demangled_name, size_bytes) for text (code) symbols."""
    raw = subprocess.run(
        ["nm", "-n", "--defined-only", str(dylib)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    rows = []
    for ln in raw:
        parts = ln.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            addr = int(parts[0], 16)
        except ValueError:
            continue
        rows.append([addr, parts[1], parts[2]])
    rows.sort(key=lambda r: r[0])
    for i in range(len(rows) - 1):
        rows[i].append(max(0, rows[i + 1][0] - rows[i][0]))
    if rows:
        rows[-1].append(0)
    code = [(n, sz) for a, t, n, sz in rows if t in ("t", "T")]

    names = [n for n, _ in code]
    dem = subprocess.check_output(
        ["rustfilt"],
        input="\n".join(names),
        text=True,
    ).splitlines()
    if len(dem) != len(names):
        sys.exit(f"rustfilt line mismatch: {len(dem)} != {len(names)}")
    # Strip the single Mach-O leading underscore so crate detection works.
    dem = [d[1:] if d.startswith("_") and not d.startswith("__") else d for d in dem]
    return [(HASH_RE.sub("", d), sz) for (n, sz), d in zip(code, dem)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dylib", nargs="?", default=DEFAULT_DYLIB)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument(
        "--head-of", default=None, help="break down symbols whose template ends with ::SUBSTR by outer type"
    )
    args = ap.parse_args()
    if not Path(args.dylib).exists():
        sys.exit(f"not found: {args.dylib} (build with cargo build --release --lib ...)")

    code = load_symbols(args.dylib)
    # for name, sz in code:
    #     if "::to_buf" in name:
    #         print(f"{name}")
    total = sum(sz for _, sz in code)

    tmpl_size, tmpl_cnt = defaultdict(int), defaultdict(int)
    crate_size, crate_cnt = defaultdict(int), defaultdict(int)
    for name, sz in code:
        # Special-case the dominant monomorphized methods: keep the wrapper
        # struct name so their rows split per op type (<ReductionOp<..>>::read_data,
        # <Compact<..>>::to_buf::<_>, ...) instead of folding into one <_>:: row.
        t = collapse_angles(name, keep_head_for=("read_data", "to_buf", "call_once"))
        tmpl_size[t] += sz
        tmpl_cnt[t] += 1
        cr = crate_of(name)
        crate_size[cr] += sz
        crate_cnt[cr] += 1

    print(f"{args.dylib}")
    print(f"code symbols: {len(code):,}   total code size (address-delta): {total / 1e6:.2f} MB\n")

    print("== BY CRATE ==")
    print(f"{'crate':<22}{'size(KB)':>11}{'symbols':>9}{'% code':>8}")
    for cr, sz in sorted(crate_size.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{cr:<22}{sz / 1024:>11.1f}{crate_cnt[cr]:>9}{100 * sz / total:>7.1f}%")

    print(f"\n== TOP {args.top} GENERIC TEMPLATES BY TOTAL SIZE (all monomorphizations summed) ==")
    print(f"{'copies':>7}{'total(KB)':>11}{'avg(B)':>9}  template")
    for t, sz in sorted(tmpl_size.items(), key=lambda kv: -kv[1])[: args.top]:
        c = tmpl_cnt[t]
        tt = t if len(t) <= 100 else t[:97] + "..."
        print(f"{c:>7}{sz / 1024:>11.1f}{sz / max(c, 1):>9.0f}  {tt}")

    print(f"\n== TOP {args.top} TEMPLATES BY COPY COUNT (monomorphization multiplier) ==")
    print(f"{'copies':>7}{'total(KB)':>11}  template")
    for t, c in sorted(tmpl_cnt.items(), key=lambda kv: -kv[1])[: args.top]:
        tt = t if len(t) <= 100 else t[:97] + "..."
        print(f"{c:>7}{tmpl_size[t] / 1024:>11.1f}  {tt}")

    if args.head_of:
        suffix = "::" + args.head_of
        head_size, head_cnt = defaultdict(int), defaultdict(int)
        for name, sz in code:
            if collapse_angles(name).endswith(suffix):
                h = outer_head(name)
                head_size[h] += sz
                head_cnt[h] += 1
        tot = sum(head_size.values())
        print(
            f"\n== {args.head_of}: {sum(head_cnt.values())} copies, "
            f"{tot / 1e6:.2f} MB, grouped by outer wrapper type =="
        )
        for h, sz in sorted(head_size.items(), key=lambda kv: -kv[1])[: args.top]:
            print(f"  {head_cnt[h]:>5} copies {sz / 1024:>9.1f} KB  {h}")


if __name__ == "__main__":
    main()
