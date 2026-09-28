#!/usr/bin/env python3
"""Static performance analysis of the byte/bit-shuffle kernels: cargo-asm + llvm-mca, per platform.

For every platform (target triple + ISA level) the probe crate is compiled in release mode, and for
every kernel:

1. `cargo asm` extracts the function's asm, and the asm loop tree is recovered from back-edges.
2. `cargo asm --llvm` emits the final (post-optimization) LLVM IR, and LLVM's own `opt` reports the
   IR loop tree (`print<loops>`) and each loop's exact trip count (`print<scalar-evolution>`).
3. The two loop trees are matched by shape, which gives every asm loop its trip count.
4. The hot loops are the loops whose total iteration count (their trip count times their
   ancestors', evaluated from SCEV for two concrete input lengths) is linear in the input length:
   e.g. a byte-shuffle main loop, or the loop over groups nested in a bit-shuffle pass's loop over
   byte planes. Tails and loops over the itemsize do not scale with the input and are ignored. This
   also gives each hot loop's input bytes per iteration, so kernels may use any step.
5. One iteration of each hot loop is flattened into a straight-line trace (inner loops repeated
   by their constant trip counts), and llvm-mca simulates that trace in a steady state on each
   representative CPU of the platform.

The objective is llvm-mca's **cycles per 4096 bytes** (= sum over the hot loops of cycles per
iteration * 4096 / bytes per iteration). Lower is better. It is a static model: memory is assumed
to hit L1, calls (e.g. to `memcpy`) are not followed - the `calls` column flags kernels whose hot
loop still calls something - and the per-call prologue/epilogue and the tails are not included.

Kernels without a hot loop (e.g. the bit-shuffle `*bitrow_eight` passes, whose loops run 8 *
itemsize `memcpy` calls whatever the input length) are reported with the cycles per iteration of
their largest innermost loop, as an informational number.

Usage (from anywhere):
    python jix/probe/analyze.py [--direction decode] [--label baseline] [--platform x86_64-v3 ...] [--fn decode_4 ...]

Outputs, under jix/probe/results/<label>/:
    summary.md                      the tables
    summary.json                    same data, machine readable
    asm/<platform>/<fn>.s           full function asm (cargo asm --simplify)
    mca/<platform>/<fn>.<cpu>.txt   full llvm-mca report (gitignored)

Requirements: the pinned nightly (rust-toolchain.toml, with the llvm-tools component for `opt`),
`cargo install cargo-show-asm`, and an llvm-mca whose LLVM major version matches rustc's (found via
$LLVM_MCA, llvm-mca-<major>, or llvm-mca on PATH). See README.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
MCA_ITERATIONS = 100
# Reporting unit: cycles per this many bytes of input.
BYTES_UNIT = 4096
# Input length (bytes) at which the outer loop's SCEV trip count is evaluated to derive its bytes
# per iteration. A multiple of any plausible step, so there is no remainder.
SCEV_EVAL_LEN = 1 << 20

# x86 levels, exactly as enabled by fearless_simd's `dispatch!` arms (`Simd::vectorize` of the
# Sse4_2 / Avx2 / Avx512 tokens). The platform names keep the x86-64-vN labels of the baseline.
X86_V2 = "+fxsr,+sse4.2,+cmpxchg16b,+popcnt"
X86_V3 = "+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave"
# fearless_simd's AVX-512 level is Ice Lake (includes VBMI), not x86-64-v4.
X86_V4 = ",".join(
    "+" + f
    for f in "adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,"
    "avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,fxsr,gfni,lzcnt,movbe,pclmulqdq,popcnt,"
    "rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves".split(",")
)


@dataclass
class Platform:
    name: str
    target: str
    rustflags: list[str]
    # CPUs llvm-mca simulates the code on (scheduling models). Pick CPUs that actually run this
    # code path in practice.
    mca_cpus: list[str]

    @property
    def isa(self) -> str:
        if self.target.startswith(("x86_64", "i686")):
            return "x86"
        if self.target.startswith("aarch64"):
            return "aarch64"
        if self.target.startswith(("arm", "thumb")):
            return "arm"
        if self.target.startswith("powerpc"):
            return "ppc"
        raise ValueError(self.target)

    @property
    def comment(self) -> str:
        """Line comment marker of the target's asm dialect."""
        if self.isa == "aarch64":
            return ";" if "apple" in self.target else "//"
        return {"x86": "#", "arm": "@", "ppc": "#"}[self.isa]


PLATFORMS = [
    Platform("x86_64-v2", "x86_64-unknown-linux-gnu", [f"-Ctarget-feature={X86_V2}"], ["sandybridge", "btver2"]),
    Platform(
        "x86_64-v3", "x86_64-unknown-linux-gnu", [f"-Ctarget-feature={X86_V3}"], ["skylake", "alderlake", "znver3"]
    ),
    Platform(
        "x86_64-v4",
        "x86_64-unknown-linux-gnu",
        [f"-Ctarget-feature={X86_V4}"],
        ["icelake-server", "sapphirerapids", "znver4"],
    ),
    # 32-bit x86 wheels (linux/windows i686): SSE2 baseline, multiversion targets are x86_64-only.
    Platform("i686", "i686-unknown-linux-gnu", [], ["skylake"]),
    # aarch64 linux/windows wheels: generic armv8-a + NEON baseline.
    Platform("aarch64", "aarch64-unknown-linux-gnu", [], ["cortex-a72", "neoverse-n1", "neoverse-v2"]),
    # macOS arm64 wheels: the target's default CPU is apple-m1.
    Platform("aarch64-apple", "aarch64-apple-darwin", [], ["apple-m1"]),
]


@dataclass
class Kernel:
    name: str
    symbol: str  # demangled, as `cargo asm` lists it
    ir_hint: str  # substring of the v0-mangled symbol, to find the function in the LLVM IR
    # Itemsize passed at runtime as the `typesize` argument (bit-shuffle passes), bound when the
    # trip counts are evaluated. None: the itemsize is a compile-time constant of the symbol.
    runtime_itemsize: int | None = None

    def scev_env(self, length: int) -> dict[str, int]:
        """Values of the function arguments, for an input of `length` bytes."""
        env = {"%src.1": length, "%dst.1": length}
        if self.runtime_itemsize is not None:
            t = self.runtime_itemsize
            env |= {"%typesize": t, "%n_full": length // t // 8 * 8}
        return env


def _byte_kernel(direction: str, itemsize: int) -> Kernel:
    sym = f"probe_byte_shuffle_{direction}_{itemsize}"
    return Kernel(f"{direction}_{itemsize}", sym, sym)


def _bit_kernel(fn: str, itemsize: int) -> Kernel:
    if fn.endswith("bit_byte"):  # the `probe_*` export of the dispatched body
        sym = f"probe_bit_shuffle_{fn}"
        return Kernel(f"{fn}_{itemsize}", sym, sym, itemsize)
    # v0 mangling length-prefixes identifiers: `18trans_bitrow_eight` does not match
    # `untrans_bitrow_eight`.
    return Kernel(f"{fn}_{itemsize}", f"jix_probe::bit_shuffle::{fn}", f"{len(fn)}{fn}", itemsize)


# Kernels per `--direction`. Must match the `probe_*` exports and modules in src/lib.rs.
ITEMSIZES = (2, 4, 8, 16)
KERNELS = {d: [_byte_kernel(d, s) for s in ITEMSIZES] for d in ("decode", "encode")}
# The bit-level passes of the bit-shuffle filter (its byte-shuffle pass is the byte-shuffle kernel).
KERNELS["bit-encode"] = [_bit_kernel(f, s) for f in ("trans_bit_byte", "trans_bitrow_eight") for s in ITEMSIZES]
KERNELS["bit-decode"] = [_bit_kernel(f, s) for f in ("untrans_bitrow_eight", "untrans_bit_byte") for s in ITEMSIZES]

# x86 feature sets of the `multiversion` clones (x86-64-v2/v3/v4), used by the baselines, which
# measured the auto-vectorized kernels (`--x86-levels multiversion`).
MULTIVERSION_V2 = "+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b"
MULTIVERSION_V3 = MULTIVERSION_V2 + ",+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave"
MULTIVERSION_V4 = MULTIVERSION_V3 + ",+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl"
MULTIVERSION_FEATURES = {"x86_64-v2": MULTIVERSION_V2, "x86_64-v3": MULTIVERSION_V3, "x86_64-v4": MULTIVERSION_V4}


# --------------------------------------------------------------------------------------------------
# Tools


def run(cmd: list[str], env: dict | None = None, cwd: Path | None = None, input: str | None = None) -> str:
    p = subprocess.run(cmd, env=env, cwd=cwd, input=input, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def llvm_major_of_rustc() -> int:
    out = run(["rustc", "-vV"], cwd=PROBE_DIR)
    return int(re.search(r"LLVM version: (\d+)", out).group(1))


def find_llvm_mca(major: int) -> str:
    for c in [os.environ.get("LLVM_MCA"), f"llvm-mca-{major}", "llvm-mca"]:
        path = c and shutil.which(c)
        if not path:
            continue
        m = re.search(r"LLVM version (\d+)", run([path, "--version"]))
        if m and int(m.group(1)) == major:
            return path
        print(f"note: skipping {path}: LLVM {m.group(1) if m else '?'} != rustc's LLVM {major}", file=sys.stderr)
    sys.exit(f"error: no llvm-mca with LLVM major version {major} found (set $LLVM_MCA)")


def find_opt() -> str:
    """`opt` of the pinned toolchain's own LLVM (rustup component llvm-tools)."""
    sysroot = run(["rustc", "--print", "sysroot"], cwd=PROBE_DIR).strip()
    host = re.search(r"host: (\S+)", run(["rustc", "-vV"], cwd=PROBE_DIR)).group(1)
    opt = Path(sysroot) / "lib" / "rustlib" / host / "bin" / "opt"
    if not opt.exists():
        sys.exit(f"error: {opt} not found, install the llvm-tools rustup component")
    return str(opt)


def cargo_env(platform: Platform, target_dir: Path) -> dict:
    env = dict(os.environ)
    env["RUSTFLAGS"] = " ".join(["-Csymbol-mangling-version=v0", *platform.rustflags])
    env["CARGO_TARGET_DIR"] = str(target_dir)
    return env


def cargo_asm(platform: Platform, symbol: str, target_dir: Path, llvm: bool = False) -> str:
    mode = ["--llvm"] if llvm else ["--simplify", *(["--att"] if platform.isa == "x86" else [])]
    cmd = ["cargo", "asm", "--release", "--lib", "--target", platform.target, *mode, symbol]
    return run(cmd, env=cargo_env(platform, target_dir), cwd=PROBE_DIR)


# --------------------------------------------------------------------------------------------------
# Loop trees


@dataclass
class Loop:
    start: int  # asm: line of the header label. IR: unused
    end: int  # asm: line of the last back-edge branch. IR: unused
    header: str
    children: list[Loop] = field(default_factory=list)
    trip_count: int | None = None  # constant trip count, if any
    trip_expr: str = ""  # IR only: SCEV backedge-taken count expression
    size: int = 0  # instructions in the loop (IR or asm), to match sibling loops of the same shape
    ir: IrValues | None = None  # IR only: the function's values, to resolve SCEV operands

    def shape(self) -> tuple:
        return tuple(c.shape() for c in self.children)


def build_tree(loops: list[Loop], contains) -> list[Loop]:
    """Nest `loops` (in program order, outer before inner) by the `contains` predicate."""
    roots: list[Loop] = []
    stack: list[Loop] = []
    for lp in loops:
        while stack and not contains(stack[-1], lp):
            stack.pop()
        (stack[-1].children if stack else roots).append(lp)
        stack.append(lp)
    return roots


LABEL_RE = re.compile(r"^\s*([.\w$]+):")
TOKEN_RE = re.compile(r"[.\w$]+")


def is_label(line: str) -> bool:
    return LABEL_RE.match(line) is not None


def mnemonic(line: str) -> str:
    return line.split()[0]


def is_branch(isa: str, mn: str) -> bool:
    if isa == "x86":
        return mn.startswith("j") or mn.startswith("loop")
    return mn.startswith(("b", "cb", "tb"))  # aarch64 / arm / ppc


def is_call(isa: str, mn: str) -> bool:
    if isa == "x86":
        return mn.startswith("call")
    if isa == "aarch64":
        return mn in ("bl", "blr")
    if isa == "arm":
        return mn in ("bl", "blx")
    return mn in ("bl", "bla", "bctrl", "blrl")  # ppc (`blr` is a return there)


def clean_lines(asm: str, comment: str) -> list[str]:
    """Strip comments, blank lines, directives and the function-name label."""
    out = []
    for line in asm.splitlines()[1:]:
        if comment in line:
            line = line[: line.index(comment)]
        s = line.strip()
        if not s or (s.startswith(".") and not is_label(s)):
            continue
        out.append(s)
    return out


def asm_loops(lines: list[str], isa: str) -> list[Loop]:
    labels = {m.group(1): i for i, line in enumerate(lines) if (m := LABEL_RE.match(line))}
    ends: dict[str, int] = {}
    for i, line in enumerate(lines):
        if is_label(line) or not is_branch(isa, mnemonic(line)):
            continue
        for tok in TOKEN_RE.findall(line)[1:]:
            j = labels.get(tok)
            if j is not None and j < i:
                ends[tok] = max(ends.get(tok, i), i)
    loops = sorted(
        (
            Loop(labels[h], e, h, size=sum(1 for x in lines[labels[h] : e + 1] if not is_label(x)))
            for h, e in ends.items()
        ),
        key=lambda lp: (lp.start, -lp.end),
    )
    for a in loops:
        for b in loops:
            if a.start < b.start <= a.end < b.end:
                raise RuntimeError(f"overlapping asm loops {a.header} and {b.header}")
    return build_tree(loops, lambda outer, inner: inner.end <= outer.end)


class IrValues:
    """Constant folding of a function's straight-line integer (and integer vector) instructions,
    for given argument values: resolves SCEV operands SCEV cannot see through, such as a scalar
    extracted from SLP-vectorized setup code."""

    BINOPS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "udiv": lambda a, b: a // b,
        "sdiv": lambda a, b: int(a / b),
        "lshr": lambda a, b: a >> b,
        "ashr": lambda a, b: a >> b,
        "shl": lambda a, b: a << b,
        "and": lambda a, b: a & b,
        "or": lambda a, b: a | b,
        "xor": lambda a, b: a ^ b,
    }
    FLAGS = {"nuw", "nsw", "exact", "disjoint"}

    def __init__(self, defs: dict[str, str]):
        self.defs = defs  # `%name` -> the instruction's right-hand side
        # Values SCEV cannot analyze but proves a small range of (`print<scalar-evolution>`).
        self.bounded: set[str] = set()

    @staticmethod
    def parse(ll: Path, ir_hint: str) -> IrValues:
        defs, in_fn = {}, False
        for line in ll.read_text().splitlines():
            if line.startswith("define "):
                in_fn = ir_hint in line.split("(")[0]
            elif in_fn and line.startswith("}"):
                in_fn = False
            elif in_fn and (m := re.match(r"^\s+(%[\w.]+) = (.*?)(?:, !.*)?$", line)):
                defs[m.group(1)] = m.group(2)
        return IrValues(defs)

    def value(self, name: str, env: dict[str, int], depth: int = 0):
        """Value of `%name` (int, or list of ints for a vector), or None."""
        if name in env:
            return env[name]
        rhs = self.defs.get(name)
        if rhs is None or depth > 64:
            return None
        try:
            return self._eval(rhs, env, depth + 1)
        except (ValueError, IndexError, ZeroDivisionError, TypeError):
            return None

    def _operand(self, t: str, env: dict[str, int], depth: int):
        t = t.strip()
        t = re.sub(r"^(<\d+ x i\d+>|i\d+)\s+", "", t)  # the operand's type
        if t.startswith("%"):
            return self.value(t, env, depth)
        if re.fullmatch(r"-?\d+", t):
            return int(t)
        if t in ("poison", "undef"):
            return 0
        if m := re.fullmatch(r"splat \(i\d+ (-?\d+)\)", t):
            return ("splat", int(m.group(1)))
        if m := re.fullmatch(r"<(.*)>", t):
            return [0 if x.strip().endswith("poison") else int(x.split()[-1]) for x in m.group(1).split(",")]
        raise ValueError(t)

    @staticmethod
    def _split(args: str) -> list[str]:
        """Split top-level commas (not inside `<...>` or `(...)`)."""
        out, depth, cur = [], 0, ""
        for ch in args:
            if ch in "<(":
                depth += 1
            elif ch in ">)":
                depth -= 1
            if ch == "," and depth == 0:
                out.append(cur)
                cur = ""
            else:
                cur += ch
        return out + [cur]

    def _eval(self, rhs: str, env: dict[str, int], depth: int):
        words = rhs.split()
        op = words[0]
        rest = " ".join(w for w in words[1:] if w not in self.FLAGS)
        if op in self.BINOPS:
            m = re.match(r"(<(\d+) x i\d+>|i\d+) (.*)$", rest)
            n = int(m.group(2)) if m.group(2) else None
            a, b = (self._operand(x, env, depth) for x in self._split(m.group(3)))

            def vec(v):
                if isinstance(v, tuple):  # splat
                    return [v[1]] * n
                return v if isinstance(v, list) else [v] * n

            f = self.BINOPS[op]
            if n is None:
                return None if a is None or b is None else f(a, b)
            a, b = vec(a), vec(b)
            return None if a is None or b is None else [f(x, y) for x, y in zip(a, b)]
        if op in ("zext", "sext", "trunc", "freeze"):
            m = re.match(r"(?:<\d+ x )?i\d+>? (\S+)", rest)
            return self._operand(m.group(1), env, depth)
        if op == "extractelement":
            v, i = self._split(re.sub(r"^<\d+ x i\d+> ", "", rest))
            v, i = self._operand(v, env, depth), self._operand(i, env, depth)
            return None if v is None or i is None else v[i]
        if op == "insertelement":
            m = re.match(r"<(\d+) x i\d+> (.*)$", rest)
            v, x, i = (self._operand(t, env, depth) for t in self._split(m.group(2)))
            v = [0] * int(m.group(1)) if not isinstance(v, list) else list(v)
            if x is None or i is None:
                return None
            v[i] = x
            return v
        if op == "shufflevector":
            m = re.match(r"<(\d+) x i\d+> (.*)$", rest)
            a, b, mask = self._split(m.group(2))
            a, b = self._operand(a, env, depth), self._operand(b, env, depth)
            n = int(m.group(1))
            a = a if isinstance(a, list) else [0] * n
            b = b if isinstance(b, list) else [0] * n
            mask = mask.strip()
            mask_len = int(re.match(r"<(\d+) x", mask).group(1))
            idx = [0] * mask_len if mask.endswith("zeroinitializer") else self._operand(mask, env, depth)
            ab = a + b
            return [ab[i] for i in idx]
        return None


def ir_block_sizes(ll: Path, ir_hint: str) -> dict[str, int]:
    """Instruction count of each basic block of the function whose name contains `ir_hint`."""
    sizes: dict[str, int] = {}
    in_fn, block = False, None
    for line in ll.read_text().splitlines():
        if line.startswith("define "):
            in_fn = ir_hint in line.split("(")[0]
            block = "start" if in_fn else None
        elif in_fn and line.startswith("}"):
            in_fn = False
        elif in_fn:
            if m := re.match(r"^([\w.$\-]+):", line):
                block = m.group(1)
            elif line.startswith("  ") and block is not None:
                sizes[block] = sizes.get(block, 0) + 1
    return sizes


def ir_loops(ll: Path, opt: str, ir_hint: str) -> list[Loop]:
    """IR loop tree of the function whose name contains `ir_hint`, with SCEV trip counts."""
    # opt prints analyses to stderr.
    p = subprocess.run(
        [opt, "-disable-output", "-passes=print<loops>,print<scalar-evolution>", str(ll)],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    text = p.stderr

    block_sizes = ir_block_sizes(ll, ir_hint)
    bounded: set[str] = set()
    cur_value = None
    fn_loops: list[tuple[int, Loop]] = []
    trips: dict[str, tuple[str, int | None]] = {}
    in_fn = False
    for line in text.splitlines():
        if m := re.match(r"Loop info for function '([^']+)'", line):
            in_fn = ir_hint in m.group(1)
        elif m := re.match(r"Classifying expressions for: @(\S+)", line):
            in_fn = ir_hint in m.group(1).strip('"')
        elif not in_fn:
            continue
        elif m := re.match(r"\s+(%[\w.]+) = ", line):
            cur_value = m.group(1)
        elif (m := re.match(r"\s+-->  (%[\w.]+) U: \[(-?\d+),(-?\d+)\)", line)) and m.group(1) == cur_value:
            lo, hi = int(m.group(2)), int(m.group(3))
            if 0 <= lo < hi <= lo + BOUNDED_RANGE:
                bounded.add(cur_value)
        elif m := re.match(r"( *)Loop at depth (\d+) containing: (.*)$", line):
            blocks = [re.sub(r"<[a-z]+>", "", b).lstrip("%") for b in m.group(3).split(",")]
            header = next(re.sub(r"<[a-z]+>", "", b).lstrip("%") for b in m.group(3).split(",") if "<header>" in b)
            lp = Loop(0, 0, header, size=sum(block_sizes.get(b, 0) for b in blocks))
            fn_loops.append((int(m.group(2)), lp))
        elif m := re.match(r"Loop %(\S+): (?:<multiple exits> )?(symbolic max )?backedge-taken count is (.*)$", line):
            # The exact count, else the symbolic max: loops whose other exits are panics (bounds
            # checks) have an unpredictable exact count, but run their max in the steady state.
            name, expr = m.group(1).strip('"'), m.group(3).strip()
            if m.group(2) and name in trips:
                continue
            c = re.fullmatch(r"i\d+ (\d+)", expr)
            trips[name] = (expr, int(c.group(1)) + 1 if c else None)

    values = IrValues.parse(ll, ir_hint)
    values.bounded = bounded
    for _, lp in fn_loops:
        lp.trip_expr, lp.trip_count = trips.get(lp.header, ("", None))
        lp.ir = values
    depth_of = {id(lp): d for d, lp in fn_loops}
    return build_tree([lp for _, lp in fn_loops], lambda outer, inner: depth_of[id(inner)] > depth_of[id(outer)])


def attach_trip_counts(asm_roots: list[Loop], ir_roots: list[Loop]) -> bool:
    """Copy trip counts from the IR loop tree onto the asm one; False if their shapes differ.

    Sibling loops are matched by shape, and siblings of the same shape by size rank: `opt` does not
    list loops in asm layout order (e.g. a main loop between an inlined tail and its vectorized
    version).
    """
    if sorted(r.shape() for r in asm_roots) != sorted(r.shape() for r in ir_roots):
        return False
    for shape in {r.shape() for r in asm_roots}:
        a_group = sorted((r for r in asm_roots if r.shape() == shape), key=lambda r: r.size)
        i_group = sorted((r for r in ir_roots if r.shape() == shape), key=lambda r: r.size)
        for a, i in zip(a_group, i_group):
            a.trip_count, a.trip_expr, a.ir = i.trip_count, i.trip_expr, i.ir
            if not attach_trip_counts(a.children, i.children):
                return False
    return True


def flatten(lines: list[str], loop: Loop) -> list[str] | None:
    """Straight-line instructions of one iteration of `loop`, inner loops unrolled by trip count."""
    out: list[str] = []
    i = loop.start
    children = sorted(loop.children, key=lambda c: c.start)
    while i <= loop.end:
        child = next((c for c in children if c.start == i), None)
        if child is not None:
            if child.trip_count is None:
                return None
            body = flatten(lines, child)
            if body is None:
                return None
            out += body * child.trip_count
            i = child.end + 1
            continue
        if not is_label(lines[i]):
            out.append(lines[i])
        i += 1
    return out


# Placeholders for SCEV operands that are neither arguments nor resolvable by `IrValues` (e.g. a
# bound derived from the bounds checks of an outer iteration, the size of a vector loop's scalar
# remainder, or the index a remainder loop starts at).
BOUNDED = "B"  # SCEV proves a small range: e.g. a remainder size, `U: [0,16)`
UNBOUNDED = "X"
BOUNDED_RANGE = 1 << 12


def eval_scev(expr: str, env: dict[str, int], ir: IrValues | None = None) -> tuple[int | None, bool]:
    """Evaluate a SCEV backedge-taken count expression, with the arguments bound by `env`.

    Supports `+ - * /u` and the n-ary `umax umin smax smin`, which SCEV always parenthesizes, so
    innermost parenthesized groups are evaluated first.

    Unknown operands are approximated where that is exact up to a bounded term: dropped from a
    min/max (bounds-check limits, which do not bind in the steady state), and taken as 0 in a sum
    when SCEV bounds their range (a vector loop's `(n - remainder) /u VF`). Otherwise the count is
    unknown (None): e.g. a remainder loop, whose count is `n - (unknown start)`. Returns
    (value, approximated).
    """
    e = re.sub(r"<[a-z]+>", "", expr)  # wrap flags: <nuw>, <nsw>, <nw>
    e = e.replace("/u", "//")
    # An add recurrence of an enclosing loop, `{start,+,step}<%loop>`: its value at the first
    # iteration (the loops this is meant for run the same count at every outer iteration).
    approx = False
    while (m := re.search(r"\{([^{}]*),\+,[^{}]*\}<%[\w.]+>", e)) is not None:
        e = e[: m.start()] + f"({m.group(1)})" + e[m.end() :]
        approx = True

    def value(m: re.Match) -> str:
        nonlocal approx
        v = env.get(m.group(0))
        if v is None and ir is not None:
            v = ir.value(m.group(0), env)
        if isinstance(v, int):
            return str(v)
        approx = True
        return BOUNDED if ir is not None and m.group(0) in ir.bounded else UNBOUNDED

    e = re.sub(r"%[\w.]+", value, e)
    # SCEV integers are 64-bit, and umin/umax//u are unsigned: compute modulo 2^64 (the
    # placeholders stay symbolic).
    mask = (1 << 64) - 1

    def signed(v: int) -> int:
        return v - (1 << 64) if v >> 63 else v

    minmax = {
        "umax": lambda vs: max(vs),
        "umin": lambda vs: min(vs),
        "smax": lambda vs: max(vs, key=signed),
        "smin": lambda vs: min(vs, key=signed),
    }
    placeholders = (BOUNDED, UNBOUNDED)

    def arith(t: str):
        t = t.strip()
        if t in placeholders:
            return t
        if re.fullmatch(r"-?\d+", t):
            return int(t) & mask
        for op, f in minmax.items():
            if f" {op} " in t:
                vals = [arith(x) for x in t.split(f" {op} ")]
                if None in vals:
                    return None
                known = [v for v in vals if v not in placeholders]
                return f(known) if known else UNBOUNDED
        if " + " in t:
            vals = [arith(x) for x in t.split(" + ")]
            if None in vals or UNBOUNDED in vals:
                return None
            return sum(v for v in vals if v != BOUNDED) & mask
        for op in (" * ", " // "):
            if op in t:
                vals = [arith(x) for x in t.split(op)]
                if None in vals:
                    return None
                if any(v in placeholders for v in vals):
                    # A product or quotient of an unknown, e.g. `-1 * %rem`: as bounded as it.
                    return UNBOUNDED if UNBOUNDED in vals else BOUNDED
                acc = vals[0]
                for v in vals[1:]:
                    if op == " * ":
                        acc = (acc * v) & mask
                    elif v == 0:
                        return None
                    else:
                        acc //= v
                return acc
        return None

    while (m := re.search(r"\(([^()]*)\)", e)) is not None:
        v = arith(m.group(1))
        if v is None:
            return None, approx
        e = e[: m.start()] + str(v) + e[m.end() :]
    v = arith(e)
    return (signed(v) if isinstance(v, int) else None), approx


def trips(lp: Loop, env: dict[str, int]) -> tuple[int | None, bool]:
    """Trip count of `lp` (per entry), with the arguments bound by `env`."""
    if lp.trip_count is not None:
        return lp.trip_count, False
    b, approx = eval_scev(lp.trip_expr, env, lp.ir)
    return (None if b is None or b < 0 else b + 1), approx


@dataclass
class HotLoop:
    loop: Loop
    bytes_per_iter: float  # input bytes per iteration, over the whole kernel
    approx: bool  # trip count approximated (see `eval_scev`)


def hot_loops(roots: list[Loop], kernel: Kernel) -> list[HotLoop]:
    """The loops whose total iteration count is linear in the input length: the steady state.

    A loop's total iteration count is its trip count times its ancestors' (evaluated for two input
    lengths). Loops whose count does not scale with the input (tails, loops over the itemsize) are
    ignored; the descendants of a hot loop are part of its (flattened) body.
    """
    env1, env2 = kernel.scev_env(SCEV_EVAL_LEN), kernel.scev_env(2 * SCEV_EVAL_LEN)
    out: list[HotLoop] = []

    def walk(lp: Loop, mult1: int, mult2: int, approx: bool) -> None:
        t1, a1 = trips(lp, env1)
        t2, a2 = trips(lp, env2)
        if t1 is None or t2 is None:
            return
        n1, n2, a = mult1 * t1, mult2 * t2, approx or a1 or a2
        # Linear up to a constant (e.g. a vector loop's `(n - remainder) / VF`): the bytes per
        # iteration are the slope between the two lengths.
        if n2 > n1 > 0 and abs(2 * n1 - n2) * 64 <= n1:
            out.append(HotLoop(lp, SCEV_EVAL_LEN / (n2 - n1), a))
            return
        for c in lp.children:
            walk(c, n1, n2, a)

    for r in roots:
        walk(r, 1, 1, False)
    return out


def innermost(loops: list[Loop]) -> list[Loop]:
    return [x for lp in loops for x in (innermost(lp.children) if lp.children else [lp])]


# --------------------------------------------------------------------------------------------------
# llvm-mca


@dataclass
class McaResult:
    cpu: str
    iterations: int
    instructions: int
    total_cycles: int
    uops: int
    block_rthroughput: float
    # Instructions llvm-mca has no scheduling info for (on this CPU), simulated as free.
    skipped: list[str] = field(default_factory=list)

    @property
    def cycles_per_iter(self) -> float:
        return self.total_cycles / self.iterations


def run_mca(mca: str, platform: Platform, cpu: str, trace: list[str], all_lines: list[str]) -> tuple[McaResult, str]:
    # llvm-mca does not follow branches, but their target labels must exist: define them as
    # stubs after the trace.
    # Call targets can be demangled Rust paths, which the assembler cannot parse: point direct
    # calls at a stub symbol instead.
    trace = [
        f"{mnemonic(x)} __probe_callee" if is_call(platform.isa, mnemonic(x)) and "*" not in x and "%" not in x else x
        for x in trace
    ]
    all_labels = {m.group(1) for line in all_lines if (m := LABEL_RE.match(line))} | {"__probe_callee"}
    referenced = {tok for line in trace for tok in TOKEN_RE.findall(line)[1:]} & all_labels
    src = "\n".join(trace + [f"{lab}:" for lab in sorted(referenced)]) + "\n"
    if platform.isa == "x86":
        src = ".att_syntax\n" + src
    cmd = [
        mca,
        f"-mtriple={platform.target}",
        f"-mcpu={cpu}",
        f"-iterations={MCA_ITERATIONS}",
        "-bottleneck-analysis",
        "-skip-unsupported-instructions=lack-sched",
    ]
    p = subprocess.run(cmd, input=src, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    out = p.stdout
    skipped = sorted({" ".join(m.split()) for m in re.findall(r"note: instruction:\s*(.*)", p.stderr)})

    def num(key: str) -> str:
        return re.search(rf"^{key}:\s*([\d.]+)", out, re.M).group(1)

    res = McaResult(
        cpu=cpu,
        iterations=int(num("Iterations")),
        instructions=int(num("Instructions")),
        total_cycles=int(num("Total Cycles")),
        uops=int(num("Total uOps")),
        block_rthroughput=float(num("Block RThroughput")),
        skipped=skipped,
    )
    return res, out


# --------------------------------------------------------------------------------------------------
# Driver


@dataclass
class KernelResult:
    platform: str
    kernel: str
    fn_instructions: int
    # `outer`: the hot loops, each one iteration flattened. `inner`: the largest innermost loop,
    # per iteration (informational, when there is no flattenable hot loop).
    mode: str
    trace_instructions: str  # per hot loop, `+`-joined
    calls: list[str]  # callees inside the hot loops
    loop_tree: str
    bytes_per_iter: str  # per hot loop, `+`-joined; `~`: approximated trip count
    warnings: list[str] = field(default_factory=list)
    mca: list[McaResult] = field(default_factory=list)  # per (hot loop, cpu)
    # `outer` mode: per cpu, cycles per BYTES_UNIT: sum over the hot loops of cycles per iteration
    # / bytes per iteration.
    costs: list[float] = field(default_factory=list)


def describe_tree(loops: list[Loop]) -> str:
    def one(lp: Loop) -> str:
        t = "?" if lp.trip_count is None else str(lp.trip_count)
        return f"{t}" + (f"[{' '.join(one(c) for c in lp.children)}]" if lp.children else "")

    return " ".join(one(lp) for lp in loops)


def analyze_kernel(
    platform: Platform, kernel: Kernel, ll: Path, opt: str, mca: str, target_dir: Path, out_dir: Path
) -> KernelResult:
    asm = cargo_asm(platform, kernel.symbol, target_dir)
    asm_path = out_dir / "asm" / platform.name / f"{kernel.name}.s"
    asm_path.parent.mkdir(parents=True, exist_ok=True)
    asm_path.write_text(asm)

    lines = clean_lines(asm, platform.comment)
    fn_instrs = sum(1 for line in lines if not is_label(line))
    roots = asm_loops(lines, platform.isa)
    warnings = []
    if not attach_trip_counts(roots, ir_loops(ll, opt, kernel.ir_hint)):
        warnings.append("asm and IR loop trees differ, trip counts unknown")

    hot = hot_loops(roots, kernel)
    traces = [flatten(lines, h.loop) for h in hot]
    if not hot:
        warnings.append("no loop whose iteration count is linear in the input length")
    elif None in traces:
        warnings.append("hot loop with an inner loop of unknown trip count")
    if hot and None not in traces:
        mode = "outer"
        fmt = [("~" if h.approx else "") + f"{h.bytes_per_iter:g}" for h in hot]
    else:
        mode, fmt = "inner", []
        inner = innermost(roots)
        traces = []
        if inner:
            best = max(inner, key=lambda lp: sum(1 for x in lines[lp.start : lp.end + 1] if not is_label(x)))
            traces = [[x for x in lines[best.start : best.end + 1] if not is_label(x)]]

    all_instrs = [x for t in traces for x in t]
    r = KernelResult(
        platform.name,
        kernel.name,
        fn_instrs,
        mode,
        "+".join(str(len(t)) for t in traces),
        sorted({x.split(None, 1)[-1] for x in all_instrs if is_call(platform.isa, mnemonic(x))}),
        describe_tree(roots),
        "+".join(fmt),
        warnings,
    )
    for w in warnings:
        print(f"warning: {platform.name}/{kernel.name}: {w}", file=sys.stderr)
    if not traces:
        return r
    for cpu in platform.mca_cpus:
        cost = 0.0
        for k, trace in enumerate(traces):
            m, report = run_mca(mca, platform, cpu, trace, lines)
            r.mca.append(m)
            if m.skipped:
                r.warnings.append(f"llvm-mca -mcpu={cpu} skipped (no sched info): {', '.join(m.skipped)}")
            if mode == "outer":
                cost += m.cycles_per_iter * BYTES_UNIT / hot[k].bytes_per_iter
            suffix = f".{k}" if len(traces) > 1 else ""
            p = out_dir / "mca" / platform.name / f"{kernel.name}{suffix}.{cpu}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(report)
        r.costs.append(cost)
    return r


def analyze_platform(
    platform: Platform, kernels: list[Kernel], opt: str, mca: str, out_dir: Path
) -> list[KernelResult]:
    # One target dir per feature set: the IR file is picked by mtime, and cargo does not rewrite
    # an up-to-date one, so builds with other RUSTFLAGS must not share the directory.
    flags = hashlib.sha1(" ".join(platform.rustflags).encode()).hexdigest()[:8]
    target_dir = PROBE_DIR / "target" / "analyze" / f"{platform.name}-{flags}"
    # Emits the whole crate's final LLVM IR next to the build artifacts.
    cargo_asm(platform, kernels[0].symbol, target_dir, llvm=True)
    ll = max(target_dir.glob(f"{platform.target}/release/**/jix_probe-*.ll"), key=lambda p: p.stat().st_mtime)
    return [analyze_kernel(platform, k, ll, opt, mca, target_dir, out_dir) for k in kernels]


def load_costs(label: str) -> dict[tuple[str, str, str], float]:
    """(platform, cpu, kernel) -> cycles per BYTES_UNIT, from a previous run's summary.json."""
    costs = {}
    for r in json.loads((PROBE_DIR / "results" / label / "summary.json").read_text()):
        if r["mode"] != "outer":
            continue
        if "costs" in r:
            cpus = list(dict.fromkeys(m["cpu"] for m in r["mca"]))
            for cpu, c in zip(cpus, r["costs"]):
                costs[(r["platform"], cpu, r["kernel"])] = c
        else:  # older results: a single hot loop
            for m in r["mca"]:
                costs[(r["platform"], m["cpu"], r["kernel"])] = m["cycles_per_iter"] * BYTES_UNIT / r["bytes_per_iter"]
    return costs


def write_summary(
    results: list[KernelResult],
    platforms: list[Platform],
    out_dir: Path,
    header: list[str],
    compare: str | None,
    direction: str,
) -> str:
    lines = [
        f"# llvm-mca summary: `{out_dir.name}` ({direction} kernels)",
        "",
        *header,
        "",
        f"llvm-mca steady-state **cycles per {BYTES_UNIT} bytes**: the hot loops (those whose total",
        "iteration count is linear in the input length), one iteration each with inner loops flattened",
        "by their trip counts, weighted by their iterations per byte. Lower is better. `geomean` is the",
        "per-CPU geometric mean over the kernels, the single number to optimize.",
        "",
        "- `B/iter`: input bytes per hot-loop iteration (from the SCEV trip counts); `~`: the trip count",
        "  depends on non-argument values (e.g. bounds-check limits), approximated up to a bounded term.",
        "- `instrs`: instructions per hot-loop iteration (flattened trace).",
        "- `loops`: asm loop tree with trip counts; `?` = runtime trip count.",
        "- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include. Values of such",
        "  kernels (and geomeans including them) are flagged `*`: the real cost is higher.",
        "",
    ]
    by_platform: dict[str, list[KernelResult]] = {}
    for r in results:
        by_platform.setdefault(r.platform, []).append(r)

    def cost(r: KernelResult, i: int) -> float:
        return r.costs[i]

    def star(rs: list[KernelResult]) -> str:
        return "*" if any(r.calls for r in rs) else ""

    def geo(xs: list[float]) -> float:
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    def geomean(rs: list[KernelResult], i: int) -> str:
        return f"{geo([cost(r, i) for r in rs]):.0f}{star(rs)}"

    base = load_costs(compare) if compare else {}

    def speedup(platform: str, cpu: str, rs: list[KernelResult], i: int) -> str:
        old = [base.get((platform, cpu, r.kernel)) for r in rs]
        if None in old:
            return "-"
        return f"{geo(old) / geo([cost(r, i) for r in rs]):.2f}x"

    fixed_names = list(dict.fromkeys(r.kernel for r in results if r.mode == "outer"))
    lines += [
        "## Overview",
        "",
        "| platform | cpu | geomean | " + (f"vs {compare} | " if compare else "") + " | ".join(fixed_names) + " |",
        "|---|---|---:|" + ("---:|" if compare else "") + "---:|" * len(fixed_names),
    ]
    for platform in platforms:
        fixed = [r for r in by_platform.get(platform.name, []) if r.mode == "outer"]
        if not fixed:
            continue
        for i, cpu in enumerate(platform.mca_cpus):
            cells = [f"{cost(r, i):.0f}{star([r])}" for r in fixed]
            vs = f"{speedup(platform.name, cpu, fixed, i)} | " if compare else ""
            lines.append(f"| {platform.name} | {cpu} | **{geomean(fixed, i)}** | {vs}" + " | ".join(cells) + " |")
    lines.append("")

    for platform in platforms:
        rs = by_platform.get(platform.name, [])
        cpus = platform.mca_cpus
        fixed = [r for r in rs if r.mode == "outer"]
        other = [r for r in rs if r.mode != "outer"]
        lines += [f"## {platform.name} (`{platform.target}`)", ""]
        if platform.rustflags:
            lines += [f"`RUSTFLAGS={' '.join(platform.rustflags)}`", ""]
        if fixed:
            lines.append("| kernel | B/iter | instrs | loops | calls | " + " | ".join(cpus) + " |")
            lines.append("|---|---:|---:|---|---|" + "---:|" * len(cpus))
            for r in fixed:
                cells = [f"{cost(r, i):.0f}{star([r])}" for i in range(len(cpus))]
                calls = ", ".join(f"`{c}`" for c in r.calls) or "-"
                lines.append(
                    f"| {r.kernel} | {r.bytes_per_iter} | {r.trace_instructions} | {r.loop_tree} | {calls} | "
                    + " | ".join(cells)
                    + " |"
                )
            geos = [geomean(fixed, i) for i in range(len(cpus))]
            lines.append("| **geomean** | | | | | " + " | ".join(f"**{g}**" for g in geos) + " |")
            lines.append("")
        if other:
            lines += [
                "Kernels without a flattenable hot loop, cycles per iteration of the largest innermost "
                "loop (informational):",
                "",
            ]
            lines.append("| kernel | loop instrs | loops | calls | " + " | ".join(cpus) + " |")
            lines.append("|---|---:|---|---|" + "---:|" * len(cpus))
            for r in other:
                cells = [f"{m.cycles_per_iter:.1f}" for m in r.mca] or ["-"] * len(cpus)
                calls = ", ".join(f"`{c}`" for c in r.calls) or "-"
                lines.append(
                    f"| {r.kernel} | {r.trace_instructions} | {r.loop_tree} | {calls} | " + " | ".join(cells) + " |"
                )
            lines.append("")
        warns = [f"- {r.kernel}: {w}" for r in rs for w in r.warnings]
        if warns:
            lines += ["Warnings:", "", *warns, ""]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    data = [
        {
            **{k: v for k, v in r.__dict__.items() if k != "mca"},
            "mca": [{**m.__dict__, "cycles_per_iter": m.cycles_per_iter} for m in r.mca],
        }
        for r in results
    ]
    (out_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="baseline", help="results sub-directory name")
    ap.add_argument("--platform", action="append", help="restrict to these platforms (repeatable)")
    ap.add_argument("--compare", help="also report the geomean speedup over this results label")
    ap.add_argument("--direction", choices=sorted(KERNELS), default="decode", help="kernels to analyze")
    ap.add_argument(
        "--x86-levels",
        choices=["fearless", "multiversion"],
        default="fearless",
        help="x86-64-v2/v3/v4 feature sets: fearless_simd's dispatch levels, or the multiversion clones",
    )
    ap.add_argument("--fn", action="append", dest="kernels", help="restrict to these kernels (repeatable)")
    args = ap.parse_args()

    platforms = [p for p in PLATFORMS if not args.platform or p.name in args.platform]
    kernels = [k for k in KERNELS[args.direction] if not args.kernels or k.name in args.kernels]
    if args.x86_levels == "multiversion":
        for p in platforms:
            if p.name in MULTIVERSION_FEATURES:
                p.rustflags = [f"-Ctarget-feature={MULTIVERSION_FEATURES[p.name]}"]
    if not platforms or not kernels:
        sys.exit("error: nothing selected")

    mca = find_llvm_mca(llvm_major_of_rustc())
    opt = find_opt()
    mca_version = re.search(r"LLVM version [\d.]+", run([mca, "--version"])).group(0)
    header = [
        f"- rustc: `{run(['rustc', '--version'], cwd=PROBE_DIR).strip()}`",
        f"- llvm-mca: `{mca_version}`",
    ]

    out_dir = PROBE_DIR / "results" / args.label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = [ex.submit(analyze_platform, p, kernels, opt, mca, out_dir) for p in platforms]
        results = [r for f in futures for r in f.result()]

    print(write_summary(results, platforms, out_dir, header, args.compare, args.direction))


if __name__ == "__main__":
    main()
