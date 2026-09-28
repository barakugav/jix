#!/usr/bin/env python3
"""Static performance analysis of the byte-shuffle kernels: cargo-asm + llvm-mca, per platform.

For every platform (target triple + ISA level) the probe crate is compiled in release mode, and for
every kernel:

1. `cargo asm` extracts the function's asm, and the asm loop tree is recovered from back-edges.
2. `cargo asm --llvm` emits the final (post-optimization) LLVM IR, and LLVM's own `opt` reports the
   IR loop tree (`print<loops>`) and each loop's exact trip count (`print<scalar-evolution>`).
3. The two loop trees are matched by shape, which gives every asm loop its trip count.
4. The bytes processed per outer-loop iteration are derived from the outer loop's SCEV trip count
   (evaluated for a concrete input length), so kernels may use any step (e.g. 512-bit variants
   processing more than 128 bytes per iteration, or loops LLVM unrolled).
5. One iteration of the outer loop is flattened into a straight-line trace (inner loops repeated
   by their constant trip counts), and llvm-mca simulates that trace in a steady state on each
   representative CPU of the platform.

The objective is llvm-mca's **cycles per 4096 bytes** (= cycles per outer iteration * 4096 /
bytes per iteration). Lower is better. It is a static model: memory is assumed to hit L1, calls
(e.g. to `memcpy`) are not followed - the `calls` column flags kernels whose hot loop still calls
something - and the per-call prologue/epilogue and the `<LANES` tail are not included.

The runtime-itemsize fallback kernels (`*_generic`) have runtime trip counts, so they cannot be
flattened; for them only the cycles per iteration of the largest innermost loop are reported, as
an informational number.

Usage (from anywhere):
    python jix/probe/analyze.py [--label baseline] [--platform x86_64-v3 ...] [--fn decode_4 ...]

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
    fixed: bool  # fixed itemsize (flattened, the objective) vs runtime itemsize (informational)


def _kernel(direction: str, itemsize: int) -> Kernel:
    sym = f"probe_byte_shuffle_{direction}_{itemsize}"
    return Kernel(f"{direction}_{itemsize}", sym, sym, True)


# Main loops of the fixed-itemsize kernels, per direction (`--direction`). Must match the
# `probe_*` exports in src/lib.rs.
KERNELS = {d: [_kernel(d, s) for s in (2, 4, 8, 16)] for d in ("decode", "encode")}

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
        elif m := re.match(r"( *)Loop at depth (\d+) containing: (.*)$", line):
            blocks = [re.sub(r"<[a-z]+>", "", b).lstrip("%") for b in m.group(3).split(",")]
            header = next(re.sub(r"<[a-z]+>", "", b).lstrip("%") for b in m.group(3).split(",") if "<header>" in b)
            lp = Loop(0, 0, header, size=sum(block_sizes.get(b, 0) for b in blocks))
            fn_loops.append((int(m.group(2)), lp))
        elif m := re.match(r"Loop %(\S+): backedge-taken count is (.*)$", line):
            expr = m.group(2).strip()
            c = re.fullmatch(r"i\d+ (\d+)", expr)
            trips[m.group(1).strip('"')] = (expr, int(c.group(1)) + 1 if c else None)

    for _, lp in fn_loops:
        lp.trip_expr, lp.trip_count = trips.get(lp.header, ("", None))
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
            a.trip_count, a.trip_expr = i.trip_count, i.trip_expr
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


def eval_scev(expr: str, length: int) -> int | None:
    """Evaluate a SCEV backedge-taken count expression of the input length (`%src.1`).

    Supports `+ - * /u` and the n-ary `umax umin smax smin`, which SCEV always parenthesizes, so
    innermost parenthesized groups are evaluated first.
    """
    e = re.sub(r"<[a-z]+>", "", expr)  # wrap flags: <nuw>, <nsw>, <nw>
    e = e.replace("/u", "//").replace("%src.1", str(length)).replace("%dst.1", str(length))
    minmax = {"umax": max, "umin": min, "smax": max, "smin": min}

    def arith(t: str) -> int | None:
        for op, f in minmax.items():
            if f" {op} " in t:
                vals = [arith(x) for x in t.split(f" {op} ")]
                return None if None in vals else f(vals)
        if not re.fullmatch(r"[\d+\-*/ ]+", t):
            return None
        try:
            return eval(t, {"__builtins__": {}})  # noqa: S307 - validated above
        except (SyntaxError, ZeroDivisionError):
            return None

    while (m := re.search(r"\(([^()]*)\)", e)) is not None:
        v = arith(m.group(1))
        if v is None:
            return None
        e = e[: m.start()] + str(v) + e[m.end() :]
    return arith(e)


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
    # `outer`: one flattened outer iteration. `inner`: the largest innermost loop, per iteration
    # (runtime-itemsize kernels, or when flattening failed).
    mode: str
    trace_instructions: int
    calls: list[str]  # callees inside the hot loop
    loop_tree: str
    bytes_per_iter: int | None
    warnings: list[str] = field(default_factory=list)
    mca: list[McaResult] = field(default_factory=list)


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

    trace, mode, bytes_per_iter = None, "inner", None
    if kernel.fixed and roots:
        # The outer `while` loop over LANES items: the top-level loop with a runtime trip count.
        # The main loop: of the top-level loops with a runtime trip count (the main loop, plus
        # e.g. an inlined scalar tail), the one processing the most bytes per iteration.
        candidates = []
        for lp in roots:
            if lp.trip_count is None:
                # The main loop's trip count is linear in the input length; a tail's is not.
                b1 = eval_scev(lp.trip_expr, SCEV_EVAL_LEN)
                b2 = eval_scev(lp.trip_expr, 2 * SCEV_EVAL_LEN)
                if b1 is not None and b2 is not None and b1 >= 0 and 2 * (b1 + 1) == b2 + 1:
                    if SCEV_EVAL_LEN % (b1 + 1) == 0:
                        candidates.append((SCEV_EVAL_LEN // (b1 + 1), lp))
        if not candidates:
            exprs = [lp.trip_expr for lp in roots if lp.trip_count is None]
            warnings.append(f"no top-level loop with a derivable bytes per iteration: {exprs!r}")
        else:
            bytes_per_iter, main_loop = max(candidates, key=lambda c: c[0])
            trace = flatten(lines, main_loop)
            mode = "outer"
            if trace is None:
                warnings.append("inner loop with unknown trip count")
    if trace is None:
        mode, bytes_per_iter = "inner", None
        inner = innermost(roots)
        if inner:
            best = max(inner, key=lambda lp: sum(1 for x in lines[lp.start : lp.end + 1] if not is_label(x)))
            trace = [x for x in lines[best.start : best.end + 1] if not is_label(x)]

    r = KernelResult(
        platform.name,
        kernel.name,
        fn_instrs,
        mode,
        len(trace or []),
        sorted({x.split(None, 1)[-1] for x in trace or [] if is_call(platform.isa, mnemonic(x))}),
        describe_tree(roots),
        bytes_per_iter,
        warnings,
    )
    for w in warnings:
        print(f"warning: {platform.name}/{kernel.name}: {w}", file=sys.stderr)
    if not trace:
        return r
    for cpu in platform.mca_cpus:
        m, report = run_mca(mca, platform, cpu, trace, lines)
        r.mca.append(m)
        if m.skipped:
            r.warnings.append(f"llvm-mca -mcpu={cpu} skipped (no sched info): {', '.join(m.skipped)}")
        p = out_dir / "mca" / platform.name / f"{kernel.name}.{cpu}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report)
    return r


def analyze_platform(
    platform: Platform, kernels: list[Kernel], opt: str, mca: str, out_dir: Path
) -> list[KernelResult]:
    target_dir = PROBE_DIR / "target" / "analyze" / platform.name
    # Emits the whole crate's final LLVM IR next to the build artifacts.
    cargo_asm(platform, kernels[0].symbol, target_dir, llvm=True)
    ll = max(target_dir.glob(f"{platform.target}/release/**/jix_probe-*.ll"), key=lambda p: p.stat().st_mtime)
    return [analyze_kernel(platform, k, ll, opt, mca, target_dir, out_dir) for k in kernels]


def load_costs(label: str) -> dict[tuple[str, str, str], float]:
    """(platform, cpu, kernel) -> cycles per BYTES_UNIT, from a previous run's summary.json."""
    costs = {}
    for r in json.loads((PROBE_DIR / "results" / label / "summary.json").read_text()):
        if r["mode"] == "outer":
            for m in r["mca"]:
                costs[(r["platform"], m["cpu"], r["kernel"])] = m["cycles_per_iter"] * BYTES_UNIT / r["bytes_per_iter"]
    return costs


def write_summary(
    results: list[KernelResult], platforms: list[Platform], out_dir: Path, header: list[str], compare: str | None
) -> str:
    lines = [
        f"# Byte-shuffle llvm-mca summary: `{out_dir.name}`",
        "",
        *header,
        "",
        f"Fixed-itemsize decode kernels: llvm-mca steady-state **cycles per {BYTES_UNIT} bytes** (one",
        "outer-loop iteration, inner loops flattened by their trip counts, scaled by the bytes per",
        "iteration). Lower is better. `geomean` is the per-CPU geometric mean over the fixed-itemsize",
        "kernels, the single number to optimize.",
        "",
        "- `B/iter`: bytes per outer-loop iteration (from the SCEV trip count).",
        "- `instrs`: instructions per outer-loop iteration (flattened trace).",
        "- `loops`: asm loop tree with trip counts; `?` = runtime trip count (the outer loop).",
        "- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include. Values of such",
        "  kernels (and geomeans including them) are flagged `*`: the real cost is higher.",
        "",
    ]
    by_platform: dict[str, list[KernelResult]] = {}
    for r in results:
        by_platform.setdefault(r.platform, []).append(r)

    def cost(r: KernelResult, i: int) -> float:
        return r.mca[i].cycles_per_iter * BYTES_UNIT / r.bytes_per_iter

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
                "Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost "
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

    print(write_summary(results, platforms, out_dir, header, args.compare))


if __name__ == "__main__":
    main()
