#!/usr/bin/env python3
"""Static performance analysis of the byte-shuffle kernels: cargo-asm + llvm-mca, per platform.

For every platform (target triple + ISA level) the probe crate is compiled in release mode, and for
every kernel:

1. `cargo asm` extracts the function's asm, and the asm loop tree is recovered from back-edges.
2. `cargo asm --llvm` emits the final (post-optimization) LLVM IR, and LLVM's own `opt` reports the
   IR loop tree (`print<loops>`) and each loop's exact trip count (`print<scalar-evolution>`).
3. The two loop trees are matched by shape, which gives every asm loop its trip count.
4. One iteration of the kernel's outer loop - the `while` loop over LANES items, i.e. 128 bytes for
   every fixed-itemsize kernel - is flattened into a straight-line trace (inner loops repeated by
   their constant trip counts), and llvm-mca simulates that trace in a steady state on each
   representative CPU of the platform.

The objective is llvm-mca's **cycles per 128 bytes** (= cycles per outer iteration). Lower is
better. It is a static model: memory is assumed to hit L1, and calls (e.g. to `memcpy`) are not
followed - the `calls` column flags kernels whose hot loop still calls something.

The runtime-itemsize fallback kernels (`*_generic`) have runtime trip counts, so they cannot be
flattened; for them only the cycles per iteration of the largest innermost loop are reported, as
an informational number.

Usage (from anywhere):
    python jix/probe/analyze.py [--label baseline] [--platform x86_64-v3 ...] [--fn encode_4 ...]

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

# x86-64 microarchitecture levels, exactly as enabled by the `multiversion` clones in
# jix/src/codec/filter/byte_shuffle/kernels.rs (target features only, generic tuning).
X86_V2 = "+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b"
X86_V3 = X86_V2 + ",+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave"
X86_V4 = X86_V3 + ",+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl"


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
    # x86-64 v1 (SSE2) runs on modern CPUs only when `multiversion` is disabled (or on pre-2009
    # CPUs); with multiversion (the default) modern CPUs take the v2/v3/v4 clones.
    Platform("x86_64-v1", "x86_64-unknown-linux-gnu", [], ["x86-64", "skylake"]),
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
    # armv7 linux wheels: the gnueabihf baseline has VFPv3-D16 and *no* NEON.
    Platform("armv7", "armv7-unknown-linux-gnueabihf", [], ["cortex-a9", "cortex-a57"]),
    # ppc64le linux wheels: POWER8 baseline (VSX / AltiVec).
    Platform("ppc64le", "powerpc64le-unknown-linux-gnu", [], ["pwr8", "pwr9"]),
]


@dataclass
class Kernel:
    name: str
    symbol: str  # demangled, as `cargo asm` lists it
    ir_hint: str  # substring of the v0-mangled symbol, to find the function in the LLVM IR
    lanes: int | None  # items per outer iteration; None for the runtime-itemsize kernels


def _fixed_kernel(direction: str, itemsize: int) -> Kernel:
    lanes = 128 // itemsize
    return Kernel(
        f"{direction}_{itemsize}",
        f"jix_probe::byte_shuffle::{direction}_impl::<{itemsize}, {lanes}>",
        f"11{direction}_implKj{itemsize:x}_Kj{lanes:x}_",
        lanes,
    )


KERNELS = [
    *[_fixed_kernel(d, s) for d in ("encode", "decode") for s in (2, 4, 8, 16)],
    Kernel("encode_generic", "jix_probe::byte_shuffle::encode_impl_generic", "19encode_impl_generic", None),
    Kernel("decode_generic", "jix_probe::byte_shuffle::decode_impl_generic", "19decode_impl_generic", None),
]


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
    loops = sorted((Loop(labels[h], e, h) for h, e in ends.items()), key=lambda lp: (lp.start, -lp.end))
    for a in loops:
        for b in loops:
            if a.start < b.start <= a.end < b.end:
                raise RuntimeError(f"overlapping asm loops {a.header} and {b.header}")
    return build_tree(loops, lambda outer, inner: inner.end <= outer.end)


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
        elif m := re.match(r"( *)Loop at depth (\d+) containing: .*?%([^,<]+)<header>", line):
            fn_loops.append((int(m.group(2)), Loop(0, 0, m.group(3))))
        elif m := re.match(r"Loop %(\S+): backedge-taken count is (.*)$", line):
            expr = m.group(2).strip()
            c = re.fullmatch(r"i\d+ (\d+)", expr)
            trips[m.group(1).strip('"')] = (expr, int(c.group(1)) + 1 if c else None)

    for _, lp in fn_loops:
        lp.trip_expr, lp.trip_count = trips.get(lp.header, ("", None))
    depth_of = {id(lp): d for d, lp in fn_loops}
    return build_tree([lp for _, lp in fn_loops], lambda outer, inner: depth_of[id(inner)] > depth_of[id(outer)])


def attach_trip_counts(asm_roots: list[Loop], ir_roots: list[Loop]) -> bool:
    """Copy trip counts from the IR loop tree onto the asm one; False if their shapes differ."""
    if tuple(r.shape() for r in asm_roots) != tuple(r.shape() for r in ir_roots):
        return False
    for a, i in zip(asm_roots, ir_roots):
        a.trip_count, a.trip_expr = i.trip_count, i.trip_expr
        attach_trip_counts(a.children, i.children)
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
    all_labels = {m.group(1) for line in all_lines if (m := LABEL_RE.match(line))}
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
    # `outer`: one flattened outer iteration (= 128 bytes). `inner`: the largest innermost loop,
    # per iteration (runtime-itemsize kernels, or when flattening failed).
    mode: str
    trace_instructions: int
    calls: int
    loop_tree: str
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

    trace, mode = None, "inner"
    if kernel.lanes is not None and roots:
        # The outer `while` loop over LANES items: the top-level loop with a runtime trip count.
        outer = [lp for lp in roots if lp.trip_count is None]
        if len(outer) != 1:
            warnings.append(f"expected one top-level loop with runtime trip count, found {len(outer)}")
        else:
            if f"/u {kernel.lanes})" not in outer[0].trip_expr:
                warnings.append(f"outer loop may not step by LANES={kernel.lanes}: {outer[0].trip_expr!r}")
            trace = flatten(lines, outer[0])
            mode = "outer"
            if trace is None:
                warnings.append("inner loop with unknown trip count")
    if trace is None:
        mode = "inner"
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
        sum(1 for x in trace or [] if is_call(platform.isa, mnemonic(x))),
        describe_tree(roots),
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


def write_summary(results: list[KernelResult], platforms: list[Platform], out_dir: Path, header: list[str]) -> str:
    lines = [
        f"# Byte-shuffle llvm-mca summary: `{out_dir.name}`",
        "",
        *header,
        "",
        "Fixed-itemsize kernels: llvm-mca steady-state **cycles per 128 bytes** (one outer-loop",
        "iteration, inner loops flattened by their trip counts). Lower is better. `geomean` is the",
        "per-CPU geometric mean over the 8 kernels, the single number to optimize.",
        "",
        "- `instrs`: instructions per 128 bytes (flattened trace).",
        "- `loops`: asm loop tree with trip counts; `?` = runtime trip count (the outer loop).",
        "- `calls`: calls inside the hot loop, whose cost llvm-mca does NOT include (flagged `*`).",
        "",
    ]
    by_platform: dict[str, list[KernelResult]] = {}
    for r in results:
        by_platform.setdefault(r.platform, []).append(r)

    def geomean(rs: list[KernelResult], i: int) -> float:
        return math.exp(sum(math.log(r.mca[i].cycles_per_iter) for r in rs) / len(rs))

    lines += [
        "## Overview",
        "",
        "| platform | cpu | geomean | encode | decode | worst kernel |",
        "|---|---|---:|---:|---:|---|",
    ]
    for platform in platforms:
        fixed = [r for r in by_platform.get(platform.name, []) if r.mode == "outer"]
        enc = [r for r in fixed if r.kernel.startswith("encode")]
        dec = [r for r in fixed if r.kernel.startswith("decode")]
        for i, cpu in enumerate(platform.mca_cpus):
            if not fixed:
                continue
            worst = max(fixed, key=lambda r: r.mca[i].cycles_per_iter)
            cells = [f"{geomean(x, i):.1f}" if x else "-" for x in (fixed, enc, dec)]
            lines.append(
                f"| {platform.name} | {cpu} | **{cells[0]}** | {cells[1]} | {cells[2]} | "
                f"{worst.kernel} ({worst.mca[i].cycles_per_iter:.1f}) |"
            )
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
            lines.append("| kernel | instrs | loops | calls | " + " | ".join(cpus) + " |")
            lines.append("|---|---:|---|---:|" + "---:|" * len(cpus))
            for r in fixed:
                star = "*" if r.calls else ""
                cells = [f"{m.cycles_per_iter:.1f}{star}" for m in r.mca]
                lines.append(
                    f"| {r.kernel} | {r.trace_instructions} | {r.loop_tree} | {r.calls} | " + " | ".join(cells) + " |"
                )
            geo = [geomean(fixed, i) for i in range(len(cpus))]
            lines.append("| **geomean** | | | | " + " | ".join(f"**{g:.1f}**" for g in geo) + " |")
            lines.append("")
        if other:
            lines += [
                "Runtime-itemsize / unflattenable kernels, cycles per iteration of the largest innermost "
                "loop (informational):",
                "",
            ]
            lines.append("| kernel | loop instrs | loops | calls | " + " | ".join(cpus) + " |")
            lines.append("|---|---:|---|---:|" + "---:|" * len(cpus))
            for r in other:
                cells = [f"{m.cycles_per_iter:.1f}" for m in r.mca] or ["-"] * len(cpus)
                lines.append(
                    f"| {r.kernel} | {r.trace_instructions} | {r.loop_tree} | {r.calls} | " + " | ".join(cells) + " |"
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
    ap.add_argument("--fn", action="append", dest="kernels", help="restrict to these kernels (repeatable)")
    args = ap.parse_args()

    platforms = [p for p in PLATFORMS if not args.platform or p.name in args.platform]
    kernels = [k for k in KERNELS if not args.kernels or k.name in args.kernels]
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

    print(write_summary(results, platforms, out_dir, header))


if __name__ == "__main__":
    main()
