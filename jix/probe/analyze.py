#!/usr/bin/env python3
"""Static performance analysis of the element-wise pipeline's inner loop: llvm-mca, per platform.

Two sources of kernels (`--source`):
- `py` (default): the `jix-py` extension crate, which instantiates every op for every dtype it
  dispatches. Its kernels are found automatically: every `inner_loop_contiguous` over a single op
  whose operands are leaves (`neg_f32`, `add_i32`, `equal_f64`, `cast_f32_i32`, ...).
- `probe`: the probe crate (`src/lib.rs`), for op chains (`chain_f32`, `longchain_f32`, ...).

For every platform (target triple + ISA level) the crate is compiled once in release mode with
`--emit=asm,llvm-ir`, and for every kernel:

1. Its asm is taken from the crate's asm, and the asm loop tree is recovered from back-edges.
2. LLVM's own `opt` reports the IR loop tree (`print<loops>`) and each loop's exact trip count
   (`print<scalar-evolution>`) of the final (post-optimization) IR, reduced to the kernels.
3. The two loop trees are matched by shape, which gives every asm loop its trip count.
4. The hot loops are the loops whose total iteration count (their trip count times their
   ancestors', evaluated from SCEV for two concrete input lengths) is linear in the input length:
   e.g. a vector main loop. Tails do not scale with the input and are ignored. This also gives each
   hot loop's input bytes per iteration, so kernels may use any step.
5. One iteration of each hot loop is flattened into a straight-line trace (inner loops repeated
   by their constant trip counts), and llvm-mca simulates that trace in a steady state on each
   representative CPU of the platform.

The objective is llvm-mca's **cycles per 4096 bytes** (= sum over the hot loops of cycles per
iteration * 4096 / bytes per iteration). Lower is better. It is a static model: memory is assumed
to hit L1, calls (e.g. to `memcpy`) are not followed - the `calls` column flags kernels whose hot
loop still calls something - and the per-call prologue/epilogue and the tails are not included.

Usage (from anywhere):
    python jix/probe/analyze.py [--source py] [--label baseline] [--platform x86_64-v3 ...] [--fn 'add_*' ...]

Outputs, under jix/probe/results/<label>/:
    summary.md                      the tables
    summary.json                    same data, machine readable (gitignored)
    asm/<platform>/<fn>.s           the function's asm (gitignored)
    mca/<platform>/<fn>.<cpu>.txt   full llvm-mca report (gitignored)

Requirements: the pinned nightly (rust-toolchain.toml, with the llvm-tools component for `opt`), and
an llvm-mca whose LLVM major version matches rustc's (found via $LLVM_MCA, llvm-mca-<major>, or
llvm-mca on PATH). See README.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
REPO_DIR = PROBE_DIR.parent.parent
MCA_ITERATIONS = 100
# Reporting unit: cycles per this many bytes of input.
BYTES_UNIT = 4096
# Input length (bytes) at which the outer loop's SCEV trip count is evaluated to derive its bytes
# per iteration. A multiple of any plausible step, so there is no remainder.
SCEV_EVAL_LEN = 1 << 20

# x86 levels, exactly as enabled by fearless_simd's `dispatch!` arms (`Simd::vectorize` of the
# Sse4_2 / Avx2 / Avx512 tokens). The platforms are named after the nearest x86-64-vN level.
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


# The element-wise pipeline dispatches on the SIMD level at runtime (fearless_simd): plain x86_64 is
# its SSE2 arm, the x86_64-v2/v3/v4 platforms its SSE4.2 / AVX2 / AVX-512 arms.
PLATFORMS = [
    Platform("x86_64", "x86_64-unknown-linux-gnu", [], ["sandybridge", "skylake", "znver3"]),
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
    # 32-bit x86 wheels (linux/windows i686): SSE2 baseline.
    Platform("i686", "i686-unknown-linux-gnu", [], ["skylake"]),
    # aarch64 linux/windows wheels: generic armv8-a + NEON baseline.
    Platform("aarch64", "aarch64-unknown-linux-gnu", [], ["cortex-a72", "neoverse-n1", "neoverse-v2"]),
    # macOS arm64 wheels: the target's default CPU is apple-m1.
    Platform("aarch64-apple", "aarch64-apple-darwin", [], ["apple-m1"]),
]


# Key of `Kernel.scev_env` binding the function's last argument, whatever its IR name.
LAST_ARG = "$last_arg"


@dataclass
class Kernel:
    name: str
    symbol: str  # demangled, without crate hashes
    # Element size of the inner loop, whose `len` argument counts elements (of the output: its
    # bytes are the "input bytes" of the cost model).
    elem_size: int

    def symbol_for(self, platform: Platform) -> str:
        """`symbol` on `platform` (the same on all platforms, for now)."""
        return self.symbol

    def scev_env(self, length: int) -> dict[str, int]:
        """Values of the function arguments, for an input of `length` bytes."""
        # `len`, the last argument, whose IR name can be lost (`%1`) to its stack slot's.
        n = length // self.elem_size
        return {"%len": n, LAST_ARG: n}


def _elementwise_kernel(name: str, ty: str, pipeline: str) -> Kernel:
    """The inner loop of the element-wise pipeline for all operands contiguous,
    `inner_loop_contiguous`. It dispatches on the SIMD level at runtime; the arm of the level that
    the platform's target features enable is inlined into it (the fallback arm without features),
    the higher levels are separate functions. (The baseline analyzed its predecessor,
    `inner_loop::<T, LANES, true, true, _>`.)"""
    size = {"f32": 4, "f64": 8, "i32": 4, "i64": 8}[ty]
    ew = "jix::storage::elementwise_pipeline"
    sym = f"{ew}::inner_loop_contiguous::<{ty}, {pipeline}>"
    return Kernel(f"{name}_{ty}", sym, size)


def _pipeline(ty: str, expr) -> str:
    """Demangled type of the pipeline of `expr`, of element type `ty`: a leaf operand (None),
    ("neg", x), ("cast", from_ty, x) or (op2, x, y)."""
    if expr is None:
        return f"jix::storage::elementwise_pipeline::OperandTyped<{ty}>"
    op1 = "<jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline"
    if expr[0] == "neg":
        return f"{op1}::Op1Pipeline<{_pipeline(ty, expr[1])}, jix::ops::op1::NegKernel, {ty}>"
    if expr[0] == "cast":
        src = expr[1]
        return f"{op1}::Op1Pipeline<{_pipeline(src, expr[2])}, jix::ops::cast::CastKernel<{ty}>, {src}>"
    a, b = _pipeline(ty, expr[1]), _pipeline(ty, expr[2])
    return (
        "<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline"
        f"::Op2Pipeline<{a}, {b}, jix::ops::op2::{expr[0].capitalize()}Kernel, {ty}, {ty}>"
    )


# Kernels: each op chain's inner loop. Must match the `probe_*` exports in src/lib.rs.
ELEMENTWISE_EXPRS = {
    "neg": ("neg", None),
    "add": ("add", None, None),
    "chain": ("mul", ("add", None, None), ("sub", None, None)),  # (a + b) * (c - d)
}
# (a + b) * (c - d) + (e + f) * (g - h)
ELEMENTWISE_EXPRS["longchain"] = ("add", ELEMENTWISE_EXPRS["chain"], ELEMENTWISE_EXPRS["chain"])
KERNELS = [
    _elementwise_kernel(name, ty, _pipeline(ty, expr))
    for ty in ("f32", "f64", "i32")
    for name, expr in ELEMENTWISE_EXPRS.items()
]
# Mixed dtypes: (name, output type, expr).
KERNELS += [
    _elementwise_kernel(name, ty, _pipeline(ty, expr))
    for name, ty, expr in [
        ("narrow", "i32", ("cast", "i64", ("add", None, None))),  # (a + b).cast::<i32>(), over i64
        ("widen", "i64", ("add", ("cast", "i32", None), None)),  # a.cast::<i64>() + b, a i32
        ("narrow", "f32", ("cast", "f64", ("add", None, None))),
        ("widen", "f64", ("add", ("cast", "f32", None), None)),
    ]
]


@dataclass
class Source:
    """A crate whose kernels are analyzed."""

    name: str
    crate_dir: Path
    lib: str  # the lib's name: the stem of the emitted `.s` / `.ll`
    env: dict[str, str]


SOURCES = {
    # pyo3's build script needs the Python version when cross compiling; nothing is linked.
    "py": Source("py", REPO_DIR / "jix-py", "jix", {"PYO3_CROSS_PYTHON_VERSION": "3.13"}),
    "probe": Source("probe", PROBE_DIR, "jix_probe", {}),
}

EW = "jix::storage::elementwise_pipeline"
OP_PIPELINE = re.compile(
    r"<jix::ops::op[12]::Op[12]<[_, ]+> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline"
    r"::Op([12])Pipeline<(.*)>"
)
# Short names of the element types, and their sizes.
TYPES = {
    "bool": ("bool", 1),
    "i8": ("i8", 1),
    "u8": ("u8", 1),
    "i16": ("i16", 2),
    "u16": ("u16", 2),
    "half::binary16::f16": ("f16", 2),
    "i32": ("i32", 4),
    "u32": ("u32", 4),
    "f32": ("f32", 4),
    "i64": ("i64", 8),
    "u64": ("u64", 8),
    "f64": ("f64", 8),
    "num_complex::Complex<f32>": ("complex64", 8),
    "num_complex::Complex<f64>": ("complex128", 16),
}


def split_generic_args(s: str) -> list[str]:
    """`a<b, c>, d` -> [`a<b, c>`, `d`]."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += {"<": 1, ">": -1}.get(ch, 0)
        cur += ch
    return [*out, cur.strip()] if cur.strip() else out


def py_kernel(symbol: str) -> Kernel | None:
    """The kernel of `symbol` if it is the contiguous inner loop of a single op over leaves."""
    m = re.fullmatch(rf"{EW}::inner_loop_contiguous::<(.+?), (<.*)>", symbol)
    if not m or m.group(1) not in TYPES:
        return None
    out, p = m.group(1), OP_PIPELINE.fullmatch(m.group(2))
    if not p:
        return None
    arity, args = int(p.group(1)), split_generic_args(p.group(2))
    leaves, kernel, inputs = args[:arity], args[arity], args[arity + 1 :]
    if len(inputs) != arity or any(leaf != f"{EW}::OperandTyped<{t}>" for leaf, t in zip(leaves, inputs)):
        return None
    if any(t not in TYPES for t in inputs):
        return None
    op = re.sub(r"Kernel$", "", kernel.split("<")[0].rsplit("::", 1)[-1])
    op = re.sub(r"(?<!^)(?=[A-Z])", "_", op).lower()
    types = [TYPES[t][0] for t in dict.fromkeys(inputs)]
    if out not in inputs:
        types.append(TYPES[out][0])
    return Kernel(f"{op}_{'_'.join(types)}", symbol, TYPES[out][1])


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


def cargo_env(platform: Platform, source: Source, target_dir: Path) -> dict:
    env = dict(os.environ) | source.env
    env["RUSTFLAGS"] = " ".join(["-Csymbol-mangling-version=v0", *platform.rustflags])
    env["CARGO_TARGET_DIR"] = str(target_dir)
    # C dependencies (zstd) are built with a stand-in toolchain: the crate is compiled to an rlib
    # that is never linked, so no C cross compiler or sysroot is needed.
    fake = str(PROBE_DIR / "fake_cc.sh")
    t = platform.target.replace("-", "_")
    env |= {f"CC_{t}": fake, f"AR_{t}": fake}
    return env


# Function labels (v0-mangled; Mach-O prefixes an underscore) in the emitted asm.
FN_LABEL_RE = re.compile(r"^(_?_R[0-9A-Za-z_]+):")


@dataclass
class Build:
    """A platform's build of a source: its functions' asm and IR, by mangled name."""

    asm: dict[str, list[str]]  # the function's asm lines, its label first
    names: dict[str, str]  # demangled name (without crate hashes) -> mangled name
    ll: dict[str, list[str]] = field(default_factory=dict)  # the function's IR, for the kernels
    opt: dict[str, list[str]] = field(default_factory=dict)  # its `opt` analyses, for the kernels


# Builds run a whole-crate rustc each: a few at a time, as each uses all cores (and much memory).
BUILD_SLOTS = threading.Semaphore(2)


def compile_crate(platform: Platform, source: Source) -> tuple[Path, Path]:
    """Compile `source` for `platform`, emitting its asm and final LLVM IR: the (`.s`, `.ll`)."""
    # One target dir per feature set: builds with other RUSTFLAGS must not share the directory.
    flags = hashlib.sha1(" ".join(platform.rustflags).encode()).hexdigest()[:8]
    target_dir = PROBE_DIR / "target" / "analyze" / source.name / f"{platform.name}-{flags}"
    out = target_dir / platform.target / "release"
    for f in [*out.rglob(f"{source.lib}-*.s"), *out.rglob(f"{source.lib}-*.ll")]:
        f.unlink()
    # cargo does not re-emit them for an up-to-date crate: rebuild it (only it).
    (source.crate_dir / "src" / "lib.rs").touch()
    # Verbose asm names each block's IR block (`.LBB0_5: # %vector.body`), to match the loops.
    emit = ["--emit=asm,llvm-ir", "-Zverbose-asm"]
    emit += ["-Cllvm-args=-x86-asm-syntax=intel"] if platform.isa == "x86" else []
    # rlib: nothing is linked, so any rustup target works without its linker.
    cmd = ["cargo", "rustc", "--release", "--lib", "--crate-type", "rlib", "--target", platform.target, "--", *emit]
    with BUILD_SLOTS:
        run(cmd, env=cargo_env(platform, source, target_dir), cwd=source.crate_dir)
    [asm] = out.rglob(f"{source.lib}-*.s")
    [ll] = out.rglob(f"{source.lib}-*.ll")
    return asm, ll


def split_asm(text: str) -> dict[str, list[str]]:
    """The functions of an asm file, by mangled name: their lines, from the label to the end."""
    fns: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in text.splitlines():
        if m := FN_LABEL_RE.match(line):
            cur = fns[m.group(1)] = [line]
        elif cur is not None:
            if re.match(r"^\.?Lfunc_end\d+:", line):
                cur = None
            else:
                cur.append(line)
    return fns


def demangle(names: list[str]) -> dict[str, str]:
    """Demangled name (without crate hashes and const generic types) -> mangled name."""
    # binutils demangles v0 with crate hashes and typed const generics: `jix[0123abcd]`, `32: usize`.
    plain = [n[1:] if n.startswith("__R") else n for n in names]
    out = run(["c++filt"], input="\n".join(plain)).splitlines()
    return {re.sub(r": (?:usize|bool)\b", "", re.sub(r"\[[0-9a-f]+\]", "", d)): n for n, d in zip(names, out)}


def reduce_ir(ll: Path, keep: list[str], opt: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The IR of the functions `keep` (mangled, as in the IR) and their `opt` loop and SCEV
    analyses, each by name. `opt` runs once, on the module reduced to those functions (and what they
    call): the whole crate's module is too large to analyze."""
    keep_set = set(keep)
    text = []
    for line in ll.read_text().splitlines():
        # Internal functions to keep become external, so that `internalize` + `globaldce` keep them.
        if line.startswith("define ") and (m := re.search(r"@(\"[^\"]+\"|[\w.$]+)\(", line)):
            if m.group(1).strip('"') in keep_set:
                line = re.sub(r"^define (?:internal |private )", "define ", line)
        text.append(line)
    with tempfile.TemporaryDirectory() as tmp:
        full, api, small = Path(tmp) / "full.ll", Path(tmp) / "api.txt", Path(tmp) / "small.ll"
        full.write_text("\n".join(text) + "\n")
        api.write_text("\n".join(keep) + "\n")
        passes = "internalize,globaldce"
        run([opt, "-S", f"-passes={passes}", f"-internalize-public-api-file={api}", str(full), "-o", str(small)])
        ir_text = small.read_text()
        # opt prints analyses to stderr.
        p = subprocess.run(
            [opt, "-disable-output", "-passes=print<loops>,print<scalar-evolution>", str(small)],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(p.stderr)
    fn_ir: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in ir_text.splitlines():
        if line.startswith("define ") and (m := re.search(r"@(\"[^\"]+\"|[\w.$]+)\(", line)):
            name = m.group(1).strip('"')
            cur = fn_ir[name] = [line] if name in keep_set else None
        elif cur is not None:
            cur.append(line)
            if line.startswith("}"):
                cur = None
    fn_opt: dict[str, list[str]] = {}
    cur = None
    for line in p.stderr.splitlines():
        m = re.match(r"Loop info for function '([^']+)'", line) or re.match(
            r"Classifying expressions for: @(\S+)", line
        )
        if m:
            name = m.group(1).strip('"')
            cur = fn_opt.setdefault(name, []) if name in keep_set else None
        if cur is not None:
            cur.append(line)
    return fn_ir, fn_opt


def build(platform: Platform, source: Source, select, opt: str) -> tuple[Build, list[Kernel]]:
    """Compile `source` for `platform` and extract the kernels `select(build)` returns."""
    asm_path, ll = compile_crate(platform, source)
    fns = split_asm(asm_path.read_text())
    b = Build(fns, demangle(list(fns)))
    kernels = select(b)
    # The IR names functions as in the asm, less Mach-O's underscore.
    keep = [b.names[k.symbol_for(platform)] for k in kernels]
    b.ll, b.opt = reduce_ir(ll, [n[1:] if n.startswith("__R") else n for n in keep], opt)
    return b, kernels


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
    # Share of the loop's instructions on vector types (IR) / registers (asm), to match a vector
    # loop and its scalar remainder, whose sizes can rank differently in IR and asm.
    vec: float = 0.0
    ir: IrValues | None = None  # IR only: the function's values, to resolve SCEV operands
    blocks: set[str] = field(default_factory=set)  # asm only: the IR names of the loop's blocks

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


def clean_lines(asm: str, comment: str, block_names: dict[int, str] | None = None) -> list[str]:
    """Strip comments, blank lines, directives and the function-name label. `block_names` gets the
    IR name of the block starting at each line, from the verbose asm comments: after a label, or on
    a comment line of its own for a block without one (`# %bb.3: # %vector.ph`)."""
    out = []
    pending = None
    for line in asm.splitlines()[1:]:
        if comment in line:
            line, note = line[: line.index(comment)], line[line.index(comment) :]
            if m := re.search(r"%([\w.\-$]+)\s*$", note):
                pending = m.group(1)
        s = line.strip()
        if not s or (s.startswith(".") and not is_label(s)):
            continue
        if pending is not None and block_names is not None:
            block_names[len(out)] = pending
        pending = None
        out.append(s)
    return out


def is_unconditional(isa: str, mn: str) -> bool:
    """A branch or return that never falls through."""
    if isa == "x86":
        return mn in ("jmp", "jmpq", "ret", "retq", "retl", "ud2")
    return mn in ("b", "br", "ret", "udf", "brk")


def asm_loops(lines: list[str], isa: str, block_names: dict[int, str] | None = None) -> list[Loop]:
    """The natural loops of the asm: back edges (to a block that dominates the branch) and the
    blocks that reach them. A loop spans the lines from its first to its last block, so a rotated
    loop entered in the middle is found, and a backward jump that is not a back edge (e.g. to a
    vector epilogue laid out before the main vector loop) is not a loop."""
    labels = {m.group(1): i for i, line in enumerate(lines) if (m := LABEL_RE.match(line))}
    ends_block = [
        not is_label(x) and (is_branch(isa, mnemonic(x)) or is_unconditional(isa, mnemonic(x))) for x in lines
    ]
    starts = sorted({0, *labels.values(), *(i + 1 for i, e in enumerate(ends_block) if e)} - {len(lines)})
    blocks = list(zip(starts, [*starts[1:], len(lines)]))  # [start, end)
    block_at = {s: b for b, (s, _) in enumerate(blocks)}
    succs: list[set[int]] = []
    for b, (_, e) in enumerate(blocks):
        last = lines[e - 1]
        out = set()
        if not is_label(last) and is_branch(isa, mnemonic(last)):
            out |= {block_at[labels[t]] for t in TOKEN_RE.findall(last)[1:] if t in labels}
        if not (not is_label(last) and is_unconditional(isa, mnemonic(last))) and b + 1 < len(blocks):
            out.add(b + 1)
        succs.append(out)
    preds: list[set[int]] = [set() for _ in blocks]
    for b, out in enumerate(succs):
        for t in out:
            preds[t].add(b)

    # Dominators, iteratively; blocks unreachable from the entry dominate nothing.
    everything = set(range(len(blocks)))
    dom = [everything.copy() for _ in blocks]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for b in range(1, len(blocks)):
            ps = [dom[p] for p in preds[b]]
            d = {b} | (set.intersection(*ps) if ps else set())
            if d != dom[b]:
                dom[b], changed = d, True

    bodies: dict[int, set[int]] = {}
    for u, out in enumerate(succs):
        for h in out:
            if h in dom[u] and dom[u] != everything:
                body, stack = {h, u}, [u] if u != h else []
                while stack:
                    for p in preds[stack.pop()]:
                        if p not in body:
                            body.add(p)
                            stack.append(p)
                bodies.setdefault(h, set()).update(body)

    vreg = re.compile(r"\b[xyz]mm\d+\b" if isa == "x86" else r"\b[vq]\d+\b")
    # x86 scalar float instructions (`vaddsd`, `vcvtsd2ss`; not the packed integer `vpminsd`) and
    # moves between general and xmm registers also use xmm registers.
    x86_scalar = re.compile(r"^(?!v?p|vbroadcast).*((ss|sd)$|s[sd]2)|^v?mov[dq]$|^v?p(insr|extr)")

    def is_vector(x: str) -> bool:
        return bool(vreg.search(x)) and not (isa == "x86" and x86_scalar.search(mnemonic(x)))

    def make(h: int, body: set[int]) -> Loop:
        start, end = min(blocks[b][0] for b in body), max(blocks[b][1] for b in body) - 1
        instrs = [x for x in lines[start : end + 1] if not is_label(x)]
        header = LABEL_RE.match(lines[blocks[h][0]]).group(1)
        names = {n for b in body for i in range(*blocks[b]) if (n := (block_names or {}).get(i)) is not None}
        vec = sum(1 for x in instrs if is_vector(x)) / len(instrs)
        return Loop(start, end, header, size=len(instrs), vec=vec, blocks=names)

    loops = sorted((make(h, body) for h, body in bodies.items()), key=lambda lp: (lp.start, -lp.end))
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

    def __init__(self, defs: dict[str, str], stores: dict[str, str | None], params: list[str]):
        self.defs = defs  # `%name` -> the instruction's right-hand side
        self.params = params  # the function's parameters
        # `%ptr` -> the value stored to it, if stored once (e.g. an argument spilled to a stack
        # slot, as when a closure capturing it by reference is passed to another function).
        self.stores = stores
        # Values SCEV cannot analyze but proves a small range of (`print<scalar-evolution>`).
        self.bounded: set[str] = set()

    @staticmethod
    def parse(ll: list[str], ir_hint: str) -> IrValues:
        defs, stores, params, in_fn = {}, {}, [], False
        for line in ll:
            if line.startswith("define "):
                in_fn = ir_hint in line.split("(")[0]
                if in_fn:
                    params = re.findall(r"(%[\w.]+)[,)]", line)
            elif in_fn and line.startswith("}"):
                in_fn = False
            elif in_fn and (m := re.match(r"^\s+(%[\w.]+) = (.*?)(?:, !.*)?$", line)):
                defs[m.group(1)] = m.group(2)
            elif in_fn and (m := re.match(r"^\s+store (i\d+ [^,]+), ptr (%[\w.]+)", line)):
                stores[m.group(2)] = None if m.group(2) in stores else m.group(1)
        return IrValues(defs, stores, params)

    def value(self, name: str, env: dict[str, int], depth: int = 0):
        """Value of `%name` (int, or list of ints for a vector), or None."""
        if name in env:
            return env[name]
        if self.params and name == self.params[-1] and LAST_ARG in env:
            return env[LAST_ARG]
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
        if op == "load":
            # A stack slot named after an argument (`%len`) holds that argument.
            m = re.match(r"i\d+, ptr (%[\w.]+)", rest)
            if m.group(1) in env:
                return env[m.group(1)]
            stored = self.stores.get(m.group(1))
            return None if stored is None else self._operand(stored, env, depth)
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


def ir_block_sizes(ll: list[str], ir_hint: str) -> tuple[dict[str, int], dict[str, int]]:
    """Instruction count, and count of instructions on vector types, of each basic block of the
    function whose name contains `ir_hint`."""
    sizes: dict[str, int] = {}
    vecs: dict[str, int] = {}
    in_fn, block = False, None
    for line in ll:
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
                vecs[block] = vecs.get(block, 0) + bool(re.search(r"<\d+ x ", line))
    return sizes, vecs


def ir_loops(ll: list[str], analyses: list[str], ir_hint: str) -> list[Loop]:
    """IR loop tree of the function whose name contains `ir_hint`, with SCEV trip counts, from its
    IR `ll` and its `opt` `analyses`."""

    block_sizes, block_vecs = ir_block_sizes(ll, ir_hint)
    bounded: set[str] = set()
    cur_value = None
    fn_loops: list[tuple[int, Loop]] = []
    trips: dict[str, tuple[str, int | None]] = {}
    in_fn = False
    for line in analyses:
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
            size = sum(block_sizes.get(b, 0) for b in blocks)
            vec = sum(block_vecs.get(b, 0) for b in blocks) / max(size, 1)
            lp = Loop(0, 0, header, size=size, vec=vec)
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

    Sibling loops are matched by shape, and siblings of the same shape by rank of vector share,
    then size: `opt` does not list loops in asm layout order (e.g. a main loop between an inlined
    tail and its vectorized version), and a vector loop can be smaller than its scalar remainder in
    IR (bounds checks) but larger in asm.
    """
    if sorted(r.shape() for r in asm_roots) != sorted(r.shape() for r in ir_roots):
        return False
    # By name: the asm loop holding the IR loop's header block (verbose asm), if one and only one.
    by_name = [[a for a in asm_roots if i.header in a.blocks] for i in ir_roots]
    if all(len(m) == 1 for m in by_name) and len({id(m[0]) for m in by_name}) == len(ir_roots):
        for i, [a] in zip(ir_roots, by_name):
            a.trip_count, a.trip_expr, a.ir = i.trip_count, i.trip_expr, i.ir
            if a.shape() != i.shape() or not attach_trip_counts(a.children, i.children):
                return False
        return True
    for shape in {r.shape() for r in asm_roots}:
        a_group = sorted((r for r in asm_roots if r.shape() == shape), key=lambda r: (r.vec > 0.25, r.size))
        i_group = sorted((r for r in ir_roots if r.shape() == shape), key=lambda r: (r.vec > 0.25, r.size))
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


POST_INDEX_STORE_RE = re.compile(r"^(st\w*)\s+(.*), \[(\w+)\], #(-?\d+)$")
PRE_INDEX_STORE_RE = re.compile(r"^(st\w*)\s+(.*), \[(\w+), #(-?\d+)\]!$")


def split_writeback_stores(trace: list[str]) -> list[str]:
    """aarch64: rewrite a store with base-register writeback as a plain store plus an add.

    llvm-mca issues an instruction when all its operands are ready, so a post-indexed store's new
    base register waits for the stored data. When the next iteration's loads use that register,
    this serializes the whole loop behind its dependency chain (5-7x too slow for a loop of
    shifts). Real cores split a store into address and data micro-ops, so the writeback depends on
    the base register only, as it does after this rewrite.
    """
    out = []
    for x in trace:
        if m := POST_INDEX_STORE_RE.match(x):
            op, data, base, imm = m.groups()
            out += [f"{op} {data}, [{base}]", f"add {base}, {base}, #{imm}"]
        elif m := PRE_INDEX_STORE_RE.match(x):
            op, data, base, imm = m.groups()
            out += [f"add {base}, {base}, #{imm}", f"{op} {data}, [{base}]"]
        else:
            out.append(x)
    return [re.sub(r"^add (\w+), (\w+), #-(\d+)$", r"sub \1, \2, #\3", x) for x in out]


def run_mca(mca: str, platform: Platform, cpu: str, trace: list[str], all_lines: list[str]) -> tuple[McaResult, str]:
    if platform.isa == "aarch64":
        trace = split_writeback_stores(trace)
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
        src = ".intel_syntax noprefix\n" + src
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
    trace_instructions: str  # per hot loop, `+`-joined
    calls: list[str]  # callees inside the hot loops
    loop_tree: str
    bytes_per_iter: str  # per hot loop, `+`-joined; `~`: approximated trip count
    warnings: list[str] = field(default_factory=list)
    mca: list[McaResult] = field(default_factory=list)  # per (hot loop, cpu)
    # Per cpu, cycles per BYTES_UNIT: sum over the hot loops of cycles per iteration / bytes per
    # iteration. Empty: no flattenable hot loop.
    costs: list[float] = field(default_factory=list)


def describe_tree(loops: list[Loop]) -> str:
    def one(lp: Loop) -> str:
        t = "?" if lp.trip_count is None else str(lp.trip_count)
        return f"{t}" + (f"[{' '.join(one(c) for c in lp.children)}]" if lp.children else "")

    return " ".join(one(lp) for lp in loops)


def analyze_kernel(platform: Platform, kernel: Kernel, b: Build, mca: str, out_dir: Path) -> KernelResult:
    mangled = b.names[kernel.symbol_for(platform)]
    asm = "\n".join(b.asm[mangled]) + "\n"
    ir_name = mangled[1:] if mangled.startswith("__R") else mangled
    asm_path = out_dir / "asm" / platform.name / f"{kernel.name}.s"
    asm_path.parent.mkdir(parents=True, exist_ok=True)
    asm_path.write_text(asm)

    block_names: dict[int, str] = {}
    lines = clean_lines(asm, platform.comment, block_names)
    fn_instrs = sum(1 for line in lines if not is_label(line))
    warnings = []
    try:
        roots = asm_loops(lines, platform.isa, block_names)
    except RuntimeError as e:  # a loop LLVM laid out in non-contiguous pieces
        roots = []
        warnings.append(str(e))
    if not attach_trip_counts(roots, ir_loops(b.ll.get(ir_name, []), b.opt.get(ir_name, []), ir_name)):
        warnings.append("asm and IR loop trees differ, trip counts unknown")

    hot = hot_loops(roots, kernel)
    traces = [flatten(lines, h.loop) for h in hot]
    if not hot:
        warnings.append("no loop whose iteration count is linear in the input length")
    elif None in traces:
        warnings.append("hot loop with an inner loop of unknown trip count")
    if not hot or None in traces:
        hot, traces = [], []
    fmt = [("~" if h.approx else "") + f"{h.bytes_per_iter:g}" for h in hot]

    all_instrs = [x for t in traces for x in t]
    r = KernelResult(
        platform.name,
        kernel.name,
        fn_instrs,
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
            cost += m.cycles_per_iter * BYTES_UNIT / hot[k].bytes_per_iter
            suffix = f".{k}" if len(traces) > 1 else ""
            p = out_dir / "mca" / platform.name / f"{kernel.name}{suffix}.{cpu}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(report)
        r.costs.append(cost)
    return r


def analyze_platform(
    platform: Platform, source: Source, select, opt: str, mca: str, out_dir: Path
) -> list[KernelResult]:
    b, kernels = build(platform, source, select, opt)
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        return list(ex.map(lambda k: analyze_kernel(platform, k, b, mca, out_dir), kernels))


def load_costs(label: str) -> dict[tuple[str, str, str], float]:
    """(platform, cpu, kernel) -> cycles per BYTES_UNIT, from a previous run's summary.json."""
    costs = {}
    for r in json.loads((PROBE_DIR / "results" / label / "summary.json").read_text()):
        cpus = list(dict.fromkeys(m["cpu"] for m in r["mca"]))
        for cpu, c in zip(cpus, r["costs"]):
            costs[(r["platform"], cpu, r["kernel"])] = c
    return costs


def write_summary(
    results: list[KernelResult],
    platforms: list[Platform],
    out_dir: Path,
    header: list[str],
    compare: str | None,
) -> str:
    lines = [
        f"# llvm-mca summary: `{out_dir.name}`",
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

    # Overview: a row per kernel, a column per (platform, cpu).
    cols = [(p, i, cpu) for p in platforms if by_platform.get(p.name) for i, cpu in enumerate(p.mca_cpus)]
    by_key = {(r.platform, r.kernel): r for r in results}
    names = list(dict.fromkeys(r.kernel for r in results if r.costs))
    lines += [
        "## Overview",
        "",
        "| kernel | " + " | ".join(f"{p.name} {cpu}" for p, _, cpu in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for name in names:
        cells = []
        for p, i, _ in cols:
            r = by_key.get((p.name, name))
            cells.append(f"{cost(r, i):.0f}{star([r])}" if r and r.costs else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    def fixed_of(p: Platform) -> list[KernelResult]:
        return [r for r in by_platform.get(p.name, []) if r.costs]

    lines.append("| **geomean** | " + " | ".join(f"**{geomean(fixed_of(p), i)}**" for p, i, _ in cols) + " |")
    if compare:
        vs = [speedup(p.name, cpu, fixed_of(p), i) for p, i, cpu in cols]
        lines.append(f"| **vs {compare}** | " + " | ".join(vs) + " |")
    lines.append("")

    for platform in platforms:
        rs = by_platform.get(platform.name, [])
        cpus = platform.mca_cpus
        fixed = [r for r in rs if r.costs]
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
    ap.add_argument(
        "--fn", action="append", dest="kernels", help="restrict to these kernels, glob patterns (repeatable)"
    )
    ap.add_argument("--source", choices=sorted(SOURCES), default="py", help="the crate whose kernels to analyze")
    args = ap.parse_args()

    platforms = [p for p in PLATFORMS if not args.platform or p.name in args.platform]
    if not platforms:
        sys.exit("error: no platform selected")
    source = SOURCES[args.source]

    def selected(name: str) -> bool:
        return not args.kernels or any(fnmatch.fnmatchcase(name, pat) for pat in args.kernels)

    def select(b: Build) -> list[Kernel]:
        """The kernels to analyze in the build `b`."""
        if source.name == "probe":
            return [k for k in KERNELS if selected(k.name)]
        found = {k.name: k for k in map(py_kernel, b.names) if k and selected(k.name)}
        return [found[n] for n in sorted(found)]

    mca = find_llvm_mca(llvm_major_of_rustc())
    opt = find_opt()
    mca_version = re.search(r"LLVM version [\d.]+", run([mca, "--version"])).group(0)
    header = [
        f"- rustc: `{run(['rustc', '--version'], cwd=PROBE_DIR).strip()}`",
        f"- llvm-mca: `{mca_version}`",
        f"- source: `{source.name}` (`{source.crate_dir.relative_to(REPO_DIR)}`)",
        "- kernels: the pipeline's inner loop for contiguous operands (`inner_loop_contiguous`, the arm of the"
        " platform's SIMD level); the bytes are output bytes",
    ]

    out_dir = PROBE_DIR / "results" / args.label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = [ex.submit(analyze_platform, p, source, select, opt, mca, out_dir) for p in platforms]
        results = [r for f in futures for r in f.result()]
    if not results:
        sys.exit("error: no kernel selected")

    print(write_summary(results, platforms, out_dir, header, args.compare))


if __name__ == "__main__":
    main()
