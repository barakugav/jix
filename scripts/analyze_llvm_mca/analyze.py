"""Static performance analysis of jix's hot loops: llvm-mca, on every platform jix ships wheels for.

Compiles the `jix-py` extension crate, which instantiates every op for every dtype it dispatches, and
simulates the steady state of its kernels' hot loops with llvm-mca. Nothing is run, so one machine
analyzes every platform.

Kernels, found in the build by their demangled names:
- Element-wise ops: the pipeline's inner loop for contiguous operands (`inner_loop::<T, P, true,
  true>`), named after the op and its operand types: `neg_f32`, `add_i32`, `clamp_f64`, with the
  output type last when it is not an operand's: `equal_f32_bool`, `cast_f32_i32`. A pipeline whose
  operands are not all leaves is named by its expression: `mul(add(f32,f32),f32)`.
- Reductions: `reduce_<op>_<type>_<loop>` (`reduce_sum_f32_cell`), the loops of `REDUCTION_LOOPS`.

Usage:
    python scripts/analyze_llvm_mca/analyze.py --out <dir> [--platform x86_64-v3 ...] [--fn 'add_*' ...]
    python scripts/analyze_llvm_mca/analyze.py --out <dir> --no-build   # reuse the previous builds
    python scripts/analyze_llvm_mca/compare.py <new out dir> <base out dir>   # per-kernel speedups

A build takes about 8 minutes per platform, two at a time. Outputs, under `--out`:
    summary.md                          the tables
    summary.json                        the same, read by compare.py
    asm/<platform>/<kernel>.s           the kernel's asm
    mca/<platform>/<kernel>.<cpu>.txt   llvm-mca's report: resource pressure, bottlenecks

Setup:
- A Rust toolchain (pick it with `RUSTUP_TOOLCHAIN`) with the targets of `PLATFORMS` and the
  `llvm-tools` component, for `opt`. The beta, 1.100, has LLVM 23:
      rustup toolchain install beta --profile minimal --component llvm-tools
      rustup target add --toolchain beta i686-unknown-linux-gnu aarch64-unknown-linux-gnu aarch64-apple-darwin
      export RUSTUP_TOOLCHAIN=beta
  Its unstable flags (`-Zverbose-asm`) are enabled by `RUSTC_BOOTSTRAP=1`, so it may be stable
  or beta.
- llvm-mca of rustc's LLVM major version (`rustc -vV`), at `$LLVM_MCA`, `llvm-mca-<major>` or
  `llvm-mca`. rustup's llvm-tools has none: take it from the LLVM release, e.g. for LLVM 23.1.4
  (beta 1.100.0-beta.7):
      curl -sSLO https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.4/LLVM-23.1.4-Linux-X64.tar.xz
      tar -xJf LLVM-23.1.4-Linux-X64.tar.xz LLVM-23.1.4-Linux-X64/bin/llvm-mca  # static binary
      sudo install LLVM-23.1.4-Linux-X64/bin/llvm-mca /usr/local/bin/llvm-mca-23
- `c++filt` (binutils), which demangles Rust's v0 symbols.

The crate is compiled as an rlib, never linked, so any rustup target builds without its linker or
a C cross toolchain: `zstd-sys`'s C is "compiled" by `fake_cc.sh` (empty objects and archives),
and pyo3's build script only needs `PYO3_CROSS_PYTHON_VERSION`.

Platforms: fearless_simd's `dispatch!` runs a loop inside a `#[target_feature]` function of the
SIMD level detected at runtime. Each x86 level is built with that level's features in `-C
target-feature`, which makes its arm the one in the loop: x86_64 and i686 analyze the `Sse2` arm,
x86_64-v2 / v3 / v4 the `Sse4_2` / `Avx2` / `Avx512` arms (fearless_simd's AVX-512 is Ice Lake's,
with VBMI), aarch64 the `Neon` arm. Windows and musl wheels share their linux counterpart's ISA.

Method: each platform's crate is compiled once with `--emit=asm,llvm-ir`, then for each kernel:
1. The asm loop tree: the natural loops of the function's asm (back edges to a dominating block),
   and the irreducible cycles.
2. The IR loop tree and trip counts: the toolchain's `opt` prints the loops (`print<loops>`) and
   their backedge-taken counts (`print<scalar-evolution>`) of the final IR, reduced to the kernels
   (`internalize` + `globaldce`).
3. Each IR loop is matched to the asm loop holding its header block, named by the verbose asm
   (`.LBB0_5: # %vector.body`), which gives each asm loop its trip count.
4. The hot loops are those whose total iteration count (trip count times their ancestors') is
   linear in the input length, evaluated at two lengths: the vector main loop. Tails do not scale
   and are ignored. The slope is the input bytes per iteration (`B/iter`), measured, so a kernel
   may use any step. Trip counts are evaluated with `%len` bound to the length in elements.
   Integer values SCEV does not see through (e.g. the length after the SIMD level dispatch) are
   folded from the IR, and a single value still unknown is the length: an argument whose IR name
   is lost (`%1`), or loaded from a closure's captures or an argument struct. A loop with panic
   exits (bounds checks) has no exact count: its symbolic max is used, which it runs in the steady
   state.
5. One iteration of each hot loop, inner loops repeated by their trip counts, is simulated by
   llvm-mca (100 iterations) on each CPU of the platform.

**Metric: cycles per 4096 bytes (sum over the hot loops of cycles per iteration * 4096 / B/iter).
Lower is better.** The bytes are the element-wise output's, the reduced items'. The per-(platform,
CPU) geomean over the kernels is the headline number.

Caveats:
- llvm-mca assumes every load hits L1 and ignores the front end. Branches are not followed.
- Calls are free in llvm-mca: a kernel whose hot loop calls something (libm's `expf`) is flagged
  `*`, its real cost is higher. x86-64's register-indirect calls are named by the GOT entry last
  loaded into the register.
- A kernel must not call a Rust function in any loop (a closure or helper LLVM did not inline,
  `half`'s software f16 arithmetic), unless `#[cold]`: summary.md lists them, and the script exits
  with an error.
- The steady state only: the per-call prologue and epilogue and the tails are not counted.
- aarch64 stores with base-register writeback (`str q0, [x0], #16`) are split into a store and an
  `add`: llvm-mca would make the new base register wait for the stored data (5-7x too slow on a
  shift-bound loop), where real cores split a store into address and data micro-ops.
- Kernels without a hot loop of known trip count, and instructions a CPU model has no scheduling
  info for (simulated as free), are listed as warnings.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
JIX_PY = SCRIPT_DIR.parent.parent / "jix-py"
MCA_ITERATIONS = 100
BYTES_UNIT = 4096  # the reported cost is cycles per this many bytes
# Input length (bytes) at which the trip counts are evaluated: a multiple of any plausible step.
SCEV_EVAL_LEN = 1 << 20

# The x86 levels, as enabled by fearless_simd's `Sse4_2` / `Avx2` / `Avx512` tokens.
X86_V2 = "+fxsr,+sse4.2,+cmpxchg16b,+popcnt"
X86_V3 = "+avx2,+bmi1,+bmi2,+cmpxchg16b,+f16c,+fma,+fxsr,+lzcnt,+movbe,+popcnt,+xsave"
X86_V4 = (
    "+adx,+aes,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,"
    "+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi1,+bmi2,+cmpxchg16b,+fma,+fxsr,+gfni,+lzcnt,+movbe,+pclmulqdq,"
    "+popcnt,+rdrand,+rdseed,+sha,+vaes,+vpclmulqdq,+xsave,+xsavec,+xsaveopt,+xsaves"
)


@dataclass
class Platform:
    name: str
    target: str
    features: str  # `-C target-feature`
    cpus: list[str]  # llvm-mca's scheduling models: CPUs that run this code path in practice
    simd: str  # the level's fearless_simd token: the instance analyzed of the loops instantiated per level

    @property
    def isa(self) -> str:
        return "aarch64" if self.target.startswith("aarch64") else "x86"

    @property
    def comment(self) -> str:
        """The line comment marker of the asm."""
        return "#" if self.isa == "x86" else ";" if "apple" in self.target else "//"


PLATFORMS = [
    Platform("x86_64", "x86_64-unknown-linux-gnu", "", ["sandybridge", "skylake", "znver3"], "Sse2"),
    Platform("x86_64-v2", "x86_64-unknown-linux-gnu", X86_V2, ["sandybridge", "btver2"], "Sse4_2"),
    Platform("x86_64-v3", "x86_64-unknown-linux-gnu", X86_V3, ["skylake", "alderlake", "znver3"], "Avx2"),
    Platform("x86_64-v4", "x86_64-unknown-linux-gnu", X86_V4, ["icelake-server", "sapphirerapids", "znver4"], "Avx512"),
    Platform("i686", "i686-unknown-linux-gnu", "", ["skylake"], "Sse2"),
    Platform("aarch64", "aarch64-unknown-linux-gnu", "", ["cortex-a72", "neoverse-n1", "neoverse-v2"], "Neon"),
    Platform("aarch64-apple", "aarch64-apple-darwin", "", ["apple-m1"], "Neon"),  # the target's default CPU
]

# --------------------------------------------------------------------------------------------------
# Kernels


@dataclass
class Kernel:
    name: str
    symbol: str  # demangled, without crate hashes
    elem_size: int  # bytes of the elements `len` counts


# Element types: short name and size.
TYPES = {
    "jix::scalar::bool::bool8": ("bool", 1),  # short name kept, to compare with runs before `Bool8`
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

EW = "jix::storage::elementwise_pipeline"
INNER_LOOP = re.compile(rf"{EW}::inner_loop::<(.+?), (<.*), true, true>")
LEAF = re.compile(rf"{EW}::OperandTyped<(.+)>")
# `<jix::ops::op2::Op2<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<..>`
NODE = re.compile(r"<[^<>]*(?:<[_, ]*>)?[^<>]* as [\w:]+>::read_as_elementwise_pipeline::(\w+)<(.*)>")

# The reduction loops over contiguous items, by name suffix; groups: the item type, the op, and the
# SIMD level of the loops instantiated per level. `cell`: the one-cell fold's leaf; `init` /
# `update`: the fold across cells, of the first item / the others.
LEAF_FOLD = (
    r"fold_pairwise::<(.+), jix::ops::reduction::(\w+)Kernel, \d+, {}, fearless_simd::(?:\w+::)*(\w+)>::\{{closure#0\}}"
)
REDUCTION_LOOPS = {
    LEAF_FOLD.format("true"): "cell",
    LEAF_FOLD.format("false"): "cell_strided",
    r"fold_multi_cell_contiguous::<(.+), jix::ops::reduction::(\w+)Kernel, true>": "init",
    r"fold_multi_cell_contiguous::<(.+), jix::ops::reduction::(\w+)Kernel, false>": "update",
    r"fold_multi_cell::<(.+), jix::ops::reduction::(\w+)Kernel, false, true>": "init_strided",
    r"fold_multi_cell::<(.+), jix::ops::reduction::(\w+)Kernel, false, false>": "update_strided",
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


def snake(path: str) -> str:
    """`jix::ops::op2::GreaterEqualKernel<f32>` -> `greater_equal`."""
    name = re.sub(r"(?:Kernel|Pipeline)$", "", path.split("<")[0].rsplit("::", 1)[-1])
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def parse_pipeline(ty: str) -> str | tuple[str, list]:
    """A pipeline type: a leaf's element type (short name), or a node's (op, operands)."""
    if m := LEAF.fullmatch(ty):
        return TYPES[m.group(1)][0]
    if not (m := NODE.fullmatch(ty)):
        raise ValueError(ty)
    node, args = m.group(1), split_generic_args(m.group(2))
    if node in ("Op1Pipeline", "Op2Pipeline"):  # <operands..., kernel, operand types...>
        arity = int(node[2])
        return snake(args[arity]), [parse_pipeline(a) for a in args[:arity]]
    return snake(node), [parse_pipeline(a) for a in args if LEAF.fullmatch(a) or NODE.fullmatch(a)]


def expr(p: str | tuple[str, list]) -> str:
    return p if isinstance(p, str) else f"{p[0]}({','.join(map(expr, p[1]))})"


def elementwise_kernel(symbol: str) -> Kernel | None:
    if not (m := INNER_LOOP.fullmatch(symbol)) or m.group(1) not in TYPES:
        return None
    try:
        p = parse_pipeline(m.group(2))
    except (KeyError, ValueError):  # an element type or a node this script does not know
        print(f"note: skipping `{symbol}`", file=sys.stderr)
        return None
    out, size = TYPES[m.group(1)]
    if isinstance(p, str) or not all(isinstance(x, str) for x in p[1]):
        return Kernel(expr(p), symbol, size)
    types = list(dict.fromkeys(p[1]))
    return Kernel("_".join([p[0], *types, *([out] if out not in types else [])]), symbol, size)


def reduction_kernel(symbol: str, simd: str) -> Kernel | None:
    """Of the loops instantiated per SIMD level, the instance of `simd`. `len` counts items."""
    for pattern, loop in REDUCTION_LOOPS.items():
        m = re.fullmatch(rf"jix::ops::reduction::{pattern}", symbol)
        if m and m.group(1) in TYPES and m.groups()[2:] in [(), (simd,)]:
            ty, size = TYPES[m.group(1)]
            return Kernel(f"reduce_{snake(m.group(2))}_{ty}_{loop}", symbol, size)
    return None


def find_kernels(symbols: list[str], simd: str, patterns: list[str] | None) -> list[Kernel]:
    found: dict[str, Kernel] = {}
    for s in symbols:
        k = reduction_kernel(s, simd) if s.startswith("jix::ops::reduction::") else elementwise_kernel(s)
        if k and (not patterns or any(fnmatch.fnmatchcase(k.name, p) for p in patterns)):
            if k.name in found:  # e.g. two closures of the same op and types
                print(f"note: `{k.name}` names two kernels, keeping `{found[k.name].symbol}`", file=sys.stderr)
                continue
            found[k.name] = k
    return [found[n] for n in sorted(found)]


# --------------------------------------------------------------------------------------------------
# Build


def run(cmd: list[str], env: dict | None = None, cwd: Path | None = None, input: str | None = None):
    p = subprocess.run(cmd, env=env, cwd=cwd, input=input, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    return p


def llvm_major() -> int:
    return int(re.search(r"LLVM version: (\d+)", run(["rustc", "-vV"]).stdout).group(1))


def find_llvm_mca(major: int) -> str:
    for c in [os.environ.get("LLVM_MCA"), f"llvm-mca-{major}", "llvm-mca"]:
        if not (path := c and shutil.which(c)):
            continue
        m = re.search(r"LLVM version (\d+)", run([path, "--version"]).stdout)
        if m and int(m.group(1)) == major:
            return path
        print(f"note: skipping {path}: not LLVM {major}, rustc's", file=sys.stderr)
    sys.exit(f"error: no llvm-mca of LLVM {major} found (set $LLVM_MCA)")


def find_opt() -> str:
    """The toolchain's own `opt` (rustup component llvm-tools)."""
    sysroot = run(["rustc", "--print", "sysroot"]).stdout.strip()
    host = re.search(r"host: (\S+)", run(["rustc", "-vV"]).stdout).group(1)
    if not (opt := Path(sysroot) / "lib" / "rustlib" / host / "bin" / "opt").exists():
        sys.exit(f"error: {opt} not found, install the llvm-tools rustup component")
    return str(opt)


# Function labels in the emitted asm: v0-mangled, Mach-O prefixes an underscore.
FN_LABEL_RE = re.compile(r"^(_?_R[0-9A-Za-z_]+):")


def ir_name(mangled: str) -> str:
    """The IR name of an asm function label: without Mach-O's underscore."""
    return mangled[1:] if mangled.startswith("__R") else mangled


@dataclass
class Build:
    """A platform's build: its functions' asm, and the kernels' IR and `opt` analyses, by mangled
    name."""

    asm: dict[str, list[str]]  # the function's lines, its label first
    names: dict[str, str]  # demangled name (without crate hashes) -> mangled name
    ir: dict[str, list[str]] = field(default_factory=dict)
    analyses: dict[str, list[str]] = field(default_factory=dict)
    cold: set[str] = field(default_factory=set)  # the `#[cold]` functions, defined or declared


# A build is a whole-crate rustc using all cores (and much memory): two at a time.
BUILD_SLOTS = threading.Semaphore(2)


def compile_crate(platform: Platform, reuse: bool) -> tuple[Path, Path]:
    """Compile jix-py for `platform`, emitting its asm and final IR: the (`.s`, `.ll`). `reuse`: the
    previous build's."""
    target_dir = JIX_PY / "target" / "llvm-mca" / platform.name
    out = target_dir / platform.target / "release"
    if not reuse:
        for f in [*out.rglob("jix-*.s"), *out.rglob("jix-*.ll")]:
            f.unlink()
        # cargo does not re-emit them for an up-to-date crate: rebuild it (only it).
        (JIX_PY / "src" / "lib.rs").touch()
        fake_cc = str(SCRIPT_DIR / "fake_cc.sh")
        t = platform.target.replace("-", "_")
        rustflags = ["-Csymbol-mangling-version=v0"]
        rustflags += [f"-Ctarget-feature={platform.features}"] if platform.features else []
        env = os.environ | {
            "RUSTFLAGS": " ".join(rustflags),
            "CARGO_TARGET_DIR": str(target_dir),
            "RUSTC_BOOTSTRAP": "1",
            "PYO3_CROSS_PYTHON_VERSION": "3.13",
            f"CC_{t}": fake_cc,
            f"AR_{t}": fake_cc,
        }
        # Verbose asm names each block's IR block (`.LBB0_5: # %vector.body`), to match the loops.
        emit = ["--emit=asm,llvm-ir", "-Zverbose-asm"]
        emit += ["-Cllvm-args=-x86-asm-syntax=intel"] if platform.isa == "x86" else []
        cmd = ["cargo", "rustc", "--release", "--lib", "--crate-type", "rlib", "--target", platform.target]
        with BUILD_SLOTS:
            run([*cmd, "--", *emit], env=env, cwd=JIX_PY)
    [asm] = out.rglob("jix-*.s")
    [ll] = out.rglob("jix-*.ll")
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


def demangle(names: list[str]) -> list[str]:
    """`names`, the Rust symbols demangled, without crate hashes and const generic types."""
    # binutils demangles v0 with crate hashes and typed const generics: `jix[0123abcd]`, `32: usize`.
    out = run(["c++filt"], input="\n".join(map(ir_name, names))).stdout.splitlines() if names else []
    return [re.sub(r": (?:usize|bool)\b", "", re.sub(r"\[[0-9a-f]+\]", "", d)) for d in out]


IR_DEFINE_RE = re.compile(r"^(?:define|declare) .*?@(\"[^\"]+\"|[\w.$]+)\(")


def reduce_ir(ll: Path, keep: set[str], opt: str) -> tuple[dict[str, list[str]], dict[str, list[str]], set[str]]:
    """The IR of the functions `keep` (IR names), their `opt` loop and SCEV analyses, and the cold
    functions of the module. `opt` runs once, on the module reduced to `keep` (and what they call):
    the crate's is too large."""
    text = []
    attributes: dict[str, str] = {}  # function -> its attribute group
    cold_groups: set[str] = set()
    for line in ll.read_text().splitlines():
        if m := IR_DEFINE_RE.match(line):
            name = m.group(1).strip('"')
            if group := re.search(r"#(\d+)", line[line.rindex(")") :]):  # after the parameters
                attributes[name] = group.group(1)
            # The kept internal functions become external, so that `internalize` + `globaldce` keep them.
            if name in keep:
                line = re.sub(r"^define (?:internal |private )", "define ", line)
        elif (m := re.match(r"^attributes #(\d+) = \{(.*)\}", line)) and "cold" in m.group(2).split():
            cold_groups.add(m.group(1))
        text.append(line)
    with tempfile.TemporaryDirectory() as tmp:
        full, api, small = Path(tmp) / "full.ll", Path(tmp) / "api.txt", Path(tmp) / "small.ll"
        full.write_text("\n".join(text) + "\n")
        api.write_text("\n".join(keep) + "\n")
        run(
            [
                opt,
                "-S",
                "-passes=internalize,globaldce",
                f"-internalize-public-api-file={api}",
                str(full),
                "-o",
                str(small),
            ]
        )
        ir_text = small.read_text()
        # opt prints analyses to stderr.
        analyses = run([opt, "-disable-output", "-passes=print<loops>,print<scalar-evolution>", str(small)]).stderr

    def by_function(lines: list[str], start: re.Pattern) -> dict[str, list[str]]:
        """The lines from each match of `start` (naming a function) to the next, of the kept ones."""
        out: dict[str, list[str]] = {}
        cur: list[str] | None = None
        for line in lines:
            if m := start.match(line):
                name = next(g for g in m.groups() if g is not None).strip('"')
                cur = out.setdefault(name, []) if name in keep else None
            if cur is not None:
                cur.append(line)
        return out

    ir = by_function(ir_text.splitlines(), IR_DEFINE_RE)
    analysis_start = re.compile(r"Loop info for function '([^']+)'|Classifying expressions for: @(\S+)")
    cold = {f for f, group in attributes.items() if group in cold_groups}
    return ir, by_function(analyses.splitlines(), analysis_start), cold


# --------------------------------------------------------------------------------------------------
# Loop trees


@dataclass
class Loop:
    header: str
    start: int = 0  # asm: line of the first block
    end: int = 0  # asm: line of the last instruction
    blocks: set[str] = field(default_factory=set)  # asm: the IR names of the loop's blocks
    children: list[Loop] = field(default_factory=list)
    trip_count: int | None = None  # if constant
    trip_expr: str = ""  # SCEV backedge-taken count
    ir: IrValues | None = None  # the function's values, to resolve SCEV operands


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
# Rust's functions that never return (panics, aborts), mangled or demangled.
NORETURN_RE = re.compile(r"panic|unwrap_failed|expect_failed|assert_failed|_Unwind_Resume|abort|handle_alloc_error")


def is_label(line: str) -> bool:
    return LABEL_RE.match(line) is not None


def mnemonic(line: str) -> str:
    return line.split()[0]


def is_branch(isa: str, mn: str) -> bool:
    return mn.startswith(("j", "loop") if isa == "x86" else ("b", "cb", "tb"))


def is_call(isa: str, mn: str) -> bool:
    return mn.startswith("call") if isa == "x86" else mn in ("bl", "blr")


def stops(isa: str, line: str) -> bool:
    """A branch, return or call that never falls through."""
    if is_label(line):
        return False
    mn = mnemonic(line)
    jumps = ("jmp", "jmpq", "ret", "retq", "retl", "ud2") if isa == "x86" else ("b", "br", "ret", "udf", "brk")
    return mn in jumps or (is_call(isa, mn) and bool(NORETURN_RE.search(line)))


def clean_lines(asm: list[str], comment: str) -> tuple[list[str], dict[int, str]]:
    """The instructions and labels of a function's `asm`, and the IR name of the block starting at
    each line, from the verbose asm comments: after a label, or on its own line for a block without
    one (`# %bb.3: # %vector.ph`)."""
    out: list[str] = []
    block_names: dict[int, str] = {}
    pending = None
    for line in asm[1:]:
        if comment in line:
            line, note = line[: line.index(comment)], line[line.index(comment) :]
            if m := re.search(r"%([\w.\-$]+)\s*$", note):
                pending = m.group(1)
        s = line.strip()
        if not s or (s.startswith(".") and not is_label(s)):  # directives
            continue
        if pending is not None:
            block_names[len(out)] = pending
        pending = None
        out.append(s)
    return out, block_names


def asm_loops(lines: list[str], isa: str, block_names: dict[int, str]) -> list[Loop]:
    """The loop tree of the asm: its natural loops (back edges to a block that dominates the branch,
    and the blocks that reach them) and the irreducible cycles outside them. A loop spans the lines
    from its first to its last block: a rotated loop entered in the middle is found, and a backward
    jump that is not a back edge (e.g. to a vector epilogue laid out before the main loop) is not a
    loop."""
    labels = {m.group(1): i for i, line in enumerate(lines) if (m := LABEL_RE.match(line))}
    stop = [stops(isa, x) for x in lines]
    ends = [s or (not is_label(x) and is_branch(isa, mnemonic(x))) for s, x in zip(stop, lines)]
    starts = sorted({0, *labels.values(), *(i + 1 for i, e in enumerate(ends) if e)} - {len(lines)})
    blocks = list(zip(starts, [*starts[1:], len(lines)]))  # [start, end)
    block_at = {s: b for b, (s, _) in enumerate(blocks)}
    succs: list[set[int]] = []
    for b, (_, e) in enumerate(blocks):
        last, out = lines[e - 1], set()
        if not is_label(last) and is_branch(isa, mnemonic(last)):
            out |= {block_at[labels[t]] for t in TOKEN_RE.findall(last)[1:] if t in labels}
        if not stop[e - 1] and b + 1 < len(blocks):
            out.add(b + 1)
        succs.append(out)
    preds: list[set[int]] = [set() for _ in blocks]
    for b, out in enumerate(succs):
        for t in out:
            preds[t].add(b)

    # Dominators, iteratively; the blocks unreachable from the entry dominate nothing.
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

    # Irreducible cycles, entered at more than one block (e.g. a rotated remainder loop entered at
    # its body from before the vector loops and at its exit test from them): the strongly connected
    # components outside the natural loops, headed by their first block.
    in_natural = set().union(*bodies.values())
    reach = []
    for b in range(len(blocks)):
        seen, stack = set(), [*succs[b]]
        while stack:
            if (x := stack.pop()) not in seen:
                seen.add(x)
                stack += succs[x]
        reach.append(seen)
    for b in range(len(blocks)):
        if b in reach[b] and b not in in_natural:
            scc = {c for c in reach[b] if b in reach[c]}
            bodies[min(scc)] = scc
            in_natural |= scc

    def make(h: int, body: set[int]) -> Loop:
        names = {block_names[i] for b in body for i in range(*blocks[b]) if i in block_names}
        header = LABEL_RE.match(lines[blocks[h][0]]).group(1)
        return Loop(header, min(blocks[b][0] for b in body), max(blocks[b][1] for b in body) - 1, names)

    loops = sorted((make(h, body) for h, body in bodies.items()), key=lambda lp: (lp.start, -lp.end))
    for a in loops:
        for b in loops:
            if a.start < b.start <= a.end < b.end:
                raise RuntimeError(f"overlapping asm loops {a.header} and {b.header}")
    return build_tree(loops, lambda outer, inner: inner.end <= outer.end)


IR_BINOPS = {
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
IR_FLAGS = {"nuw", "nsw", "exact", "disjoint"}


class IrValues:
    """Constant folding of a function's straight-line integer instructions, for given values of its
    leaves (arguments, loads, phis...): resolves the SCEV operands SCEV does not see through
    (`%_4.i.i`)."""

    def __init__(self, ir: list[str]):
        self.defs: dict[str, str] = {}  # `%name` -> the instruction's right-hand side
        for line in ir:
            if m := re.match(r"^\s+(%[\w.]+) = (.*?)(?:, !.*)?$", line):
                self.defs[m.group(1)] = m.group(2)

    def value(self, name: str, env: dict[str, int], leaves: set[str], depth: int = 0) -> int | None:
        """The value of `%name`, or None, and the leaves it depends on not in `env` added to `leaves`."""
        if name in env:
            return env[name]
        words = [w for w in self.defs.get(name, "").split() if w not in IR_FLAGS]
        op, rest = (words[0], " ".join(words[1:])) if words and depth < 64 else ("", "")
        if op in IR_BINOPS and (m := re.fullmatch(r"i\d+ (\S+), (\S+)", rest)):
            a, b = [self.operand(x, env, leaves, depth + 1) for x in m.groups()]
            try:
                return None if a is None or b is None else IR_BINOPS[op](a, b)
            except ZeroDivisionError:
                return None
        if op in ("zext", "sext", "trunc", "freeze") and (m := re.match(r"i\d+ (\S+)", rest)):
            return self.operand(m.group(1), env, leaves, depth + 1)
        # a stack slot named after an argument (`%len`) holds it
        if op == "load" and (m := re.fullmatch(r"i\d+, ptr (%[\w.]+)(?:, align \d+)?", rest)) and m.group(1) in env:
            return env[m.group(1)]
        leaves.add(name)
        return None

    def operand(self, t: str, env: dict[str, int], leaves: set[str], depth: int) -> int | None:
        if t.startswith("%"):
            return self.value(t, env, leaves, depth)
        return int(t) if re.fullmatch(r"-?\d+", t) else None


def ir_loops(ir: list[str], analyses: list[str]) -> list[Loop]:
    """The IR loop tree of a function, with SCEV trip counts, from its IR and `opt` analyses."""
    loops: list[tuple[int, Loop]] = []
    trips: dict[str, tuple[str, int | None]] = {}
    for line in analyses:
        if m := re.match(r" *Loop at depth (\d+) containing: (.*)$", line):
            header = next(re.sub(r"<[a-z]+>", "", b).lstrip("%") for b in m.group(2).split(",") if "<header>" in b)
            loops.append((int(m.group(1)), Loop(header)))
        elif m := re.match(r"Loop %(\S+): (?:<multiple exits> )?(symbolic max )?backedge-taken count is (.*)$", line):
            # The exact count, else the symbolic max: the loops whose other exits are panics have
            # no exact count, but run their max in the steady state.
            name, expr = m.group(1).strip('"'), m.group(3).strip()
            if not (m.group(2) and name in trips):
                c = re.fullmatch(r"i\d+ (\d+)", expr)
                trips[name] = (expr, int(c.group(1)) + 1 if c else None)
    values = IrValues(ir)
    for _, lp in loops:
        lp.trip_expr, lp.trip_count = trips.get(lp.header, ("", None))
        lp.ir = values
    depth = {id(lp): d for d, lp in loops}
    return build_tree([lp for _, lp in loops], lambda outer, inner: depth[id(inner)] > depth[id(outer)])


def attach_trip_counts(asm_roots: list[Loop], ir_roots: list[Loop]) -> bool:
    """Copy the trip counts of the IR loop tree onto the asm one, matching each IR loop to the asm
    loop holding its header block. False if a loop has no match, or not a single one."""
    if len(asm_roots) != len(ir_roots):
        return False
    matched = set()
    for i in ir_roots:
        ms = [a for a in asm_roots if i.header in a.blocks]
        if len(ms) != 1 or id(ms[0]) in matched:
            return False
        a = ms[0]
        matched.add(id(a))
        a.trip_count, a.trip_expr, a.ir = i.trip_count, i.trip_expr, i.ir
        if not attach_trip_counts(a.children, i.children):
            return False
    return True


def eval_scev(expr: str, env: dict[str, int], ir: IrValues | None = None) -> int | None:
    """Evaluate a SCEV expression, with the values of `env` (and the others folded by `ir`); a
    single leaf value still unknown is `env["%len"]`. None if not evaluable.

    Supports `+ * /u`, the n-ary `umax umin smax smin` and the casts `zext sext trunc`, which SCEV
    always parenthesizes, so innermost groups are evaluated first. SCEV integers are 64-bit, and
    umin, umax and /u unsigned: computed modulo 2^64.
    """
    mask = (1 << 64) - 1
    leaves: set[str] = set()  # unknown

    def signed(v: int) -> int:
        return v - (1 << 64) if v >> 63 else v

    def value(m: re.Match) -> str:
        v = ir.value(m.group(0), env, leaves) if ir else env.get(m.group(0))
        if v is None and not ir:
            leaves.add(m.group(0))
        return str(v)

    ops = {
        " umax ": max,
        " umin ": min,
        " smax ": lambda vs: max(vs, key=signed),
        " smin ": lambda vs: min(vs, key=signed),
        " + ": lambda vs: sum(vs) & mask,
        " * ": lambda vs: math.prod(vs) & mask,
        " /u ": lambda vs: vs[0] // vs[1] if len(vs) == 2 and vs[1] else None,
    }

    def arith(t: str) -> int | None:
        if re.fullmatch(r"-?\d+", t := t.strip()):
            return int(t) & mask
        if m := re.fullmatch(r"(zext|sext|trunc) i(\d+) (.*) to i(\d+)", t):
            if (v := arith(m.group(3))) is None:
                return None
            bits = int(m.group(4) if m.group(1) == "trunc" else m.group(2))
            v &= (1 << bits) - 1
            return (v - (1 << bits) if m.group(1) == "sext" and v >> (bits - 1) else v) & mask
        for op, f in ops.items():
            if op in t:
                vals = [arith(x) for x in t.split(op)]
                return None if None in vals else f(vals)
        return None

    e = re.sub(r"%[\w.]+", value, re.sub(r"<[a-z]+>", "", expr))  # without the wrap flags (<nuw>)
    if leaves:
        if len(leaves) > 1:  # which one is the length?
            return None
        return eval_scev(expr, env | {leaves.pop(): env["%len"]}, ir)
    while (m := re.search(r"\(([^()]*)\)", e)) is not None:
        if (v := arith(m.group(1))) is None:
            return None
        e = e[: m.start()] + str(v) + e[m.end() :]
    v = arith(e)
    return None if v is None else signed(v)


def trips(lp: Loop, env: dict[str, int]) -> int | None:
    """The trip count of `lp`, per entry."""
    if lp.trip_count is not None:
        return lp.trip_count
    b = eval_scev(lp.trip_expr, env, lp.ir)
    return None if b is None or b < 0 else b + 1


def hot_loops(roots: list[Loop], kernel: Kernel) -> list[tuple[Loop, float]]:
    """The loops whose total iteration count is linear in the input length, with their input bytes
    per iteration. A loop's total iteration count is its trip count times its ancestors'. The
    descendants of a hot loop are part of its (flattened) body."""
    env1, env2 = ({"%len": n // kernel.elem_size} for n in (SCEV_EVAL_LEN, 2 * SCEV_EVAL_LEN))
    out = []

    def walk(lp: Loop, mult1: int, mult2: int) -> None:
        if (t1 := trips(lp, env1)) is None or (t2 := trips(lp, env2)) is None:
            return
        n1, n2 = mult1 * t1, mult2 * t2
        # Linear up to a constant (e.g. a vector loop's `(n - remainder) / VF`): the bytes per
        # iteration are the slope between the two lengths.
        if n2 > n1 > 0 and abs(2 * n1 - n2) * 64 <= n1:
            out.append((lp, SCEV_EVAL_LEN / (n2 - n1)))
            return
        for c in lp.children:
            walk(c, n1, n2)

    for r in roots:
        walk(r, 1, 1)
    return out


def flatten(lines: list[str], loop: Loop) -> list[str] | None:
    """The instructions of one iteration of `loop`, inner loops repeated by their trip counts."""
    out: list[str] = []
    i = loop.start
    while i <= loop.end:
        if child := next((c for c in loop.children if c.start == i), None):
            if child.trip_count is None or (body := flatten(lines, child)) is None:
                return None
            out += body * child.trip_count
            i = child.end + 1
            continue
        if not is_label(lines[i]):
            out.append(lines[i])
        i += 1
    return out


# x86-64 loads the GOT entry of a function a loop calls into a register before the loop.
GOT_LOAD_RE = re.compile(r"^mov\s+(\w+), qword ptr \[rip \+ (\S+)@GOTPCREL\]$")


def callees(isa: str, lines: list[str], loops: list[Loop]) -> list[str]:
    """The functions called in `loops`: a register-indirect call calls the function whose GOT entry
    was last loaded into the register."""
    in_loops = {i for lp in loops for i in range(lp.start, lp.end + 1)}
    got: dict[str, str] = {}
    out = set()
    for i, x in enumerate(lines):
        if m := GOT_LOAD_RE.match(x):
            got[m.group(1)] = m.group(2)
        elif i in in_loops and not is_label(x) and is_call(isa, mnemonic(x)):
            target = x.split(None, 1)[1].strip().removesuffix("@PLT")
            out.add(got.get(target, target))
    return sorted(out)


# --------------------------------------------------------------------------------------------------
# llvm-mca


def split_writeback_stores(trace: list[str]) -> list[str]:
    """aarch64: a store with base-register writeback as a plain store and an add (see Caveats)."""
    out = []
    for x in trace:
        if m := re.match(r"^(st\w*)\s+(.*), \[(\w+)\], #(-?\d+)$", x):  # post-index
            op, data, base, imm = m.groups()
            out += [f"{op} {data}, [{base}]", f"add {base}, {base}, #{imm}"]
        elif m := re.match(r"^(st\w*)\s+(.*), \[(\w+), #(-?\d+)\]!$", x):  # pre-index
            op, data, base, imm = m.groups()
            out += [f"add {base}, {base}, #{imm}", f"{op} {data}, [{base}]"]
        else:
            out.append(x)
    return [re.sub(r"^add (\w+), (\w+), #-(\d+)$", r"sub \1, \2, #\3", x) for x in out]


def run_mca(mca: str, platform: Platform, cpu: str, trace: list[str], labels: set[str]) -> tuple[float, list[str], str]:
    """Cycles per iteration of `trace` on `cpu`, the instructions skipped for lack of scheduling
    info, and the report."""
    if platform.isa == "aarch64":
        trace = split_writeback_stores(trace)
    # Direct calls may name demangled Rust paths, which the assembler cannot parse: a stub instead.
    trace = [
        f"{mnemonic(x)} __callee" if is_call(platform.isa, mnemonic(x)) and not re.search(r"[*%]", x) else x
        for x in trace
    ]
    # llvm-mca does not follow branches, but their targets must exist: stub labels after the trace.
    referenced = {t for x in trace for t in TOKEN_RE.findall(x)[1:]} & (labels | {"__callee"})
    src = "\n".join([*trace, *(f"{t}:" for t in sorted(referenced))]) + "\n"
    if platform.isa == "x86":
        src = ".intel_syntax noprefix\n" + src
    cmd = [mca, f"-mtriple={platform.target}", f"-mcpu={cpu}", f"-iterations={MCA_ITERATIONS}"]
    p = run([*cmd, "-bottleneck-analysis", "-skip-unsupported-instructions=lack-sched"], input=src)
    cycles = int(re.search(r"^Total Cycles:\s*(\d+)", p.stdout, re.MULTILINE).group(1))
    skipped = sorted({" ".join(m.split()) for m in re.findall(r"note: instruction:\s*(.*)", p.stderr)})
    return cycles / MCA_ITERATIONS, skipped, p.stdout


# --------------------------------------------------------------------------------------------------
# Driver


@dataclass
class Result:
    platform: str
    kernel: str
    costs: dict[str, float] = field(default_factory=dict)  # cpu -> cycles per BYTES_UNIT; empty: no hot loop
    bytes_per_iter: str = ""  # per hot loop, `+`-joined, as the next two
    instructions: str = ""
    stack: str = ""  # instructions accessing the stack (spills / reloads): `<vector>/<general>` registers
    loops: str = ""  # the asm loop tree, with the constant trip counts
    calls: list[str] = field(default_factory=list)  # in the hot loops
    rust_calls: list[str] = field(default_factory=list)  # Rust functions called in any loop
    warnings: list[str] = field(default_factory=list)


# A stack memory operand (no frame pointer on x86; Apple's aarch64 has one), and a vector register.
STACK_OPERAND = {"x86": re.compile(r"\[(?:rsp|esp)\b"), "aarch64": re.compile(r"\[(?:sp|x29)\b")}
VECTOR_REGISTER = {"x86": re.compile(r"\b[xyz]mm\d+\b"), "aarch64": re.compile(r"\b(?:[qdsbh]\d+|v\d+\.)")}


def stack_ops(isa: str, trace: list[str]) -> str:
    """`<vector>/<general>`: the instructions of `trace` accessing the stack, by the kind of their
    register operand (vector: a vector spill or reload; general: e.g. a reloaded pointer)."""
    ops = [x for x in trace if STACK_OPERAND[isa].search(x)]
    vector = sum(1 for x in ops if VECTOR_REGISTER[isa].search(x))
    return f"{vector}/{len(ops) - vector}"


def describe_tree(loops: list[Loop]) -> str:
    def one(lp: Loop) -> str:
        t = "?" if lp.trip_count is None else str(lp.trip_count)
        return t + (f"[{' '.join(map(one, lp.children))}]" if lp.children else "")

    return " ".join(map(one, loops))


def analyze_kernel(platform: Platform, kernel: Kernel, b: Build, mca: str, out_dir: Path) -> Result:
    r = Result(platform.name, kernel.name)
    mangled = b.names[kernel.symbol]
    asm_path = out_dir / "asm" / platform.name / f"{kernel.name}.s"
    asm_path.parent.mkdir(parents=True, exist_ok=True)
    asm_path.write_text("\n".join(b.asm[mangled]) + "\n")

    lines, block_names = clean_lines(b.asm[mangled], platform.comment)
    try:
        roots = asm_loops(lines, platform.isa, block_names)
    except RuntimeError as e:  # a loop LLVM laid out in non-contiguous pieces
        roots = []
        r.warnings.append(str(e))
    name = ir_name(mangled)
    if not attach_trip_counts(roots, ir_loops(b.ir.get(name, []), b.analyses.get(name, []))):
        r.warnings.append("asm loops not matched to IR loops by block name, trip counts unknown")
    r.loops = describe_tree(roots)
    hot = hot_loops(roots, kernel)
    traces = [flatten(lines, lp) for lp, _ in hot]
    if not hot:
        r.warnings.append("no loop whose iteration count is linear in the input length")
    elif None in traces:
        r.warnings.append("a hot loop with an inner loop of unknown trip count")
    # Rust functions called in any loop, hot or not: they should all be inlined, but for `#[cold]` ones.
    calls = callees(platform.isa, lines, roots)
    if rust := [c for c in calls if re.match(r"_?(_R|_ZN)", c) and ir_name(c) not in b.cold]:
        r.rust_calls = demangle(rust)
        r.warnings += [f"out-of-line call to a Rust function in a loop: `{n}`" for n in r.rust_calls]
    if hot and None not in traces:
        r.bytes_per_iter = "+".join(f"{bpi:g}" for _, bpi in hot)
        r.instructions = "+".join(str(len(t)) for t in traces)
        r.stack = "+".join(stack_ops(platform.isa, t) for t in traces)
        r.calls = demangle(callees(platform.isa, lines, [lp for lp, _ in hot]))
        labels = {m.group(1) for x in lines if (m := LABEL_RE.match(x))}
        for cpu in platform.cpus:
            r.costs[cpu] = 0.0
            for k, ((_, bpi), trace) in enumerate(zip(hot, traces)):
                cycles, skipped, report = run_mca(mca, platform, cpu, trace, labels)
                r.costs[cpu] += cycles * BYTES_UNIT / bpi
                if skipped:
                    r.warnings.append(f"llvm-mca -mcpu={cpu} skipped (no scheduling info): {', '.join(skipped)}")
                p = out_dir / "mca" / platform.name / f"{kernel.name}{f'.{k}' if len(traces) > 1 else ''}.{cpu}.txt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(report)
    for w in r.warnings:
        print(f"warning: {platform.name}/{kernel.name}: {w}", file=sys.stderr)
    return r


def analyze_platform(platform: Platform, args: argparse.Namespace, opt: str, mca: str) -> list[Result]:
    asm_path, ll = compile_crate(platform, args.no_build)
    fns = split_asm(asm_path.read_text())
    b = Build(fns, dict(zip(demangle(list(fns)), fns)))
    kernels = find_kernels(list(b.names), platform.simd, args.kernels)
    # The IR names functions as the asm does, less Mach-O's underscore.
    b.ir, b.analyses, b.cold = reduce_ir(ll, {ir_name(b.names[k.symbol]) for k in kernels}, opt)
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        return list(ex.map(lambda k: analyze_kernel(platform, k, b, mca, args.out), kernels))


def geomean(xs: list[float]) -> float:
    return math.exp(sum(map(math.log, xs)) / len(xs))


def summary(results: list[Result], platforms: list[Platform], header: list[str]) -> str:
    """summary.md: an overview (a row per kernel, a column per platform and CPU), then a table per
    platform."""
    rust_calls = [
        f"- {r.platform}/{r.kernel}: " + ", ".join(f"`{c}`" for c in r.rust_calls) for r in results if r.rust_calls
    ]
    lines = [
        *header,
        "",
        f"llvm-mca steady-state **cycles per {BYTES_UNIT} bytes** of the kernels' hot loops (see",
        "`analyze.py`). Lower is better. `geomean`: per CPU, over the kernels.",
        "",
        "- `B/iter`: input bytes per hot-loop iteration (from the SCEV trip counts).",
        "- `instrs`: instructions per hot-loop iteration (inner loops flattened).",
        "- `stack`: of those, the ones accessing the stack (spills / reloads), `vector/general` registers.",
        "- `loops`: the asm loop tree with trip counts; `?` = runtime trip count.",
        "- `calls`: calls in the hot loop, which llvm-mca counts as free: such costs (and geomeans",
        "  including them) are flagged `*`, the real cost is higher.",
        "",
        "## Out-of-line Rust calls",
        "",
        *(rust_calls or ["None."]),
        "",
    ]
    measured = {p.name: [r for r in results if r.platform == p.name and r.costs] for p in platforms}
    cols = [(p, cpu) for p in platforms if measured[p.name] for cpu in p.cpus]
    by_key = {(r.platform, r.kernel): r for r in results}

    def cost(r: Result | None, cpu: str) -> str:
        return f"{r.costs[cpu]:.0f}{'*' * bool(r.calls)}" if r and r.costs else "-"

    def geo(rs: list[Result], cpu: str) -> str:
        return f"**{geomean([r.costs[cpu] for r in rs]):.0f}{'*' * any(r.calls for r in rs)}**"

    lines += ["## Overview", "", "| kernel | " + " | ".join(f"{p.name} {cpu}" for p, cpu in cols) + " |"]
    lines.append("|---|" + "---:|" * len(cols))
    for name in dict.fromkeys(r.kernel for r in results if r.costs):
        lines.append(f"| {name} | " + " | ".join(cost(by_key.get((p.name, name)), cpu) for p, cpu in cols) + " |")
    lines.append("| **geomean** | " + " | ".join(geo(measured[p.name], cpu) for p, cpu in cols) + " |")
    lines.append("")

    for p in platforms:
        lines += [f"## {p.name} (`{p.target}`)", ""]
        if p.features:
            lines += [f"`-Ctarget-feature={p.features}`", ""]
        if rs := measured[p.name]:
            lines += ["| kernel | B/iter | instrs | stack | loops | calls | " + " | ".join(p.cpus) + " |"]
            lines.append("|---|---:|---:|---:|---|---|" + "---:|" * len(p.cpus))
            for r in rs:
                calls = ", ".join(f"`{c}`" for c in r.calls) or "-"
                row = [
                    r.kernel,
                    r.bytes_per_iter,
                    r.instructions,
                    r.stack,
                    r.loops,
                    calls,
                    *(cost(r, c) for c in p.cpus),
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines += ["| **geomean** | | | | | | " + " | ".join(geo(rs, c) for c in p.cpus) + " |", ""]
        if warns := [f"- {r.kernel}: {w}" for r in results if r.platform == p.name for w in r.warnings]:
            lines += ["Warnings:", "", *warns, ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory (replaced)")
    ap.add_argument("--platform", action="append", choices=[p.name for p in PLATFORMS], help="only these (repeatable)")
    ap.add_argument("--fn", action="append", dest="kernels", help="only the kernels matching these globs (repeatable)")
    ap.add_argument("--no-build", action="store_true", help="reuse the previous builds' asm and IR")
    args = ap.parse_args()
    platforms = [p for p in PLATFORMS if not args.platform or p.name in args.platform]

    mca, opt = find_llvm_mca(llvm_major()), find_opt()
    mca_version = re.search(r"LLVM version [\d.]+", run([mca, "--version"]).stdout).group(0)
    header = [
        f"# llvm-mca: `{args.out.name}`",
        "",
        f"- rustc: `{run(['rustc', '--version']).stdout.strip()}`",
        f"- llvm-mca: `{mca_version}`",
    ]
    if args.out.exists():
        if any(args.out.iterdir()) and not (args.out / "summary.md").exists():
            sys.exit(f"error: {args.out} is not empty and not a previous output")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = [ex.submit(analyze_platform, p, args, opt, mca) for p in platforms]
        results = [r for f in futures for r in f.result()]
    if not results:
        sys.exit("error: no kernel found")

    text = summary(results, platforms, header)
    (args.out / "summary.md").write_text(text + "\n")
    (args.out / "summary.json").write_text(json.dumps([asdict(r) for r in results], indent=1) + "\n")
    print(text)
    if rust := [r for r in results if r.rust_calls]:
        # A kernel must inline its whole op chain: a call in a loop blocks vectorization.
        sys.exit(f"error: {len(rust)} kernels call a Rust function in a loop (see summary.md)")


if __name__ == "__main__":
    main()
