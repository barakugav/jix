import json
import subprocess
import sys

from analyze_llvm_mca import analyze as a

ASM = """\
_RNvCs123_3foo:
\txor eax, eax
.LBB0_1:                                # %outer
\txor ecx, ecx
\t.p2align 4
.LBB0_2:                                # %inner
\tadd ecx, 1                            # comment
\tcmp ecx, 4
\tjne .LBB0_2
# %bb.3:                                # %outer.latch
\tadd eax, 1
\tcmp eax, edi
\tjne .LBB0_1
# %bb.4:                                # %exit
\tret
"""
ANALYSES = """\
Loop info for function 'foo':
Loop at depth 1 containing: %outer<header>,%inner,%outer.latch<latch><exiting>
    Loop at depth 2 containing: %inner<header><latch><exiting>
Classifying expressions for: @foo
Loop %inner: backedge-taken count is i32 3
Loop %outer: backedge-taken count is (-1 + %len)
Loop %outer: symbolic max backedge-taken count is (-1 + %len)
"""


def test_loops_trip_counts_and_flattened_hot_loop():
    lines, block_names = a.clean_lines(ASM.splitlines(), "#")
    assert ".p2align 4" not in lines and "add ecx, 1" in lines
    roots = a.asm_loops(lines, "x86", block_names)
    assert [lp.header for lp in roots] == [".LBB0_1"] and [lp.header for lp in roots[0].children] == [".LBB0_2"]
    assert a.attach_trip_counts(roots, a.ir_loops([], ANALYSES.splitlines()))
    assert a.describe_tree(roots) == "?[4]"
    [(hot, bytes_per_iter)] = a.hot_loops(roots, a.Kernel("foo", "foo", elem_size=4))
    assert hot is roots[0] and bytes_per_iter == 4
    trace = a.flatten(lines, hot)
    assert len(trace) == 1 + 3 * 4 + 3 and trace[-1] == "jne .LBB0_1"


def test_eval_scev():
    env = {"%len": 1000}
    assert a.eval_scev("((-1 + %len) /u 8)", env) == 124
    assert a.eval_scev("(-1 + (8 umax %len))<nsw>", env) == 999
    assert a.eval_scev("(zext i32 (-1 + (trunc i64 %len to i32)) to i64)", env) == 999
    assert a.eval_scev("(zext i3 (trunc i64 %len to i3) to i64)", env) == 1000 % 8
    assert a.eval_scev("(sext i8 (trunc i64 %len to i8) to i64)", env) == 1000 % 256 - 256
    # A single unknown value is the length; two are not evaluable.
    assert a.eval_scev("(%1 /u 4)", env) == 250
    assert a.eval_scev("(%a /u %b)", env) is None
    # Folded from the IR: a value SCEV does not see through.
    ir = a.IrValues(["define void @f(ptr %p, i64 %0) {", "  %n.i = and i64 %0, -16", "  %m = lshr exact i64 %n.i, 4"])
    assert a.eval_scev("(-1 + %m)", env | {"%0": 1000}, ir) == 61
    assert a.eval_scev("(-1 + %m)", env, ir) == 61  # `%0`, the single unknown leaf, is the length
    assert a.eval_scev("(%m + %phi)", env, ir) is None


def test_kernel_names():
    ew = a.EW
    leaf = f"{ew}::OperandTyped<f32>"
    op2 = "<jix::ops::op2::Op2<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline"
    add = f"{op2}<{leaf}, {leaf}, jix::ops::op2::AddKernel, f32, f32>"
    assert a.elementwise_kernel(f"{ew}::inner_loop::<f32, {add}, true, true>").name == "add_f32"
    assert a.elementwise_kernel(f"{ew}::inner_loop::<f32, {add}, true, false>") is None
    less = f"{op2}<{leaf}, {leaf}, jix::ops::cmp::LessKernel, f32, f32>"
    assert (
        a.elementwise_kernel(f"{ew}::inner_loop::<jix::scalar::bool::bool8, {less}, true, true>").name
        == "less_f32_bool"
    )
    clamp = (
        "<jix::ops::cmp::Clamp<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline"
        f"::ClampPipeline<{leaf}, {leaf}, {leaf}, f32>"
    )
    assert a.elementwise_kernel(f"{ew}::inner_loop::<f32, {clamp}, true, true>").name == "clamp_f32"
    nested = f"{op2}<{add}, {leaf}, jix::ops::op2::MulKernel, f32, f32>"
    assert a.elementwise_kernel(f"{ew}::inner_loop::<f32, {nested}, true, true>").name == "mul(add(f32,f32),f32)"

    leaf_fold = "jix::ops::reduction::fold_pairwise::<f32, jix::ops::reduction::ArgMaxKernel, 16, true, fearless_simd::generated::avx2::Avx2>::{closure#0}"
    assert a.reduction_kernel(leaf_fold, "Avx2").name == "reduce_arg_max_f32_cell"
    assert a.reduction_kernel(leaf_fold, "Sse4_2") is None
    update = "jix::ops::reduction::fold_multi_cell_contiguous::<u8, jix::ops::reduction::SumKernel, false>"
    k = a.reduction_kernel(update, "Neon")
    assert (k.name, k.elem_size) == ("reduce_sum_u8_update", 1)


def test_split_writeback_stores():
    trace = ["str q0, [x0], #16", "stp q0, q1, [x1, #-32]!", "ldr q2, [x2], #16"]
    assert a.split_writeback_stores(trace) == [
        "str q0, [x0]",
        "add x0, x0, #16",
        "sub x1, x1, #32",
        "stp q0, q1, [x1]",
        "ldr q2, [x2], #16",
    ]


def test_compare(tmp_path):
    def write(name, costs):
        d = tmp_path / name
        d.mkdir()
        rows = [{"platform": "x86_64", "kernel": k, "costs": {"skylake": c}, "calls": []} for k, c in costs.items()]
        (d / "summary.json").write_text(json.dumps(rows))
        return d

    new, base = write("new", {"add_f32": 100.0, "neg_f32": 50.0}), write("base", {"add_f32": 200.0, "neg_f32": 50.0})
    out = subprocess.check_output([sys.executable, str(a.SCRIPT_DIR / "compare.py"), new, base, "--kernels"], text=True)
    assert "| x86_64 | skylake | 2 | **1.41** | 50% | 0% |" in out
    assert "| add_f32 | 2.00 |" in out
