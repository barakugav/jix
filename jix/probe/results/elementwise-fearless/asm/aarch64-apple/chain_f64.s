jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
Lfunc_begin168:
	mov x8, #0
	ands x9, x4, #0xfffffffffffffff0
	b.eq LBB168_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	ldr x12, [x0, #448]
	add x10, x10, #64
	ldr x13, [x0, #608]
	add x11, x11, #64
	add x12, x12, #64
	add x13, x13, #64
	add x14, x1, #64
LBB168_2:
	ldp q0, q1, [x10, #-64]
	ldp q2, q3, [x10, #-32]
	ldp q4, q5, [x10]
	ldp q6, q7, [x10, #32]
	ldp q16, q17, [x11, #-64]
	ldp q18, q19, [x11, #-32]
	ldp q20, q21, [x11]
	ldp q22, q23, [x11, #32]
	fadd.2d v0, v0, v16
	fadd.2d v1, v1, v17
	fadd.2d v2, v2, v18
	fadd.2d v3, v3, v19
	fadd.2d v4, v4, v20
	fadd.2d v5, v5, v21
	fadd.2d v6, v6, v22
	fadd.2d v7, v7, v23
	ldp q16, q17, [x12, #-64]
	ldp q18, q19, [x12, #-32]
	ldp q20, q21, [x12]
	ldp q22, q23, [x12, #32]
	fmul.2d v0, v0, v16
	fmul.2d v1, v1, v17
	fmul.2d v2, v2, v18
	fmul.2d v3, v3, v19
	fmul.2d v4, v4, v20
	fmul.2d v5, v5, v21
	fmul.2d v6, v6, v22
	fmul.2d v7, v7, v23
	ldp q16, q17, [x13, #-64]
	ldp q18, q19, [x13, #-32]
	ldp q20, q21, [x13]
	ldp q22, q23, [x13, #32]
	fsub.2d v0, v0, v16
	fsub.2d v1, v1, v17
	fsub.2d v2, v2, v18
	fsub.2d v3, v3, v19
	fsub.2d v4, v4, v20
	fsub.2d v5, v5, v21
	fsub.2d v6, v6, v22
	fsub.2d v7, v7, v23
	stp q0, q1, [x14, #-64]
	stp q2, q3, [x14, #-32]
	stp q4, q5, [x14]
	add x8, x8, #16
	add x10, x10, #128
	add x11, x11, #128
	stp q6, q7, [x14, #32]
	add x12, x12, #128
	add x13, x13, #128
	add x14, x14, #128
	cmp x8, x9
	b.lo LBB168_2
LBB168_3:
	subs x13, x4, x8
	b.ls LBB168_11
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	ldr x11, [x0, #448]
	ldr x12, [x0, #608]
	cmp x13, #7
	b.hi LBB168_6
	mov x14, x8
	b LBB168_9
LBB168_6:
	and x15, x13, #0xfffffffffffffff8
	add x14, x8, x15
	lsl x8, x8, #3
	add x2, x8, #32
	add x8, x9, x2
	add x16, x10, x2
	add x17, x1, x2
	add x0, x11, x2
	add x2, x12, x2
	and x3, x13, #0xfffffffffffffff8
LBB168_7:
	ldp q0, q1, [x8, #-32]
	ldp q2, q3, [x8], #64
	ldp q4, q5, [x16, #-32]
	ldp q6, q7, [x16], #64
	fadd.2d v0, v0, v4
	fadd.2d v1, v1, v5
	fadd.2d v2, v2, v6
	fadd.2d v3, v3, v7
	ldp q4, q5, [x0, #-32]
	ldp q6, q7, [x0], #64
	fmul.2d v0, v0, v4
	fmul.2d v1, v1, v5
	fmul.2d v2, v2, v6
	fmul.2d v3, v3, v7
	ldp q4, q5, [x2, #-32]
	ldp q6, q7, [x2], #64
	fsub.2d v0, v0, v4
	fsub.2d v1, v1, v5
	fsub.2d v2, v2, v6
	fsub.2d v3, v3, v7
	stp q0, q1, [x17, #-32]
	stp q2, q3, [x17], #64
	subs x3, x3, #8
	b.ne LBB168_7
	cmp x13, x15
	b.eq LBB168_11
LBB168_9:
	sub x8, x4, x14
	lsl x14, x14, #3
	add x13, x1, x14
	add x12, x12, x14
	add x11, x11, x14
	add x10, x10, x14
	add x9, x9, x14
LBB168_10:
	ldr d0, [x9], #8
	ldr d1, [x10], #8
	fadd d0, d0, d1
	ldr d1, [x11], #8
	fmul d0, d0, d1
	ldr d1, [x12], #8
	fsub d0, d0, d1
	str d0, [x13], #8
	subs x8, x8, #1
	b.ne LBB168_10
LBB168_11:
	ret
