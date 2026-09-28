jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
Lfunc_begin170:
	mov x8, #0
	ands x9, x4, #0xfffffffffffffff0
	b.eq LBB170_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x10, x10, #64
	add x11, x11, #64
	add x12, x1, #64
LBB170_2:
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
	stp q0, q1, [x12, #-64]
	stp q2, q3, [x12, #-32]
	stp q4, q5, [x12]
	add x8, x8, #16
	add x10, x10, #128
	add x11, x11, #128
	stp q6, q7, [x12, #32]
	add x12, x12, #128
	cmp x8, x9
	b.lo LBB170_2
LBB170_3:
	subs x11, x4, x8
	b.ls LBB170_11
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x11, #7
	b.hi LBB170_6
	mov x12, x8
	b LBB170_9
LBB170_6:
	and x13, x11, #0xfffffffffffffff8
	add x12, x8, x13
	lsl x8, x8, #3
	add x15, x8, #32
	add x8, x9, x15
	add x14, x10, x15
	add x15, x1, x15
	and x16, x11, #0xfffffffffffffff8
LBB170_7:
	ldp q0, q1, [x8, #-32]
	ldp q2, q3, [x8], #64
	ldp q4, q5, [x14, #-32]
	ldp q6, q7, [x14], #64
	fadd.2d v0, v0, v4
	fadd.2d v1, v1, v5
	fadd.2d v2, v2, v6
	fadd.2d v3, v3, v7
	stp q0, q1, [x15, #-32]
	stp q2, q3, [x15], #64
	subs x16, x16, #8
	b.ne LBB170_7
	cmp x11, x13
	b.eq LBB170_11
LBB170_9:
	sub x8, x4, x12
	lsl x12, x12, #3
	add x11, x1, x12
	add x10, x10, x12
	add x9, x9, x12
LBB170_10:
	ldr d0, [x9], #8
	ldr d1, [x10], #8
	fadd d0, d0, d1
	str d0, [x11], #8
	subs x8, x8, #1
	b.ne LBB170_10
LBB170_11:
	ret
