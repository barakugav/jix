jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
Lfunc_begin167:
	mov x8, #0
	ands x9, x4, #0xfffffffffffffff0
	b.eq LBB167_3
	ldr x10, [x0, #136]
	add x10, x10, #64
	add x11, x1, #64
LBB167_2:
	ldp q0, q1, [x10, #-64]
	ldp q2, q3, [x10, #-32]
	ldp q4, q5, [x10]
	ldp q6, q7, [x10, #32]
	fneg.2d v0, v0
	fneg.2d v1, v1
	fneg.2d v2, v2
	fneg.2d v3, v3
	fneg.2d v4, v4
	fneg.2d v5, v5
	fneg.2d v6, v6
	fneg.2d v7, v7
	stp q0, q1, [x11, #-64]
	stp q2, q3, [x11, #-32]
	stp q4, q5, [x11]
	add x8, x8, #16
	add x10, x10, #128
	stp q6, q7, [x11, #32]
	add x11, x11, #128
	cmp x8, x9
	b.lo LBB167_2
LBB167_3:
	subs x10, x4, x8
	b.ls LBB167_11
	ldr x9, [x0, #136]
	cmp x10, #7
	b.hi LBB167_6
	mov x11, x8
	b LBB167_9
LBB167_6:
	and x12, x10, #0xfffffffffffffff8
	add x11, x8, x12
	lsl x8, x8, #3
	add x13, x8, #32
	add x8, x9, x13
	add x13, x1, x13
	and x14, x10, #0xfffffffffffffff8
LBB167_7:
	ldp q0, q1, [x8, #-32]
	ldp q2, q3, [x8], #64
	fneg.2d v0, v0
	fneg.2d v1, v1
	fneg.2d v2, v2
	fneg.2d v3, v3
	stp q0, q1, [x13, #-32]
	stp q2, q3, [x13], #64
	subs x14, x14, #8
	b.ne LBB167_7
	cmp x10, x12
	b.eq LBB167_11
LBB167_9:
	sub x8, x4, x11
	lsl x11, x11, #3
	add x10, x1, x11
	add x9, x9, x11
LBB167_10:
	ldr d0, [x9], #8
	fneg d0, d0
	str d0, [x10], #8
	subs x8, x8, #1
	b.ne LBB167_10
LBB167_11:
	ret
