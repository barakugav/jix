jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	mov x8, xzr
	ands x9, x4, #0xfffffffffffffff0
	b.eq .LBB170_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x12, x1, #64
	add x10, x10, #64
	add x11, x11, #64
.LBB170_2:
	ldp q0, q1, [x10, #-64]
	add x8, x8, #16
	ldp q2, q5, [x11, #-64]
	cmp x8, x9
	ldp q3, q4, [x10, #-32]
	ldp q7, q16, [x11]
	fadd v1.2d, v1.2d, v5.2d
	ldp q6, q5, [x11, #-32]
	fadd v0.2d, v0.2d, v2.2d
	fadd v3.2d, v3.2d, v6.2d
	ldp q2, q6, [x10]
	fadd v4.2d, v4.2d, v5.2d
	stp q0, q1, [x12, #-64]
	fadd v2.2d, v2.2d, v7.2d
	fadd v6.2d, v6.2d, v16.2d
	ldp q0, q7, [x10, #32]
	add x10, x10, #128
	ldp q5, q16, [x11, #32]
	stp q3, q4, [x12, #-32]
	add x11, x11, #128
	stp q2, q6, [x12]
	fadd v0.2d, v0.2d, v5.2d
	fadd v1.2d, v7.2d, v16.2d
	stp q0, q1, [x12, #32]
	add x12, x12, #128
	b.lo .LBB170_2
.LBB170_3:
	subs x11, x4, x8
	b.ls .LBB170_10
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x11, #3
	b.ls .LBB170_8
	lsl x13, x8, #3
	and x12, x11, #0xfffffffffffffffc
	and x16, x11, #0xfffffffffffffffc
	add x8, x8, x12
	add x15, x13, #16
	add x13, x9, x15
	add x14, x1, x15
	add x15, x10, x15
.LBB170_6:
	ldp q0, q3, [x15, #-16]
	subs x16, x16, #4
	ldp q1, q2, [x13, #-16]
	add x13, x13, #32
	add x15, x15, #32
	fadd v0.2d, v1.2d, v0.2d
	fadd v1.2d, v2.2d, v3.2d
	stp q0, q1, [x14, #-16]
	add x14, x14, #32
	b.ne .LBB170_6
	cmp x11, x12
	b.eq .LBB170_10
.LBB170_8:
	lsl x12, x8, #3
	sub x8, x4, x8
	add x11, x1, x12
	add x10, x10, x12
	add x9, x9, x12
.LBB170_9:
	ldr d0, [x9], #8
	subs x8, x8, #1
	ldr d1, [x10], #8
	fadd d0, d0, d1
	str d0, [x11], #8
	b.ne .LBB170_9
.LBB170_10:
	ret
