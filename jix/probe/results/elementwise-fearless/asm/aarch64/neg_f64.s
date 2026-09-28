jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
	mov x8, xzr
	ands x9, x4, #0xfffffffffffffff0
	b.eq .LBB167_3
	ldr x10, [x0, #136]
	add x11, x1, #64
	add x10, x10, #64
.LBB167_2:
	ldp q0, q1, [x10, #-64]
	add x8, x8, #16
	ldp q2, q3, [x10, #-32]
	cmp x8, x9
	ldp q4, q5, [x10]
	fneg v0.2d, v0.2d
	fneg v1.2d, v1.2d
	fneg v2.2d, v2.2d
	fneg v3.2d, v3.2d
	fneg v4.2d, v4.2d
	fneg v5.2d, v5.2d
	stp q0, q1, [x11, #-64]
	ldp q0, q6, [x10, #32]
	stp q2, q3, [x11, #-32]
	add x10, x10, #128
	stp q4, q5, [x11]
	fneg v0.2d, v0.2d
	fneg v1.2d, v6.2d
	stp q0, q1, [x11, #32]
	add x11, x11, #128
	b.lo .LBB167_2
.LBB167_3:
	subs x10, x4, x8
	b.ls .LBB167_10
	ldr x9, [x0, #136]
	cmp x10, #3
	b.ls .LBB167_8
	lsl x12, x8, #3
	and x11, x10, #0xfffffffffffffffc
	and x14, x10, #0xfffffffffffffffc
	add x8, x8, x11
	add x13, x12, #16
	add x12, x1, x13
	add x13, x9, x13
.LBB167_6:
	ldp q0, q1, [x13, #-16]
	subs x14, x14, #4
	add x13, x13, #32
	fneg v0.2d, v0.2d
	fneg v1.2d, v1.2d
	stp q0, q1, [x12, #-16]
	add x12, x12, #32
	b.ne .LBB167_6
	cmp x10, x11
	b.eq .LBB167_10
.LBB167_8:
	lsl x11, x8, #3
	sub x8, x4, x8
	add x10, x1, x11
	add x9, x9, x11
.LBB167_9:
	ldr d0, [x9], #8
	subs x8, x8, #1
	fneg d0, d0
	str d0, [x10], #8
	b.ne .LBB167_9
.LBB167_10:
	ret
