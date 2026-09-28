jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
	mov x8, xzr
	ands x9, x4, #0xffffffffffffffe0
	b.eq .LBB171_3
	ldr x10, [x0, #136]
	add x11, x1, #64
	add x10, x10, #64
.LBB171_2:
	ldp q0, q1, [x10, #-64]
	add x8, x8, #32
	ldp q2, q3, [x10, #-32]
	cmp x8, x9
	ldp q4, q5, [x10]
	fneg v0.4s, v0.4s
	fneg v1.4s, v1.4s
	fneg v2.4s, v2.4s
	fneg v3.4s, v3.4s
	fneg v4.4s, v4.4s
	fneg v5.4s, v5.4s
	stp q0, q1, [x11, #-64]
	ldp q0, q6, [x10, #32]
	stp q2, q3, [x11, #-32]
	add x10, x10, #128
	stp q4, q5, [x11]
	fneg v0.4s, v0.4s
	fneg v1.4s, v6.4s
	stp q0, q1, [x11, #32]
	add x11, x11, #128
	b.lo .LBB171_2
.LBB171_3:
	subs x10, x4, x8
	b.ls .LBB171_10
	ldr x9, [x0, #136]
	cmp x10, #7
	b.ls .LBB171_8
	lsl x12, x8, #2
	and x11, x10, #0xfffffffffffffff8
	and x14, x10, #0xfffffffffffffff8
	add x8, x8, x11
	add x13, x12, #16
	add x12, x1, x13
	add x13, x9, x13
.LBB171_6:
	ldp q0, q1, [x13, #-16]
	subs x14, x14, #8
	add x13, x13, #32
	fneg v0.4s, v0.4s
	fneg v1.4s, v1.4s
	stp q0, q1, [x12, #-16]
	add x12, x12, #32
	b.ne .LBB171_6
	cmp x10, x11
	b.eq .LBB171_10
.LBB171_8:
	lsl x11, x8, #2
	sub x8, x4, x8
	add x10, x1, x11
	add x9, x9, x11
.LBB171_9:
	ldr s0, [x9], #4
	subs x8, x8, #1
	fneg s0, s0
	str s0, [x10], #4
	b.ne .LBB171_9
.LBB171_10:
	ret
