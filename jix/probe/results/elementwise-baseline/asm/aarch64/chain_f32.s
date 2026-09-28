jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
	mov x8, xzr
	ands x9, x4, #0xffffffffffffffe0
	b.eq .LBB144_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x14, x1, #64
	ldr x12, [x0, #448]
	ldr x13, [x0, #608]
	add x10, x10, #64
	add x11, x11, #64
	add x12, x12, #64
	add x13, x13, #64
.LBB144_2:
	ldp q0, q3, [x11, #-64]
	add x8, x8, #32
	ldp q1, q2, [x10, #-64]
	cmp x8, x9
	ldp q4, q5, [x12, #-64]
	ldp q6, q17, [x11, #-32]
	fadd v0.4s, v1.4s, v0.4s
	ldp q1, q16, [x10, #-32]
	fadd v2.4s, v2.4s, v3.4s
	ldp q3, q7, [x13, #-64]
	fadd v1.4s, v1.4s, v6.4s
	ldp q6, q18, [x10]
	fmul v0.4s, v0.4s, v4.4s
	ldp q4, q19, [x11]
	fadd v16.4s, v16.4s, v17.4s
	fmul v2.4s, v2.4s, v5.4s
	ldp q5, q17, [x12, #-32]
	fadd v4.4s, v6.4s, v4.4s
	fadd v18.4s, v18.4s, v19.4s
	fsub v0.4s, v0.4s, v3.4s
	ldp q6, q3, [x11, #32]
	fmul v1.4s, v1.4s, v5.4s
	ldp q5, q19, [x10, #32]
	fsub v2.4s, v2.4s, v7.4s
	add x10, x10, #128
	add x11, x11, #128
	fadd v5.4s, v5.4s, v6.4s
	fmul v6.4s, v16.4s, v17.4s
	fadd v3.4s, v19.4s, v3.4s
	ldp q7, q16, [x12]
	ldp q17, q19, [x13, #-32]
	stp q0, q2, [x14, #-64]
	fmul v4.4s, v4.4s, v7.4s
	fmul v16.4s, v18.4s, v16.4s
	ldp q7, q18, [x12, #32]
	fsub v1.4s, v1.4s, v17.4s
	fsub v6.4s, v6.4s, v19.4s
	add x12, x12, #128
	fmul v5.4s, v5.4s, v7.4s
	ldp q7, q17, [x13]
	fmul v3.4s, v3.4s, v18.4s
	stp q1, q6, [x14, #-32]
	fsub v4.4s, v4.4s, v7.4s
	ldp q7, q2, [x13, #32]
	fsub v0.4s, v16.4s, v17.4s
	add x13, x13, #128
	fsub v5.4s, v5.4s, v7.4s
	fsub v1.4s, v3.4s, v2.4s
	stp q4, q0, [x14]
	stp q5, q1, [x14, #32]
	add x14, x14, #128
	b.lo .LBB144_2
.LBB144_3:
	subs x13, x4, x8
	b.ls .LBB144_10
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x13, #7
	ldr x11, [x0, #448]
	ldr x12, [x0, #608]
	b.ls .LBB144_8
	lsl x15, x8, #2
	and x14, x13, #0xfffffffffffffff8
	and x2, x13, #0xfffffffffffffff8
	add x8, x8, x14
	add x0, x15, #16
	add x15, x9, x0
	add x16, x1, x0
	add x17, x10, x0
	add x18, x12, x0
	add x0, x11, x0
.LBB144_6:
	ldp q0, q3, [x17, #-16]
	subs x2, x2, #8
	ldp q1, q2, [x15, #-16]
	add x15, x15, #32
	add x17, x17, #32
	fadd v0.4s, v1.4s, v0.4s
	fadd v1.4s, v2.4s, v3.4s
	ldp q2, q3, [x0, #-16]
	add x0, x0, #32
	fmul v0.4s, v0.4s, v2.4s
	fmul v1.4s, v1.4s, v3.4s
	ldp q2, q3, [x18, #-16]
	add x18, x18, #32
	fsub v0.4s, v0.4s, v2.4s
	fsub v1.4s, v1.4s, v3.4s
	stp q0, q1, [x16, #-16]
	add x16, x16, #32
	b.ne .LBB144_6
	cmp x13, x14
	b.eq .LBB144_10
.LBB144_8:
	lsl x14, x8, #2
	sub x8, x4, x8
	add x13, x1, x14
	add x12, x12, x14
	add x11, x11, x14
	add x10, x10, x14
	add x9, x9, x14
.LBB144_9:
	ldr s0, [x9], #4
	subs x8, x8, #1
	ldr s1, [x10], #4
	fadd s0, s0, s1
	ldr s1, [x11], #4
	fmul s0, s0, s1
	ldr s1, [x12], #4
	fsub s0, s0, s1
	str s0, [x13], #4
	b.ne .LBB144_9
.LBB144_10:
	ret
