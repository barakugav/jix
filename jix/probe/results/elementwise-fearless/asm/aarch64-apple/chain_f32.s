jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
Lfunc_begin172:
	mov x8, #0
	ands x9, x4, #0xffffffffffffffe0
	b.eq LBB172_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	ldr x12, [x0, #448]
	add x10, x10, #64
	ldr x13, [x0, #608]
	add x11, x11, #64
	add x12, x12, #64
	add x13, x13, #64
	add x14, x1, #64
LBB172_2:
	ldp q0, q1, [x10, #-64]
	ldp q2, q3, [x10, #-32]
	ldp q4, q5, [x10]
	ldp q6, q7, [x10, #32]
	ldp q16, q17, [x11, #-64]
	ldp q18, q19, [x11, #-32]
	ldp q20, q21, [x11]
	ldp q22, q23, [x11, #32]
	fadd.4s v0, v0, v16
	fadd.4s v1, v1, v17
	fadd.4s v2, v2, v18
	fadd.4s v3, v3, v19
	fadd.4s v4, v4, v20
	fadd.4s v5, v5, v21
	fadd.4s v6, v6, v22
	fadd.4s v7, v7, v23
	ldp q16, q17, [x12, #-64]
	ldp q18, q19, [x12, #-32]
	ldp q20, q21, [x12]
	ldp q22, q23, [x12, #32]
	fmul.4s v0, v0, v16
	fmul.4s v1, v1, v17
	fmul.4s v2, v2, v18
	fmul.4s v3, v3, v19
	fmul.4s v4, v4, v20
	fmul.4s v5, v5, v21
	fmul.4s v6, v6, v22
	fmul.4s v7, v7, v23
	ldp q16, q17, [x13, #-64]
	ldp q18, q19, [x13, #-32]
	ldp q20, q21, [x13]
	ldp q22, q23, [x13, #32]
	fsub.4s v0, v0, v16
	fsub.4s v1, v1, v17
	fsub.4s v2, v2, v18
	fsub.4s v3, v3, v19
	fsub.4s v4, v4, v20
	fsub.4s v5, v5, v21
	fsub.4s v6, v6, v22
	fsub.4s v7, v7, v23
	stp q0, q1, [x14, #-64]
	stp q2, q3, [x14, #-32]
	add x8, x8, #32
	add x10, x10, #128
	stp q4, q5, [x14]
	add x11, x11, #128
	add x12, x12, #128
	add x13, x13, #128
	stp q6, q7, [x14, #32]
	add x14, x14, #128
	cmp x8, x9
	b.lo LBB172_2
LBB172_3:
	subs x13, x4, x8
	b.ls LBB172_17
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	ldr x11, [x0, #448]
	ldr x12, [x0, #608]
	cmp x13, #3
	b.hi LBB172_6
	mov x14, x8
	b LBB172_15
LBB172_6:
	cmp x13, #16
	b.hs LBB172_8
	mov x15, #0
	b LBB172_12
LBB172_8:
	and x16, x13, #0xc
	and x15, x13, #0xfffffffffffffff0
	add x14, x8, x15
	lsl x17, x8, #2
	add x5, x17, #32
	add x17, x9, x5
	add x0, x10, x5
	add x2, x1, x5
	add x3, x11, x5
	add x5, x12, x5
	and x6, x13, #0xfffffffffffffff0
LBB172_9:
	ldp q0, q1, [x17, #-32]
	ldp q2, q3, [x17], #64
	ldp q4, q5, [x0, #-32]
	ldp q6, q7, [x0], #64
	fadd.4s v0, v0, v4
	fadd.4s v1, v1, v5
	fadd.4s v2, v2, v6
	fadd.4s v3, v3, v7
	ldp q4, q5, [x3, #-32]
	ldp q6, q7, [x3], #64
	fmul.4s v0, v0, v4
	fmul.4s v1, v1, v5
	fmul.4s v2, v2, v6
	fmul.4s v3, v3, v7
	ldp q4, q5, [x5, #-32]
	ldp q6, q7, [x5], #64
	fsub.4s v0, v0, v4
	fsub.4s v1, v1, v5
	fsub.4s v2, v2, v6
	fsub.4s v3, v3, v7
	stp q0, q1, [x2, #-32]
	stp q2, q3, [x2], #64
	subs x6, x6, #16
	b.ne LBB172_9
	cmp x13, x15
	b.eq LBB172_17
	cbz x16, LBB172_15
LBB172_12:
	and x16, x13, #0xfffffffffffffffc
	add x14, x8, x16
	sub x17, x15, x16
	add x8, x15, x8
	lsl x3, x8, #2
	add x8, x1, x3
	add x15, x12, x3
	add x0, x11, x3
	add x2, x10, x3
	add x3, x9, x3
LBB172_13:
	ldr q0, [x3], #16
	ldr q1, [x2], #16
	fadd.4s v0, v0, v1
	ldr q1, [x0], #16
	fmul.4s v0, v0, v1
	ldr q1, [x15], #16
	fsub.4s v0, v0, v1
	str q0, [x8], #16
	adds x17, x17, #4
	b.ne LBB172_13
	cmp x13, x16
	b.eq LBB172_17
LBB172_15:
	sub x8, x4, x14
	lsl x14, x14, #2
	add x13, x1, x14
	add x12, x12, x14
	add x11, x11, x14
	add x10, x10, x14
	add x9, x9, x14
LBB172_16:
	ldr s0, [x9], #4
	ldr s1, [x10], #4
	fadd s0, s0, s1
	ldr s1, [x11], #4
	fmul s0, s0, s1
	ldr s1, [x12], #4
	fsub s0, s0, s1
	str s0, [x13], #4
	subs x8, x8, #1
	b.ne LBB172_16
LBB172_17:
	ret
