jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>:
Lfunc_begin174:
	mov x8, #0
	ands x9, x4, #0xffffffffffffffe0
	b.eq LBB174_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x10, x10, #64
	add x11, x11, #64
	add x12, x1, #64
LBB174_2:
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
	stp q0, q1, [x12, #-64]
	stp q2, q3, [x12, #-32]
	stp q4, q5, [x12]
	add x8, x8, #32
	add x10, x10, #128
	add x11, x11, #128
	stp q6, q7, [x12, #32]
	add x12, x12, #128
	cmp x8, x9
	b.lo LBB174_2
LBB174_3:
	subs x11, x4, x8
	b.ls LBB174_17
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x11, #3
	b.hi LBB174_6
	mov x12, x8
	b LBB174_15
LBB174_6:
	cmp x11, #16
	b.hs LBB174_8
	mov x13, #0
	b LBB174_12
LBB174_8:
	and x14, x11, #0xc
	and x13, x11, #0xfffffffffffffff0
	add x12, x8, x13
	lsl x15, x8, #2
	add x17, x15, #32
	add x15, x9, x17
	add x16, x10, x17
	add x17, x1, x17
	and x0, x11, #0xfffffffffffffff0
LBB174_9:
	ldp q0, q1, [x15, #-32]
	ldp q2, q3, [x15], #64
	ldp q4, q5, [x16, #-32]
	ldp q6, q7, [x16], #64
	fadd.4s v0, v0, v4
	fadd.4s v1, v1, v5
	fadd.4s v2, v2, v6
	fadd.4s v3, v3, v7
	stp q0, q1, [x17, #-32]
	stp q2, q3, [x17], #64
	subs x0, x0, #16
	b.ne LBB174_9
	cmp x11, x13
	b.eq LBB174_17
	cbz x14, LBB174_15
LBB174_12:
	and x14, x11, #0xfffffffffffffffc
	add x12, x8, x14
	sub x15, x13, x14
	add x8, x13, x8
	lsl x16, x8, #2
	add x8, x1, x16
	add x13, x10, x16
	add x16, x9, x16
LBB174_13:
	ldr q0, [x16], #16
	ldr q1, [x13], #16
	fadd.4s v0, v0, v1
	str q0, [x8], #16
	adds x15, x15, #4
	b.ne LBB174_13
	cmp x11, x14
	b.eq LBB174_17
LBB174_15:
	sub x8, x4, x12
	lsl x12, x12, #2
	add x11, x1, x12
	add x10, x10, x12
	add x9, x9, x12
LBB174_16:
	ldr s0, [x9], #4
	ldr s1, [x10], #4
	fadd s0, s0, s1
	str s0, [x11], #4
	subs x8, x8, #1
	b.ne LBB174_16
LBB174_17:
	ret
