jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
Lfunc_begin143:
	mov x8, #0
	ands x9, x4, #0xffffffffffffffe0
	b.eq LBB143_3
	ldr x10, [x0, #136]
	add x10, x10, #64
	add x11, x1, #64
LBB143_2:
	ldp q0, q1, [x10, #-64]
	fneg.4s v0, v0
	fneg.4s v1, v1
	stp q0, q1, [x11, #-64]
	ldp q0, q1, [x10, #-32]
	fneg.4s v0, v0
	fneg.4s v1, v1
	stp q0, q1, [x11, #-32]
	ldp q0, q1, [x10]
	fneg.4s v0, v0
	fneg.4s v1, v1
	stp q0, q1, [x11]
	ldp q0, q1, [x10, #32]
	fneg.4s v0, v0
	fneg.4s v1, v1
	add x8, x8, #32
	add x10, x10, #128
	stp q0, q1, [x11, #32]
	add x11, x11, #128
	cmp x8, x9
	b.lo LBB143_2
LBB143_3:
	subs x10, x4, x8
	b.ls LBB143_17
	ldr x9, [x0, #136]
	cmp x10, #3
	b.hi LBB143_6
	mov x11, x8
	b LBB143_15
LBB143_6:
	cmp x10, #16
	b.hs LBB143_8
	mov x12, #0
	b LBB143_12
LBB143_8:
	and x13, x10, #0xc
	and x12, x10, #0xfffffffffffffff0
	add x11, x8, x12
	lsl x14, x8, #2
	add x15, x14, #32
	add x14, x9, x15
	add x15, x1, x15
	and x16, x10, #0xfffffffffffffff0
LBB143_9:
	ldp q0, q1, [x14, #-32]
	ldp q2, q3, [x14], #64
	fneg.4s v0, v0
	fneg.4s v1, v1
	fneg.4s v2, v2
	fneg.4s v3, v3
	stp q0, q1, [x15, #-32]
	stp q2, q3, [x15], #64
	subs x16, x16, #16
	b.ne LBB143_9
	cmp x10, x12
	b.eq LBB143_17
	cbz x13, LBB143_15
LBB143_12:
	and x13, x10, #0xfffffffffffffffc
	add x11, x8, x13
	sub x14, x12, x13
	add x8, x12, x8
	lsl x12, x8, #2
	add x8, x1, x12
	add x12, x9, x12
LBB143_13:
	ldr q0, [x12], #16
	fneg.4s v0, v0
	str q0, [x8], #16
	adds x14, x14, #4
	b.ne LBB143_13
	cmp x10, x13
	b.eq LBB143_17
LBB143_15:
	sub x8, x4, x11
	lsl x11, x11, #2
	add x10, x1, x11
	add x9, x9, x11
LBB143_16:
	ldr s0, [x9], #4
	fneg s0, s0
	str s0, [x10], #4
	subs x8, x8, #1
	b.ne LBB143_16
LBB143_17:
	ret
