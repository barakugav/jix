jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
Lfunc_begin160:
	mov x8, #0
	ands x9, x4, #0xffffffffffffffe0
	b.eq LBB160_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	ldr x12, [x0, #448]
	add x10, x10, #64
	ldr x13, [x0, #608]
	add x11, x11, #64
	add x12, x12, #64
	add x13, x13, #64
	add x14, x1, #64
LBB160_2:
	ldp q0, q1, [x10, #-64]
	ldp q2, q3, [x11, #-64]
	add.4s v2, v2, v0
	ldp q0, q4, [x13, #-64]
	neg.4s v0, v0
	ldp q5, q6, [x12, #-64]
	mla.4s v0, v5, v2
	add.4s v2, v3, v1
	neg.4s v1, v4
	mla.4s v1, v6, v2
	ldp q2, q3, [x10, #-32]
	ldp q4, q5, [x11, #-32]
	add.4s v2, v4, v2
	ldp q4, q6, [x13, #-32]
	neg.4s v4, v4
	ldp q7, q16, [x12, #-32]
	mla.4s v4, v7, v2
	add.4s v2, v5, v3
	neg.4s v3, v6
	mla.4s v3, v16, v2
	ldp q2, q5, [x10]
	ldp q6, q7, [x11]
	add.4s v2, v6, v2
	ldp q6, q16, [x13]
	neg.4s v6, v6
	ldp q17, q18, [x12]
	mla.4s v6, v17, v2
	add.4s v2, v7, v5
	neg.4s v5, v16
	mla.4s v5, v18, v2
	ldp q2, q7, [x10, #32]
	ldp q16, q17, [x11, #32]
	add.4s v2, v16, v2
	ldp q16, q18, [x13, #32]
	neg.4s v16, v16
	ldp q19, q20, [x12, #32]
	mla.4s v16, v19, v2
	add.4s v2, v17, v7
	neg.4s v7, v18
	mla.4s v7, v20, v2
	stp q0, q1, [x14, #-64]
	stp q4, q3, [x14, #-32]
	add x8, x8, #32
	add x10, x10, #128
	stp q6, q5, [x14]
	add x11, x11, #128
	add x12, x12, #128
	add x13, x13, #128
	stp q16, q7, [x14, #32]
	add x14, x14, #128
	cmp x8, x9
	b.lo LBB160_2
LBB160_3:
	subs x13, x4, x8
	b.ls LBB160_17
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	ldr x11, [x0, #448]
	ldr x12, [x0, #608]
	cmp x13, #3
	b.hi LBB160_6
	mov x14, x8
	b LBB160_15
LBB160_6:
	cmp x13, #16
	b.hs LBB160_8
	mov x15, #0
	b LBB160_12
LBB160_8:
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
LBB160_9:
	ldp q0, q1, [x17, #-32]
	ldp q2, q3, [x17], #64
	ldp q4, q5, [x0, #-32]
	ldp q6, q7, [x0], #64
	ldp q16, q17, [x3, #-32]
	ldp q18, q19, [x3], #64
	ldp q20, q21, [x5, #-32]
	ldp q22, q23, [x5], #64
	add.4s v0, v4, v0
	add.4s v1, v5, v1
	add.4s v2, v6, v2
	add.4s v3, v7, v3
	neg.4s v4, v20
	mla.4s v4, v16, v0
	neg.4s v0, v21
	mla.4s v0, v17, v1
	neg.4s v1, v22
	mla.4s v1, v18, v2
	neg.4s v2, v23
	mla.4s v2, v19, v3
	stp q4, q0, [x2, #-32]
	stp q1, q2, [x2], #64
	subs x6, x6, #16
	b.ne LBB160_9
	cmp x13, x15
	b.eq LBB160_17
	cbz x16, LBB160_15
LBB160_12:
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
LBB160_13:
	ldr q0, [x3], #16
	ldr q1, [x2], #16
	ldr q2, [x0], #16
	ldr q3, [x15], #16
	add.4s v0, v1, v0
	neg.4s v1, v3
	mla.4s v1, v2, v0
	str q1, [x8], #16
	adds x17, x17, #4
	b.ne LBB160_13
	cmp x13, x16
	b.eq LBB160_17
LBB160_15:
	sub x8, x4, x14
	lsl x14, x14, #2
	add x13, x1, x14
	add x12, x12, x14
	add x11, x11, x14
	add x10, x10, x14
	add x9, x9, x14
LBB160_16:
	ldr w14, [x9], #4
	ldr w15, [x10], #4
	ldr w16, [x11], #4
	ldr w17, [x12], #4
	add w14, w15, w14
	neg w15, w17
	madd w14, w16, w14, w15
	str w14, [x13], #4
	subs x8, x8, #1
	b.ne LBB160_16
LBB160_17:
	ret
