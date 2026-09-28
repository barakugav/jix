jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
Lfunc_begin178:
	mov x8, #0
	ands x9, x4, #0xffffffffffffffe0
	b.eq LBB178_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x10, x10, #64
	add x11, x11, #64
	add x12, x1, #64
LBB178_2:
	ldp q0, q1, [x10, #-64]
	ldp q2, q3, [x10, #-32]
	ldp q4, q5, [x10]
	ldp q6, q7, [x10, #32]
	ldp q16, q17, [x11, #-64]
	ldp q18, q19, [x11, #-32]
	ldp q20, q21, [x11]
	ldp q22, q23, [x11, #32]
	add.4s v0, v16, v0
	add.4s v1, v17, v1
	add.4s v2, v18, v2
	add.4s v3, v19, v3
	add.4s v4, v20, v4
	add.4s v5, v21, v5
	add.4s v6, v22, v6
	add.4s v7, v23, v7
	stp q0, q1, [x12, #-64]
	stp q2, q3, [x12, #-32]
	stp q4, q5, [x12]
	add x8, x8, #32
	add x10, x10, #128
	add x11, x11, #128
	stp q6, q7, [x12, #32]
	add x12, x12, #128
	cmp x8, x9
	b.lo LBB178_2
LBB178_3:
	subs x11, x4, x8
	b.ls LBB178_17
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x11, #3
	b.hi LBB178_6
	mov x12, x8
	b LBB178_15
LBB178_6:
	cmp x11, #16
	b.hs LBB178_8
	mov x13, #0
	b LBB178_12
LBB178_8:
	and x14, x11, #0xc
	and x13, x11, #0xfffffffffffffff0
	add x12, x8, x13
	lsl x15, x8, #2
	add x17, x15, #32
	add x15, x9, x17
	add x16, x10, x17
	add x17, x1, x17
	and x0, x11, #0xfffffffffffffff0
LBB178_9:
	ldp q0, q1, [x15, #-32]
	ldp q2, q3, [x15], #64
	ldp q4, q5, [x16, #-32]
	ldp q6, q7, [x16], #64
	add.4s v0, v4, v0
	add.4s v1, v5, v1
	add.4s v2, v6, v2
	add.4s v3, v7, v3
	stp q0, q1, [x17, #-32]
	stp q2, q3, [x17], #64
	subs x0, x0, #16
	b.ne LBB178_9
	cmp x11, x13
	b.eq LBB178_17
	cbz x14, LBB178_15
LBB178_12:
	and x14, x11, #0xfffffffffffffffc
	add x12, x8, x14
	sub x15, x13, x14
	add x8, x13, x8
	lsl x16, x8, #2
	add x8, x1, x16
	add x13, x10, x16
	add x16, x9, x16
LBB178_13:
	ldr q0, [x16], #16
	ldr q1, [x13], #16
	add.4s v0, v1, v0
	str q0, [x8], #16
	adds x15, x15, #4
	b.ne LBB178_13
	cmp x11, x14
	b.eq LBB178_17
LBB178_15:
	sub x8, x4, x12
	lsl x12, x12, #2
	add x11, x1, x12
	add x10, x10, x12
	add x9, x9, x12
LBB178_16:
	ldr w12, [x9], #4
	ldr w13, [x10], #4
	add w12, w13, w12
	str w12, [x11], #4
	subs x8, x8, #1
	b.ne LBB178_16
LBB178_17:
	ret
