jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
	mov x8, xzr
	ands x9, x4, #0xffffffffffffffe0
	b.eq .LBB162_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x12, x1, #64
	add x10, x10, #64
	add x11, x11, #64
.LBB162_2:
	ldp q0, q1, [x10, #-64]
	add x8, x8, #32
	ldp q2, q3, [x11, #-64]
	cmp x8, x9
	ldp q4, q5, [x10, #-32]
	ldp q6, q7, [x10]
	add v0.4s, v2.4s, v0.4s
	add v1.4s, v3.4s, v1.4s
	ldp q2, q3, [x11, #-32]
	ldp q16, q17, [x11]
	stp q0, q1, [x12, #-64]
	add v2.4s, v2.4s, v4.4s
	add v3.4s, v3.4s, v5.4s
	add v4.4s, v16.4s, v6.4s
	ldp q1, q6, [x11, #32]
	add v5.4s, v17.4s, v7.4s
	add x11, x11, #128
	stp q2, q3, [x12, #-32]
	ldp q0, q2, [x10, #32]
	add x10, x10, #128
	stp q4, q5, [x12]
	add v0.4s, v1.4s, v0.4s
	add v1.4s, v6.4s, v2.4s
	stp q0, q1, [x12, #32]
	add x12, x12, #128
	b.lo .LBB162_2
.LBB162_3:
	subs x11, x4, x8
	b.ls .LBB162_10
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x11, #7
	b.ls .LBB162_8
	lsl x13, x8, #2
	and x12, x11, #0xfffffffffffffff8
	and x16, x11, #0xfffffffffffffff8
	add x8, x8, x12
	add x15, x13, #16
	add x13, x9, x15
	add x14, x1, x15
	add x15, x10, x15
.LBB162_6:
	ldp q0, q3, [x15, #-16]
	subs x16, x16, #8
	ldp q1, q2, [x13, #-16]
	add x13, x13, #32
	add x15, x15, #32
	add v0.4s, v0.4s, v1.4s
	add v1.4s, v3.4s, v2.4s
	stp q0, q1, [x14, #-16]
	add x14, x14, #32
	b.ne .LBB162_6
	cmp x11, x12
	b.eq .LBB162_10
.LBB162_8:
	lsl x12, x8, #2
	sub x8, x4, x8
	add x11, x1, x12
	add x10, x10, x12
	add x9, x9, x12
.LBB162_9:
	ldr w12, [x9], #4
	subs x8, x8, #1
	ldr w13, [x10], #4
	add w12, w13, w12
	str w12, [x11], #4
	b.ne .LBB162_9
.LBB162_10:
	ret
