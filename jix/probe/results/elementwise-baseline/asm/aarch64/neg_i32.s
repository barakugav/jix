jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
	mov x8, xzr
	ands x9, x4, #0xffffffffffffffe0
	b.eq .LBB159_3
	ldr x10, [x0, #136]
	add x11, x1, #64
	add x10, x10, #64
.LBB159_2:
	ldp q0, q1, [x10, #-64]
	add x8, x8, #32
	ldp q2, q3, [x10, #-32]
	cmp x8, x9
	ldp q4, q5, [x10]
	neg v0.4s, v0.4s
	neg v1.4s, v1.4s
	neg v2.4s, v2.4s
	neg v3.4s, v3.4s
	neg v5.4s, v5.4s
	stp q0, q1, [x11, #-64]
	neg v0.4s, v4.4s
	ldp q1, q4, [x10, #32]
	stp q2, q3, [x11, #-32]
	add x10, x10, #128
	stp q0, q5, [x11]
	neg v1.4s, v1.4s
	neg v2.4s, v4.4s
	stp q1, q2, [x11, #32]
	add x11, x11, #128
	b.lo .LBB159_2
.LBB159_3:
	subs x10, x4, x8
	b.ls .LBB159_10
	ldr x9, [x0, #136]
	cmp x10, #7
	b.ls .LBB159_8
	lsl x12, x8, #2
	and x11, x10, #0xfffffffffffffff8
	and x14, x10, #0xfffffffffffffff8
	add x8, x8, x11
	add x13, x12, #16
	add x12, x1, x13
	add x13, x9, x13
.LBB159_6:
	ldp q0, q1, [x13, #-16]
	subs x14, x14, #8
	add x13, x13, #32
	neg v0.4s, v0.4s
	neg v1.4s, v1.4s
	stp q0, q1, [x12, #-16]
	add x12, x12, #32
	b.ne .LBB159_6
	cmp x10, x11
	b.eq .LBB159_10
.LBB159_8:
	lsl x11, x8, #2
	sub x8, x4, x8
	add x10, x1, x11
	add x9, x9, x11
.LBB159_9:
	ldr w11, [x9], #4
	subs x8, x8, #1
	neg w11, w11
	str w11, [x10], #4
	b.ne .LBB159_9
.LBB159_10:
	ret
