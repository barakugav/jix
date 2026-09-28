jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
	mov x8, xzr
	ands x9, x4, #0xffffffffffffffe0
	b.eq .LBB176_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x14, x1, #64
	ldr x12, [x0, #448]
	ldr x13, [x0, #608]
	add x10, x10, #64
	add x11, x11, #64
	add x12, x12, #64
	add x13, x13, #64
.LBB176_2:
	ldp q0, q1, [x10, #-64]
	add x8, x8, #32
	ldp q2, q3, [x11, #-64]
	cmp x8, x9
	ldp q4, q5, [x10, #-32]
	ldp q6, q7, [x11, #-32]
	add v0.4s, v2.4s, v0.4s
	add v1.4s, v3.4s, v1.4s
	ldp q16, q17, [x11]
	add v3.4s, v6.4s, v4.4s
	add v4.4s, v7.4s, v5.4s
	ldp q2, q5, [x13, #-64]
	ldp q6, q7, [x10]
	ldp q18, q19, [x10, #32]
	add x10, x10, #128
	ldp q20, q21, [x11, #32]
	neg v2.4s, v2.4s
	add v6.4s, v16.4s, v6.4s
	add v7.4s, v17.4s, v7.4s
	neg v5.4s, v5.4s
	ldp q24, q25, [x12]
	add x11, x11, #128
	add v17.4s, v20.4s, v18.4s
	ldp q16, q20, [x12, #-64]
	add v18.4s, v21.4s, v19.4s
	ldp q21, q22, [x12, #-32]
	mla v2.4s, v16.4s, v0.4s
	ldp q19, q0, [x13, #-32]
	ldp q16, q23, [x13]
	mla v5.4s, v20.4s, v1.4s
	ldp q1, q20, [x13, #32]
	add x13, x13, #128
	neg v19.4s, v19.4s
	neg v0.4s, v0.4s
	neg v16.4s, v16.4s
	neg v1.4s, v1.4s
	stp q2, q5, [x14, #-64]
	mla v19.4s, v21.4s, v3.4s
	neg v3.4s, v23.4s
	mla v0.4s, v22.4s, v4.4s
	mla v16.4s, v24.4s, v6.4s
	neg v4.4s, v20.4s
	ldp q2, q6, [x12, #32]
	add x12, x12, #128
	mla v3.4s, v25.4s, v7.4s
	stp q19, q0, [x14, #-32]
	mla v1.4s, v2.4s, v17.4s
	mla v4.4s, v6.4s, v18.4s
	stp q16, q3, [x14]
	stp q1, q4, [x14, #32]
	add x14, x14, #128
	b.lo .LBB176_2
.LBB176_3:
	subs x13, x4, x8
	b.ls .LBB176_10
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x13, #7
	ldr x11, [x0, #448]
	ldr x12, [x0, #608]
	b.ls .LBB176_8
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
.LBB176_6:
	ldp q0, q4, [x17, #-16]
	subs x2, x2, #8
	ldp q1, q5, [x18, #-16]
	add x17, x17, #32
	ldp q2, q3, [x15, #-16]
	add x15, x15, #32
	add x18, x18, #32
	neg v1.4s, v1.4s
	add v0.4s, v0.4s, v2.4s
	add v2.4s, v4.4s, v3.4s
	neg v3.4s, v5.4s
	ldp q4, q5, [x0, #-16]
	add x0, x0, #32
	mla v1.4s, v4.4s, v0.4s
	mla v3.4s, v5.4s, v2.4s
	stp q1, q3, [x16, #-16]
	add x16, x16, #32
	b.ne .LBB176_6
	cmp x13, x14
	b.eq .LBB176_10
.LBB176_8:
	lsl x14, x8, #2
	sub x8, x4, x8
	add x13, x1, x14
	add x12, x12, x14
	add x11, x11, x14
	add x10, x10, x14
	add x9, x9, x14
.LBB176_9:
	ldr w14, [x9], #4
	subs x8, x8, #1
	ldr w15, [x10], #4
	ldr w16, [x11], #4
	ldr w17, [x12], #4
	add w14, w15, w14
	neg w15, w17
	madd w14, w16, w14, w15
	str w14, [x13], #4
	b.ne .LBB176_9
.LBB176_10:
	ret
