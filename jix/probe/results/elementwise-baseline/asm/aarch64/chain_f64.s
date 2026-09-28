jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	mov x8, xzr
	ands x9, x4, #0xfffffffffffffff0
	b.eq .LBB128_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x14, x1, #64
	ldr x12, [x0, #448]
	ldr x13, [x0, #608]
	add x10, x10, #64
	add x11, x11, #64
	add x12, x12, #64
	add x13, x13, #64
.LBB128_2:
	ldp q0, q3, [x11, #-64]
	add x8, x8, #16
	ldp q1, q2, [x10, #-64]
	cmp x8, x9
	ldp q4, q5, [x12, #-64]
	ldp q6, q17, [x11, #-32]
	fadd v0.2d, v1.2d, v0.2d
	ldp q1, q16, [x10, #-32]
	fadd v2.2d, v2.2d, v3.2d
	ldp q3, q7, [x13, #-64]
	fadd v1.2d, v1.2d, v6.2d
	ldp q6, q18, [x10]
	fmul v0.2d, v0.2d, v4.2d
	ldp q4, q19, [x11]
	fadd v16.2d, v16.2d, v17.2d
	fmul v2.2d, v2.2d, v5.2d
	ldp q5, q17, [x12, #-32]
	fadd v4.2d, v6.2d, v4.2d
	fadd v18.2d, v18.2d, v19.2d
	fsub v0.2d, v0.2d, v3.2d
	ldp q6, q3, [x11, #32]
	fmul v1.2d, v1.2d, v5.2d
	ldp q5, q19, [x10, #32]
	fsub v2.2d, v2.2d, v7.2d
	add x10, x10, #128
	add x11, x11, #128
	fadd v5.2d, v5.2d, v6.2d
	fmul v6.2d, v16.2d, v17.2d
	fadd v3.2d, v19.2d, v3.2d
	ldp q7, q16, [x12]
	ldp q17, q19, [x13, #-32]
	stp q0, q2, [x14, #-64]
	fmul v4.2d, v4.2d, v7.2d
	fmul v16.2d, v18.2d, v16.2d
	ldp q7, q18, [x12, #32]
	fsub v1.2d, v1.2d, v17.2d
	fsub v6.2d, v6.2d, v19.2d
	add x12, x12, #128
	fmul v5.2d, v5.2d, v7.2d
	ldp q7, q17, [x13]
	fmul v3.2d, v3.2d, v18.2d
	stp q1, q6, [x14, #-32]
	fsub v4.2d, v4.2d, v7.2d
	ldp q7, q2, [x13, #32]
	fsub v0.2d, v16.2d, v17.2d
	add x13, x13, #128
	fsub v5.2d, v5.2d, v7.2d
	fsub v1.2d, v3.2d, v2.2d
	stp q4, q0, [x14]
	stp q5, q1, [x14, #32]
	add x14, x14, #128
	b.lo .LBB128_2
.LBB128_3:
	subs x13, x4, x8
	b.ls .LBB128_10
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x13, #3
	ldr x11, [x0, #448]
	ldr x12, [x0, #608]
	b.ls .LBB128_8
	lsl x15, x8, #3
	and x14, x13, #0xfffffffffffffffc
	and x2, x13, #0xfffffffffffffffc
	add x8, x8, x14
	add x0, x15, #16
	add x15, x9, x0
	add x16, x1, x0
	add x17, x10, x0
	add x18, x12, x0
	add x0, x11, x0
.LBB128_6:
	ldp q0, q3, [x17, #-16]
	subs x2, x2, #4
	ldp q1, q2, [x15, #-16]
	add x15, x15, #32
	add x17, x17, #32
	fadd v0.2d, v1.2d, v0.2d
	fadd v1.2d, v2.2d, v3.2d
	ldp q2, q3, [x0, #-16]
	add x0, x0, #32
	fmul v0.2d, v0.2d, v2.2d
	fmul v1.2d, v1.2d, v3.2d
	ldp q2, q3, [x18, #-16]
	add x18, x18, #32
	fsub v0.2d, v0.2d, v2.2d
	fsub v1.2d, v1.2d, v3.2d
	stp q0, q1, [x16, #-16]
	add x16, x16, #32
	b.ne .LBB128_6
	cmp x13, x14
	b.eq .LBB128_10
.LBB128_8:
	lsl x14, x8, #3
	sub x8, x4, x8
	add x13, x1, x14
	add x12, x12, x14
	add x11, x11, x14
	add x10, x10, x14
	add x9, x9, x14
.LBB128_9:
	ldr d0, [x9], #8
	subs x8, x8, #1
	ldr d1, [x10], #8
	fadd d0, d0, d1
	ldr d1, [x11], #8
	fmul d0, d0, d1
	ldr d1, [x12], #8
	fsub d0, d0, d1
	str d0, [x13], #8
	b.ne .LBB128_9
.LBB128_10:
	ret
