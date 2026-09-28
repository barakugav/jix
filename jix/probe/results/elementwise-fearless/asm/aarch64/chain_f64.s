jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	mov x8, xzr
	ands x9, x4, #0xfffffffffffffff0
	b.eq .LBB168_3
	ldr x10, [x0, #136]
	ldr x11, [x0, #288]
	add x14, x1, #64
	ldr x12, [x0, #448]
	ldr x13, [x0, #608]
	add x10, x10, #64
	add x11, x11, #64
	add x12, x12, #64
	add x13, x13, #64
.LBB168_2:
	ldp q0, q1, [x10, #-64]
	add x8, x8, #16
	ldp q2, q3, [x11, #-64]
	cmp x8, x9
	ldp q4, q7, [x11, #-32]
	ldp q5, q6, [x10, #-32]
	fadd v0.2d, v0.2d, v2.2d
	fadd v1.2d, v1.2d, v3.2d
	ldp q16, q17, [x10, #32]
	fadd v2.2d, v5.2d, v4.2d
	fadd v3.2d, v6.2d, v7.2d
	ldp q4, q5, [x10], #128
	ldp q6, q7, [x11]
	ldp q18, q19, [x11, #32]
	add x11, x11, #128
	fadd v4.2d, v4.2d, v6.2d
	fadd v5.2d, v5.2d, v7.2d
	fadd v7.2d, v16.2d, v18.2d
	fadd v16.2d, v17.2d, v19.2d
	ldp q6, q17, [x12, #-64]
	fmul v0.2d, v0.2d, v6.2d
	ldp q6, q18, [x12, #-32]
	fmul v1.2d, v1.2d, v17.2d
	ldp q17, q19, [x12]
	fmul v2.2d, v2.2d, v6.2d
	ldp q6, q21, [x13, #-64]
	fmul v3.2d, v3.2d, v18.2d
	ldp q18, q20, [x12, #32]
	fmul v4.2d, v4.2d, v17.2d
	fmul v5.2d, v5.2d, v19.2d
	add x12, x12, #128
	fsub v0.2d, v0.2d, v6.2d
	fsub v1.2d, v1.2d, v21.2d
	ldp q17, q6, [x13, #-32]
	fmul v7.2d, v7.2d, v18.2d
	fmul v16.2d, v16.2d, v20.2d
	fsub v2.2d, v2.2d, v17.2d
	ldp q18, q17, [x13]
	fsub v3.2d, v3.2d, v6.2d
	stp q0, q1, [x14, #-64]
	ldp q0, q6, [x13, #32]
	add x13, x13, #128
	fsub v4.2d, v4.2d, v18.2d
	fsub v5.2d, v5.2d, v17.2d
	fsub v0.2d, v7.2d, v0.2d
	fsub v1.2d, v16.2d, v6.2d
	stp q2, q3, [x14, #-32]
	stp q4, q5, [x14]
	stp q0, q1, [x14, #32]
	add x14, x14, #128
	b.lo .LBB168_2
.LBB168_3:
	subs x13, x4, x8
	b.ls .LBB168_10
	ldr x9, [x0, #136]
	ldr x10, [x0, #288]
	cmp x13, #3
	ldr x11, [x0, #448]
	ldr x12, [x0, #608]
	b.ls .LBB168_8
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
.LBB168_6:
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
	b.ne .LBB168_6
	cmp x13, x14
	b.eq .LBB168_10
.LBB168_8:
	lsl x14, x8, #3
	sub x8, x4, x8
	add x13, x1, x14
	add x12, x12, x14
	add x11, x11, x14
	add x10, x10, x14
	add x9, x9, x14
.LBB168_9:
	ldr d0, [x9], #8
	subs x8, x8, #1
	ldr d1, [x10], #8
	fadd d0, d0, d1
	ldr d1, [x11], #8
	fmul d0, d0, d1
	ldr d1, [x12], #8
	fsub d0, d0, d1
	str d0, [x13], #8
	b.ne .LBB168_9
.LBB168_10:
	ret
