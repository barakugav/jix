jix_probe::bit_shuffle::untrans_bit_byte:
	sub sp, sp, #352
	stp d15, d14, [sp, #192]
	stp d13, d12, [sp, #208]
	stp d11, d10, [sp, #224]
	stp d9, d8, [sp, #240]
	stp x29, x30, [sp, #256]
	stp x28, x27, [sp, #272]
	stp x26, x25, [sp, #288]
	stp x24, x23, [sp, #304]
	stp x22, x21, [sp, #320]
	stp x20, x19, [sp, #336]
	add x29, sp, #256
	str x5, [sp, #72]
	cbz x5, .LBB1_20
	lsr x10, x4, #3
	cbz x10, .LBB1_20
	ldr x8, [sp, #72]
	mov x14, x2
	mov x24, xzr
	mov x12, xzr
	mov x2, xzr
	mov x5, xzr
	mul x22, x8, x10
	dup v0.2d, x1
	movi v5.8h, #170
	mov w16, #-252645136
	mneg x8, x8, x10
	lsl x7, x22, #2
	lsl x18, x22, #1
	add x23, x0, x22
	str x22, [sp, #64]
	sub x11, x22, x7
	stur x8, [x29, #-80]
	lsl x8, x22, #3
	fmov d1, x11
	add x20, x7, x22
	neg x9, x18
	sub x15, x18, x8
	sub x17, x8, x22
	sub x6, x22, x8
	neg x8, x20
	add x19, x18, x22
	fmov d2, x18
	mov v1.d[1], x8
	fmov d3, x9
	fmov d4, x19
	mov x11, #-1
	mov w8, #7
	lsl x9, x19, #1
	stp x8, x11, [x29, #-96]
	mov w8, #1
	mov x11, #52428
	neg x13, x7
	movk w8, #4096, lsl #16
	movk x11, #52428, lsl #32
	mov v2.d[1], x7
	add x25, x0, x17
	add x26, x0, x20
	mov v4.d[1], x20
	mov v3.d[1], x13
	add x27, x0, x9
	add x28, x0, x7
	add x30, x0, x19
	add x21, x0, x18
	dup v6.2s, w8
	dup v7.2d, x11
	str x6, [sp, #40]
	stur x6, [x29, #-104]
	mov x11, x17
	mov x6, x9
	stp x17, x15, [sp, #48]
	stur x15, [x29, #-112]
	mov w15, #8
	stp x4, x9, [sp, #24]
	str q1, [sp]
.LBB1_3:
	mul x8, x10, x5
	add x13, x3, #1
	stp x15, x5, [sp, #120]
	cmp x15, x13
	stur x2, [x29, #-120]
	mul x9, x4, x5
	csel x5, x15, x13, hi
	cmn x2, #8
	mov x4, #-8
	csel x15, x2, x4, hi
	cmp x1, x22
	dup v16.2d, x8
	str x15, [sp, #80]
	csel x15, x1, x22, hi
	cmp x1, x6
	str x15, [sp, #112]
	csel x15, x1, x6, hi
	cmp x1, x11
	add v17.2d, v2.2d, v16.2d
	add v18.2d, v4.2d, v16.2d
	str x15, [sp, #88]
	csel x15, x1, x11, hi
	cmp x1, x12
	str x15, [sp, #104]
	add x15, x9, #8
	csel x17, x1, x12, hi
	cmhi v19.2d, v0.2d, v18.2d
	cmhi v20.2d, v0.2d, v17.2d
	cmp x15, x13
	str x17, [sp, #96]
	csel x15, x15, x13, hi
	mvn x17, x9
	ldr x13, [sp, #64]
	add x15, x15, x17
	subs x9, x4, x9
	bit v18.16b, v0.16b, v19.16b
	bit v17.16b, v0.16b, v20.16b
	sub v19.2d, v1.2d, v16.2d
	sub v16.2d, v3.2d, v16.2d
	lsr x17, x15, #3
	add x15, x13, x8
	ldr x13, [sp, #32]
	csel x9, xzr, x9, lo
	subs x15, x1, x15
	csel x4, xzr, x15, lo
	add x9, x9, #7
	add v16.2d, v17.2d, v16.2d
	add v17.2d, v18.2d, v19.2d
	add x2, x13, x8
	ldr x13, [sp, #48]
	cmp x1, x2
	lsr x9, x9, #3
	csel x2, x1, x2, hi
	cmhi v18.2d, v17.2d, v16.2d
	add x15, x13, x8
	mov x13, x24
	ldr x24, [sp, #56]
	cmp x1, x15
	csel x15, x1, x15, hi
	bif v16.16b, v17.16b, v18.16b
	sub x24, x24, x8
	add x2, x2, x24
	ldr x24, [sp, #40]
	sub x24, x24, x8
	subs x8, x1, x8
	mov d17, v16.d[1]
	add x24, x15, x24
	csel x8, xzr, x8, lo
	cmhi d18, d17, d16
	bif v16.8b, v17.8b, v18.8b
	fmov x15, d16
	cmp x15, x2
	csel x2, x15, x2, lo
	cmp x24, x4
	csel x4, x24, x4, lo
	mov x24, x13
	cmp x8, x17
	sub x13, x10, #1
	csel x8, x8, x17, lo
	cmp x9, x13
	csel x9, x9, x13, lo
	cmp x2, x4
	csel x17, x2, x4, lo
	cmp x8, x9
	mov x4, #52428
	csel x8, x8, x9, lo
	movk x4, #52428, lsl #32
	cmp x17, x8
	csel x9, x17, x8, lo
	mov x17, xzr
	cmp x9, #16
	b.lo .LBB1_7
	ldur x13, [x29, #-88]
	ldr x2, [sp, #80]
	add x9, x9, #1
	mov x8, xzr
	add x17, x5, x13
	ldur x13, [x29, #-96]
	mov w5, #16
	lsr x17, x17, #3
	add x13, x2, x13
	ands x2, x9, #0xf
	lsr x13, x13, #3
	csel x2, x5, x2, eq
	cmp x15, x17
	csel x15, x15, x17, lo
	ldur x17, [x29, #-80]
	ldr x5, [sp, #112]
	cmp x15, x13
	add x17, x5, x17
	csel x13, x15, x13, lo
	ldr x5, [sp, #88]
	cmp x13, x17
	csel x13, x13, x17, lo
	ldp x15, x17, [x29, #-112]
	add x15, x5, x15
	ldr x5, [sp, #104]
	cmp x13, x15
	add x17, x5, x17
	csel x13, x13, x15, lo
	ldr x15, [sp, #96]
	cmp x13, x17
	add x15, x15, x24
	csel x13, x13, x17, lo
	sub x17, x9, x2
	cmp x13, x15
	csel x13, x13, x15, lo
	sub x15, x10, #1
	cmp x13, x15
	csel x13, x13, x15, lo
	sub x13, x13, x2
	mov x2, x14
	add x9, x13, #1
.LBB1_5:
	ldr q16, [x25, x8]
	ldr q30, [x26, x8]
	ushll v17.8h, v16.8b, #0
	ushll2 v18.8h, v16.16b, #0
	ldr q16, [x27, x8]
	ushll v19.8h, v16.8b, #0
	ushll2 v20.8h, v16.16b, #0
	ldr q16, [x23, x8]
	ushll v21.4s, v17.4h, #0
	ushll v22.4s, v18.4h, #0
	ushll2 v23.4s, v17.8h, #0
	ushll2 v24.4s, v18.8h, #0
	ldr q18, [x28, x8]
	ldr q17, [x21, x8]
	ushll v27.4s, v19.4h, #0
	ushll2 v28.4s, v19.8h, #0
	ushll2 v29.4s, v20.8h, #0
	ushll2 v26.2d, v22.4s, #0
	ushll2 v31.2d, v23.4s, #0
	ushll v8.2d, v21.2s, #0
	ushll v25.2d, v24.2s, #0
	ushll2 v24.2d, v24.4s, #0
	ushll2 v21.2d, v21.4s, #0
	ushll v9.2d, v23.2s, #0
	ushll v20.4s, v20.4h, #0
	ushll v10.2d, v27.2s, #0
	ushll v11.2d, v28.2s, #0
	ushll v22.2d, v22.2s, #0
	ushll2 v27.2d, v27.4s, #0
	shl v23.2d, v24.2d, #56
	shl v24.2d, v26.2d, #56
	shl v26.2d, v31.2d, #56
	shl v31.2d, v9.2d, #56
	shl v9.2d, v21.2d, #56
	ldr q19, [x30, x8]
	ushll2 v21.2d, v29.4s, #0
	ushll v12.2d, v20.2s, #0
	shl v8.2d, v8.2d, #56
	ushll v29.2d, v29.2s, #0
	ushll2 v28.2d, v28.4s, #0
	ushll2 v20.2d, v20.4s, #0
	shl v11.2d, v11.2d, #48
	shl v25.2d, v25.2d, #56
	shl v22.2d, v22.2d, #56
	shl v13.2d, v21.2d, #48
	shl v21.2d, v10.2d, #48
	shl v10.2d, v12.2d, #48
	ushll v12.8h, v30.8b, #0
	shl v27.2d, v27.2d, #48
	shl v29.2d, v29.2d, #48
	shl v14.2d, v20.2d, #48
	shl v28.2d, v28.2d, #48
	ushll2 v30.8h, v30.16b, #0
	orr v21.16b, v8.16b, v21.16b
	orr v20.16b, v31.16b, v11.16b
	ushll v8.8h, v18.8b, #0
	ushll v31.4s, v12.4h, #0
	orr v27.16b, v9.16b, v27.16b
	orr v22.16b, v22.16b, v10.16b
	orr v26.16b, v26.16b, v28.16b
	ushll2 v28.4s, v30.8h, #0
	orr v25.16b, v25.16b, v29.16b
	ushll v29.4s, v30.4h, #0
	ushll2 v30.4s, v12.8h, #0
	ushll v10.4s, v8.4h, #0
	ushll v9.2d, v31.2s, #0
	ushll2 v31.2d, v31.4s, #0
	ushll2 v18.8h, v18.16b, #0
	orr v24.16b, v24.16b, v14.16b
	orr v23.16b, v23.16b, v13.16b
	ushll2 v11.2d, v28.4s, #0
	ushll v28.2d, v28.2s, #0
	ushll v12.2d, v29.2s, #0
	ushll v13.2d, v30.2s, #0
	shl v9.2d, v9.2d, #40
	ushll2 v8.4s, v8.8h, #0
	shll v14.2d, v10.2s, #32
	ushll2 v29.2d, v29.4s, #0
	ushll2 v30.2d, v30.4s, #0
	shl v31.2d, v31.2d, #40
	ushll v15.4s, v18.4h, #0
	shll2 v10.2d, v10.4s, #32
	ushll2 v18.4s, v18.8h, #0
	shl v13.2d, v13.2d, #40
	shl v12.2d, v12.2d, #40
	orr v9.16b, v9.16b, v14.16b
	shll v14.2d, v8.2s, #32
	shl v30.2d, v30.2d, #40
	shl v29.2d, v29.2d, #40
	shl v28.2d, v28.2d, #40
	orr v31.16b, v31.16b, v10.16b
	shll v10.2d, v15.2s, #32
	shll2 v8.2d, v8.4s, #32
	shll v1.2d, v18.2s, #32
	shll2 v15.2d, v15.4s, #32
	shl v11.2d, v11.2d, #40
	orr v13.16b, v13.16b, v14.16b
	shll2 v14.2d, v18.4s, #32
	orr v18.16b, v21.16b, v9.16b
	orr v21.16b, v27.16b, v31.16b
	orr v27.16b, v12.16b, v10.16b
	orr v30.16b, v30.16b, v8.16b
	ushll v31.8h, v19.8b, #0
	orr v29.16b, v29.16b, v15.16b
	orr v1.16b, v28.16b, v1.16b
	ushll v8.8h, v17.8b, #0
	ushll2 v28.8h, v17.16b, #0
	orr v22.16b, v22.16b, v27.16b
	ushll2 v27.8h, v19.16b, #0
	orr v20.16b, v20.16b, v13.16b
	orr v17.16b, v26.16b, v30.16b
	orr v26.16b, v11.16b, v14.16b
	ushll v30.4s, v31.4h, #0
	orr v19.16b, v24.16b, v29.16b
	orr v24.16b, v25.16b, v1.16b
	ushll v1.4s, v8.4h, #0
	ushll2 v29.4s, v27.8h, #0
	ushll2 v9.4s, v28.8h, #0
	ushll2 v11.8h, v16.16b, #0
	orr v23.16b, v23.16b, v26.16b
	ushll v26.2d, v30.2s, #24
	ushll v25.4s, v27.4h, #0
	ushll v10.2d, v1.2s, #16
	ushll v28.4s, v28.4h, #0
	ushll2 v31.4s, v31.8h, #0
	ushll v12.2d, v29.2s, #24
	ushll v13.2d, v9.2s, #16
	ushll2 v29.2d, v29.4s, #24
	ushll2 v9.2d, v9.4s, #16
	ushll2 v30.2d, v30.4s, #24
	ushll2 v14.2d, v25.4s, #24
	orr v26.16b, v26.16b, v10.16b
	ushll2 v10.4s, v11.8h, #0
	ushll2 v8.4s, v8.8h, #0
	ushll2 v1.2d, v1.4s, #16
	orr v12.16b, v12.16b, v13.16b
	ushll2 v13.2d, v28.4s, #16
	ushll v11.4s, v11.4h, #0
	ushll v16.8h, v16.8b, #0
	orr v29.16b, v29.16b, v9.16b
	ushll v9.2d, v10.2s, #8
	ushll2 v10.2d, v10.4s, #8
	ushll v27.2d, v31.2s, #24
	ushll2 v31.2d, v31.4s, #24
	orr v1.16b, v30.16b, v1.16b
	ushll2 v30.2d, v8.4s, #16
	ushll v8.2d, v8.2s, #16
	orr v13.16b, v14.16b, v13.16b
	ushll2 v14.4s, v16.8h, #0
	orr v29.16b, v29.16b, v10.16b
	ushll2 v10.2d, v11.4s, #8
	orr v9.16b, v12.16b, v9.16b
	ldr q12, [x0, x8]
	add x8, x8, #16
	ushll v25.2d, v25.2s, #24
	cmp x9, x8
	ushll v28.2d, v28.2s, #16
	ushll v16.4s, v16.4h, #0
	orr v27.16b, v27.16b, v8.16b
	orr v30.16b, v31.16b, v30.16b
	ushll2 v31.2d, v14.4s, #8
	orr v8.16b, v13.16b, v10.16b
	ushll2 v10.8h, v12.16b, #0
	orr v23.16b, v23.16b, v29.16b
	orr v25.16b, v25.16b, v28.16b
	orr v24.16b, v24.16b, v9.16b
	ushll2 v28.2d, v16.4s, #8
	ushll v29.2d, v11.2s, #8
	ushll v9.8h, v12.8b, #0
	ushll v16.2d, v16.2s, #8
	orr v8.16b, v19.16b, v8.16b
	orr v19.16b, v30.16b, v31.16b
	ushll2 v30.4s, v10.8h, #0
	ushll v31.2d, v14.2s, #8
	orr v1.16b, v1.16b, v28.16b
	ushll v10.4s, v10.4h, #0
	orr v25.16b, v25.16b, v29.16b
	ushll2 v29.4s, v9.8h, #0
	ushll v9.4s, v9.4h, #0
	ushll2 v28.2d, v30.4s, #0
	ushll v30.2d, v30.2s, #0
	orr v16.16b, v26.16b, v16.16b
	orr v27.16b, v27.16b, v31.16b
	orr v1.16b, v21.16b, v1.16b
	ushll2 v31.2d, v10.4s, #0
	ushll2 v21.2d, v9.4s, #0
	ushll2 v26.2d, v29.4s, #0
	ushll v10.2d, v10.2s, #0
	ushll v29.2d, v29.2s, #0
	orr v11.16b, v17.16b, v19.16b
	orr v25.16b, v22.16b, v25.16b
	orr v27.16b, v20.16b, v27.16b
	orr v12.16b, v18.16b, v16.16b
	orr v19.16b, v23.16b, v28.16b
	ushr v16.2d, v23.2d, #7
	orr v17.16b, v1.16b, v21.16b
	orr v21.16b, v24.16b, v30.16b
	ushr v24.2d, v24.2d, #7
	ushll v9.2d, v9.2s, #0
	orr v20.16b, v8.16b, v31.16b
	orr v18.16b, v27.16b, v29.16b
	ushr v28.2d, v8.2d, #7
	orr v22.16b, v25.16b, v10.16b
	ushr v25.2d, v25.2d, #7
	eor v29.16b, v16.16b, v19.16b
	orr v23.16b, v11.16b, v26.16b
	ushr v26.2d, v11.2d, #7
	eor v24.16b, v24.16b, v21.16b
	ushr v1.2d, v1.2d, #7
	ushr v27.2d, v27.2d, #7
	eor v28.16b, v28.16b, v20.16b
	orr v16.16b, v12.16b, v9.16b
	eor v25.16b, v25.16b, v22.16b
	and v29.16b, v29.16b, v5.16b
	ushr v30.2d, v12.2d, #7
	eor v26.16b, v26.16b, v23.16b
	and v24.16b, v24.16b, v5.16b
	eor v1.16b, v1.16b, v17.16b
	eor v27.16b, v27.16b, v18.16b
	and v28.16b, v28.16b, v5.16b
	and v25.16b, v25.16b, v5.16b
	shl v31.2d, v29.2d, #7
	eor v30.16b, v30.16b, v16.16b
	and v26.16b, v26.16b, v5.16b
	shl v8.2d, v24.2d, #7
	and v1.16b, v1.16b, v5.16b
	and v27.16b, v27.16b, v5.16b
	shl v9.2d, v28.2d, #7
	add v29.2d, v31.2d, v29.2d
	shl v31.2d, v25.2d, #7
	and v30.16b, v30.16b, v5.16b
	add v24.2d, v8.2d, v24.2d
	shl v8.2d, v26.2d, #7
	shl v10.2d, v1.2d, #7
	add v28.2d, v9.2d, v28.2d
	shl v9.2d, v27.2d, #7
	add v31.2d, v31.2d, v25.2d
	eor v25.16b, v29.16b, v19.16b
	add v29.2d, v8.2d, v26.2d
	shl v8.2d, v30.2d, #7
	eor v26.16b, v24.16b, v21.16b
	add v27.2d, v9.2d, v27.2d
	add v1.2d, v10.2d, v1.2d
	eor v19.16b, v28.16b, v20.16b
	eor v24.16b, v31.16b, v22.16b
	ushr v28.2d, v25.2d, #14
	add v30.2d, v8.2d, v30.2d
	eor v21.16b, v29.16b, v23.16b
	ushr v29.2d, v26.2d, #14
	eor v20.16b, v1.16b, v17.16b
	eor v22.16b, v27.16b, v18.16b
	ushr v1.2d, v19.2d, #14
	ushr v17.2d, v24.2d, #14
	eor v18.16b, v28.16b, v25.16b
	eor v23.16b, v30.16b, v16.16b
	ushr v16.2d, v21.2d, #14
	eor v27.16b, v29.16b, v26.16b
	ushr v28.2d, v22.2d, #14
	ushr v29.2d, v20.2d, #14
	eor v1.16b, v1.16b, v19.16b
	eor v17.16b, v17.16b, v24.16b
	and v18.16b, v18.16b, v7.16b
	eor v16.16b, v16.16b, v21.16b
	and v27.16b, v27.16b, v7.16b
	ushr v30.2d, v23.2d, #14
	eor v29.16b, v29.16b, v20.16b
	eor v28.16b, v28.16b, v22.16b
	and v1.16b, v1.16b, v7.16b
	and v17.16b, v17.16b, v7.16b
	shl v31.2d, v18.2d, #14
	and v8.16b, v16.16b, v7.16b
	shl v16.2d, v27.2d, #14
	eor v30.16b, v30.16b, v23.16b
	and v28.16b, v28.16b, v7.16b
	and v29.16b, v29.16b, v7.16b
	shl v9.2d, v1.2d, #14
	shl v10.2d, v17.2d, #14
	add v18.2d, v31.2d, v18.2d
	shl v31.2d, v8.2d, #14
	add v27.2d, v16.2d, v27.2d
	and v30.16b, v30.16b, v7.16b
	shl v11.2d, v29.2d, #14
	add v1.2d, v9.2d, v1.2d
	shl v9.2d, v28.2d, #14
	add v10.2d, v10.2d, v17.2d
	eor v16.16b, v18.16b, v25.16b
	add v25.2d, v31.2d, v8.2d
	eor v17.16b, v27.16b, v26.16b
	shl v26.2d, v30.2d, #14
	add v27.2d, v11.2d, v29.2d
	add v28.2d, v9.2d, v28.2d
	eor v19.16b, v1.16b, v19.16b
	eor v18.16b, v10.16b, v24.16b
	ushr v1.2d, v16.2d, #28
	eor v21.16b, v25.16b, v21.16b
	ushr v25.2d, v17.2d, #28
	add v24.2d, v26.2d, v30.2d
	eor v22.16b, v28.16b, v22.16b
	eor v20.16b, v27.16b, v20.16b
	dup v28.2d, x16
	ushr v27.2d, v18.2d, #28
	eor v1.16b, v1.16b, v16.16b
	ushr v26.2d, v19.2d, #28
	eor v25.16b, v25.16b, v17.16b
	eor v23.16b, v24.16b, v23.16b
	ushr v24.2d, v21.2d, #28
	ushr v29.2d, v22.2d, #28
	ushr v30.2d, v20.2d, #28
	eor v27.16b, v27.16b, v18.16b
	and v1.16b, v1.16b, v28.16b
	eor v26.16b, v26.16b, v19.16b
	and v25.16b, v25.16b, v28.16b
	ushr v31.2d, v23.2d, #28
	eor v24.16b, v24.16b, v21.16b
	eor v29.16b, v29.16b, v22.16b
	eor v30.16b, v30.16b, v20.16b
	and v27.16b, v27.16b, v28.16b
	xtn v1.2s, v1.2d
	and v26.16b, v26.16b, v28.16b
	xtn v25.2s, v25.2d
	eor v31.16b, v31.16b, v23.16b
	and v24.16b, v24.16b, v28.16b
	and v29.16b, v29.16b, v28.16b
	and v30.16b, v30.16b, v28.16b
	xtn v27.2s, v27.2d
	xtn v26.2s, v26.2d
	and v28.16b, v31.16b, v28.16b
	umull v1.2d, v1.2s, v6.2s
	xtn v24.2s, v24.2d
	umull v25.2d, v25.2s, v6.2s
	xtn v29.2s, v29.2d
	xtn v30.2s, v30.2d
	xtn v28.2s, v28.2d
	umull v27.2d, v27.2s, v6.2s
	umull v26.2d, v26.2s, v6.2s
	eor v1.16b, v1.16b, v16.16b
	umull v16.2d, v24.2s, v6.2s
	eor v17.16b, v25.16b, v17.16b
	umull v24.2d, v29.2s, v6.2s
	umull v25.2d, v30.2s, v6.2s
	eor v19.16b, v26.16b, v19.16b
	stp q17, q1, [x2, #96]
	eor v1.16b, v27.16b, v18.16b
	umull v18.2d, v28.2s, v6.2s
	eor v16.16b, v16.16b, v21.16b
	eor v17.16b, v24.16b, v22.16b
	stp q1, q19, [x2, #64]
	eor v1.16b, v25.16b, v20.16b
	stp q17, q16, [x2, #32]
	eor v18.16b, v18.16b, v23.16b
	stp q18, q1, [x2], #128
	b.ne .LBB1_5
	ldr q1, [sp]
.LBB1_7:
	ldur x2, [x29, #-120]
	ldr x5, [sp, #128]
	add x8, x2, x17, lsl #3
	add x5, x5, #1
.LBB1_8:
	add x9, x12, x17
	cmp x9, x1
	b.hs .LBB1_22
	add x9, x22, x17
	cmp x9, x1
	b.hs .LBB1_22
	add x9, x18, x17
	cmp x9, x1
	b.hs .LBB1_22
	add x9, x19, x17
	cmp x9, x1
	b.hs .LBB1_22
	add x9, x7, x17
	cmp x9, x1
	b.hs .LBB1_22
	add x9, x20, x17
	cmp x9, x1
	b.hs .LBB1_22
	add x9, x6, x17
	cmp x9, x1
	b.hs .LBB1_22
	add x9, x11, x17
	cmp x9, x1
	b.hs .LBB1_22
	cmn x8, #9
	add x9, x8, #8
	b.hi .LBB1_21
	cmp x9, x3
	b.hi .LBB1_21
	ldrb w9, [x27, x17]
	ldrb w13, [x25, x17]
	add x8, x8, #8
	ldrb w15, [x26, x17]
	lsl x9, x9, #48
	orr x9, x9, x13, lsl #56
	ldrb w13, [x28, x17]
	orr x9, x9, x15, lsl #40
	ldrb w15, [x30, x17]
	orr x9, x9, x13, lsl #32
	ldrb w13, [x21, x17]
	orr x9, x9, x15, lsl #24
	ldrb w15, [x23, x17]
	orr x9, x9, x13, lsl #16
	ldrb w13, [x0, x17]
	orr x9, x9, x15, lsl #8
	mov x15, #-6148914691236517206
	and x15, x15, #0x1fe01fe01fe01fe
	orr x13, x9, x13
	eor x9, x13, x9, lsr #7
	and x9, x9, x15
	add x15, x17, #1
	orr x9, x9, x9, lsl #7
	cmp x10, x15
	eor x9, x9, x13
	eor x13, x9, x9, lsr #14
	and x13, x13, x4
	orr x13, x13, x13, lsl #14
	eor x9, x13, x9
	lsr x13, x9, #28
	eor w13, w13, w9
	and x13, x13, x16
	orr x13, x13, x13, lsl #28
	eor x9, x13, x9
	str x9, [x14, x17, lsl #3]
	mov x17, x15
	b.ne .LBB1_8
	ldr x4, [sp, #24]
	ldur x8, [x29, #-88]
	add x22, x22, x10
	ldr x15, [sp, #120]
	add x6, x6, x10
	add x11, x11, x10
	sub x8, x8, x4
	add x2, x2, x4
	add x12, x12, x10
	stur x8, [x29, #-88]
	ldur x8, [x29, #-96]
	add x15, x15, x4
	sub x24, x24, x10
	add x25, x25, x10
	add x14, x14, x4
	sub x8, x8, x4
	add x27, x27, x10
	add x26, x26, x10
	stur x8, [x29, #-96]
	ldur x8, [x29, #-80]
	add x28, x28, x10
	add x30, x30, x10
	add x21, x21, x10
	add x23, x23, x10
	sub x8, x8, x10
	add x0, x0, x10
	add x20, x20, x10
	stur x8, [x29, #-80]
	ldur x8, [x29, #-112]
	add x7, x7, x10
	add x19, x19, x10
	add x18, x18, x10
	sub x8, x8, x10
	stur x8, [x29, #-112]
	ldur x8, [x29, #-104]
	sub x8, x8, x10
	stur x8, [x29, #-104]
	ldr x8, [sp, #72]
	cmp x5, x8
	b.ne .LBB1_3
.LBB1_20:
	ldp x20, x19, [sp, #336]
	ldp x22, x21, [sp, #320]
	ldp x24, x23, [sp, #304]
	ldp x26, x25, [sp, #288]
	ldp x28, x27, [sp, #272]
	ldp x29, x30, [sp, #256]
	ldp d9, d8, [sp, #240]
	ldp d11, d10, [sp, #224]
	ldp d13, d12, [sp, #208]
	ldp d15, d14, [sp, #192]
	add sp, sp, #352
	ret
.LBB1_21:
	adrp x10, .Lanon.5e183fe0a34aeb9c6619c58bb96c331f.7
	add x10, x10, :lo12:.Lanon.5e183fe0a34aeb9c6619c58bb96c331f.7
	mov x0, x8
	mov x1, x9
	mov x2, x3
	mov x3, x10
	bl core::slice::index::slice_index_fail
.LBB1_22:
	adrp x2, .Lanon.5e183fe0a34aeb9c6619c58bb96c331f.8
	add x2, x2, :lo12:.Lanon.5e183fe0a34aeb9c6619c58bb96c331f.8
	mov x0, x9
	bl core::panicking::panic_bounds_check
