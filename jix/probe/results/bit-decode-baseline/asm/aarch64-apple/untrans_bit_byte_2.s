_probe_bit_shuffle_untrans_bit_byte:
Lfunc_begin9:
	sub sp, sp, #352
	stp d13, d12, [sp, #208]
	stp d11, d10, [sp, #224]
	stp d9, d8, [sp, #240]
	stp x28, x27, [sp, #256]
	stp x26, x25, [sp, #272]
	stp x24, x23, [sp, #288]
	stp x22, x21, [sp, #304]
	stp x20, x19, [sp, #320]
	stp x29, x30, [sp, #336]
	add x29, sp, #336
	stur x3, [x29, #-144]
	str x5, [sp, #64]
	cbz x5, LBB9_19
	lsr x10, x4, #3
	cbz x10, LBB9_19
	str xzr, [sp, #168]
	mov x12, #0
	mov x15, #0
	ldur x8, [x29, #-144]
	add x8, x8, #1
	str x8, [sp, #56]
	ldr x13, [sp, #64]
	mul x5, x13, x10
	lsl x17, x5, #1
	fmov d0, x17
	neg x8, x17
	fmov d1, x8
	add x6, x17, x5
	add x7, x0, x6
	fmov d2, x6
	lsl x19, x5, #2
	sub x8, x5, x19
	fmov d3, x8
	neg x8, x19
	add x20, x19, x5
	neg x9, x20
	lsl x26, x6, #1
	lsl x11, x5, #3
	sub x16, x17, x11
	sub x3, x5, x11
	dup.2d v4, x1
	mov.d v2[1], x20
	mov.d v0[1], x19
	mov.d v3[1], x9
	mov x9, #0
	mov.d v1[1], x8
	mneg x8, x13, x10
	str x8, [sp, #160]
	sub x11, x11, x5
	add x27, x0, x11
	add x28, x0, x26
	add x30, x0, x20
	add x22, x0, x19
	add x23, x0, x17
	add x21, x0, x5
	mov x8, #-1
	str x8, [sp, #152]
	mov w13, #8
	mov w8, #7
	stp x3, x8, [sp, #136]
	mov x14, #52428
	movk x14, #52428, lsl #32
	mov w25, #-252645136
	mov w8, #1
	movk w8, #4096, lsl #16
	dup.2s v5, w8
	dup.2d v6, x14
	movi.8h v7, #170
	stp x11, x3, [sp, #16]
	mov x3, x11
	stp x16, x26, [sp, #32]
	str x16, [sp, #128]
	str x5, [sp, #48]
	str x4, [sp, #8]
LBB9_3:
	mov x11, #0
	ldr x14, [sp, #56]
	cmp x13, x14
	str x13, [sp, #120]
	csel x8, x13, x14, hi
	str x8, [sp, #72]
	cmn x9, #8
	stur x9, [x29, #-160]
	mov x16, #-8
	csel x8, x9, x16, hi
	str x8, [sp, #80]
	cmp x1, x5
	csel x9, x1, x5, hi
	cmp x1, x26
	csel x8, x1, x26, hi
	stp x9, x8, [sp, #88]
	cmp x1, x3
	csel x9, x1, x3, hi
	cmp x1, x12
	csel x8, x1, x12, hi
	stp x9, x8, [sp, #104]
	mul x8, x4, x15
	add x13, x8, #8
	cmp x13, x14
	csel x13, x13, x14, hi
	mvn x24, x8
	add x13, x13, x24
	subs x8, x16, x8
	csel x8, xzr, x8, lo
	stur x15, [x29, #-152]
	mul x24, x10, x15
	ldp x14, x9, [sp, #40]
	add x9, x9, x24
	subs x9, x1, x9
	csel x9, xzr, x9, lo
	dup.2d v16, x24
	add.2d v17, v0, v16
	add.2d v18, v2, v16
	cmhi.2d v19, v4, v18
	bit.16b v18, v4, v19
	cmhi.2d v19, v4, v17
	bit.16b v17, v4, v19
	sub.2d v19, v3, v16
	sub.2d v16, v1, v16
	add.2d v16, v17, v16
	add.2d v17, v18, v19
	add x14, x14, x24
	cmp x1, x14
	csel x14, x1, x14, hi
	ldr x15, [sp, #32]
	sub x16, x15, x24
	add x16, x14, x16
	ldp x14, x15, [sp, #16]
	add x14, x14, x24
	cmp x1, x14
	csel x14, x1, x14, hi
	sub x15, x15, x24
	add x15, x14, x15
	subs x14, x1, x24
	csel x14, xzr, x14, lo
	cmhi.2d v18, v17, v16
	bif.16b v16, v17, v18
	mov d17, v16[1]
	cmhi d18, d17, d16
	bif.8b v16, v17, v18
	fmov x24, d16
	cmp x24, x16
	csel x16, x24, x16, lo
	cmp x15, x9
	csel x9, x15, x9, lo
	lsr x13, x13, #3
	add x8, x8, #7
	lsr x8, x8, #3
	cmp x14, x13
	csel x13, x14, x13, lo
	sub x14, x10, #1
	cmp x8, x14
	csel x8, x8, x14, lo
	cmp x16, x9
	csel x9, x16, x9, lo
	cmp x13, x8
	csel x8, x13, x8, lo
	cmp x9, x8
	csel x13, x9, x8, lo
	cmp x13, #16
	mov x4, x2
	b.lo LBB9_6
	mov x8, #0
	ldr x9, [sp, #152]
	ldp x11, x15, [sp, #72]
	add x9, x11, x9
	lsr x9, x9, #3
	add x11, x13, #1
	ands x13, x11, #0xf
	mov w14, #16
	csel x13, x14, x13, eq
	cmp x24, x9
	csel x9, x24, x9, lo
	ldr x14, [sp, #144]
	add x15, x15, x14
	lsr x15, x15, #3
	sub x11, x11, x13
	cmp x9, x15
	csel x9, x9, x15, lo
	ldr x14, [sp, #160]
	ldr x15, [sp, #88]
	add x15, x15, x14
	cmp x9, x15
	csel x9, x9, x15, lo
	ldr x14, [sp, #128]
	ldr x15, [sp, #96]
	add x15, x15, x14
	cmp x9, x15
	csel x9, x9, x15, lo
	ldr x14, [sp, #136]
	ldr x15, [sp, #104]
	add x15, x15, x14
	cmp x9, x15
	csel x9, x9, x15, lo
	ldr x14, [sp, #168]
	ldr x15, [sp, #112]
	add x15, x15, x14
	cmp x9, x15
	csel x9, x9, x15, lo
	sub x14, x10, #1
	cmp x9, x14
	csel x9, x9, x14, lo
	sub x9, x9, x13
	add x9, x9, #1
	mov x2, x4
LBB9_5:
	ldr q17, [x27, x8]
	ushll.8h v16, v17, #0
	ushll.4s v18, v16, #0
	ushll.2d v20, v18, #0
	ushll2.4s v21, v16, #0
	ldr q16, [x23, x8]
	ldr q19, [x7, x8]
	ushll2.2d v22, v18, #0
	ushll.2d v23, v21, #0
	ushll2.8h v18, v17, #0
	ushll.4s v24, v18, #0
	ldr q17, [x22, x8]
	ushll.2d v25, v24, #0
	ushll2.2d v26, v21, #0
	ldr q21, [x30, x8]
	ldr q27, [x28, x8]
	ushll2.2d v24, v24, #0
	ushll2.4s v18, v18, #0
	ushll.2d v28, v18, #0
	ushll2.2d v18, v18, #0
	shl.2d v18, v18, #56
	shl.2d v29, v28, #56
	shl.2d v24, v24, #56
	shl.2d v28, v26, #56
	shl.2d v25, v25, #56
	ushll.8h v26, v27, #0
	ushll.4s v30, v26, #0
	ushll.2d v31, v30, #0
	shl.2d v23, v23, #56
	ushll2.2d v30, v30, #0
	ushll2.4s v26, v26, #0
	ushll.2d v8, v26, #0
	ushll2.8h v27, v27, #0
	ushll.4s v9, v27, #0
	shl.2d v22, v22, #56
	ushll.2d v10, v9, #0
	ushll2.2d v26, v26, #0
	ushll2.2d v9, v9, #0
	ushll2.4s v27, v27, #0
	ushll.2d v11, v27, #0
	shl.2d v20, v20, #56
	ushll2.2d v27, v27, #0
	shl.2d v12, v27, #48
	shl.2d v11, v11, #48
	shl.2d v9, v9, #48
	shl.2d v13, v26, #48
	shl.2d v10, v10, #48
	shl.2d v8, v8, #48
	shl.2d v26, v30, #48
	shl.2d v27, v31, #48
	orr.16b v27, v20, v27
	orr.16b v26, v22, v26
	orr.16b v20, v23, v8
	orr.16b v22, v25, v10
	orr.16b v25, v28, v13
	ushll2.8h v23, v21, #0
	ushll2.4s v30, v23, #0
	ushll2.2d v28, v30, #0
	orr.16b v24, v24, v9
	ushll.2d v30, v30, #0
	ushll.4s v31, v23, #0
	ushll2.2d v8, v31, #0
	ushll.8h v23, v21, #0
	ushll2.4s v9, v23, #0
	orr.16b v21, v29, v11
	ushll.4s v29, v23, #0
	ushll.2d v23, v29, #0
	shl.2d v10, v23, #40
	ushll.8h v11, v17, #0
	ushll.4s v13, v11, #0
	orr.16b v23, v18, v12
	shll.2d v18, v13, #32
	orr.16b v18, v10, v18
	ushll2.2d v10, v9, #0
	ushll.2d v31, v31, #0
	ushll.2d v9, v9, #0
	ushll2.2d v29, v29, #0
	shl.2d v29, v29, #40
	shl.2d v9, v9, #40
	shl.2d v31, v31, #40
	shl.2d v10, v10, #40
	ushll2.4s v11, v11, #0
	shl.2d v8, v8, #40
	ushll2.8h v12, v17, #0
	orr.16b v17, v27, v18
	ushll.4s v27, v12, #0
	ushll2.4s v18, v12, #0
	shll2.2d v12, v13, #32
	shl.2d v30, v30, #40
	orr.16b v29, v29, v12
	shll2.2d v12, v18, #32
	shll.2d v13, v18, #32
	orr.16b v18, v26, v29
	shll.2d v26, v11, #32
	shl.2d v28, v28, #40
	orr.16b v26, v9, v26
	shll2.2d v29, v27, #32
	shll2.2d v9, v11, #32
	shll.2d v27, v27, #32
	orr.16b v31, v31, v27
	orr.16b v20, v20, v26
	orr.16b v9, v10, v9
	orr.16b v8, v8, v29
	orr.16b v27, v30, v13
	orr.16b v26, v28, v12
	ushll2.8h v28, v19, #0
	orr.16b v22, v22, v31
	ushll.4s v29, v28, #0
	ushll.8h v19, v19, #0
	ushll2.4s v31, v19, #0
	ushll.4s v30, v19, #0
	ushll.2d v10, v30, #24
	orr.16b v19, v25, v9
	ushll2.2d v25, v30, #24
	ushll.2d v30, v31, #24
	ushll.2d v9, v29, #24
	ushll.8h v11, v16, #0
	ushll.4s v12, v11, #0
	orr.16b v24, v24, v8
	ushll2.4s v8, v11, #0
	ushll2.8h v11, v16, #0
	ushll.2d v16, v12, #16
	orr.16b v16, v10, v16
	ushll.4s v10, v11, #0
	ushll2.2d v12, v12, #16
	orr.16b v25, v25, v12
	ushll.2d v12, v8, #16
	orr.16b v30, v30, v12
	ushll.2d v12, v10, #16
	orr.16b v9, v9, v12
	orr.16b v21, v21, v27
	ldr q27, [x21, x8]
	ushll2.4s v28, v28, #0
	ushll2.2d v31, v31, #24
	ushll2.2d v29, v29, #24
	orr.16b v23, v23, v26
	ushll.2d v26, v28, #24
	ushll2.2d v28, v28, #24
	ushll2.4s v11, v11, #0
	ushll2.2d v8, v8, #16
	orr.16b v31, v31, v8
	ushll.2d v8, v11, #16
	ushll2.2d v10, v10, #16
	orr.16b v29, v29, v10
	ushll2.8h v10, v27, #0
	orr.16b v26, v26, v8
	ushll2.4s v8, v10, #0
	ushll2.2d v11, v11, #16
	orr.16b v28, v28, v11
	ushll2.2d v11, v8, #8
	orr.16b v28, v28, v11
	orr.16b v23, v23, v28
	ldr q28, [x0, x8]
	ushll.4s v10, v10, #0
	ushll.8h v27, v27, #0
	ushll.2d v8, v8, #8
	orr.16b v26, v26, v8
	ushll2.4s v8, v27, #0
	ushll.4s v27, v27, #0
	orr.16b v21, v21, v26
	ushll2.2d v26, v10, #8
	orr.16b v26, v29, v26
	ushll.2d v29, v27, #8
	ushll2.2d v27, v27, #8
	orr.16b v26, v24, v26
	ushll.2d v24, v8, #8
	ushll.2d v10, v10, #8
	ushll2.2d v8, v8, #8
	orr.16b v31, v31, v8
	ushll2.8h v8, v28, #0
	orr.16b v19, v19, v31
	ushll2.4s v31, v8, #0
	orr.16b v9, v9, v10
	ushll2.2d v10, v31, #0
	ushll.2d v31, v31, #0
	ushll.4s v8, v8, #0
	orr.16b v9, v22, v9
	ushll2.2d v11, v8, #0
	ushll.8h v22, v28, #0
	orr.16b v24, v30, v24
	ushll2.4s v28, v22, #0
	orr.16b v30, v20, v24
	ushll2.2d v20, v28, #0
	ushll.2d v8, v8, #0
	ushll.2d v24, v28, #0
	ushll.4s v22, v22, #0
	orr.16b v25, v25, v27
	ushll2.2d v27, v22, #0
	ushll.2d v22, v22, #0
	orr.16b v28, v18, v25
	orr.16b v16, v16, v29
	orr.16b v12, v17, v16
	orr.16b v29, v12, v22
	orr.16b v25, v28, v27
	orr.16b v24, v30, v24
	orr.16b v22, v9, v8
	orr.16b v20, v19, v20
	orr.16b v18, v26, v11
	orr.16b v17, v21, v31
	orr.16b v16, v23, v10
	ushr.2d v27, v12, #7
	ushr.2d v28, v28, #7
	ushr.2d v30, v30, #7
	ushr.2d v31, v9, #7
	ushr.2d v19, v19, #7
	ushr.2d v26, v26, #7
	ushr.2d v21, v21, #7
	ushr.2d v23, v23, #7
	eor.16b v23, v23, v16
	eor.16b v21, v21, v17
	eor.16b v26, v26, v18
	eor.16b v19, v19, v20
	eor.16b v31, v31, v22
	eor.16b v30, v30, v24
	eor.16b v28, v28, v25
	eor.16b v27, v27, v29
	and.16b v8, v27, v7
	and.16b v9, v28, v7
	and.16b v28, v30, v7
	and.16b v27, v31, v7
	and.16b v30, v19, v7
	and.16b v26, v26, v7
	and.16b v21, v21, v7
	and.16b v19, v23, v7
	shl.2d v23, v19, #7
	add.2d v19, v23, v19
	shl.2d v23, v21, #7
	add.2d v21, v23, v21
	shl.2d v23, v26, #7
	shl.2d v31, v30, #7
	add.2d v23, v23, v26
	add.2d v26, v31, v30
	shl.2d v30, v27, #7
	add.2d v27, v30, v27
	shl.2d v30, v28, #7
	shl.2d v31, v9, #7
	add.2d v28, v30, v28
	add.2d v30, v31, v9
	shl.2d v31, v8, #7
	add.2d v10, v31, v8
	eor.16b v31, v10, v29
	ushr.2d v8, v31, #14
	eor3.16b v8, v10, v29, v8
	and.16b v8, v8, v6
	shl.2d v9, v8, #14
	add.2d v9, v9, v8
	eor.16b v8, v30, v25
	eor3.16b v29, v10, v29, v9
	ushr.2d v29, v29, #28
	eor3.16b v10, v9, v31, v29
	dup.2d v29, x25
	and.16b v10, v10, v29
	xtn.2s v10, v10
	umull.2d v10, v10, v5
	eor3.16b v31, v9, v31, v10
	ushr.2d v9, v8, #14
	eor3.16b v9, v30, v25, v9
	and.16b v9, v9, v6
	shl.2d v10, v9, #14
	add.2d v9, v10, v9
	eor3.16b v25, v30, v25, v9
	eor.16b v30, v28, v24
	ushr.2d v25, v25, #28
	eor3.16b v25, v9, v8, v25
	and.16b v25, v25, v29
	xtn.2s v25, v25
	umull.2d v25, v25, v5
	eor3.16b v25, v9, v8, v25
	ushr.2d v8, v30, #14
	eor3.16b v8, v28, v24, v8
	and.16b v8, v8, v6
	shl.2d v9, v8, #14
	add.2d v8, v9, v8
	eor3.16b v24, v28, v24, v8
	eor.16b v28, v27, v22
	ushr.2d v24, v24, #28
	eor3.16b v24, v8, v30, v24
	and.16b v24, v24, v29
	xtn.2s v24, v24
	umull.2d v24, v24, v5
	eor3.16b v24, v8, v30, v24
	ushr.2d v30, v28, #14
	eor3.16b v30, v27, v22, v30
	and.16b v30, v30, v6
	shl.2d v8, v30, #14
	add.2d v30, v8, v30
	eor3.16b v22, v27, v22, v30
	eor.16b v27, v26, v20
	ushr.2d v22, v22, #28
	eor3.16b v22, v30, v28, v22
	and.16b v22, v22, v29
	xtn.2s v22, v22
	umull.2d v22, v22, v5
	eor3.16b v22, v30, v28, v22
	ushr.2d v28, v27, #14
	eor3.16b v28, v26, v20, v28
	and.16b v28, v28, v6
	shl.2d v30, v28, #14
	add.2d v28, v30, v28
	eor3.16b v20, v26, v20, v28
	eor.16b v26, v23, v18
	ushr.2d v20, v20, #28
	eor3.16b v20, v28, v27, v20
	and.16b v20, v20, v29
	xtn.2s v20, v20
	umull.2d v20, v20, v5
	eor3.16b v20, v28, v27, v20
	ushr.2d v27, v26, #14
	eor3.16b v27, v23, v18, v27
	and.16b v27, v27, v6
	shl.2d v28, v27, #14
	add.2d v27, v28, v27
	eor3.16b v18, v23, v18, v27
	eor.16b v23, v21, v17
	ushr.2d v18, v18, #28
	eor3.16b v18, v27, v26, v18
	and.16b v18, v18, v29
	xtn.2s v18, v18
	umull.2d v18, v18, v5
	eor3.16b v18, v27, v26, v18
	ushr.2d v26, v23, #14
	eor3.16b v26, v21, v17, v26
	and.16b v26, v26, v6
	shl.2d v27, v26, #14
	add.2d v26, v27, v26
	eor3.16b v17, v21, v17, v26
	eor.16b v21, v19, v16
	ushr.2d v17, v17, #28
	eor3.16b v17, v26, v23, v17
	and.16b v17, v17, v29
	xtn.2s v17, v17
	umull.2d v17, v17, v5
	eor3.16b v17, v26, v23, v17
	ushr.2d v23, v21, #14
	eor3.16b v23, v19, v16, v23
	and.16b v23, v23, v6
	shl.2d v26, v23, #14
	add.2d v23, v26, v23
	eor3.16b v16, v19, v16, v23
	ushr.2d v16, v16, #28
	eor3.16b v16, v23, v21, v16
	and.16b v16, v16, v29
	xtn.2s v16, v16
	umull.2d v16, v16, v5
	eor3.16b v16, v23, v21, v16
	stp q17, q16, [x2, #96]
	stp q22, q18, [x2, #64]
	stp q24, q20, [x2, #32]
	stp q31, q25, [x2], #128
	add x8, x8, #16
	cmp x9, x8
	b.ne LBB9_5
LBB9_6:
	ldur x8, [x29, #-152]
	add x8, x8, #1
	stur x8, [x29, #-152]
	ldur x8, [x29, #-160]
	add x8, x8, x11, lsl #3
LBB9_7:
	add x9, x12, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x5, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x17, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x6, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x19, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x20, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x26, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x3, x11
	cmp x9, x1
	b.hs LBB9_21
	add x9, x8, #8
	cmn x8, #9
	b.hi LBB9_20
	ldur x13, [x29, #-144]
	cmp x9, x13
	b.hi LBB9_20
	ldrb w9, [x0, x11]
	ldrb w13, [x21, x11]
	ldrb w15, [x23, x11]
	ldrb w16, [x7, x11]
	ldrb w2, [x22, x11]
	ldrb w24, [x30, x11]
	ldrb w14, [x28, x11]
	ldrb w25, [x27, x11]
	lsl x14, x14, #48
	orr x14, x14, x25, lsl #56
	mov w25, #-252645136
	orr x14, x14, x24, lsl #40
	orr x14, x14, x2, lsl #32
	orr x14, x14, x16, lsl #24
	orr x14, x14, x15, lsl #16
	orr x13, x14, x13, lsl #8
	orr x9, x13, x9
	eor x13, x9, x13, lsr #7
	mov x14, #-6148914691236517206
	and x14, x14, #0x1fe01fe01fe01fe
	and x13, x13, x14
	orr x13, x13, x13, lsl #7
	eor x9, x13, x9
	eor x13, x9, x9, lsr #14
	mov x14, #52428
	movk x14, #52428, lsl #32
	and x13, x13, x14
	orr x13, x13, x13, lsl #14
	eor x9, x13, x9
	lsr x13, x9, #28
	eor w13, w13, w9
	and x13, x13, x25
	orr x13, x13, x13, lsl #28
	eor x9, x13, x9
	str x9, [x4, x11, lsl #3]
	add x11, x11, #1
	add x8, x8, #8
	cmp x10, x11
	b.ne LBB9_7
	mov x2, x4
	ldr x4, [sp, #8]
	ldr x13, [sp, #120]
	add x13, x13, x4
	ldr x8, [sp, #152]
	sub x8, x8, x4
	str x8, [sp, #152]
	ldur x9, [x29, #-160]
	add x9, x9, x4
	ldr x8, [sp, #144]
	sub x8, x8, x4
	str x8, [sp, #144]
	add x5, x5, x10
	ldr x8, [sp, #160]
	sub x8, x8, x10
	str x8, [sp, #160]
	add x26, x26, x10
	ldr x8, [sp, #128]
	sub x8, x8, x10
	str x8, [sp, #128]
	add x3, x3, x10
	ldr x8, [sp, #136]
	sub x8, x8, x10
	str x8, [sp, #136]
	add x12, x12, x10
	ldr x8, [sp, #168]
	sub x8, x8, x10
	str x8, [sp, #168]
	add x27, x27, x10
	add x2, x2, x4
	add x28, x28, x10
	add x30, x30, x10
	add x22, x22, x10
	add x7, x7, x10
	add x23, x23, x10
	add x21, x21, x10
	add x0, x0, x10
	add x20, x20, x10
	add x19, x19, x10
	add x6, x6, x10
	add x17, x17, x10
	ldr x8, [sp, #64]
	ldur x15, [x29, #-152]
	cmp x15, x8
	b.ne LBB9_3
LBB9_19:
	ldp x29, x30, [sp, #336]
	ldp x20, x19, [sp, #320]
	ldp x22, x21, [sp, #304]
	ldp x24, x23, [sp, #288]
	ldp x26, x25, [sp, #272]
	ldp x28, x27, [sp, #256]
	ldp d9, d8, [sp, #240]
	ldp d11, d10, [sp, #224]
	ldp d13, d12, [sp, #208]
	add sp, sp, #352
	ret
LBB9_20:
Lloh94:
	adrp x3, l_anon.2649d3bd1fda821ec307976ec34fb359.11@PAGE
Lloh95:
	add x3, x3, l_anon.2649d3bd1fda821ec307976ec34fb359.11@PAGEOFF
	mov x0, x8
	mov x1, x9
	ldur x2, [x29, #-144]
	bl core::slice::index::slice_index_fail
LBB9_21:
Lloh96:
	adrp x2, l_anon.2649d3bd1fda821ec307976ec34fb359.12@PAGE
Lloh97:
	add x2, x2, l_anon.2649d3bd1fda821ec307976ec34fb359.12@PAGEOFF
	mov x0, x9
	bl core::panicking::panic_bounds_check
