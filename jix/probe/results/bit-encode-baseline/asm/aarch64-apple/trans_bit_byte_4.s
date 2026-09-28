jix_probe::bit_shuffle::trans_bit_byte:
Lfunc_begin0:
	stp d15, d14, [sp, #-160]!
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x28, x27, [sp, #64]
	stp x26, x25, [sp, #80]
	stp x24, x23, [sp, #96]
	stp x22, x21, [sp, #112]
	stp x20, x19, [sp, #128]
	stp x29, x30, [sp, #144]
	add x29, sp, #144
	sub sp, sp, #416
	str x5, [sp, #288]
	cbz x5, LBB0_26
	lsr x10, x4, #3
	cbz x10, LBB0_26
	stp xzr, xzr, [x29, #-232]
	mov x13, #0
	mov x5, #0
	ldr x17, [sp, #288]
	mul x19, x17, x10
	fmov d6, x19
	lsl x21, x19, #1
	fmov d1, x21
	add x8, x21, x19
	add x6, x2, x8
	fmov d2, x8
	lsl x7, x19, #2
	add x9, x21, x19
	stur x9, [x29, #-160]
	lsl x23, x8, #1
	lsl x8, x19, #3
	str x4, [sp, #280]
	neg x9, x21
	fmov d3, x9
	sub x9, x19, x7
	fmov d4, x9
	neg x9, x7
	add x20, x7, x19
	neg x11, x20
	sub x4, x21, x8
	sub x24, x19, x8
	mov.d v6[1], x21
	mov.16b v0, v2
	mov.d v0[1], x7
	dup.2d v5, x1
	dup.2d v21, x3
	mov.d v2[1], x20
	movi.2d v7, #0xffffffffffffffff
	add.2d v0, v0, v7
	add.2d v6, v6, v7
	mov w12, #15
	dup.2d v7, x12
	sub x12, x20, #1
	sub x14, x23, #1
	sub x27, x8, x19
	add x25, x2, x27
	sub x8, x27, #1
	cmhi.2d v6, v7, v6
	cmhi.2d v0, v7, v0
	uzp1.4s v0, v6, v0
	xtn.4h v0, v0
	fmov x15, d0
	mov.d v1[1], x7
	cmp x15, #0
	ccmp x12, #15, #0, eq
	cset w12, lo
	cmp x14, #15
	mov.d v4[1], x11
	ccmp x8, #15, #0, hs
	csinc w8, w12, wzr, hs
	str w8, [sp, #236]
	mneg x8, x17, x10
	stur x8, [x29, #-240]
	add x28, x2, x23
	mov.d v3[1], x9
	ldr x9, [sp, #280]
	add x30, x2, x20
	add x26, x2, x7
	mov x16, x21
	add x21, x2, x21
	add x22, x2, x19
	add x8, x0, #63
	stur x8, [x29, #-248]
Lloh0:
	adrp x8, lCPI0_0@PAGE
Lloh1:
	ldr q22, [x8, lCPI0_0@PAGEOFF]
Lloh2:
	adrp x8, lCPI0_1@PAGE
Lloh3:
	ldr q23, [x8, lCPI0_1@PAGEOFF]
Lloh4:
	adrp x8, lCPI0_2@PAGE
Lloh5:
	ldr q24, [x8, lCPI0_2@PAGEOFF]
Lloh6:
	adrp x8, lCPI0_3@PAGE
Lloh7:
	ldr q25, [x8, lCPI0_3@PAGEOFF]
Lloh8:
	adrp x8, lCPI0_4@PAGE
Lloh9:
	ldr q26, [x8, lCPI0_4@PAGEOFF]
Lloh10:
	adrp x8, lCPI0_5@PAGE
Lloh11:
	ldr q27, [x8, lCPI0_5@PAGEOFF]
Lloh12:
	adrp x8, lCPI0_6@PAGE
Lloh13:
	ldr q28, [x8, lCPI0_6@PAGEOFF]
	mov x17, #52428
	movk x17, #52428, lsl #32
Lloh14:
	adrp x8, lCPI0_7@PAGE
Lloh15:
	ldr q29, [x8, lCPI0_7@PAGEOFF]
	str x24, [sp, #248]
	stur x24, [x29, #-256]
	str x27, [sp, #240]
	mov x12, x27
	stp x4, x23, [sp, #256]
	str x4, [sp, #296]
	mov x11, x23
	str x19, [sp, #272]
	stp q2, q1, [sp, #192]
	stp q4, q3, [sp, #160]
	stp q21, q5, [sp, #128]
	stp q23, q22, [sp, #96]
	stp q25, q24, [sp, #64]
	stp q27, q26, [sp, #32]
	stp q29, q28, [sp]
LBB0_3:
	mov x4, #0
	cmp x3, x19
	csel x14, x3, x19, hi
	cmp x3, x11
	csel x8, x3, x11, hi
	stur x8, [x29, #-208]
	cmp x3, x12
	csel x8, x3, x12, hi
	stur x8, [x29, #-192]
	cmp x3, x13
	csel x8, x3, x13, hi
	stur x8, [x29, #-176]
	mul x8, x9, x5
	dup.2d v0, x8
	add.2d v6, v0, v22
	add.2d v7, v0, v23
	add.2d v16, v0, v24
	add.2d v17, v0, v25
	cmhi.2d v18, v5, v17
	bit.16b v17, v5, v18
	cmhi.2d v18, v5, v16
	bit.16b v16, v5, v18
	cmhi.2d v18, v5, v7
	bit.16b v7, v5, v18
	cmhi.2d v18, v5, v6
	bit.16b v6, v5, v18
	sub.2d v6, v6, v0
	sub.2d v7, v7, v0
	sub.2d v16, v16, v0
	sub.2d v0, v17, v0
	add.2d v0, v0, v26
	add.2d v16, v16, v27
	add.2d v7, v7, v28
	add.2d v6, v6, v29
	ushr.2d v6, v6, #3
	ushr.2d v7, v7, #3
	ushr.2d v16, v16, #3
	ushr.2d v0, v0, #3
	stur x5, [x29, #-216]
	mul x8, x10, x5
	ldp x15, x9, [sp, #264]
	add x9, x9, x8
	subs x9, x3, x9
	csel x9, xzr, x9, lo
	dup.2d v17, x8
	add.2d v18, v1, v17
	add.2d v19, v2, v17
	cmhi.2d v20, v21, v19
	bit.16b v19, v21, v20
	cmhi.2d v20, v21, v18
	bit.16b v18, v21, v20
	sub.2d v20, v4, v17
	sub.2d v17, v3, v17
	add.2d v17, v18, v17
	add.2d v18, v19, v20
	add x24, x15, x8
	cmp x3, x24
	csel x24, x3, x24, hi
	ldr x15, [sp, #256]
	sub x23, x15, x8
	add x23, x24, x23
	ldr x15, [sp, #240]
	add x24, x15, x8
	cmp x3, x24
	csel x24, x3, x24, hi
	ldr x15, [sp, #248]
	sub x27, x15, x8
	add x27, x24, x27
	subs x8, x3, x8
	csel x8, xzr, x8, lo
	cmhi.2d v19, v18, v0
	bif.16b v0, v18, v19
	cmhi.2d v18, v17, v16
	bif.16b v16, v17, v18
	cmhi.2d v17, v7, v16
	bit.16b v7, v16, v17
	cmhi.2d v16, v6, v0
	bif.16b v0, v6, v16
	cmhi.2d v6, v0, v7
	bit.16b v0, v7, v6
	mov d6, v0[1]
	cmhi d7, d6, d0
	bif.8b v0, v6, v7
	fmov x24, d0
	cmp x24, x23
	csel x23, x24, x23, lo
	cmp x27, x9
	csel x9, x27, x9, lo
	sub x15, x10, #1
	cmp x8, x15
	csel x8, x8, x15, lo
	cmp x23, x9
	csel x9, x23, x9, lo
	cmp x9, x8
	csel x9, x9, x8, lo
	cmp x9, #16
	cset w8, lo
	ldr w15, [sp, #236]
	orr w8, w8, w15
	mov w5, #-252645136
	tbnz w8, #0, LBB0_7
	mov x8, #0
	add x9, x9, #1
	ands x4, x9, #0xf
	mov w15, #16
	csel x23, x15, x4, eq
	ldur x15, [x29, #-240]
	add x14, x14, x15
	cmp x24, x14
	csel x14, x24, x14, lo
	sub x4, x9, x23
	ldr x9, [sp, #296]
	ldur x15, [x29, #-208]
	add x9, x15, x9
	cmp x14, x9
	csel x9, x14, x9, lo
	ldp x14, x27, [x29, #-256]
	ldur x15, [x29, #-192]
	add x14, x15, x14
	cmp x9, x14
	csel x9, x9, x14, lo
	ldur x14, [x29, #-232]
	ldur x15, [x29, #-176]
	add x14, x15, x14
	cmp x9, x14
	csel x9, x9, x14, lo
	sub x14, x10, #1
	cmp x9, x14
	csel x9, x9, x14, lo
	sub x9, x9, x23
	add x9, x9, #1
	mov w15, #1
	movk w15, #4096, lsl #16
	adrp x24, lCPI0_8@PAGE
LBB0_5:
	sub x14, x27, #48
	ldur b23, [x27, #-56]
	ld1.b { v23 }[1], [x14]
	sub x14, x27, #40
	ld1.b { v23 }[2], [x14]
	sub x14, x27, #32
	ld1.b { v23 }[3], [x14]
	sub x14, x27, #24
	ld1.b { v23 }[4], [x14]
	sub x14, x27, #16
	ld1.b { v23 }[5], [x14]
	sub x14, x27, #8
	ld1.b { v23 }[6], [x14]
	mov x23, x27
	ld1.b { v23 }[7], [x23], #1
	add x14, x27, #16
	ldr b24, [x27, #8]
	ld1.b { v24 }[1], [x14]
	add x14, x27, #24
	ld1.b { v24 }[2], [x14]
	sub x14, x27, #49
	ldur b25, [x27, #-57]
	ld1.b { v25 }[1], [x14]
	add x14, x27, #32
	ld1.b { v24 }[3], [x14]
	sub x14, x27, #41
	ld1.b { v25 }[2], [x14]
	add x14, x27, #40
	ld1.b { v24 }[4], [x14]
	sub x14, x27, #33
	ld1.b { v25 }[3], [x14]
	sub x14, x27, #25
	ld1.b { v25 }[4], [x14]
	add x14, x27, #15
	ldr b26, [x27, #7]
	ld1.b { v26 }[1], [x14]
	add x14, x27, #23
	ld1.b { v26 }[2], [x14]
	add x14, x27, #31
	ld1.b { v26 }[3], [x14]
	add x14, x27, #14
	ldr b27, [x27, #6]
	ld1.b { v27 }[1], [x14]
	add x14, x27, #39
	ld1.b { v26 }[4], [x14]
	add x14, x27, #22
	ld1.b { v27 }[2], [x14]
	add x14, x27, #48
	ld1.b { v24 }[5], [x14]
	add x14, x27, #30
	ld1.b { v27 }[3], [x14]
	sub x14, x27, #17
	ld1.b { v25 }[5], [x14]
	add x14, x27, #38
	ld1.b { v27 }[4], [x14]
	sub x14, x27, #50
	ldur b28, [x27, #-58]
	ld1.b { v28 }[1], [x14]
	sub x14, x27, #42
	ld1.b { v28 }[2], [x14]
	sub x14, x27, #34
	ld1.b { v28 }[3], [x14]
	sub x14, x27, #26
	ld1.b { v28 }[4], [x14]
	sub x14, x27, #51
	ldur b29, [x27, #-59]
	ld1.b { v29 }[1], [x14]
	add x14, x27, #47
	ld1.b { v26 }[5], [x14]
	sub x14, x27, #43
	ld1.b { v29 }[2], [x14]
	add x14, x27, #46
	ld1.b { v27 }[5], [x14]
	sub x14, x27, #35
	ld1.b { v29 }[3], [x14]
	sub x14, x27, #18
	ld1.b { v28 }[5], [x14]
	sub x14, x27, #27
	ld1.b { v29 }[4], [x14]
	add x14, x27, #56
	ld1.b { v24 }[6], [x14]
	sub x14, x27, #19
	ld1.b { v29 }[5], [x14]
	sub x14, x27, #9
	ld1.b { v25 }[6], [x14]
	add x14, x27, #13
	ldr b30, [x27, #5]
	ld1.b { v30 }[1], [x14]
	add x14, x27, #21
	ld1.b { v30 }[2], [x14]
	add x14, x27, #29
	ld1.b { v30 }[3], [x14]
	add x14, x27, #37
	ld1.b { v30 }[4], [x14]
	add x14, x27, #45
	ld1.b { v30 }[5], [x14]
	add x14, x27, #12
	ldr b31, [x27, #4]
	ld1.b { v31 }[1], [x14]
	add x14, x27, #55
	ld1.b { v26 }[6], [x14]
	add x14, x27, #20
	ld1.b { v31 }[2], [x14]
	add x14, x27, #54
	ld1.b { v27 }[6], [x14]
	add x14, x27, #28
	ld1.b { v31 }[3], [x14]
	sub x14, x27, #10
	ld1.b { v28 }[6], [x14]
	add x14, x27, #36
	ld1.b { v31 }[4], [x14]
	sub x14, x27, #11
	ld1.b { v29 }[6], [x14]
	add x14, x27, #44
	ld1.b { v31 }[5], [x14]
	add x14, x27, #53
	ld1.b { v30 }[6], [x14]
	sub x14, x27, #52
	ldur b8, [x27, #-60]
	ld1.b { v8 }[1], [x14]
	sub x14, x27, #44
	ld1.b { v8 }[2], [x14]
	sub x14, x27, #36
	ld1.b { v8 }[3], [x14]
	sub x14, x27, #28
	ld1.b { v8 }[4], [x14]
	sub x14, x27, #20
	ld1.b { v8 }[5], [x14]
	sub x14, x27, #53
	ldur b9, [x27, #-61]
	ld1.b { v9 }[1], [x14]
	add x14, x27, #52
	ld1.b { v31 }[6], [x14]
	sub x14, x27, #45
	ld1.b { v9 }[2], [x14]
	sub x14, x27, #12
	ld1.b { v8 }[6], [x14]
	sub x14, x27, #37
	ld1.b { v9 }[3], [x14]
	add x14, x27, #64
	ld1.b { v24 }[7], [x14]
	sub x14, x27, #29
	ld1.b { v9 }[4], [x14]
	sub x14, x27, #1
	ld1.b { v25 }[7], [x14]
	sub x14, x27, #21
	ld1.b { v9 }[5], [x14]
	add x14, x27, #63
	ld1.b { v26 }[7], [x14]
	sub x14, x27, #13
	ld1.b { v9 }[6], [x14]
	add x14, x27, #11
	ldr b10, [x27, #3]
	ld1.b { v10 }[1], [x14]
	add x14, x27, #19
	ld1.b { v10 }[2], [x14]
	add x14, x27, #27
	ld1.b { v10 }[3], [x14]
	add x14, x27, #35
	ld1.b { v10 }[4], [x14]
	add x14, x27, #43
	ld1.b { v10 }[5], [x14]
	add x14, x27, #51
	ld1.b { v10 }[6], [x14]
	add x14, x27, #10
	ldr b11, [x27, #2]
	ld1.b { v11 }[1], [x14]
	add x14, x27, #62
	ld1.b { v27 }[7], [x14]
	add x14, x27, #18
	ld1.b { v11 }[2], [x14]
	sub x14, x27, #2
	ld1.b { v28 }[7], [x14]
	add x14, x27, #26
	ld1.b { v11 }[3], [x14]
	sub x14, x27, #3
	ld1.b { v29 }[7], [x14]
	add x14, x27, #34
	ld1.b { v11 }[4], [x14]
	add x14, x27, #61
	ld1.b { v30 }[7], [x14]
	add x14, x27, #42
	ld1.b { v11 }[5], [x14]
	add x14, x27, #60
	ld1.b { v31 }[7], [x14]
	add x14, x27, #50
	ld1.b { v11 }[6], [x14]
	sub x14, x27, #4
	ld1.b { v8 }[7], [x14]
	sub x14, x27, #54
	ldur b12, [x27, #-62]
	ld1.b { v12 }[1], [x14]
	sub x14, x27, #46
	ld1.b { v12 }[2], [x14]
	sub x14, x27, #38
	ld1.b { v12 }[3], [x14]
	sub x14, x27, #30
	ld1.b { v12 }[4], [x14]
	sub x14, x27, #22
	ld1.b { v12 }[5], [x14]
	sub x14, x27, #14
	ld1.b { v12 }[6], [x14]
	ldr b13, [x23]
	sub x14, x27, #5
	ld1.b { v9 }[7], [x14]
	add x14, x27, #9
	ld1.b { v13 }[1], [x14]
	add x14, x27, #59
	ld1.b { v10 }[7], [x14]
	add x14, x27, #17
	ld1.b { v13 }[2], [x14]
	add x14, x27, #58
	ld1.b { v11 }[7], [x14]
	add x14, x27, #25
	ld1.b { v13 }[3], [x14]
	sub x14, x27, #6
	ld1.b { v12 }[7], [x14]
	add x14, x27, #33
	ld1.b { v13 }[4], [x14]
	add x14, x27, #41
	ld1.b { v13 }[5], [x14]
	add x14, x27, #49
	ld1.b { v13 }[6], [x14]
	add x14, x27, #57
	ld1.b { v13 }[7], [x14]
	sub x14, x27, #55
	ldur b14, [x27, #-63]
	ld1.b { v14 }[1], [x14]
	sub x14, x27, #47
	ld1.b { v14 }[2], [x14]
	sub x14, x27, #39
	ld1.b { v14 }[3], [x14]
	sub x14, x27, #31
	ld1.b { v14 }[4], [x14]
	sub x14, x27, #23
	ld1.b { v14 }[5], [x14]
	sub x14, x27, #15
	ld1.b { v14 }[6], [x14]
	sub x14, x27, #7
	ld1.b { v14 }[7], [x14]
	ushll.8h v0, v23, #0
	ushll.4s v6, v0, #0
	ushll.2d v7, v6, #0
	ushll2.2d v6, v6, #0
	ushll2.4s v0, v0, #0
	ushll.2d v16, v0, #0
	ushll2.2d v17, v0, #0
	ushll.8h v0, v24, #0
	ushll.4s v18, v0, #0
	ushll.2d v19, v18, #0
	ushll2.2d v18, v18, #0
	ushll2.4s v0, v0, #0
	ushll.2d v20, v0, #0
	ushll2.2d v0, v0, #0
	shl.2d v0, v0, #56
	stur q0, [x29, #-176]
	shl.2d v17, v17, #56
	shl.2d v16, v16, #56
	shl.2d v6, v6, #56
	shl.2d v20, v20, #56
	shl.2d v7, v7, #56
	ushll.8h v21, v25, #0
	ushll.4s v23, v21, #0
	ushll.2d v24, v23, #0
	ushll2.2d v23, v23, #0
	shl.2d v18, v18, #56
	ushll2.4s v21, v21, #0
	ushll.2d v25, v21, #0
	ushll2.2d v21, v21, #0
	ushll.8h v26, v26, #0
	ushll.4s v15, v26, #0
	shl.2d v19, v19, #56
	ushll.2d v2, v15, #0
	ushll2.2d v15, v15, #0
	ushll2.4s v26, v26, #0
	ushll.2d v3, v26, #0
	ushll2.2d v26, v26, #0
	shl.2d v26, v26, #48
	shl.2d v3, v3, #48
	shl.2d v15, v15, #48
	shl.2d v2, v2, #48
	shl.2d v21, v21, #48
	shl.2d v25, v25, #48
	shl.2d v23, v23, #48
	shl.2d v24, v24, #48
	orr.16b v7, v7, v24
	ushll.8h v24, v27, #0
	ushll2.4s v27, v24, #0
	ushll2.2d v1, v27, #0
	orr.16b v6, v6, v23
	ushll.2d v23, v27, #0
	ushll.4s v24, v24, #0
	ushll2.2d v27, v24, #0
	ushll.2d v24, v24, #0
	ushll.8h v28, v28, #0
	orr.16b v25, v16, v25
	ushll2.4s v16, v28, #0
	ushll2.2d v5, v16, #0
	ushll.2d v16, v16, #0
	ushll.4s v28, v28, #0
	ushll2.2d v4, v28, #0
	orr.16b v17, v17, v21
	ushll.2d v21, v28, #0
	shl.2d v21, v21, #40
	shl.2d v4, v4, #40
	shl.2d v28, v16, #40
	shl.2d v5, v5, #40
	orr.16b v2, v19, v2
	shl.2d v19, v24, #40
	shl.2d v24, v27, #40
	shl.2d v23, v23, #40
	shl.2d v1, v1, #40
	ushll.8h v16, v29, #0
	orr.16b v18, v18, v15
	ushll.4s v27, v16, #0
	ushll2.4s v29, v16, #0
	ushll.8h v16, v30, #0
	ushll.4s v30, v16, #0
	ushll2.4s v16, v16, #0
	orr.16b v3, v20, v3
	shll2.2d v20, v16, #32
	shll.2d v15, v16, #32
	shll2.2d v22, v30, #32
	shll.2d v30, v30, #32
	shll2.2d v0, v29, #32
	ldur q16, [x29, #-176]
	orr.16b v16, v16, v26
	shll.2d v26, v29, #32
	shll2.2d v29, v27, #32
	shll.2d v27, v27, #32
	orr.16b v21, v21, v27
	orr.16b v4, v4, v29
	orr.16b v26, v28, v26
	orr.16b v5, v5, v0
	orr.16b v19, v19, v30
	orr.16b v22, v24, v22
	orr.16b v24, v23, v15
	orr.16b v1, v1, v20
	orr.16b v0, v7, v21
	stur q0, [x29, #-176]
	ushll.8h v0, v31, #0
	ushll2.4s v7, v0, #0
	ushll.4s v20, v0, #0
	ushll.8h v21, v8, #0
	ushll2.4s v27, v21, #0
	orr.16b v0, v6, v4
	ushll.4s v4, v21, #0
	ushll.2d v21, v4, #24
	ushll2.2d v4, v4, #24
	ushll.2d v28, v27, #24
	ushll2.2d v27, v27, #24
	orr.16b v6, v25, v26
	ushll.2d v25, v20, #24
	ushll2.2d v20, v20, #24
	ushll.2d v26, v7, #24
	ushll2.2d v29, v7, #24
	ushll.8h v30, v9, #0
	orr.16b v7, v17, v5
	ushll.4s v5, v30, #0
	ushll2.4s v17, v30, #0
	ushll.8h v30, v10, #0
	ushll.4s v31, v30, #0
	ushll2.4s v30, v30, #0
	orr.16b v2, v2, v19
	ushll2.2d v19, v30, #16
	ushll.2d v30, v30, #16
	ushll2.2d v8, v31, #16
	ushll.2d v31, v31, #16
	ushll2.2d v9, v17, #16
	orr.16b v18, v18, v22
	ushll.2d v17, v17, #16
	ushll2.2d v22, v5, #16
	ushll.2d v5, v5, #16
	ushll.8h v10, v11, #0
	ushll2.4s v11, v10, #0
	orr.16b v3, v3, v24
	ushll.4s v24, v10, #0
	ushll.8h v10, v12, #0
	ushll2.4s v12, v10, #0
	ushll.4s v10, v10, #0
	ushll.2d v15, v10, #8
	ushll2.2d v10, v10, #8
	orr.16b v23, v21, v5
	ushll.2d v21, v12, #8
	ushll2.2d v12, v12, #8
	orr.16b v4, v4, v22
	ushll.2d v22, v24, #8
	ushll2.2d v24, v24, #8
	orr.16b v17, v28, v17
	ushll.2d v28, v11, #8
	ushll2.2d v11, v11, #8
	orr.16b v27, v27, v9
	orr.16b v25, v25, v31
	orr.16b v1, v16, v1
	orr.16b v16, v20, v8
	orr.16b v20, v26, v30
	orr.16b v19, v29, v19
	ushll.8h v26, v13, #0
	ushll2.4s v29, v26, #0
	ushll2.2d v5, v29, #0
	ushll.2d v8, v29, #0
	ushll.4s v26, v26, #0
	ushll.8h v29, v14, #0
	ushll2.4s v30, v29, #0
	ushll2.2d v9, v30, #0
	ushll2.2d v13, v26, #0
	ushll.2d v30, v30, #0
	ushll.4s v29, v29, #0
	ushll2.2d v14, v29, #0
	ushll.2d v29, v29, #0
	orr.16b v19, v19, v11
	ushll.2d v26, v26, #0
	orr.16b v1, v1, v19
	orr.16b v19, v20, v28
	orr.16b v3, v3, v19
	orr.16b v16, v16, v24
	orr.16b v19, v25, v22
	orr.16b v16, v18, v16
	orr.16b v2, v2, v19
	orr.16b v18, v27, v12
	orr.16b v7, v7, v18
	orr.16b v17, v17, v21
	orr.16b v4, v4, v10
	orr.16b v6, v6, v17
	orr.16b v0, v0, v4
	orr.16b v4, v23, v15
	ldur q17, [x29, #-176]
	orr.16b v4, v17, v4
	orr.16b v31, v4, v29
	orr.16b v27, v0, v14
	orr.16b v28, v6, v30
	orr.16b v29, v7, v9
	orr.16b v30, v2, v26
	orr.16b v12, v16, v13
	orr.16b v13, v3, v8
	orr.16b v14, v1, v5
	ushr.2d v4, v4, #7
	ushr.2d v0, v0, #7
	ushr.2d v5, v6, #7
	ushr.2d v6, v7, #7
	ushr.2d v2, v2, #7
	ushr.2d v7, v16, #7
	ushr.2d v3, v3, #7
	ushr.2d v1, v1, #7
	eor.16b v1, v1, v14
	eor.16b v3, v3, v13
	eor.16b v7, v7, v12
	eor.16b v2, v2, v30
	eor.16b v6, v6, v29
	eor.16b v5, v5, v28
	eor.16b v0, v0, v27
	eor.16b v4, v4, v31
	movi.8h v16, #170
	and.16b v4, v4, v16
	and.16b v0, v0, v16
	and.16b v5, v5, v16
	and.16b v6, v6, v16
	and.16b v2, v2, v16
	and.16b v7, v7, v16
	and.16b v3, v3, v16
	and.16b v1, v1, v16
	shl.2d v16, v1, #7
	add.2d v25, v16, v1
	shl.2d v1, v3, #7
	add.2d v24, v1, v3
	shl.2d v1, v7, #7
	shl.2d v3, v2, #7
	add.2d v23, v1, v7
	add.2d v20, v3, v2
	shl.2d v3, v6, #7
	add.2d v18, v3, v6
	shl.2d v6, v5, #7
	shl.2d v7, v0, #7
	add.2d v5, v6, v5
	add.2d v22, v7, v0
	shl.2d v0, v4, #7
	add.2d v4, v0, v4
	eor.16b v1, v4, v31
	eor.16b v2, v22, v27
	stp q1, q2, [x29, #-192]
	eor.16b v26, v5, v28
	eor.16b v3, v18, v29
	stur q3, [x29, #-208]
	eor.16b v8, v20, v30
	eor.16b v9, v23, v12
	eor.16b v10, v24, v13
	eor.16b v11, v25, v14
	ushr.2d v0, v11, #14
	ushr.2d v6, v10, #14
	ushr.2d v7, v9, #14
	ushr.2d v16, v8, #14
	ushr.2d v17, v3, #14
	ushr.2d v19, v26, #14
	ushr.2d v21, v2, #14
	ushr.2d v15, v1, #14
	eor3.16b v15, v4, v31, v15
	eor3.16b v21, v22, v27, v21
	eor3.16b v19, v5, v28, v19
	eor3.16b v17, v18, v29, v17
	eor3.16b v16, v20, v30, v16
	eor3.16b v7, v23, v12, v7
	eor3.16b v6, v24, v13, v6
	eor3.16b v1, v25, v14, v0
	dup.2d v0, x17
	and.16b v3, v1, v0
	and.16b v2, v6, v0
	and.16b v1, v7, v0
	and.16b v16, v16, v0
	and.16b v7, v17, v0
	and.16b v17, v19, v0
	and.16b v6, v21, v0
	and.16b v0, v15, v0
	shl.2d v19, v0, #14
	add.2d v0, v19, v0
	shl.2d v19, v6, #14
	add.2d v6, v19, v6
	shl.2d v19, v17, #14
	add.2d v15, v19, v17
	shl.2d v17, v7, #14
	add.2d v7, v17, v7
	shl.2d v17, v16, #14
	add.2d v16, v17, v16
	shl.2d v17, v1, #14
	add.2d v17, v17, v1
	shl.2d v1, v2, #14
	add.2d v19, v1, v2
	shl.2d v1, v3, #14
	add.2d v21, v1, v3
	eor3.16b v1, v25, v14, v21
	eor3.16b v2, v24, v13, v19
	eor3.16b v3, v23, v12, v17
	eor3.16b v20, v20, v30, v16
	eor3.16b v18, v18, v29, v7
	eor3.16b v5, v5, v28, v15
	eor3.16b v22, v22, v27, v6
	eor3.16b v4, v4, v31, v0
	ushr.2d v4, v4, #28
	ushr.2d v22, v22, #28
	ushr.2d v5, v5, #28
	ushr.2d v18, v18, #28
	ushr.2d v20, v20, #28
	ushr.2d v3, v3, #28
	ushr.2d v2, v2, #28
	ushr.2d v1, v1, #28
	eor3.16b v1, v21, v11, v1
	eor3.16b v2, v19, v10, v2
	eor3.16b v3, v17, v9, v3
	eor3.16b v20, v16, v8, v20
	ldp q24, q27, [x29, #-208]
	eor3.16b v18, v7, v24, v18
	eor3.16b v5, v15, v26, v5
	dup.2d v23, x5
	ldur q25, [x29, #-176]
	eor3.16b v22, v6, v25, v22
	eor3.16b v4, v0, v27, v4
	and.16b v4, v4, v23
	xtn.2s v4, v4
	and.16b v22, v22, v23
	and.16b v5, v5, v23
	xtn.2s v22, v22
	xtn.2s v5, v5
	and.16b v18, v18, v23
	xtn.2s v18, v18
	and.16b v20, v20, v23
	and.16b v3, v3, v23
	xtn.2s v20, v20
	xtn.2s v3, v3
	and.16b v2, v2, v23
	and.16b v1, v1, v23
	dup.2s v23, w15
	xtn.2s v2, v2
	xtn.2s v1, v1
	umull.2d v1, v1, v23
	umull.2d v2, v2, v23
	umull.2d v3, v3, v23
	umull.2d v20, v20, v23
	umull.2d v18, v18, v23
	umull.2d v5, v5, v23
	umull.2d v22, v22, v23
	umull.2d v4, v4, v23
	eor3.16b v27, v0, v27, v4
	eor3.16b v28, v6, v25, v22
	eor3.16b v29, v15, v26, v5
	eor3.16b v30, v7, v24, v18
	eor3.16b v23, v16, v8, v20
	eor3.16b v24, v17, v9, v3
	eor3.16b v25, v19, v10, v2
	eor3.16b v26, v21, v11, v1
	ushr.2d v3, v26, #8
	ushr.2d v2, v25, #8
	ushr.2d v1, v24, #8
	ushr.2d v0, v23, #8
	ushr.2d v7, v30, #8
	ushr.2d v6, v29, #8
	ushr.2d v5, v28, #8
	ushr.2d v4, v27, #8
	ldr q31, [x24, lCPI0_8@PAGEOFF]
	ushr.2d v19, v26, #16
	ushr.2d v18, v25, #16
	ushr.2d v17, v24, #16
	tbl.16b v8, { v4, v5, v6, v7 }, v31
	ushr.2d v16, v23, #16
	ushr.2d v7, v30, #16
	ushr.2d v6, v29, #16
	ushr.2d v5, v28, #16
	ushr.2d v4, v27, #16
	tbl.16b v10, { v0, v1, v2, v3 }, v31
	tbl.16b v9, { v4, v5, v6, v7 }, v31
	ushr.2d v3, v26, #24
	ushr.2d v2, v25, #24
	ushr.2d v1, v24, #24
	ushr.2d v0, v23, #24
	tbl.16b v11, { v16, v17, v18, v19 }, v31
	ushr.2d v7, v30, #24
	ushr.2d v6, v29, #24
	ushr.2d v5, v28, #24
	ushr.2d v4, v27, #24
	tbl.16b v12, { v4, v5, v6, v7 }, v31
	tbl.16b v13, { v0, v1, v2, v3 }, v31
	ushr.2d v3, v26, #32
	ushr.2d v2, v25, #32
	ushr.2d v1, v24, #32
	ushr.2d v0, v23, #32
	ushr.2d v7, v30, #32
	ushr.2d v6, v29, #32
	ushr.2d v5, v28, #32
	ushr.2d v4, v27, #32
	tbl.16b v14, { v4, v5, v6, v7 }, v31
	ushr.2d v7, v26, #40
	ushr.2d v6, v25, #40
	tbl.16b v15, { v0, v1, v2, v3 }, v31
	ushr.2d v5, v24, #40
	ushr.2d v4, v23, #40
	ushr.2d v3, v30, #40
	ushr.2d v2, v29, #40
	ushr.2d v1, v28, #40
	ushr.2d v0, v27, #40
	tbl.16b v0, { v0, v1, v2, v3 }, v31
	tbl.16b v6, { v4, v5, v6, v7 }, v31
	ushr.2d v4, v30, #48
	ushr.2d v3, v29, #48
	ushr.2d v2, v28, #48
	ushr.2d v1, v27, #48
	tbl.16b v7, { v1, v2, v3, v4 }, v31
	ushr.2d v4, v26, #48
	ushr.2d v3, v25, #48
	ushr.2d v2, v24, #48
	ushr.2d v1, v23, #48
	tbl.16b v1, { v1, v2, v3, v4 }, v31
	ushr.2d v5, v30, #56
	ushr.2d v4, v29, #56
	ushr.2d v3, v28, #56
	ushr.2d v2, v27, #56
	tbl.16b v16, { v27, v28, v29, v30 }, v31
	tbl.16b v2, { v2, v3, v4, v5 }, v31
	ushr.2d v20, v26, #56
	ushr.2d v19, v25, #56
	ushr.2d v18, v24, #56
	ushr.2d v17, v23, #56
	tbl.16b v3, { v23, v24, v25, v26 }, v31
	tbl.16b v4, { v17, v18, v19, v20 }, v31
	mov.d v8[1], v10[0]
	mov.d v16[1], v3[0]
	str q16, [x2, x8]
	str q8, [x22, x8]
	mov.d v9[1], v11[0]
	str q9, [x21, x8]
	mov.d v12[1], v13[0]
	mov.d v14[1], v15[0]
	str q12, [x6, x8]
	str q14, [x26, x8]
	mov.d v0[1], v6[0]
	str q0, [x30, x8]
	mov.d v7[1], v1[0]
	mov.d v2[1], v4[0]
	str q7, [x28, x8]
	str q2, [x25, x8]
	add x8, x8, #16
	add x27, x27, #128
	cmp x9, x8
	b.ne LBB0_5
	ldp q2, q1, [sp, #192]
	ldp q4, q3, [sp, #160]
	ldp q21, q5, [sp, #128]
	ldp q23, q22, [sp, #96]
	ldp q25, q24, [sp, #64]
	ldp q27, q26, [sp, #32]
	ldp q29, q28, [sp]
LBB0_7:
	ldur x8, [x29, #-216]
	add x8, x8, #1
	stur x8, [x29, #-216]
	ldur x8, [x29, #-224]
	add x8, x8, x4, lsl #3
LBB0_8:
	cmp x8, x1
	b.hs LBB0_29
	add x9, x8, #1
	cmp x9, x1
	b.hs LBB0_28
	add x9, x8, #2
	cmp x9, x1
	b.hs LBB0_28
	add x9, x8, #3
	cmp x9, x1
	b.hs LBB0_28
	add x9, x8, #4
	cmp x9, x1
	b.hs LBB0_28
	add x9, x8, #5
	cmp x9, x1
	b.hs LBB0_28
	add x9, x8, #6
	cmp x9, x1
	b.hs LBB0_28
	add x9, x8, #7
	cmp x9, x1
	b.hs LBB0_28
	add x9, x13, x4
	cmp x9, x3
	b.hs LBB0_27
	ldrb w9, [x0, x8]
	add x14, x0, x8
	ldrb w23, [x14, #1]
	ldrb w14, [x14, #2]
	add x24, x0, x8
	ldrb w27, [x24, #3]
	ldrb w24, [x24, #4]
	add x5, x0, x8
	ldrb w5, [x5, #5]
	add x15, x0, x8
	ldrb w17, [x15, #6]
	ldrb w15, [x15, #7]
	lsl x17, x17, #48
	orr x15, x17, x15, lsl #56
	mov x17, #52428
	movk x17, #52428, lsl #32
	orr x15, x15, x5, lsl #40
	mov w5, #-252645136
	orr x15, x15, x24, lsl #32
	orr x15, x15, x27, lsl #24
	orr x14, x15, x14, lsl #16
	orr x14, x14, x23, lsl #8
	orr x9, x14, x9
	eor x14, x9, x14, lsr #7
	mov x15, #-6148914691236517206
	and x15, x15, #0x1fe01fe01fe01fe
	and x14, x14, x15
	orr x14, x14, x14, lsl #7
	eor x9, x14, x9
	eor x14, x9, x9, lsr #14
	and x14, x14, x17
	orr x14, x14, x14, lsl #14
	eor x9, x14, x9
	lsr x14, x9, #28
	eor w14, w14, w9
	and x14, x14, x5
	orr x14, x14, x14, lsl #28
	eor x14, x14, x9
	strb w14, [x2, x4]
	add x9, x19, x4
	cmp x9, x3
	b.hs LBB0_27
	lsr x9, x14, #8
	strb w9, [x22, x4]
	add x9, x16, x4
	cmp x9, x3
	b.hs LBB0_27
	lsr x9, x14, #16
	strb w9, [x21, x4]
	ldur x9, [x29, #-160]
	add x9, x9, x4
	cmp x9, x3
	b.hs LBB0_27
	lsr x9, x14, #24
	strb w9, [x6, x4]
	add x9, x7, x4
	cmp x9, x3
	b.hs LBB0_27
	lsr x9, x14, #32
	strb w9, [x26, x4]
	add x9, x20, x4
	cmp x9, x3
	b.hs LBB0_27
	lsr x9, x14, #40
	strb w9, [x30, x4]
	add x9, x11, x4
	cmp x9, x3
	b.hs LBB0_27
	lsr x9, x14, #48
	strb w9, [x28, x4]
	add x9, x12, x4
	cmp x9, x3
	b.hs LBB0_27
	lsr x9, x14, #56
	strb w9, [x25, x4]
	add x4, x4, #1
	add x8, x8, #8
	cmp x10, x4
	b.ne LBB0_8
	add x19, x19, x10
	ldur x8, [x29, #-240]
	sub x8, x8, x10
	stur x8, [x29, #-240]
	add x11, x11, x10
	ldr x8, [sp, #296]
	sub x8, x8, x10
	str x8, [sp, #296]
	add x12, x12, x10
	ldur x8, [x29, #-256]
	sub x8, x8, x10
	stur x8, [x29, #-256]
	add x13, x13, x10
	ldur x8, [x29, #-232]
	sub x8, x8, x10
	stur x8, [x29, #-232]
	add x25, x25, x10
	add x28, x28, x10
	add x30, x30, x10
	add x26, x26, x10
	add x6, x6, x10
	add x21, x21, x10
	add x22, x22, x10
	add x2, x2, x10
	ldr x9, [sp, #280]
	ldur x8, [x29, #-248]
	add x8, x8, x9
	stur x8, [x29, #-248]
	add x20, x20, x10
	add x7, x7, x10
	ldur x8, [x29, #-160]
	add x8, x8, x10
	stur x8, [x29, #-160]
	add x16, x16, x10
	ldp x8, x5, [x29, #-224]
	add x8, x8, x9
	stur x8, [x29, #-224]
	ldr x8, [sp, #288]
	cmp x5, x8
	b.ne LBB0_3
LBB0_26:
	add sp, sp, #416
	ldp x29, x30, [sp, #144]
	ldp x20, x19, [sp, #128]
	ldp x22, x21, [sp, #112]
	ldp x24, x23, [sp, #96]
	ldp x26, x25, [sp, #80]
	ldp x28, x27, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #160
	ret
LBB0_27:
Lloh16:
	adrp x2, l_anon.86fcf0a1111718708cdca7b99afbb18a.5@PAGE
Lloh17:
	add x2, x2, l_anon.86fcf0a1111718708cdca7b99afbb18a.5@PAGEOFF
	mov x0, x9
	mov x1, x3
	bl core::panicking::panic_bounds_check
LBB0_28:
	mov x8, x9
LBB0_29:
Lloh18:
	adrp x2, l_anon.86fcf0a1111718708cdca7b99afbb18a.6@PAGE
Lloh19:
	add x2, x2, l_anon.86fcf0a1111718708cdca7b99afbb18a.6@PAGEOFF
	mov x0, x8
	bl core::panicking::panic_bounds_check
