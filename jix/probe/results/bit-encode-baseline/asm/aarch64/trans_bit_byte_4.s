jix_probe::bit_shuffle::trans_bit_byte:
	stp d15, d14, [sp, #-160]!
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	stp x28, x27, [sp, #80]
	stp x26, x25, [sp, #96]
	stp x24, x23, [sp, #112]
	stp x22, x21, [sp, #128]
	stp x20, x19, [sp, #144]
	add x29, sp, #64
	sub sp, sp, #784
	str x5, [sp, #288]
	cbz x5, .LBB0_26
	lsr x10, x4, #3
	cbz x10, .LBB0_26
	ldr x9, [sp, #288]
	movi v2.2d, #0xffffffffffffffff
	mov w8, #15
	dup v3.2d, x8
	adrp x8, .LCPI0_0
	adrp x11, .LCPI0_2
	mul x19, x9, x10
	ldr q23, [x8, :lo12:.LCPI0_0]
	adrp x14, .LCPI0_4
	adrp x17, .LCPI0_6
	ldr q25, [x11, :lo12:.LCPI0_2]
	ldr q27, [x14, :lo12:.LCPI0_4]
	mneg x9, x9, x10
	ldr q29, [x17, :lo12:.LCPI0_6]
	adrp x12, .LCPI0_3
	ldr q26, [x12, :lo12:.LCPI0_3]
	dup v21.2d, x1
	dup v22.2d, x3
	adrp x15, .LCPI0_5
	adrp x5, .LCPI0_7
	mov x16, #52428
	lsl x6, x19, #1
	fmov d0, x19
	lsl x28, x19, #2
	lsl x8, x19, #3
	ldr q28, [x15, :lo12:.LCPI0_5]
	ldr q30, [x5, :lo12:.LCPI0_7]
	add x25, x6, x19
	str x9, [sp, #360]
	add x9, x0, #63
	fmov d20, x25
	mov v0.d[1], x6
	stp xzr, x9, [sp, #344]
	adrp x9, .LCPI0_1
	sub x17, x8, x19
	sub x11, x19, x28
	ldr q24, [x9, :lo12:.LCPI0_1]
	neg x9, x6
	sub x27, x6, x8
	mov v1.16b, v20.16b
	sub x14, x19, x8
	fmov d31, x6
	add v0.2d, v0.2d, v2.2d
	fmov d8, x9
	fmov d9, x11
	add x22, x28, x19
	lsl x7, x25, #1
	neg x12, x28
	mov v1.d[1], x28
	neg x9, x22
	sub x11, x22, #1
	cmhi v0.2d, v3.2d, v0.2d
	mov v31.d[1], x28
	mov v20.d[1], x22
	mov v9.d[1], x9
	mov v8.d[1], x12
	stur x6, [x29, #-80]
	add x23, x2, x6
	sub x6, x17, #1
	mov x21, xzr
	add v1.2d, v1.2d, v2.2d
	mov x20, xzr
	mov x24, xzr
	movk x16, #52428, lsl #32
	mov w18, #-252645136
	add x26, x2, x19
	add x30, x2, x28
	mov x13, x28
	add x5, x2, x17
	cmhi v1.2d, v3.2d, v1.2d
	add x28, x2, x22
	add x15, x2, x25
	stp x14, x27, [sp, #248]
	stp x27, x14, [sp, #328]
	mov x14, x17
	uzp1 v0.4s, v0.4s, v1.4s
	stp x17, x19, [sp, #264]
	add x17, x2, x7
	str x7, [sp, #240]
	str x4, [sp, #280]
	stp q9, q8, [sp]
	xtn v0.4h, v0.4s
	stp q31, q30, [sp, #32]
	stp q29, q28, [sp, #64]
	stp q27, q26, [sp, #96]
	stp q25, q24, [sp, #128]
	fmov x8, d0
	stp q23, q22, [sp, #160]
	stp q21, q20, [sp, #192]
	cmp x8, #0
	sub x8, x7, #1
	ccmp x11, #15, #0, eq
	cset w9, lo
	cmp x8, #15
	ccmp x6, #15, #0, hs
	csinc w8, w9, wzr, hs
	str w8, [sp, #236]
.LBB0_3:
	mul x8, x4, x24
	cmp x3, x19
	stp x24, x21, [sp, #312]
	ldr x4, [sp, #240]
	mov x27, x20
	stp x22, x25, [sp, #296]
	dup v0.2d, x8
	mul x8, x10, x24
	csel x24, x3, x19, hi
	cmp x3, x7
	csel x9, x3, x7, hi
	cmp x3, x14
	add v1.2d, v0.2d, v25.2d
	add v2.2d, v0.2d, v26.2d
	add v7.2d, v0.2d, v24.2d
	add v6.2d, v0.2d, v23.2d
	csel x6, x3, x14, hi
	cmp x3, x20
	dup v3.2d, x8
	csel x12, x3, x20, hi
	add x20, x4, x8
	cmhi v4.2d, v21.2d, v2.2d
	cmhi v5.2d, v21.2d, v1.2d
	ldp x4, x11, [sp, #264]
	cmhi v18.2d, v21.2d, v6.2d
	add x11, x11, x8
	add v16.2d, v31.2d, v3.2d
	subs x11, x3, x11
	add x25, x4, x8
	bit v2.16b, v21.16b, v4.16b
	bit v1.16b, v21.16b, v5.16b
	add v4.2d, v20.2d, v3.2d
	cmhi v5.2d, v21.2d, v7.2d
	bit v6.16b, v21.16b, v18.16b
	ldr x4, [sp, #256]
	cmhi v17.2d, v22.2d, v16.2d
	csel x11, xzr, x11, lo
	cmp x3, x20
	cmhi v19.2d, v22.2d, v4.2d
	csel x20, x3, x20, hi
	sub x21, x4, x8
	bsl v5.16b, v21.16b, v7.16b
	sub v1.2d, v1.2d, v0.2d
	sub v2.2d, v2.2d, v0.2d
	mov v7.16b, v17.16b
	cmp x3, x25
	add x20, x20, x21
	bit v4.16b, v22.16b, v19.16b
	csel x21, x3, x25, hi
	ldr x4, [sp, #248]
	add v2.2d, v2.2d, v27.2d
	add v1.2d, v1.2d, v28.2d
	bsl v7.16b, v22.16b, v16.16b
	sub v16.2d, v8.2d, v3.2d
	sub v3.2d, v9.2d, v3.2d
	sub x22, x4, x8
	subs x8, x3, x8
	sub x4, x10, #1
	ushr v1.2d, v1.2d, #3
	ushr v2.2d, v2.2d, #3
	add x21, x21, x22
	add v3.2d, v4.2d, v3.2d
	sub v4.2d, v6.2d, v0.2d
	sub v0.2d, v5.2d, v0.2d
	add v7.2d, v7.2d, v16.2d
	csel x8, xzr, x8, lo
	cmhi v5.2d, v3.2d, v2.2d
	add v0.2d, v0.2d, v29.2d
	add v4.2d, v4.2d, v30.2d
	cmhi v6.2d, v7.2d, v1.2d
	bif v2.16b, v3.16b, v5.16b
	ushr v3.2d, v4.2d, #3
	ushr v0.2d, v0.2d, #3
	bif v1.16b, v7.16b, v6.16b
	cmhi v5.2d, v3.2d, v2.2d
	cmhi v4.2d, v0.2d, v1.2d
	bit v0.16b, v1.16b, v4.16b
	mov v1.16b, v5.16b
	bsl v1.16b, v2.16b, v3.16b
	cmhi v2.2d, v1.2d, v0.2d
	bif v0.16b, v1.16b, v2.16b
	mov d1, v0.d[1]
	cmhi d2, d1, d0
	bif v0.8b, v1.8b, v2.8b
	fmov x25, d0
	cmp x25, x20
	csel x20, x25, x20, lo
	cmp x21, x11
	csel x11, x21, x11, lo
	cmp x8, x4
	csel x8, x8, x4, lo
	cmp x20, x11
	csel x11, x20, x11, lo
	cmp x11, x8
	csel x20, x11, x8, lo
	ldr w11, [sp, #236]
	cmp x20, #16
	cset w8, lo
	orr w8, w8, w11
	mov x11, xzr
	tbnz w8, #0, .LBB0_7
	add x11, x20, #1
	ldr x4, [sp, #360]
	mov w16, #16
	ands x21, x11, #0xf
	mov w18, #1
	mov x8, xzr
	add x20, x24, x4
	csel x21, x16, x21, eq
	ldr x16, [sp, #328]
	cmp x25, x20
	mov x4, #52428
	sub x11, x11, x21
	csel x20, x25, x20, lo
	add x9, x9, x16
	ldr x16, [sp, #336]
	cmp x20, x9
	movk x4, #52428, lsl #32
	movk w18, #4096, lsl #16
	add x6, x6, x16
	csel x9, x20, x9, lo
	ldr x16, [sp, #344]
	cmp x9, x6
	adrp x22, .LCPI0_8
	add x12, x12, x16
	csel x9, x9, x6, lo
	ldr x6, [sp, #352]
	cmp x9, x12
	mov w16, #-252645136
	csel x9, x9, x12, lo
	sub x12, x10, #1
	cmp x9, x12
	csel x9, x9, x12, lo
	sub x9, x9, x21
	add x9, x9, #1
.LBB0_5:
	ldur b8, [x6, #-56]
	sub x12, x6, #48
	add x25, x6, #16
	sub x24, x6, #49
	add x20, x6, #15
	add x21, x6, #14
	ld1 { v8.b }[1], [x12]
	sub x12, x6, #40
	ld1 { v8.b }[2], [x12]
	sub x12, x6, #32
	ld1 { v8.b }[3], [x12]
	sub x12, x6, #24
	ld1 { v8.b }[4], [x12]
	sub x12, x6, #16
	ld1 { v8.b }[5], [x12]
	sub x12, x6, #8
	ld1 { v8.b }[6], [x12]
	mov x12, x6
	ld1 { v8.b }[7], [x12], #1
	ldr b9, [x6, #8]
	ldur b23, [x6, #-57]
	ldur b27, [x6, #-59]
	ldr b24, [x6, #7]
	ld1 { v9.b }[1], [x25]
	ldr b25, [x6, #6]
	ldur b26, [x6, #-58]
	ld1 { v23.b }[1], [x24]
	sub x24, x6, #51
	ldr b29, [x6, #4]
	ld1 { v27.b }[1], [x24]
	add x24, x6, #24
	ldur b31, [x6, #-61]
	ld1 { v9.b }[2], [x24]
	ld1 { v24.b }[1], [x20]
	ld1 { v25.b }[1], [x21]
	sub x25, x6, #50
	add x20, x6, #12
	ldur b30, [x6, #-60]
	sub x21, x6, #53
	ldr b28, [x6, #5]
	ld1 { v26.b }[1], [x25]
	ld1 { v29.b }[1], [x20]
	sub x20, x6, #52
	ld1 { v31.b }[1], [x21]
	add x21, x6, #32
	add x25, x6, #13
	ld1 { v30.b }[1], [x20]
	sub x20, x6, #41
	ld1 { v9.b }[3], [x21]
	add x21, x6, #23
	ld1 { v28.b }[1], [x25]
	ld1 { v23.b }[2], [x20]
	add x20, x6, #22
	ld1 { v24.b }[2], [x21]
	sub x21, x6, #42
	ld1 { v25.b }[2], [x20]
	sub x20, x6, #43
	ld1 { v26.b }[2], [x21]
	add x21, x6, #40
	ld1 { v27.b }[2], [x20]
	add x20, x6, #21
	ld1 { v9.b }[4], [x21]
	ld1 { v28.b }[2], [x20]
	add x20, x6, #20
	sub x21, x6, #44
	ld1 { v29.b }[2], [x20]
	add x20, x6, #48
	ld1 { v30.b }[2], [x21]
	sub x21, x6, #45
	ld1 { v9.b }[5], [x20]
	sub x20, x6, #33
	ld1 { v31.b }[2], [x21]
	add x21, x6, #31
	ld1 { v23.b }[3], [x20]
	add x20, x6, #30
	ld1 { v24.b }[3], [x21]
	add x21, x6, #56
	ld1 { v25.b }[3], [x20]
	sub x20, x6, #34
	ld1 { v9.b }[6], [x21]
	sub x21, x6, #35
	ld1 { v26.b }[3], [x20]
	add x20, x6, #29
	ld1 { v27.b }[3], [x21]
	add x21, x6, #64
	ld1 { v28.b }[3], [x20]
	add x20, x6, #28
	ld1 { v9.b }[7], [x21]
	sub x21, x6, #25
	ld1 { v29.b }[3], [x20]
	add x20, x6, #39
	ld1 { v23.b }[4], [x21]
	add x21, x6, #38
	ld1 { v24.b }[4], [x20]
	sub x20, x6, #26
	ld1 { v25.b }[4], [x21]
	sub x21, x6, #27
	ld1 { v26.b }[4], [x20]
	sub x20, x6, #17
	ld1 { v27.b }[4], [x21]
	add x21, x6, #47
	ld1 { v23.b }[5], [x20]
	add x20, x6, #37
	ld1 { v24.b }[5], [x21]
	add x21, x6, #46
	ld1 { v28.b }[4], [x20]
	sub x20, x6, #18
	ld1 { v25.b }[5], [x21]
	sub x21, x6, #9
	ld1 { v26.b }[5], [x20]
	add x20, x6, #55
	ld1 { v23.b }[6], [x21]
	add x21, x6, #54
	ld1 { v24.b }[6], [x20]
	sub x20, x6, #19
	ushll v10.8h, v8.8b, #0
	ld1 { v25.b }[6], [x21]
	sub x21, x6, #10
	ld1 { v27.b }[5], [x20]
	sub x20, x6, #1
	ld1 { v26.b }[6], [x21]
	add x21, x6, #62
	ld1 { v23.b }[7], [x20]
	add x20, x6, #63
	ushll v9.8h, v9.8b, #0
	ld1 { v24.b }[7], [x20]
	sub x20, x6, #2
	ld1 { v25.b }[7], [x21]
	sub x21, x6, #36
	ld1 { v26.b }[7], [x20]
	add x20, x6, #45
	ld1 { v28.b }[5], [x20]
	add x20, x6, #36
	ld1 { v30.b }[3], [x21]
	sub x21, x6, #37
	ld1 { v29.b }[4], [x20]
	sub x20, x6, #11
	ld1 { v27.b }[6], [x20]
	sub x20, x6, #28
	ld1 { v31.b }[3], [x21]
	add x21, x6, #44
	ld1 { v30.b }[4], [x20]
	add x20, x6, #53
	ld1 { v28.b }[6], [x20]
	sub x20, x6, #29
	ld1 { v29.b }[5], [x21]
	sub x21, x6, #20
	ld1 { v31.b }[4], [x20]
	sub x20, x6, #3
	ld1 { v27.b }[7], [x20]
	add x20, x6, #52
	ld1 { v30.b }[5], [x21]
	ushll v5.8h, v23.8b, #0
	sub x21, x6, #21
	ld1 { v29.b }[6], [x20]
	add x20, x6, #61
	ld1 { v31.b }[5], [x21]
	ushll v14.4s, v10.4h, #0
	ld1 { v28.b }[7], [x20]
	sub x20, x6, #12
	ushll v6.8h, v24.8b, #0
	ld1 { v30.b }[6], [x20]
	ushll v18.4s, v5.4h, #0
	sub x21, x6, #13
	add x20, x6, #60
	ldr b8, [x6, #3]
	add x24, x6, #11
	ld1 { v29.b }[7], [x20]
	ld1 { v31.b }[6], [x21]
	sub x20, x6, #4
	ldr b11, [x6, #2]
	ld1 { v30.b }[7], [x20]
	ld1 { v8.b }[1], [x24]
	add x20, x6, #10
	ushll v0.4s, v9.4h, #0
	ushll v1.2d, v14.2s, #0
	ushll2 v2.4s, v10.8h, #0
	ushll2 v7.2d, v14.4s, #0
	ushll2 v19.4s, v6.8h, #0
	ushll2 v5.4s, v5.8h, #0
	ushll v6.4s, v6.4h, #0
	ushll v20.2d, v18.2s, #0
	ushll2 v18.2d, v18.4s, #0
	ld1 { v11.b }[1], [x20]
	sub x20, x6, #5
	ld1 { v31.b }[7], [x20]
	add x20, x6, #19
	ushll2 v4.4s, v9.8h, #0
	ld1 { v8.b }[2], [x20]
	add x20, x6, #18
	ldur b12, [x6, #-62]
	ushll v16.2d, v0.2s, #0
	ushll2 v0.2d, v0.4s, #0
	ushll v17.2d, v2.2s, #0
	shl v7.2d, v7.2d, #56
	ushll v24.2d, v5.2s, #0
	shl v1.2d, v1.2d, #56
	ushll v9.2d, v6.2s, #0
	ushll2 v6.2d, v6.4s, #0
	shl v20.2d, v20.2d, #48
	shl v18.2d, v18.2d, #48
	ushll v26.8h, v26.8b, #0
	ld1 { v11.b }[2], [x20]
	sub x20, x6, #54
	ldr b13, [x12]
	add x12, x6, #9
	ld1 { v12.b }[1], [x20]
	ldur b15, [x6, #-63]
	sub x20, x6, #55
	ld1 { v13.b }[1], [x12]
	add x12, x6, #27
	ushll2 v3.2d, v4.4s, #0
	ushll2 v2.2d, v2.4s, #0
	shl v0.2d, v0.2d, #56
	shl v17.2d, v17.2d, #56
	ushll2 v23.2d, v19.4s, #0
	ushll2 v5.2d, v5.4s, #0
	shl v24.2d, v24.2d, #48
	shl v6.2d, v6.2d, #48
	orr v1.16b, v1.16b, v20.16b
	orr v7.16b, v7.16b, v18.16b
	ushll v18.4s, v26.4h, #0
	ushll v20.8h, v27.8b, #0
	ld1 { v8.b }[3], [x12]
	sub x12, x6, #46
	ld1 { v15.b }[1], [x20]
	add x20, x6, #17
	ld1 { v12.b }[2], [x12]
	add x12, x6, #26
	ld1 { v13.b }[2], [x20]
	ld1 { v11.b }[3], [x12]
	sub x12, x6, #47
	ushll v4.2d, v4.2s, #0
	shl v3.2d, v3.2d, #56
	shl v16.2d, v16.2d, #56
	shl v2.2d, v2.2d, #56
	ushll v19.2d, v19.2s, #0
	shl v23.2d, v23.2d, #48
	shl v5.2d, v5.2d, #48
	shl v9.2d, v9.2d, #48
	ushll v25.8h, v25.8b, #0
	orr v17.16b, v17.16b, v24.16b
	orr v6.16b, v0.16b, v6.16b
	ushll2 v0.4s, v26.8h, #0
	ushll2 v24.2d, v18.4s, #0
	ushll v18.2d, v18.2s, #0
	ushll v26.4s, v20.4h, #0
	sub x20, x6, #38
	ld1 { v15.b }[2], [x12]
	add x12, x6, #35
	ld1 { v12.b }[3], [x20]
	ld1 { v8.b }[4], [x12]
	add x12, x6, #25
	sub x20, x6, #39
	ld1 { v13.b }[3], [x12]
	add x12, x6, #34
	shl v4.2d, v4.2d, #56
	shl v19.2d, v19.2d, #48
	orr v2.16b, v2.16b, v5.16b
	orr v5.16b, v16.16b, v9.16b
	ushll2 v16.4s, v25.8h, #0
	orr v23.16b, v3.16b, v23.16b
	ushll v3.4s, v25.4h, #0
	ushll v25.2d, v0.2s, #0
	ushll2 v0.2d, v0.4s, #0
	ushll v28.8h, v28.8b, #0
	shl v18.2d, v18.2d, #40
	ushll2 v20.4s, v20.8h, #0
	shll v9.2d, v26.2s, #32
	ld1 { v11.b }[4], [x12]
	sub x12, x6, #30
	ld1 { v15.b }[3], [x20]
	add x20, x6, #33
	ld1 { v12.b }[4], [x12]
	add x12, x6, #43
	ld1 { v8.b }[5], [x12]
	sub x12, x6, #31
	ld1 { v13.b }[4], [x20]
	orr v19.16b, v4.16b, v19.16b
	ushll2 v4.2d, v16.4s, #0
	ushll v27.2d, v3.2s, #0
	ushll2 v3.2d, v3.4s, #0
	shl v24.2d, v24.2d, #40
	shl v25.2d, v25.2d, #40
	shl v10.2d, v0.2d, #40
	ushll v0.4s, v28.4h, #0
	ushll2 v28.4s, v28.8h, #0
	shll2 v26.2d, v26.4s, #32
	shll v14.2d, v20.2s, #32
	orr v18.16b, v18.16b, v9.16b
	ld1 { v15.b }[4], [x12]
	add x12, x6, #42
	sub x20, x6, #22
	ld1 { v11.b }[5], [x12]
	add x12, x6, #41
	ushll v16.2d, v16.2s, #0
	ld1 { v12.b }[5], [x20]
	sub x20, x6, #23
	ld1 { v13.b }[5], [x12]
	add x12, x6, #51
	shl v27.2d, v27.2d, #40
	shl v3.2d, v3.2d, #40
	shl v4.2d, v4.2d, #40
	shll v9.2d, v0.2s, #32
	shll2 v21.2d, v0.4s, #32
	orr v24.16b, v24.16b, v26.16b
	shll2 v26.2d, v28.4s, #32
	orr v0.16b, v1.16b, v18.16b
	orr v18.16b, v25.16b, v14.16b
	ld1 { v8.b }[6], [x12]
	add x12, x6, #50
	ld1 { v15.b }[5], [x20]
	ld1 { v11.b }[6], [x12]
	sub x12, x6, #14
	shl v16.2d, v16.2d, #40
	shll v22.2d, v28.2s, #32
	ld1 { v12.b }[6], [x12]
	sub x12, x6, #15
	shll2 v20.2d, v20.4s, #32
	orr v25.16b, v27.16b, v9.16b
	orr v1.16b, v7.16b, v24.16b
	orr v7.16b, v3.16b, v21.16b
	orr v3.16b, v17.16b, v18.16b
	orr v17.16b, v4.16b, v26.16b
	ld1 { v15.b }[6], [x12]
	add x12, x6, #59
	ld1 { v8.b }[7], [x12]
	sub x12, x6, #6
	orr v16.16b, v16.16b, v22.16b
	add x20, x6, #49
	ld1 { v12.b }[7], [x12]
	orr v20.16b, v10.16b, v20.16b
	ushll v27.8h, v30.8b, #0
	orr v4.16b, v5.16b, v25.16b
	orr v5.16b, v6.16b, v7.16b
	orr v6.16b, v23.16b, v17.16b
	ushll v23.8h, v31.8b, #0
	ld1 { v13.b }[6], [x20]
	add x20, x6, #58
	ushll v18.8h, v29.8b, #0
	orr v16.16b, v19.16b, v16.16b
	ld1 { v11.b }[7], [x20]
	ushll v19.8h, v8.8b, #0
	orr v2.16b, v2.16b, v20.16b
	ushll v20.4s, v27.4h, #0
	ushll2 v7.4s, v27.8h, #0
	ushll v27.8h, v12.8b, #0
	ushll v28.4s, v23.4h, #0
	sub x20, x6, #7
	ushll v17.4s, v18.4h, #0
	ushll2 v25.4s, v19.8h, #0
	ushll v19.4s, v19.4h, #0
	ushll v30.8h, v11.8b, #0
	ld1 { v15.b }[7], [x20]
	ushll v21.2d, v20.2s, #24
	ushll2 v23.4s, v23.8h, #0
	ushll v8.4s, v27.4h, #0
	ushll2 v10.2d, v28.4s, #16
	ushll v28.2d, v28.2s, #16
	add x12, x6, #57
	ushll2 v18.4s, v18.8h, #0
	ushll2 v20.2d, v20.4s, #24
	ld1 { v13.b }[7], [x12]
	ushll v24.2d, v17.2s, #24
	ushll2 v31.2d, v19.4s, #16
	ushll v19.2d, v19.2s, #16
	ushll v11.4s, v30.4h, #0
	ushll v22.2d, v7.2s, #24
	ushll2 v7.2d, v7.4s, #24
	ushll2 v17.2d, v17.4s, #24
	ushll2 v9.2d, v23.4s, #16
	ushll2 v27.4s, v27.8h, #0
	ushll v12.2d, v8.2s, #8
	orr v21.16b, v21.16b, v28.16b
	ushll v15.8h, v15.8b, #0
	ushll v26.2d, v18.2s, #24
	ushll2 v18.2d, v18.4s, #24
	ushll2 v29.2d, v25.4s, #16
	ushll v25.2d, v25.2s, #16
	ushll2 v30.4s, v30.8h, #0
	ushll v23.2d, v23.2s, #16
	ushll2 v8.2d, v8.4s, #8
	ushll v13.8h, v13.8b, #0
	ushll v28.2d, v11.2s, #8
	orr v20.16b, v20.16b, v10.16b
	orr v19.16b, v24.16b, v19.16b
	ushll v14.2d, v27.2s, #8
	ushll2 v27.2d, v27.4s, #8
	ushll2 v11.2d, v11.4s, #8
	orr v7.16b, v7.16b, v9.16b
	orr v17.16b, v17.16b, v31.16b
	ushll v31.4s, v15.4h, #0
	orr v21.16b, v21.16b, v12.16b
	ushll v10.2d, v30.2s, #8
	ushll2 v30.2d, v30.4s, #8
	orr v25.16b, v26.16b, v25.16b
	orr v18.16b, v18.16b, v29.16b
	orr v22.16b, v22.16b, v23.16b
	ushll v24.4s, v13.4h, #0
	orr v20.16b, v20.16b, v8.16b
	orr v19.16b, v19.16b, v28.16b
	ushll v12.2d, v31.2s, #0
	orr v17.16b, v17.16b, v11.16b
	orr v7.16b, v7.16b, v27.16b
	orr v21.16b, v0.16b, v21.16b
	ushll2 v23.4s, v13.8h, #0
	orr v18.16b, v18.16b, v30.16b
	orr v25.16b, v25.16b, v10.16b
	ushll2 v26.4s, v15.8h, #0
	ushll2 v9.2d, v24.4s, #0
	ushll v24.2d, v24.2s, #0
	orr v22.16b, v22.16b, v14.16b
	ushll2 v0.2d, v31.4s, #0
	orr v20.16b, v1.16b, v20.16b
	orr v19.16b, v4.16b, v19.16b
	orr v17.16b, v5.16b, v17.16b
	orr v27.16b, v2.16b, v7.16b
	orr v2.16b, v21.16b, v12.16b
	ushr v21.2d, v21.2d, #7
	ushll2 v29.2d, v23.4s, #0
	ushll v23.2d, v23.2s, #0
	orr v6.16b, v6.16b, v18.16b
	orr v18.16b, v16.16b, v25.16b
	ushll2 v13.2d, v26.4s, #0
	ushll v26.2d, v26.2s, #0
	orr v22.16b, v3.16b, v22.16b
	orr v1.16b, v20.16b, v0.16b
	orr v4.16b, v19.16b, v24.16b
	ushr v20.2d, v20.2d, #7
	ushr v19.2d, v19.2d, #7
	movi v25.8h, #170
	orr v16.16b, v17.16b, v9.16b
	ushr v17.2d, v17.2d, #7
	eor v21.16b, v21.16b, v2.16b
	orr v5.16b, v18.16b, v23.16b
	ushr v18.2d, v18.2d, #7
	orr v0.16b, v22.16b, v26.16b
	orr v3.16b, v27.16b, v13.16b
	orr v7.16b, v6.16b, v29.16b
	ushr v22.2d, v22.2d, #7
	ushr v23.2d, v27.2d, #7
	ushr v6.2d, v6.2d, #7
	eor v19.16b, v19.16b, v4.16b
	eor v20.16b, v20.16b, v1.16b
	eor v17.16b, v17.16b, v16.16b
	and v21.16b, v21.16b, v25.16b
	eor v18.16b, v18.16b, v5.16b
	eor v22.16b, v22.16b, v0.16b
	sub x12, x29, #160
	eor v23.16b, v23.16b, v3.16b
	eor v6.16b, v6.16b, v7.16b
	and v19.16b, v19.16b, v25.16b
	and v20.16b, v20.16b, v25.16b
	and v17.16b, v17.16b, v25.16b
	shl v24.2d, v21.2d, #7
	and v18.16b, v18.16b, v25.16b
	and v22.16b, v22.16b, v25.16b
	add x6, x6, #128
	and v23.16b, v23.16b, v25.16b
	and v6.16b, v6.16b, v25.16b
	shl v25.2d, v19.2d, #7
	shl v26.2d, v20.2d, #7
	shl v27.2d, v17.2d, #7
	add v21.2d, v24.2d, v21.2d
	shl v24.2d, v18.2d, #7
	shl v29.2d, v22.2d, #7
	shl v28.2d, v23.2d, #7
	shl v30.2d, v6.2d, #7
	add v19.2d, v25.2d, v19.2d
	add v20.2d, v26.2d, v20.2d
	add v17.2d, v27.2d, v17.2d
	eor v2.16b, v21.16b, v2.16b
	add v18.2d, v24.2d, v18.2d
	add v22.2d, v29.2d, v22.2d
	add v21.2d, v28.2d, v23.2d
	add v23.2d, v30.2d, v6.2d
	eor v6.16b, v19.16b, v4.16b
	eor v1.16b, v20.16b, v1.16b
	eor v4.16b, v17.16b, v16.16b
	ushr v19.2d, v2.2d, #14
	eor v18.16b, v18.16b, v5.16b
	eor v16.16b, v22.16b, v0.16b
	dup v20.2d, x4
	eor v17.16b, v21.16b, v3.16b
	eor v7.16b, v23.16b, v7.16b
	ushr v0.2d, v6.2d, #14
	ushr v3.2d, v1.2d, #14
	ushr v5.2d, v4.2d, #14
	eor v19.16b, v19.16b, v2.16b
	ushr v21.2d, v18.2d, #14
	ushr v23.2d, v16.2d, #14
	ushr v22.2d, v17.2d, #14
	ushr v24.2d, v7.2d, #14
	eor v0.16b, v0.16b, v6.16b
	eor v3.16b, v3.16b, v1.16b
	eor v5.16b, v5.16b, v4.16b
	and v19.16b, v19.16b, v20.16b
	eor v21.16b, v21.16b, v18.16b
	eor v23.16b, v23.16b, v16.16b
	eor v22.16b, v22.16b, v17.16b
	eor v24.16b, v24.16b, v7.16b
	and v0.16b, v0.16b, v20.16b
	and v3.16b, v3.16b, v20.16b
	and v5.16b, v5.16b, v20.16b
	shl v25.2d, v19.2d, #14
	and v21.16b, v21.16b, v20.16b
	and v23.16b, v23.16b, v20.16b
	and v22.16b, v22.16b, v20.16b
	and v20.16b, v24.16b, v20.16b
	shl v26.2d, v0.2d, #14
	shl v24.2d, v3.2d, #14
	add v19.2d, v25.2d, v19.2d
	shl v25.2d, v5.2d, #14
	shl v29.2d, v21.2d, #14
	shl v27.2d, v23.2d, #14
	shl v28.2d, v22.2d, #14
	add v24.2d, v24.2d, v3.2d
	add v3.2d, v26.2d, v0.2d
	shl v26.2d, v20.2d, #14
	add v25.2d, v25.2d, v5.2d
	eor v0.16b, v19.16b, v2.16b
	add v21.2d, v29.2d, v21.2d
	add v19.2d, v27.2d, v23.2d
	add v22.2d, v28.2d, v22.2d
	ldr q29, [x22, :lo12:.LCPI0_8]
	add v20.2d, v26.2d, v20.2d
	eor v3.16b, v3.16b, v6.16b
	eor v5.16b, v24.16b, v1.16b
	eor v2.16b, v25.16b, v4.16b
	ushr v23.2d, v0.2d, #28
	eor v1.16b, v21.16b, v18.16b
	eor v4.16b, v22.16b, v17.16b
	eor v16.16b, v19.16b, v16.16b
	eor v6.16b, v20.16b, v7.16b
	ushr v17.2d, v3.2d, #28
	ushr v7.2d, v5.2d, #28
	ushr v18.2d, v2.2d, #28
	eor v19.16b, v23.16b, v0.16b
	ushr v23.2d, v1.2d, #28
	dup v20.2d, x16
	ushr v21.2d, v16.2d, #28
	ushr v22.2d, v4.2d, #28
	ushr v24.2d, v6.2d, #28
	eor v17.16b, v17.16b, v3.16b
	eor v7.16b, v7.16b, v5.16b
	eor v18.16b, v18.16b, v2.16b
	eor v23.16b, v23.16b, v1.16b
	eor v22.16b, v22.16b, v4.16b
	eor v21.16b, v21.16b, v16.16b
	and v19.16b, v19.16b, v20.16b
	eor v24.16b, v24.16b, v6.16b
	and v17.16b, v17.16b, v20.16b
	and v7.16b, v7.16b, v20.16b
	and v18.16b, v18.16b, v20.16b
	and v23.16b, v23.16b, v20.16b
	and v21.16b, v21.16b, v20.16b
	and v22.16b, v22.16b, v20.16b
	xtn v19.2s, v19.2d
	xtn v17.2s, v17.2d
	and v20.16b, v24.16b, v20.16b
	dup v24.2s, w18
	xtn v18.2s, v18.2d
	xtn v23.2s, v23.2d
	xtn v7.2s, v7.2d
	xtn v21.2s, v21.2d
	xtn v22.2s, v22.2d
	xtn v20.2s, v20.2d
	umull v17.2d, v17.2s, v24.2s
	umull v19.2d, v19.2s, v24.2s
	umull v18.2d, v18.2s, v24.2s
	umull v23.2d, v23.2s, v24.2s
	umull v7.2d, v7.2s, v24.2s
	umull v21.2d, v21.2s, v24.2s
	umull v22.2d, v22.2s, v24.2s
	umull v20.2d, v20.2s, v24.2s
	eor v8.16b, v17.16b, v3.16b
	eor v24.16b, v19.16b, v0.16b
	eor v9.16b, v18.16b, v2.16b
	eor v25.16b, v7.16b, v5.16b
	eor v10.16b, v23.16b, v1.16b
	eor v26.16b, v21.16b, v16.16b
	eor v11.16b, v20.16b, v6.16b
	eor v27.16b, v22.16b, v4.16b
	ushr v3.2d, v11.2d, #8
	ushr v7.2d, v27.2d, #8
	ushr v19.2d, v11.2d, #32
	ushr v2.2d, v10.2d, #8
	ushr v6.2d, v26.2d, #8
	ushr v18.2d, v10.2d, #32
	ushr v1.2d, v9.2d, #8
	ushr v5.2d, v25.2d, #8
	ushr v17.2d, v9.2d, #32
	ushr v0.2d, v8.2d, #8
	ushr v4.2d, v24.2d, #8
	ushr v16.2d, v8.2d, #32
	ushr v15.2d, v27.2d, #16
	tbl v30.16b, { v8.16b, v9.16b, v10.16b, v11.16b }, v29.16b
	ushr v23.2d, v27.2d, #32
	ushr v14.2d, v26.2d, #16
	ushr v22.2d, v26.2d, #32
	tbl v31.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v29.16b
	tbl v0.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v29.16b
	tbl v28.16b, { v4.16b, v5.16b, v6.16b, v7.16b }, v29.16b
	ushr v7.2d, v11.2d, #24
	ushr v6.2d, v10.2d, #24
	ushr v13.2d, v25.2d, #16
	ushr v21.2d, v25.2d, #32
	ushr v5.2d, v9.2d, #24
	ushr v12.2d, v24.2d, #16
	ushr v20.2d, v24.2d, #32
	ushr v4.2d, v8.2d, #24
	mov v31.d[1], v30.d[0]
	stur q0, [x29, #-96]
	ushr v3.2d, v11.2d, #16
	ushr v2.2d, v10.2d, #16
	tbl v20.16b, { v20.16b, v21.16b, v22.16b, v23.16b }, v29.16b
	ushr v1.2d, v9.2d, #16
	str q31, [x2, x8]
	ushr v0.2d, v8.2d, #16
	st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x12]
	add x12, sp, #560
	ushr v3.2d, v11.2d, #40
	ushr v2.2d, v10.2d, #40
	ushr v1.2d, v9.2d, #40
	ushr v0.2d, v8.2d, #40
	st1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x12]
	sub x12, x29, #224
	ushr v7.2d, v11.2d, #48
	ushr v6.2d, v10.2d, #48
	ushr v5.2d, v9.2d, #48
	ushr v4.2d, v8.2d, #48
	st1 { v16.2d, v17.2d, v18.2d, v19.2d }, [x12]
	ushr v19.2d, v11.2d, #56
	add x12, sp, #432
	ushr v18.2d, v10.2d, #56
	ushr v17.2d, v9.2d, #56
	ushr v16.2d, v8.2d, #56
	ushr v11.2d, v27.2d, #24
	st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x12]
	add x12, sp, #368
	ushr v10.2d, v26.2d, #24
	ushr v9.2d, v25.2d, #24
	ushr v3.2d, v27.2d, #40
	ushr v8.2d, v24.2d, #24
	ushr v2.2d, v26.2d, #40
	ushr v1.2d, v25.2d, #40
	st1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x12]
	add x12, sp, #496
	ushr v7.2d, v27.2d, #48
	ushr v6.2d, v26.2d, #48
	ushr v0.2d, v24.2d, #40
	ushr v5.2d, v25.2d, #48
	ushr v4.2d, v24.2d, #48
	st1 { v16.2d, v17.2d, v18.2d, v19.2d }, [x12]
	ushr v19.2d, v27.2d, #56
	sub x12, x29, #160
	ushr v18.2d, v26.2d, #56
	tbl v0.16b, { v0.16b, v1.16b, v2.16b, v3.16b }, v29.16b
	ldur q3, [x29, #-96]
	ushr v17.2d, v25.2d, #56
	tbl v2.16b, { v4.16b, v5.16b, v6.16b, v7.16b }, v29.16b
	ushr v16.2d, v24.2d, #56
	tbl v24.16b, { v12.16b, v13.16b, v14.16b, v15.16b }, v29.16b
	ld1 { v12.2d, v13.2d, v14.2d, v15.2d }, [x12]
	add x12, sp, #560
	tbl v26.16b, { v8.16b, v9.16b, v10.16b, v11.16b }, v29.16b
	mov v28.d[1], v3.d[0]
	ld1 { v8.2d, v9.2d, v10.2d, v11.2d }, [x12]
	sub x12, x29, #224
	tbl v25.16b, { v12.16b, v13.16b, v14.16b, v15.16b }, v29.16b
	str q28, [x26, x8]
	tbl v27.16b, { v8.16b, v9.16b, v10.16b, v11.16b }, v29.16b
	ld1 { v8.2d, v9.2d, v10.2d, v11.2d }, [x12]
	add x12, sp, #432
	mov v24.d[1], v25.d[0]
	tbl v21.16b, { v8.16b, v9.16b, v10.16b, v11.16b }, v29.16b
	ld1 { v8.2d, v9.2d, v10.2d, v11.2d }, [x12]
	add x12, sp, #368
	ld1 { v3.2d, v4.2d, v5.2d, v6.2d }, [x12]
	add x12, sp, #496
	mov v26.d[1], v27.d[0]
	str q24, [x23, x8]
	tbl v1.16b, { v8.16b, v9.16b, v10.16b, v11.16b }, v29.16b
	mov v20.d[1], v21.d[0]
	str q26, [x15, x8]
	tbl v3.16b, { v3.16b, v4.16b, v5.16b, v6.16b }, v29.16b
	tbl v4.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v29.16b
	ld1 { v16.2d, v17.2d, v18.2d, v19.2d }, [x12]
	str q20, [x30, x8]
	mov v0.d[1], v1.d[0]
	tbl v5.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v29.16b
	mov v2.d[1], v3.d[0]
	str q0, [x28, x8]
	str q2, [x17, x8]
	mov v4.d[1], v5.d[0]
	str q4, [x5, x8]
	add x8, x8, #16
	cmp x9, x8
	b.ne .LBB0_5
	ldp q21, q20, [sp, #192]
	mov x16, #52428
	ldp q23, q22, [sp, #160]
	mov w18, #-252645136
	ldp q25, q24, [sp, #128]
	movk x16, #52428, lsl #32
	ldp q27, q26, [sp, #96]
	ldp q29, q28, [sp, #64]
	ldp q31, q30, [sp, #32]
	ldp q9, q8, [sp]
.LBB0_7:
	ldp x24, x21, [sp, #312]
	ldr x4, [sp, #280]
	ldp x22, x25, [sp, #296]
	add x8, x21, x11, lsl #3
	add x24, x24, #1
.LBB0_8:
	cmp x8, x1
	b.hs .LBB0_29
	add x9, x8, #1
	cmp x9, x1
	b.hs .LBB0_28
	add x9, x8, #2
	cmp x9, x1
	b.hs .LBB0_28
	add x9, x8, #3
	cmp x9, x1
	b.hs .LBB0_28
	add x9, x8, #4
	cmp x9, x1
	b.hs .LBB0_28
	add x9, x8, #5
	cmp x9, x1
	b.hs .LBB0_28
	add x9, x8, #6
	cmp x9, x1
	b.hs .LBB0_28
	add x9, x8, #7
	cmp x9, x1
	b.hs .LBB0_28
	add x9, x27, x11
	cmp x9, x3
	b.hs .LBB0_27
	add x9, x0, x8
	add x6, x0, x8
	ldrb w12, [x9, #6]
	ldrb w9, [x9, #7]
	ldrb w6, [x6, #5]
	lsl x12, x12, #48
	orr x9, x12, x9, lsl #56
	add x12, x0, x8
	ldrb w20, [x12, #4]
	ldrb w12, [x12, #3]
	orr x9, x9, x6, lsl #40
	add x6, x0, x8
	orr x9, x9, x20, lsl #32
	ldrb w20, [x6, #2]
	orr x9, x9, x12, lsl #24
	ldrb w12, [x6, #1]
	ldrb w6, [x0, x8]
	orr x9, x9, x20, lsl #16
	orr x9, x9, x12, lsl #8
	orr x12, x9, x6
	mov x6, #-6148914691236517206
	eor x9, x12, x9, lsr #7
	and x6, x6, #0x1fe01fe01fe01fe
	and x9, x9, x6
	orr x9, x9, x9, lsl #7
	eor x9, x9, x12
	eor x12, x9, x9, lsr #14
	and x12, x12, x16
	orr x12, x12, x12, lsl #14
	eor x12, x12, x9
	lsr x9, x12, #28
	eor w9, w9, w12
	and x9, x9, x18
	orr x6, x9, x9, lsl #28
	add x9, x19, x11
	cmp x9, x3
	eor x12, x6, x12
	strb w12, [x2, x11]
	b.hs .LBB0_27
	ldur x9, [x29, #-80]
	lsr x6, x12, #8
	strb w6, [x26, x11]
	add x9, x9, x11
	cmp x9, x3
	b.hs .LBB0_27
	add x9, x25, x11
	lsr x6, x12, #16
	cmp x9, x3
	strb w6, [x23, x11]
	b.hs .LBB0_27
	add x9, x13, x11
	lsr x6, x12, #24
	cmp x9, x3
	strb w6, [x15, x11]
	b.hs .LBB0_27
	add x9, x22, x11
	lsr x6, x12, #32
	cmp x9, x3
	strb w6, [x30, x11]
	b.hs .LBB0_27
	add x9, x7, x11
	lsr x6, x12, #40
	cmp x9, x3
	strb w6, [x28, x11]
	b.hs .LBB0_27
	add x9, x14, x11
	lsr x6, x12, #48
	cmp x9, x3
	strb w6, [x17, x11]
	b.hs .LBB0_27
	add x9, x11, #1
	lsr x12, x12, #56
	add x8, x8, #8
	cmp x10, x9
	strb w12, [x5, x11]
	mov x11, x9
	b.ne .LBB0_8
	ldr x8, [sp, #360]
	add x19, x19, x10
	add x7, x7, x10
	add x14, x14, x10
	add x20, x27, x10
	add x5, x5, x10
	sub x8, x8, x10
	add x17, x17, x10
	add x28, x28, x10
	str x8, [sp, #360]
	ldr x8, [sp, #328]
	add x30, x30, x10
	add x15, x15, x10
	add x23, x23, x10
	add x26, x26, x10
	sub x8, x8, x10
	add x2, x2, x10
	add x22, x22, x10
	str x8, [sp, #328]
	ldr x8, [sp, #336]
	add x13, x13, x10
	add x25, x25, x10
	add x21, x21, x4
	sub x8, x8, x10
	str x8, [sp, #336]
	ldr x8, [sp, #344]
	sub x8, x8, x10
	str x8, [sp, #344]
	ldr x8, [sp, #352]
	add x8, x8, x4
	str x8, [sp, #352]
	ldr x8, [sp, #288]
	cmp x24, x8
	ldur x8, [x29, #-80]
	add x8, x8, x10
	stur x8, [x29, #-80]
	b.ne .LBB0_3
.LBB0_26:
	add sp, sp, #784
	ldp x20, x19, [sp, #144]
	ldp x22, x21, [sp, #128]
	ldp x24, x23, [sp, #112]
	ldp x26, x25, [sp, #96]
	ldp x28, x27, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #160
	ret
.LBB0_27:
	adrp x2, .Lanon.5e183fe0a34aeb9c6619c58bb96c331f.5
	add x2, x2, :lo12:.Lanon.5e183fe0a34aeb9c6619c58bb96c331f.5
	mov x0, x9
	mov x1, x3
	bl core::panicking::panic_bounds_check
.LBB0_28:
	mov x8, x9
.LBB0_29:
	adrp x2, .Lanon.5e183fe0a34aeb9c6619c58bb96c331f.6
	add x2, x2, :lo12:.Lanon.5e183fe0a34aeb9c6619c58bb96c331f.6
	mov x0, x8
	bl core::panicking::panic_bounds_check
