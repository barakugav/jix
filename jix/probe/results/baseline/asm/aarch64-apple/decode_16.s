jix_probe::byte_shuffle::decode_impl::<16, 8>:
Lfunc_begin0:
	stp x22, x21, [sp, #-32]!
	stp x20, x19, [sp, #16]
	lsr x19, x1, #4
	mov x5, #0
	ands x8, x19, #0x7fffffffffffff8
	b.eq LBB0_3
	lsl x12, x19, #1
	lsl x14, x19, #2
	mov w9, #6
	lsl x21, x19, #3
	mov w20, #10
	mov w6, #11
	mov w4, #12
	mov w3, #13
	madd x9, x19, x9, x0
	mov w17, #14
	add x10, x0, x12
	add x11, x0, x19
	add x12, x11, x12
	add x13, x0, x14
	add x14, x11, x14
	sub x15, x21, x19
	add x15, x0, x15
	add x16, x2, #64
	sub x7, x0, x19
	madd x17, x19, x17, x0
	madd x3, x19, x3, x0
	madd x4, x19, x4, x0
	madd x6, x19, x6, x0
	add x7, x7, x19, lsl #4
	madd x19, x19, x20, x0
Lloh0:
	adrp x20, lCPI0_0@PAGE
Lloh1:
	ldr q0, [x20, lCPI0_0@PAGEOFF]
	add x20, x11, x21
	add x21, x0, x21
LBB0_2:
	add x22, x11, x5
	ldr d1, [x0, x5]
	ld1.d { v1 }[1], [x22]
	add x22, x12, x5
	ldr d2, [x10, x5]
	ld1.d { v2 }[1], [x22]
	add x22, x14, x5
	ldr d3, [x13, x5]
	ld1.d { v3 }[1], [x22]
	add x22, x15, x5
	ldr d4, [x9, x5]
	ld1.d { v4 }[1], [x22]
	tbl.16b v5, { v1, v2, v3, v4 }, v0
	add x22, x20, x5
	ldr d16, [x21, x5]
	ld1.d { v16 }[1], [x22]
	add x22, x6, x5
	ldr d17, [x19, x5]
	ld1.d { v17 }[1], [x22]
	add x22, x3, x5
	ldr d18, [x4, x5]
	ld1.d { v18 }[1], [x22]
	ldr d19, [x17, x5]
	add x22, x7, x5
	ld1.d { v19 }[1], [x22]
	tbl.16b v6, { v16, v17, v18, v19 }, v0
	mov.d v5[1], v6[0]
	ushr.2d v23, v19, #8
	ushr.2d v22, v18, #8
	ushr.2d v21, v17, #8
	ushr.2d v20, v16, #8
	ushr.2d v27, v4, #8
	ushr.2d v26, v3, #8
	ushr.2d v25, v2, #8
	ushr.2d v24, v1, #8
	tbl.16b v6, { v24, v25, v26, v27 }, v0
	tbl.16b v7, { v20, v21, v22, v23 }, v0
	mov.d v6[1], v7[0]
	ushr.2d v23, v19, #16
	ushr.2d v22, v18, #16
	ushr.2d v21, v17, #16
	ushr.2d v20, v16, #16
	ushr.2d v27, v4, #16
	ushr.2d v26, v3, #16
	ushr.2d v25, v2, #16
	ushr.2d v24, v1, #16
	tbl.16b v7, { v24, v25, v26, v27 }, v0
	tbl.16b v20, { v20, v21, v22, v23 }, v0
	mov.d v7[1], v20[0]
	ushr.2d v24, v19, #24
	ushr.2d v23, v18, #24
	ushr.2d v22, v17, #24
	ushr.2d v21, v16, #24
	ushr.2d v28, v4, #24
	ushr.2d v27, v3, #24
	ushr.2d v26, v2, #24
	ushr.2d v25, v1, #24
	tbl.16b v20, { v25, v26, v27, v28 }, v0
	tbl.16b v21, { v21, v22, v23, v24 }, v0
	mov.d v20[1], v21[0]
	ushr.2d v25, v19, #32
	ushr.2d v24, v18, #32
	ushr.2d v23, v17, #32
	ushr.2d v22, v16, #32
	ushr.2d v29, v4, #32
	ushr.2d v28, v3, #32
	ushr.2d v27, v2, #32
	ushr.2d v26, v1, #32
	tbl.16b v21, { v26, v27, v28, v29 }, v0
	tbl.16b v22, { v22, v23, v24, v25 }, v0
	mov.d v21[1], v22[0]
	ushr.2d v26, v19, #40
	ushr.2d v25, v18, #40
	ushr.2d v24, v17, #40
	ushr.2d v23, v16, #40
	ushr.2d v30, v4, #40
	ushr.2d v29, v3, #40
	ushr.2d v28, v2, #40
	ushr.2d v27, v1, #40
	tbl.16b v22, { v27, v28, v29, v30 }, v0
	tbl.16b v23, { v23, v24, v25, v26 }, v0
	mov.d v22[1], v23[0]
	ushr.2d v26, v19, #48
	ushr.2d v25, v18, #48
	ushr.2d v24, v17, #48
	ushr.2d v23, v16, #48
	ushr.2d v30, v4, #48
	ushr.2d v29, v3, #48
	ushr.2d v28, v2, #48
	ushr.2d v27, v1, #48
	tbl.16b v27, { v27, v28, v29, v30 }, v0
	tbl.16b v23, { v23, v24, v25, v26 }, v0
	mov.d v27[1], v23[0]
	stp q5, q6, [x16, #-64]
	ushr.2d v26, v19, #56
	ushr.2d v25, v18, #56
	ushr.2d v24, v17, #56
	stp q7, q20, [x16, #-32]
	ushr.2d v23, v16, #56
	ushr.2d v19, v4, #56
	ushr.2d v18, v3, #56
	ushr.2d v17, v2, #56
	ushr.2d v16, v1, #56
	tbl.16b v1, { v16, v17, v18, v19 }, v0
	tbl.16b v2, { v23, v24, v25, v26 }, v0
	mov.d v1[1], v2[0]
	stp q21, q22, [x16]
	stp q27, q1, [x16, #32]
	add x5, x5, #8
	add x16, x16, #128
	cmp x5, x8
	b.lo LBB0_2
LBB0_3:
	mov w4, #16
	ldp x20, x19, [sp, #16]
	ldp x22, x21, [sp], #32
	b jix_probe::byte_shuffle::decode_impl_generic
