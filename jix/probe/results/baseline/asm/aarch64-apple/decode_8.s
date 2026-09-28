jix_probe::byte_shuffle::decode_impl::<8, 16>:
Lfunc_begin3:
	sub sp, sp, #112
	stp x28, x27, [sp, #16]
	stp x26, x25, [sp, #32]
	stp x24, x23, [sp, #48]
	stp x22, x21, [sp, #64]
	stp x20, x19, [sp, #80]
	stp x29, x30, [sp, #96]
	stp x1, x2, [sp]
	lsr x8, x1, #3
	ands x1, x8, #0xffffffffffffff0
	mov x5, #0
	b.eq LBB3_3
	lsl x11, x8, #1
	lsl x16, x8, #2
	mov w13, #6
	add x9, x0, x11
	add x10, x0, x8
	add x11, x10, x11
	add x12, x2, #64
Lloh4:
	adrp x14, lCPI3_0@PAGE
Lloh5:
	ldr q0, [x14, lCPI3_0@PAGEOFF]
Lloh6:
	adrp x14, lCPI3_1@PAGE
Lloh7:
	ldr q1, [x14, lCPI3_1@PAGEOFF]
	sub x14, x0, x8
	madd x13, x8, x13, x0
	add x14, x14, x8, lsl #3
Lloh8:
	adrp x8, lCPI3_2@PAGE
Lloh9:
	ldr q2, [x8, lCPI3_2@PAGEOFF]
Lloh10:
	adrp x8, lCPI3_3@PAGE
Lloh11:
	ldr q3, [x8, lCPI3_3@PAGEOFF]
	add x15, x0, x16
	add x16, x10, x16
LBB3_2:
	add x17, x0, x5
	add x27, x10, x5
	add x23, x27, #8
	add x3, x27, #12
	add x28, x9, x5
	add x24, x28, #8
	add x4, x28, #12
	add x30, x11, x5
	add x25, x30, #8
	add x6, x30, #12
	add x7, x15, x5
	add x26, x16, x5
	add x22, x26, #8
	add x19, x26, #12
	ldr s17, [x17]
	ld1.s { v17 }[1], [x27], #4
	ld1.s { v17 }[2], [x28], #4
	ld1.s { v17 }[3], [x30], #4
	add x2, x13, x5
	add x8, x2, #8
	ldr s18, [x7]
	ld1.s { v18 }[1], [x26], #4
	add x20, x2, #12
	add x21, x14, x5
	ld1.s { v18 }[2], [x2], #4
	ldp s5, s7, [x21]
	tbl.16b v4, { v17, v18 }, v0
	tbl.16b v16, { v4, v5 }, v1
	tbl.16b v4, { v17, v18 }, v2
	ldp s17, s19, [x17, #4]
	ld1.s { v17 }[1], [x27]
	ld1.s { v17 }[2], [x28]
	ld1.s { v17 }[3], [x30]
	ld1.s { v19 }[1], [x23]
	ld1.s { v19 }[2], [x24]
	ld1.s { v19 }[3], [x25]
	ldp s18, s20, [x7, #4]
	ld1.s { v18 }[1], [x26]
	ld1.s { v18 }[2], [x2]
	tbl.16b v6, { v17, v18 }, v0
	tbl.16b v21, { v6, v7 }, v1
	ld1.s { v20 }[1], [x22]
	ld1.s { v20 }[2], [x8]
	tbl.16b v6, { v17, v18 }, v2
	ldp s18, s23, [x21, #8]
	tbl.16b v17, { v19, v20 }, v2
	tbl.16b v24, { v17, v18 }, v3
	tbl.16b v17, { v19, v20 }, v0
	tbl.16b v4, { v4, v5 }, v3
	stp q16, q4, [x12, #-64]
	tbl.16b v4, { v6, v7 }, v3
	stp q21, q4, [x12, #-32]
	tbl.16b v4, { v17, v18 }, v1
	ldr s5, [x17, #12]
	ld1.s { v5 }[1], [x3]
	ld1.s { v5 }[2], [x4]
	ld1.s { v5 }[3], [x6]
	ldr s6, [x7, #12]
	ld1.s { v6 }[1], [x19]
	stp q4, q24, [x12]
	ld1.s { v6 }[2], [x20]
	tbl.16b v22, { v5, v6 }, v0
	tbl.16b v4, { v22, v23 }, v1
	tbl.16b v22, { v5, v6 }, v2
	tbl.16b v5, { v22, v23 }, v3
	stp q4, q5, [x12, #32]
	add x5, x5, #16
	add x12, x12, #128
	cmp x5, x1
	b.lo LBB3_2
LBB3_3:
	ldp x1, x2, [sp]
	mov w4, #8
	ldp x29, x30, [sp, #96]
	ldp x20, x19, [sp, #80]
	ldp x22, x21, [sp, #64]
	ldp x24, x23, [sp, #48]
	ldp x26, x25, [sp, #32]
	ldp x28, x27, [sp, #16]
	add sp, sp, #112
	b jix_probe::byte_shuffle::decode_impl_generic
