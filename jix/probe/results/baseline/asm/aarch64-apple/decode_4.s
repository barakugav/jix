jix_probe::byte_shuffle::decode_impl::<4, 32>:
Lfunc_begin2:
	sub sp, sp, #112
	stp x28, x27, [sp, #16]
	stp x26, x25, [sp, #32]
	stp x24, x23, [sp, #48]
	stp x22, x21, [sp, #64]
	stp x20, x19, [sp, #80]
	stp x29, x30, [sp, #96]
	stp x1, x2, [sp]
	lsr x10, x1, #2
	ands x1, x10, #0x1fffffffffffffe0
	mov x5, #0
	b.eq LBB2_3
	lsl x11, x10, #1
	add x9, x0, x11
	add x10, x0, x10
	add x11, x10, x11
	add x12, x2, #64
Lloh2:
	adrp x13, lCPI2_0@PAGE
Lloh3:
	ldr q0, [x13, lCPI2_0@PAGEOFF]
LBB2_2:
	add x13, x0, x5
	add x26, x10, x5
	add x23, x26, #8
	add x19, x26, #12
	add x6, x26, #16
	add x16, x26, #20
	add x15, x26, #24
	add x14, x26, #28
	ldp s1, s2, [x13]
	ld1.s { v1 }[1], [x26], #4
	add x27, x9, x5
	add x28, x27, #8
	add x25, x27, #12
	add x22, x27, #16
	add x20, x27, #20
	add x4, x27, #24
	add x17, x27, #28
	ld1.s { v1 }[2], [x27], #4
	add x30, x11, x5
	add x8, x30, #8
	add x2, x30, #12
	add x24, x30, #16
	add x21, x30, #20
	add x7, x30, #24
	add x3, x30, #28
	ld1.s { v1 }[3], [x30], #4
	ldr s3, [x13, #8]
	ld1.s { v2 }[1], [x26]
	ld1.s { v2 }[2], [x27]
	ld1.s { v2 }[3], [x30]
	tbl.16b v1, { v1 }, v0
	tbl.16b v2, { v2 }, v0
	stp q1, q2, [x12, #-64]
	ld1.s { v3 }[1], [x23]
	ld1.s { v3 }[2], [x28]
	ld1.s { v3 }[3], [x8]
	ldp s1, s2, [x13, #12]
	ld1.s { v1 }[1], [x19]
	ld1.s { v1 }[2], [x25]
	ld1.s { v1 }[3], [x2]
	tbl.16b v3, { v3 }, v0
	tbl.16b v1, { v1 }, v0
	stp q3, q1, [x12, #-32]
	ld1.s { v2 }[1], [x6]
	ld1.s { v2 }[2], [x22]
	ld1.s { v2 }[3], [x24]
	ldp s1, s3, [x13, #20]
	ld1.s { v1 }[1], [x16]
	ld1.s { v1 }[2], [x20]
	ld1.s { v1 }[3], [x21]
	tbl.16b v2, { v2 }, v0
	tbl.16b v1, { v1 }, v0
	stp q2, q1, [x12]
	ld1.s { v3 }[1], [x15]
	ld1.s { v3 }[2], [x4]
	ld1.s { v3 }[3], [x7]
	tbl.16b v1, { v3 }, v0
	ldr s2, [x13, #28]
	ld1.s { v2 }[1], [x14]
	ld1.s { v2 }[2], [x17]
	ld1.s { v2 }[3], [x3]
	tbl.16b v2, { v2 }, v0
	stp q1, q2, [x12, #32]
	add x5, x5, #32
	add x12, x12, #128
	cmp x5, x1
	b.lo LBB2_2
LBB2_3:
	ldp x1, x2, [sp]
	mov w4, #4
	ldp x29, x30, [sp, #96]
	ldp x20, x19, [sp, #80]
	ldp x22, x21, [sp, #64]
	ldp x24, x23, [sp, #48]
	ldp x26, x25, [sp, #32]
	ldp x28, x27, [sp, #16]
	add sp, sp, #112
	b jix_probe::byte_shuffle::decode_impl_generic
