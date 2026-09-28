jix_probe::byte_shuffle::decode_impl::<4, 32>:
	sub sp, sp, #112
	stp x1, x30, [sp, #8]
	stp x28, x27, [sp, #32]
	stp x26, x25, [sp, #48]
	stp x24, x23, [sp, #64]
	stp x22, x21, [sp, #80]
	stp x20, x19, [sp, #96]
	lsr x8, x1, #2
	mov x5, xzr
	str x2, [sp, #24]
	ands x1, x8, #0x1fffffffffffffe0
	b.eq .LBB2_3
	lsl x11, x8, #1
	adrp x10, .LCPI2_0
	add x9, x0, x8
	ldr q0, [x10, :lo12:.LCPI2_0]
	add x12, x2, #64
	add x10, x0, x11
	add x11, x9, x11
.LBB2_2:
	add x13, x0, x5
	add x8, x9, x5
	add x14, x10, x5
	ldr s1, [x13]
	add x28, x8, #8
	add x30, x8, #12
	add x27, x8, #16
	add x4, x8, #20
	add x17, x8, #24
	add x15, x8, #28
	ld1 { v1.s }[1], [x8], #4
	add x25, x14, #8
	add x24, x14, #12
	add x21, x14, #16
	add x19, x14, #20
	add x6, x14, #24
	add x16, x14, #28
	ld1 { v1.s }[2], [x14], #4
	add x3, x11, x5
	add x26, x3, #8
	add x23, x3, #12
	add x22, x3, #16
	add x20, x3, #20
	add x7, x3, #24
	add x18, x3, #28
	ld1 { v1.s }[3], [x3], #4
	ldr s16, [x13, #28]
	ldp s2, s3, [x13, #4]
	add x5, x5, #32
	ldp s4, s5, [x13, #12]
	ld1 { v16.s }[1], [x15]
	ldp s6, s7, [x13, #20]
	tbl v1.16b, { v1.16b }, v0.16b
	ld1 { v2.s }[1], [x8]
	ld1 { v3.s }[1], [x28]
	cmp x5, x1
	ld1 { v4.s }[1], [x30]
	ld1 { v5.s }[1], [x27]
	ld1 { v16.s }[2], [x16]
	ld1 { v6.s }[1], [x4]
	ld1 { v7.s }[1], [x17]
	ld1 { v2.s }[2], [x14]
	ld1 { v3.s }[2], [x25]
	ld1 { v4.s }[2], [x24]
	ld1 { v5.s }[2], [x21]
	ld1 { v16.s }[3], [x18]
	ld1 { v6.s }[2], [x19]
	ld1 { v7.s }[2], [x6]
	ld1 { v2.s }[3], [x3]
	ld1 { v3.s }[3], [x26]
	ld1 { v4.s }[3], [x23]
	ld1 { v5.s }[3], [x22]
	ld1 { v6.s }[3], [x20]
	ld1 { v7.s }[3], [x7]
	tbl v2.16b, { v2.16b }, v0.16b
	tbl v3.16b, { v3.16b }, v0.16b
	tbl v4.16b, { v4.16b }, v0.16b
	tbl v5.16b, { v5.16b }, v0.16b
	stp q1, q2, [x12, #-64]
	tbl v1.16b, { v6.16b }, v0.16b
	tbl v2.16b, { v7.16b }, v0.16b
	stp q3, q4, [x12, #-32]
	tbl v3.16b, { v16.16b }, v0.16b
	stp q5, q1, [x12]
	stp q2, q3, [x12, #32]
	add x12, x12, #128
	b.lo .LBB2_2
.LBB2_3:
	ldp x1, x30, [sp, #8]
	mov w4, #4
	ldp x20, x19, [sp, #96]
	ldr x2, [sp, #24]
	ldp x22, x21, [sp, #80]
	ldp x24, x23, [sp, #64]
	ldp x26, x25, [sp, #48]
	ldp x28, x27, [sp, #32]
	add sp, sp, #112
	b jix_probe::byte_shuffle::decode_impl_generic
