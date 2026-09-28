jix_probe::byte_shuffle::decode_impl::<8, 16>:
	lsr x12, x1, #3
	ands x8, x12, #0xffffffffffffff0
	b.eq .LBB3_4
	str x30, [sp, #-96]!
	stp x28, x27, [sp, #16]
	stp x26, x25, [sp, #32]
	stp x24, x23, [sp, #48]
	stp x22, x21, [sp, #64]
	stp x20, x19, [sp, #80]
	mov w10, #6
	sub x14, x0, x12
	lsl x13, x12, #1
	madd x10, x12, x10, x0
	lsl x16, x12, #2
	add x9, x0, x12
	adrp x15, .LCPI3_0
	add x12, x14, x12, lsl #3
	adrp x14, .LCPI3_1
	ldr q0, [x15, :lo12:.LCPI3_0]
	ldr q1, [x14, :lo12:.LCPI3_1]
	adrp x14, .LCPI3_2
	adrp x15, .LCPI3_3
	ldr q2, [x14, :lo12:.LCPI3_2]
	mov x5, xzr
	ldr q3, [x15, :lo12:.LCPI3_3]
	add x11, x0, x13
	add x13, x9, x13
	add x14, x2, #64
	add x15, x0, x16
	add x16, x9, x16
.LBB3_2:
	add x17, x0, x5
	add x27, x9, x5
	add x28, x11, x5
	ldr s4, [x17]
	add x3, x27, #8
	add x18, x27, #12
	add x30, x28, #8
	add x21, x28, #12
	add x25, x13, x5
	ld1 { v4.s }[1], [x27], #4
	add x4, x15, x5
	add x26, x25, #8
	add x22, x25, #12
	add x23, x16, x5
	add x20, x23, #8
	add x6, x23, #12
	add x24, x10, x5
	ld1 { v4.s }[2], [x28], #4
	add x19, x24, #8
	add x7, x24, #12
	ld1 { v4.s }[3], [x25], #4
	ldr s5, [x4]
	ld1 { v5.s }[1], [x23], #4
	ld1 { v5.s }[2], [x24], #4
	ldr s17, [x30]
	ldp s6, s16, [x17, #4]
	ldr s18, [x17, #12]
	ldr s19, [x21]
	ld1 { v17.s }[1], [x26]
	add x17, x12, x5
	ld1 { v18.s }[1], [x18]
	ldp s21, s23, [x17]
	ld1 { v6.s }[1], [x27]
	ld1 { v16.s }[1], [x3]
	ld1 { v19.s }[1], [x22]
	tbl v20.16b, { v4.16b, v5.16b }, v0.16b
	add x5, x5, #16
	cmp x5, x8
	ld1 { v6.s }[2], [x28]
	zip1 v16.2d, v16.2d, v17.2d
	zip1 v18.2d, v18.2d, v19.2d
	ldr s19, [x4, #12]
	tbl v26.16b, { v20.16b, v21.16b }, v1.16b
	tbl v20.16b, { v4.16b, v5.16b }, v2.16b
	ld1 { v6.s }[3], [x25]
	ld1 { v19.s }[1], [x6]
	ldp s7, s17, [x4, #4]
	ldp s25, s5, [x17, #8]
	ld1 { v19.s }[2], [x7]
	ld1 { v7.s }[1], [x23]
	ld1 { v17.s }[1], [x20]
	tbl v4.16b, { v18.16b, v19.16b }, v0.16b
	ld1 { v7.s }[2], [x24]
	ld1 { v17.s }[2], [x19]
	tbl v22.16b, { v6.16b, v7.16b }, v0.16b
	tbl v24.16b, { v16.16b, v17.16b }, v2.16b
	tbl v27.16b, { v22.16b, v23.16b }, v1.16b
	tbl v22.16b, { v6.16b, v7.16b }, v2.16b
	tbl v6.16b, { v24.16b, v25.16b }, v3.16b
	tbl v24.16b, { v16.16b, v17.16b }, v0.16b
	tbl v16.16b, { v4.16b, v5.16b }, v1.16b
	tbl v4.16b, { v18.16b, v19.16b }, v2.16b
	tbl v7.16b, { v20.16b, v21.16b }, v3.16b
	tbl v17.16b, { v22.16b, v23.16b }, v3.16b
	tbl v18.16b, { v24.16b, v25.16b }, v1.16b
	tbl v4.16b, { v4.16b, v5.16b }, v3.16b
	stp q26, q7, [x14, #-64]
	stp q27, q17, [x14, #-32]
	stp q18, q6, [x14]
	stp q16, q4, [x14, #32]
	add x14, x14, #128
	b.lo .LBB3_2
	ldp x20, x19, [sp, #80]
	ldp x22, x21, [sp, #64]
	ldp x24, x23, [sp, #48]
	ldp x26, x25, [sp, #32]
	ldp x28, x27, [sp, #16]
	ldr x30, [sp], #96
	mov w4, #8
	b jix_probe::byte_shuffle::decode_impl_generic
.LBB3_4:
	mov x5, xzr
	mov w4, #8
	b jix_probe::byte_shuffle::decode_impl_generic
