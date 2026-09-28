_probe_byte_shuffle_decode_16:
Lfunc_begin8:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	and x8, x1, #0x7ffffffffffffff0
	cmp x3, x8
	b.lo LBB8_5
	lsr x9, x1, #8
	lsl x8, x9, #4
	cbz x9, LBB8_4
	mov x9, #0
	lsr x6, x1, #4
	lsl x13, x6, #1
	lsl x15, x6, #2
	mov w10, #6
	lsl x19, x6, #3
	mov w17, #10
	mov w1, #11
	mov w3, #12
	mov w4, #13
	madd x10, x6, x10, x0
	mov w7, #14
	add x11, x0, x13
	add x12, x0, x6
	add x13, x12, x13
	add x14, x0, x15
	add x15, x12, x15
	sub x16, x19, x6
	add x16, x0, x16
	madd x17, x6, x17, x0
	madd x1, x6, x1, x0
	madd x3, x6, x3, x0
	madd x4, x6, x4, x0
	sub x5, x0, x6
	add x5, x5, x6, lsl #4
	madd x6, x6, x7, x0
	add x7, x0, x19
	add x19, x12, x19
	add x2, x2, #128
LBB8_3:
	ldr q0, [x0, x9]
	ldr q1, [x7, x9]
	zip1.16b v2, v0, v1
	ldr q3, [x12, x9]
	zip2.16b v0, v0, v1
	ldr q1, [x19, x9]
	zip1.16b v4, v3, v1
	ldr q5, [x11, x9]
	zip2.16b v1, v3, v1
	ldr q3, [x17, x9]
	zip1.16b v6, v5, v3
	ldr q7, [x13, x9]
	zip2.16b v3, v5, v3
	ldr q5, [x1, x9]
	zip1.16b v16, v7, v5
	ldr q17, [x14, x9]
	zip2.16b v5, v7, v5
	ldr q7, [x3, x9]
	zip1.16b v18, v17, v7
	ldr q19, [x15, x9]
	zip2.16b v7, v17, v7
	ldr q17, [x4, x9]
	zip1.16b v20, v19, v17
	ldr q21, [x10, x9]
	zip2.16b v17, v19, v17
	ldr q19, [x6, x9]
	zip1.16b v22, v21, v19
	ldr q23, [x16, x9]
	zip2.16b v19, v21, v19
	ldr q21, [x5, x9]
	zip1.16b v24, v23, v21
	zip2.16b v21, v23, v21
	zip1.16b v23, v2, v18
	zip2.16b v18, v2, v18
	zip1.16b v25, v0, v7
	zip2.16b v0, v0, v7
	zip1.16b v7, v4, v20
	zip2.16b v20, v4, v20
	zip1.16b v4, v1, v17
	zip2.16b v1, v1, v17
	zip1.16b v17, v6, v22
	zip2.16b v6, v6, v22
	zip1.16b v22, v3, v19
	zip2.16b v2, v3, v19
	zip1.16b v19, v16, v24
	zip2.16b v16, v16, v24
	zip1.16b v24, v5, v21
	zip2.16b v3, v5, v21
	zip1.16b v5, v23, v17
	zip2.16b v17, v23, v17
	zip1.16b v21, v18, v6
	zip2.16b v6, v18, v6
	zip1.16b v18, v25, v22
	zip2.16b v22, v25, v22
	zip1.16b v23, v7, v19
	zip1.16b v25, v5, v23
	zip2.16b v5, v5, v23
	zip1.16b v23, v0, v2
	stp q25, q5, [x2, #-128]
	zip2.16b v5, v7, v19
	zip1.16b v7, v17, v5
	zip2.16b v5, v17, v5
	stp q7, q5, [x2, #-96]
	zip1.16b v5, v20, v16
	zip1.16b v7, v21, v5
	zip2.16b v5, v21, v5
	stp q7, q5, [x2, #-64]
	zip2.16b v5, v20, v16
	zip1.16b v7, v6, v5
	zip2.16b v5, v6, v5
	stp q7, q5, [x2, #-32]
	zip1.16b v5, v4, v24
	zip1.16b v6, v18, v5
	zip2.16b v5, v18, v5
	stp q6, q5, [x2]
	zip2.16b v4, v4, v24
	zip1.16b v5, v22, v4
	zip2.16b v4, v22, v4
	stp q5, q4, [x2, #32]
	zip1.16b v4, v1, v3
	zip1.16b v5, v23, v4
	zip2.16b v4, v23, v4
	stp q5, q4, [x2, #64]
	zip2.16b v0, v0, v2
	zip2.16b v1, v1, v3
	zip1.16b v2, v0, v1
	zip2.16b v0, v0, v1
	stp q2, q0, [x2, #96]
	add x9, x9, #16
	add x2, x2, #256
	cmp x8, x9
	b.ne LBB8_3
LBB8_4:
	mov x0, x8
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret
LBB8_5:
Lloh8:
	adrp x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGE
Lloh9:
	add x0, x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGEOFF
Lloh10:
	adrp x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGE
Lloh11:
	add x2, x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
