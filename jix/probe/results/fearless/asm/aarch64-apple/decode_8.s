_probe_byte_shuffle_decode_8:
Lfunc_begin11:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffff8
	cmp x3, x8
	b.lo LBB11_5
	lsr x8, x1, #3
	lsr x10, x1, #8
	cbz x10, LBB11_4
	mov x9, #0
	lsl x14, x8, #1
	lsl x16, x8, #2
	mov w1, #6
	lsl x10, x10, #5
	add x11, x0, #16
	add x12, x11, x14
	add x13, x11, x8
	add x14, x13, x14
	add x15, x11, x16
	add x16, x13, x16
	add x17, x2, #128
	sub x2, x11, x8
	madd x0, x8, x1, x11
	add x1, x2, x8, lsl #3
LBB11_3:
	add x2, x11, x9
	add x3, x15, x9
	ldp q0, q1, [x2, #-16]
	ldp q2, q3, [x3, #-16]
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	add x2, x13, x9
	add x3, x16, x9
	ldp q2, q5, [x2, #-16]
	ldp q6, q7, [x3, #-16]
	zip1.16b v16, v2, v6
	zip2.16b v2, v2, v6
	add x2, x12, x9
	add x3, x0, x9
	ldp q6, q17, [x2, #-16]
	ldp q18, q19, [x3, #-16]
	zip1.16b v20, v6, v18
	zip2.16b v6, v6, v18
	add x2, x14, x9
	add x3, x1, x9
	ldp q18, q21, [x2, #-16]
	ldp q22, q23, [x3, #-16]
	zip1.16b v24, v18, v22
	zip2.16b v18, v18, v22
	zip1.16b v22, v4, v20
	zip2.16b v4, v4, v20
	zip1.16b v20, v0, v6
	zip2.16b v0, v0, v6
	zip1.16b v6, v16, v24
	zip2.16b v16, v16, v24
	zip1.16b v24, v2, v18
	zip2.16b v2, v2, v18
	zip1.16b v18, v22, v6
	zip2.16b v6, v22, v6
	zip1.16b v22, v4, v16
	zip2.16b v4, v4, v16
	zip1.16b v16, v20, v24
	zip2.16b v20, v20, v24
	zip1.16b v24, v0, v2
	zip2.16b v0, v0, v2
	zip1.16b v2, v1, v3
	zip2.16b v1, v1, v3
	stp q18, q6, [x17, #-128]
	stp q22, q4, [x17, #-96]
	zip1.16b v3, v5, v7
	zip2.16b v4, v5, v7
	zip1.16b v5, v17, v19
	zip2.16b v6, v17, v19
	zip1.16b v7, v21, v23
	zip2.16b v17, v21, v23
	stp q16, q20, [x17, #-64]
	zip1.16b v16, v2, v5
	stp q24, q0, [x17, #-32]
	zip2.16b v0, v2, v5
	zip1.16b v2, v1, v6
	zip2.16b v1, v1, v6
	zip1.16b v5, v3, v7
	zip1.16b v6, v16, v5
	zip2.16b v5, v16, v5
	stp q6, q5, [x17]
	zip2.16b v3, v3, v7
	zip1.16b v5, v0, v3
	zip2.16b v0, v0, v3
	stp q5, q0, [x17, #32]
	zip1.16b v0, v4, v17
	zip1.16b v3, v2, v0
	zip2.16b v0, v2, v0
	stp q3, q0, [x17, #64]
	add x9, x9, #32
	zip2.16b v0, v4, v17
	zip1.16b v2, v1, v0
	zip2.16b v0, v1, v0
	stp q2, q0, [x17, #96]
	add x17, x17, #256
	cmp x10, x9
	b.ne LBB11_3
LBB11_4:
	and x0, x8, #0xfffffffffffffe0
	ldp x29, x30, [sp], #16
	ret
LBB11_5:
Lloh20:
	adrp x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGE
Lloh21:
	add x0, x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGEOFF
Lloh22:
	adrp x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGE
Lloh23:
	add x2, x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
