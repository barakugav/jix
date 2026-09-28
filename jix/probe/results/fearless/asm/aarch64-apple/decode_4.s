_probe_byte_shuffle_decode_4:
Lfunc_begin10:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffc
	cmp x3, x8
	b.lo LBB10_5
	lsr x8, x1, #2
	lsr x9, x1, #8
	cbz x9, LBB10_4
	lsl x14, x8, #1
	add x10, x2, #128
	add x11, x0, #32
	add x12, x11, x8
	add x13, x12, x14
	add x14, x11, x14
LBB10_3:
	ldp q0, q1, [x11, #-32]
	ldp q2, q3, [x14, #-32]
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	ldp q2, q5, [x12, #-32]
	ldp q6, q7, [x13, #-32]
	zip1.16b v16, v2, v6
	zip2.16b v2, v2, v6
	zip1.16b v6, v4, v16
	zip2.16b v4, v4, v16
	zip1.16b v16, v0, v2
	zip2.16b v0, v0, v2
	stp q6, q4, [x10, #-128]
	stp q16, q0, [x10, #-96]
	zip1.16b v0, v1, v3
	zip2.16b v1, v1, v3
	zip1.16b v2, v5, v7
	zip2.16b v3, v5, v7
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	zip1.16b v2, v1, v3
	zip2.16b v1, v1, v3
	stp q4, q0, [x10, #-64]
	stp q2, q1, [x10, #-32]
	ldp q0, q1, [x11], #64
	ldp q2, q3, [x14], #64
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	ldp q2, q5, [x12], #64
	ldp q6, q7, [x13], #64
	zip1.16b v16, v2, v6
	zip2.16b v2, v2, v6
	zip1.16b v6, v4, v16
	zip2.16b v4, v4, v16
	zip1.16b v16, v0, v2
	zip2.16b v0, v0, v2
	stp q6, q4, [x10]
	stp q16, q0, [x10, #32]
	zip1.16b v0, v1, v3
	zip2.16b v1, v1, v3
	zip1.16b v2, v5, v7
	zip2.16b v3, v5, v7
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	zip1.16b v2, v1, v3
	zip2.16b v1, v1, v3
	stp q4, q0, [x10, #64]
	stp q2, q1, [x10, #96]
	add x10, x10, #256
	subs x9, x9, #1
	b.ne LBB10_3
LBB10_4:
	and x0, x8, #0x1fffffffffffffc0
	ldp x29, x30, [sp], #16
	ret
LBB10_5:
Lloh16:
	adrp x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGE
Lloh17:
	add x0, x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGEOFF
Lloh18:
	adrp x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGE
Lloh19:
	add x2, x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
