_probe_byte_shuffle_decode_2:
Lfunc_begin9:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffe
	cmp x3, x8
	b.lo LBB9_5
	lsr x8, x1, #1
	lsr x9, x1, #8
	cbz x9, LBB9_4
	add x10, x2, #128
	add x11, x0, #64
	add x12, x11, x8
LBB9_3:
	ldp q0, q1, [x11, #-64]
	ldp q2, q3, [x12, #-64]
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	stp q4, q0, [x10, #-128]
	zip1.16b v0, v1, v3
	zip2.16b v1, v1, v3
	stp q0, q1, [x10, #-96]
	ldp q0, q1, [x11, #-32]
	ldp q2, q3, [x12, #-32]
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	stp q4, q0, [x10, #-64]
	zip1.16b v0, v1, v3
	zip2.16b v1, v1, v3
	stp q0, q1, [x10, #-32]
	ldp q0, q1, [x11]
	ldp q2, q3, [x12]
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	stp q4, q0, [x10]
	zip1.16b v0, v1, v3
	zip2.16b v1, v1, v3
	stp q0, q1, [x10, #32]
	ldp q0, q1, [x11, #32]
	ldp q2, q3, [x12, #32]
	zip1.16b v4, v0, v2
	zip2.16b v0, v0, v2
	zip1.16b v2, v1, v3
	zip2.16b v1, v1, v3
	stp q4, q0, [x10, #64]
	stp q2, q1, [x10, #96]
	add x10, x10, #256
	add x12, x12, #128
	add x11, x11, #128
	subs x9, x9, #1
	b.ne LBB9_3
LBB9_4:
	and x0, x8, #0x3fffffffffffff80
	ldp x29, x30, [sp], #16
	ret
LBB9_5:
Lloh12:
	adrp x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGE
Lloh13:
	add x0, x0, l_anon.b46df6bc8687e9d54c3982d97b2605b8.0@PAGEOFF
Lloh14:
	adrp x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGE
Lloh15:
	add x2, x2, l_anon.b46df6bc8687e9d54c3982d97b2605b8.2@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
