probe_byte_shuffle_decode_2:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffe
	cmp x3, x8
	b.lo .LBB9_5
	lsr x9, x1, #8
	lsr x8, x1, #1
	cbz x9, .LBB9_4
	add x10, x0, #64
	add x11, x2, #128
	add x12, x10, x8
.LBB9_3:
	ldp q0, q3, [x12, #-64]
	subs x9, x9, #1
	ldp q1, q2, [x10, #-64]
	zip1 v4.16b, v1.16b, v0.16b
	zip2 v0.16b, v1.16b, v0.16b
	zip1 v1.16b, v2.16b, v3.16b
	zip2 v2.16b, v2.16b, v3.16b
	ldp q3, q6, [x10, #-32]
	stp q4, q0, [x11, #-128]
	ldp q5, q4, [x12, #-32]
	stp q1, q2, [x11, #-96]
	zip1 v7.16b, v3.16b, v5.16b
	zip2 v3.16b, v3.16b, v5.16b
	zip1 v5.16b, v6.16b, v4.16b
	zip2 v2.16b, v6.16b, v4.16b
	ldp q0, q4, [x10]
	ldp q1, q6, [x12]
	stp q7, q3, [x11, #-64]
	zip1 v16.16b, v0.16b, v1.16b
	zip2 v0.16b, v0.16b, v1.16b
	zip1 v1.16b, v4.16b, v6.16b
	stp q5, q2, [x11, #-32]
	zip2 v2.16b, v4.16b, v6.16b
	ldp q3, q4, [x10, #32]
	add x10, x10, #128
	ldp q7, q5, [x12, #32]
	add x12, x12, #128
	stp q16, q0, [x11]
	stp q1, q2, [x11, #32]
	zip1 v6.16b, v3.16b, v7.16b
	zip2 v0.16b, v3.16b, v7.16b
	zip1 v3.16b, v4.16b, v5.16b
	zip2 v1.16b, v4.16b, v5.16b
	stp q6, q0, [x11, #64]
	stp q3, q1, [x11, #96]
	add x11, x11, #256
	b.ne .LBB9_3
.LBB9_4:
	and x0, x8, #0x3fffffffffffff80
	ldp x29, x30, [sp], #16
	ret
.LBB9_5:
	adrp x0, .Lanon.198375b9867b10ba11d34f490504f5a9.0
	add x0, x0, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.0
	adrp x2, .Lanon.198375b9867b10ba11d34f490504f5a9.2
	add x2, x2, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.2
	mov w1, #48
	bl core::panicking::panic
