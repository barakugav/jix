probe_byte_shuffle_decode_4:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffc
	cmp x3, x8
	b.lo .LBB10_5
	lsr x9, x1, #8
	lsr x8, x1, #2
	cbz x9, .LBB10_4
	lsl x14, x8, #1
	add x10, x0, #32
	add x12, x2, #128
	add x11, x10, x8
	add x13, x11, x14
	add x14, x10, x14
.LBB10_3:
	ldp q0, q4, [x10, #-32]
	subs x9, x9, #1
	ldp q1, q5, [x14, #-32]
	ldp q2, q7, [x11, #-32]
	ldp q3, q16, [x13, #-32]
	zip1 v6.16b, v0.16b, v1.16b
	zip2 v0.16b, v0.16b, v1.16b
	zip1 v22.16b, v4.16b, v5.16b
	zip2 v4.16b, v4.16b, v5.16b
	zip1 v17.16b, v2.16b, v3.16b
	zip2 v1.16b, v2.16b, v3.16b
	zip1 v23.16b, v7.16b, v16.16b
	zip2 v5.16b, v7.16b, v16.16b
	ldp q18, q19, [x10], #64
	ldp q2, q3, [x11], #64
	zip1 v24.16b, v6.16b, v17.16b
	zip2 v6.16b, v6.16b, v17.16b
	zip1 v7.16b, v0.16b, v1.16b
	zip2 v0.16b, v0.16b, v1.16b
	zip1 v1.16b, v22.16b, v23.16b
	ldp q20, q21, [x14], #64
	ldp q25, q26, [x13], #64
	stp q24, q6, [x12, #-128]
	zip2 v6.16b, v22.16b, v23.16b
	zip1 v16.16b, v18.16b, v20.16b
	zip2 v18.16b, v18.16b, v20.16b
	zip1 v20.16b, v4.16b, v5.16b
	zip1 v17.16b, v2.16b, v25.16b
	zip2 v2.16b, v2.16b, v25.16b
	stp q7, q0, [x12, #-96]
	zip2 v0.16b, v4.16b, v5.16b
	zip1 v5.16b, v19.16b, v21.16b
	zip1 v7.16b, v3.16b, v26.16b
	stp q1, q6, [x12, #-64]
	zip2 v6.16b, v19.16b, v21.16b
	zip2 v3.16b, v3.16b, v26.16b
	zip1 v4.16b, v16.16b, v17.16b
	zip2 v1.16b, v16.16b, v17.16b
	zip1 v16.16b, v18.16b, v2.16b
	stp q20, q0, [x12, #-32]
	zip2 v0.16b, v18.16b, v2.16b
	zip1 v2.16b, v5.16b, v7.16b
	stp q4, q1, [x12]
	zip2 v1.16b, v5.16b, v7.16b
	zip1 v4.16b, v6.16b, v3.16b
	stp q16, q0, [x12, #32]
	zip2 v0.16b, v6.16b, v3.16b
	stp q2, q1, [x12, #64]
	stp q4, q0, [x12, #96]
	add x12, x12, #256
	b.ne .LBB10_3
.LBB10_4:
	and x0, x8, #0x1fffffffffffffc0
	ldp x29, x30, [sp], #16
	ret
.LBB10_5:
	adrp x0, .Lanon.198375b9867b10ba11d34f490504f5a9.0
	add x0, x0, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.0
	adrp x2, .Lanon.198375b9867b10ba11d34f490504f5a9.2
	add x2, x2, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.2
	mov w1, #48
	bl core::panicking::panic
