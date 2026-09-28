probe_byte_shuffle_decode_8:
	stp x29, x30, [sp, #-32]!
	str x19, [sp, #16]
	mov x29, sp
	and x8, x1, #0x7ffffffffffffff8
	cmp x3, x8
	b.lo .LBB11_5
	lsr x12, x1, #8
	lsr x8, x1, #3
	cbz x12, .LBB11_4
	mov w11, #6
	add x10, x0, #16
	lsl x15, x8, #1
	madd x11, x8, x11, x10
	lsl x18, x8, #2
	sub x16, x10, x8
	lsl x12, x12, #5
	add x13, x10, x8
	add x16, x16, x8, lsl #3
	mov x9, xzr
	add x14, x10, x15
	add x15, x13, x15
	add x17, x10, x18
	add x18, x13, x18
	add x0, x2, #128
.LBB11_3:
	add x1, x10, x9
	add x2, x13, x9
	add x3, x14, x9
	add x4, x15, x9
	add x5, x17, x9
	add x6, x18, x9
	add x7, x11, x9
	add x19, x16, x9
	add x9, x9, #32
	ldp q0, q16, [x1, #-16]
	cmp x12, x9
	ldp q1, q19, [x2, #-16]
	ldp q2, q22, [x3, #-16]
	ldp q3, q25, [x4, #-16]
	ldp q4, q17, [x5, #-16]
	ldp q5, q20, [x6, #-16]
	ldp q6, q23, [x7, #-16]
	ldp q7, q26, [x19, #-16]
	zip1 v18.16b, v0.16b, v4.16b
	zip1 v21.16b, v1.16b, v5.16b
	zip2 v0.16b, v0.16b, v4.16b
	zip2 v1.16b, v1.16b, v5.16b
	zip1 v24.16b, v2.16b, v6.16b
	zip2 v2.16b, v2.16b, v6.16b
	zip1 v27.16b, v3.16b, v7.16b
	zip2 v3.16b, v3.16b, v7.16b
	zip1 v7.16b, v16.16b, v17.16b
	zip1 v28.16b, v18.16b, v24.16b
	zip2 v4.16b, v18.16b, v24.16b
	zip1 v24.16b, v0.16b, v2.16b
	zip1 v29.16b, v21.16b, v27.16b
	zip2 v5.16b, v21.16b, v27.16b
	zip1 v21.16b, v19.16b, v20.16b
	zip1 v27.16b, v1.16b, v3.16b
	zip2 v0.16b, v0.16b, v2.16b
	zip1 v2.16b, v25.16b, v26.16b
	zip2 v1.16b, v1.16b, v3.16b
	zip1 v6.16b, v28.16b, v29.16b
	zip2 v18.16b, v28.16b, v29.16b
	zip1 v28.16b, v22.16b, v23.16b
	zip1 v3.16b, v4.16b, v5.16b
	zip2 v4.16b, v4.16b, v5.16b
	zip2 v5.16b, v16.16b, v17.16b
	zip2 v17.16b, v22.16b, v23.16b
	zip1 v16.16b, v24.16b, v27.16b
	zip2 v22.16b, v24.16b, v27.16b
	stp q6, q18, [x0, #-128]
	zip2 v6.16b, v19.16b, v20.16b
	zip2 v18.16b, v25.16b, v26.16b
	zip1 v19.16b, v7.16b, v28.16b
	zip1 v20.16b, v21.16b, v2.16b
	stp q3, q4, [x0, #-96]
	zip2 v3.16b, v7.16b, v28.16b
	zip1 v7.16b, v0.16b, v1.16b
	zip2 v2.16b, v21.16b, v2.16b
	zip2 v0.16b, v0.16b, v1.16b
	zip1 v1.16b, v5.16b, v17.16b
	zip2 v5.16b, v5.16b, v17.16b
	zip1 v4.16b, v6.16b, v18.16b
	zip2 v6.16b, v6.16b, v18.16b
	stp q16, q22, [x0, #-64]
	zip1 v21.16b, v19.16b, v20.16b
	zip2 v16.16b, v19.16b, v20.16b
	zip1 v17.16b, v3.16b, v2.16b
	stp q7, q0, [x0, #-32]
	zip2 v0.16b, v3.16b, v2.16b
	zip1 v2.16b, v1.16b, v4.16b
	zip2 v1.16b, v1.16b, v4.16b
	zip1 v3.16b, v5.16b, v6.16b
	zip2 v4.16b, v5.16b, v6.16b
	stp q21, q16, [x0]
	stp q17, q0, [x0, #32]
	stp q2, q1, [x0, #64]
	stp q3, q4, [x0, #96]
	add x0, x0, #256
	b.ne .LBB11_3
.LBB11_4:
	and x0, x8, #0xfffffffffffffe0
	ldr x19, [sp, #16]
	ldp x29, x30, [sp], #32
	ret
.LBB11_5:
	adrp x0, .Lanon.198375b9867b10ba11d34f490504f5a9.0
	add x0, x0, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.0
	adrp x2, .Lanon.198375b9867b10ba11d34f490504f5a9.2
	add x2, x2, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.2
	mov w1, #48
	bl core::panicking::panic
