probe_byte_shuffle_decode_16:
	stp d9, d8, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	and x8, x1, #0x7ffffffffffffff0
	cmp x3, x8
	b.lo .LBB8_5
	lsr x9, x1, #8
	lsl x8, x9, #4
	cbz x9, .LBB8_4
	lsr x5, x1, #4
	mov w10, #6
	mov w11, #10
	mov w12, #11
	mov w13, #12
	mov w14, #13
	mov w15, #14
	madd x10, x5, x10, x0
	lsl x7, x5, #3
	lsl x18, x5, #1
	lsl x3, x5, #2
	sub x6, x0, x5
	madd x11, x5, x11, x0
	sub x4, x7, x5
	add x16, x0, x5
	mov x9, xzr
	add x17, x0, x18
	add x18, x16, x18
	madd x12, x5, x12, x0
	add x1, x0, x3
	add x3, x16, x3
	add x4, x0, x4
	add x2, x2, #128
	madd x13, x5, x13, x0
	madd x14, x5, x14, x0
	madd x15, x5, x15, x0
	add x5, x6, x5, lsl #4
	add x6, x0, x7
	add x7, x16, x7
.LBB8_3:
	ldr q0, [x0, x9]
	ldr q1, [x16, x9]
	ldr q2, [x6, x9]
	ldr q3, [x7, x9]
	ldr q4, [x17, x9]
	ldr q5, [x18, x9]
	zip1 v6.16b, v0.16b, v2.16b
	zip2 v0.16b, v0.16b, v2.16b
	ldr q16, [x1, x9]
	ldr q17, [x3, x9]
	zip1 v7.16b, v1.16b, v3.16b
	zip2 v1.16b, v1.16b, v3.16b
	ldr q18, [x10, x9]
	ldr q19, [x4, x9]
	ldr q20, [x11, x9]
	ldr q21, [x12, x9]
	ldr q22, [x13, x9]
	ldr q23, [x14, x9]
	zip1 v26.16b, v4.16b, v20.16b
	zip2 v4.16b, v4.16b, v20.16b
	ldr q24, [x15, x9]
	ldr q25, [x5, x9]
	zip1 v27.16b, v5.16b, v21.16b
	zip1 v28.16b, v16.16b, v22.16b
	zip1 v29.16b, v17.16b, v23.16b
	zip1 v30.16b, v18.16b, v24.16b
	zip2 v5.16b, v5.16b, v21.16b
	zip1 v31.16b, v19.16b, v25.16b
	zip2 v16.16b, v16.16b, v22.16b
	zip2 v17.16b, v17.16b, v23.16b
	zip2 v18.16b, v18.16b, v24.16b
	zip2 v19.16b, v19.16b, v25.16b
	add x9, x9, #16
	zip1 v2.16b, v6.16b, v28.16b
	zip2 v6.16b, v6.16b, v28.16b
	cmp x8, x9
	zip1 v3.16b, v7.16b, v29.16b
	zip1 v8.16b, v26.16b, v30.16b
	zip2 v7.16b, v7.16b, v29.16b
	zip1 v9.16b, v27.16b, v31.16b
	zip2 v20.16b, v26.16b, v30.16b
	zip2 v21.16b, v27.16b, v31.16b
	zip1 v24.16b, v0.16b, v16.16b
	zip2 v0.16b, v0.16b, v16.16b
	zip1 v16.16b, v1.16b, v17.16b
	zip1 v25.16b, v4.16b, v18.16b
	zip1 v26.16b, v5.16b, v19.16b
	zip2 v1.16b, v1.16b, v17.16b
	zip1 v22.16b, v2.16b, v8.16b
	zip2 v2.16b, v2.16b, v8.16b
	zip2 v4.16b, v4.16b, v18.16b
	zip1 v23.16b, v3.16b, v9.16b
	zip2 v3.16b, v3.16b, v9.16b
	zip1 v27.16b, v6.16b, v20.16b
	zip1 v28.16b, v7.16b, v21.16b
	zip2 v6.16b, v6.16b, v20.16b
	zip2 v7.16b, v7.16b, v21.16b
	zip2 v5.16b, v5.16b, v19.16b
	zip1 v19.16b, v24.16b, v25.16b
	zip1 v29.16b, v22.16b, v23.16b
	zip2 v17.16b, v22.16b, v23.16b
	zip1 v18.16b, v2.16b, v3.16b
	zip2 v2.16b, v2.16b, v3.16b
	zip1 v3.16b, v16.16b, v26.16b
	zip2 v16.16b, v16.16b, v26.16b
	zip1 v22.16b, v6.16b, v7.16b
	zip2 v6.16b, v6.16b, v7.16b
	zip1 v7.16b, v1.16b, v5.16b
	zip2 v1.16b, v1.16b, v5.16b
	zip1 v20.16b, v27.16b, v28.16b
	zip2 v21.16b, v27.16b, v28.16b
	stp q29, q17, [x2, #-128]
	zip2 v17.16b, v24.16b, v25.16b
	stp q18, q2, [x2, #-96]
	zip1 v2.16b, v0.16b, v4.16b
	zip1 v18.16b, v19.16b, v3.16b
	stp q22, q6, [x2, #-32]
	zip2 v0.16b, v0.16b, v4.16b
	zip2 v3.16b, v19.16b, v3.16b
	stp q20, q21, [x2, #-64]
	zip1 v4.16b, v17.16b, v16.16b
	zip2 v5.16b, v17.16b, v16.16b
	zip1 v6.16b, v2.16b, v7.16b
	zip2 v2.16b, v2.16b, v7.16b
	stp q18, q3, [x2]
	zip1 v3.16b, v0.16b, v1.16b
	zip2 v0.16b, v0.16b, v1.16b
	stp q4, q5, [x2, #32]
	stp q6, q2, [x2, #64]
	stp q3, q0, [x2, #96]
	add x2, x2, #256
	b.ne .LBB8_3
.LBB8_4:
	mov x0, x8
	ldp x29, x30, [sp, #16]
	ldp d9, d8, [sp], #32
	ret
.LBB8_5:
	adrp x0, .Lanon.198375b9867b10ba11d34f490504f5a9.0
	add x0, x0, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.0
	adrp x2, .Lanon.198375b9867b10ba11d34f490504f5a9.2
	add x2, x2, :lo12:.Lanon.198375b9867b10ba11d34f490504f5a9.2
	mov w1, #48
	bl core::panicking::panic
