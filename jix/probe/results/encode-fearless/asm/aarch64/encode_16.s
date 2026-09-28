probe_byte_shuffle_encode_16:
	stp d13, d12, [sp, #-80]!
	stp d11, d10, [sp, #16]
	stp d9, d8, [sp, #32]
	stp x29, x30, [sp, #48]
	str x19, [sp, #64]
	add x29, sp, #48
	and x8, x1, #0x7ffffffffffffff0
	cmp x3, x8
	b.lo .LBB8_5
	lsr x9, x1, #8
	mov x8, x0
	lsl x0, x9, #4
	cbz x9, .LBB8_4
	lsr x17, x1, #4
	mov w10, #14
	mov w11, #13
	mov w12, #12
	mov w13, #11
	mov w14, #10
	mov w15, #6
	madd x10, x17, x10, x2
	lsl x1, x17, #3
	lsl x7, x17, #1
	lsl x5, x17, #2
	sub x18, x2, x17
	madd x11, x17, x11, x2
	sub x3, x1, x17
	add x16, x2, x17
	mov x9, xzr
	add x3, x2, x3
	add x4, x16, x5
	madd x12, x17, x12, x2
	add x5, x2, x5
	add x6, x16, x7
	add x7, x2, x7
	add x8, x8, #128
	madd x13, x17, x13, x2
	madd x14, x17, x14, x2
	madd x15, x17, x15, x2
	add x17, x18, x17, lsl #4
	add x18, x16, x1
	add x1, x2, x1
.LBB8_3:
	ldp q17, q18, [x8, #-128]
	add x19, x2, x9
	ldp q19, q20, [x8]
	ldr q16, [x8, #32]
	ldp q1, q0, [x8, #-96]
	ldr q7, [x8, #112]
	ldp q22, q23, [x8, #-64]
	zip1 v21.16b, v17.16b, v19.16b
	zip1 v4.16b, v18.16b, v20.16b
	zip2 v19.16b, v17.16b, v19.16b
	ldp q3, q2, [x8, #-32]
	zip1 v26.16b, v1.16b, v16.16b
	ldp q5, q24, [x8, #48]
	zip2 v17.16b, v18.16b, v20.16b
	ldp q25, q6, [x8, #80]
	zip2 v1.16b, v1.16b, v16.16b
	zip1 v31.16b, v2.16b, v7.16b
	zip2 v2.16b, v2.16b, v7.16b
	add x8, x8, #256
	zip1 v27.16b, v22.16b, v24.16b
	zip1 v29.16b, v0.16b, v5.16b
	zip2 v18.16b, v22.16b, v24.16b
	zip1 v28.16b, v3.16b, v6.16b
	zip1 v30.16b, v23.16b, v25.16b
	zip2 v3.16b, v3.16b, v6.16b
	zip2 v20.16b, v23.16b, v25.16b
	zip2 v0.16b, v0.16b, v5.16b
	zip1 v8.16b, v21.16b, v27.16b
	zip1 v11.16b, v29.16b, v31.16b
	zip2 v21.16b, v21.16b, v27.16b
	zip1 v9.16b, v26.16b, v28.16b
	zip1 v10.16b, v4.16b, v30.16b
	zip2 v22.16b, v26.16b, v28.16b
	zip2 v4.16b, v4.16b, v30.16b
	zip2 v23.16b, v29.16b, v31.16b
	zip1 v16.16b, v19.16b, v18.16b
	zip1 v27.16b, v1.16b, v3.16b
	zip1 v26.16b, v17.16b, v20.16b
	zip2 v18.16b, v19.16b, v18.16b
	zip2 v1.16b, v1.16b, v3.16b
	zip1 v12.16b, v8.16b, v9.16b
	zip2 v24.16b, v8.16b, v9.16b
	zip1 v6.16b, v21.16b, v22.16b
	zip1 v13.16b, v10.16b, v11.16b
	zip2 v25.16b, v10.16b, v11.16b
	zip2 v21.16b, v21.16b, v22.16b
	zip1 v7.16b, v4.16b, v23.16b
	zip2 v22.16b, v4.16b, v23.16b
	zip2 v5.16b, v12.16b, v13.16b
	st2 { v12.8b, v13.8b }, [x19]
	add x19, x7, x9
	zip2 v28.16b, v24.16b, v25.16b
	zip2 v4.16b, v6.16b, v7.16b
	zip2 v3.16b, v21.16b, v22.16b
	str q5, [x16, x9]
	zip1 v5.16b, v0.16b, v2.16b
	zip2 v0.16b, v0.16b, v2.16b
	st2 { v24.8b, v25.8b }, [x19]
	zip1 v23.16b, v16.16b, v27.16b
	add x19, x5, x9
	str q28, [x6, x9]
	zip1 v24.16b, v26.16b, v5.16b
	st2 { v6.8b, v7.8b }, [x19]
	zip2 v6.16b, v17.16b, v20.16b
	zip2 v16.16b, v16.16b, v27.16b
	add x19, x15, x9
	zip2 v17.16b, v26.16b, v5.16b
	str q4, [x4, x9]
	st2 { v21.8b, v22.8b }, [x19]
	add x19, x1, x9
	zip2 v2.16b, v23.16b, v24.16b
	str q3, [x3, x9]
	zip1 v3.16b, v18.16b, v1.16b
	st2 { v23.8b, v24.8b }, [x19]
	zip1 v4.16b, v6.16b, v0.16b
	zip2 v5.16b, v16.16b, v17.16b
	add x19, x14, x9
	str q2, [x18, x9]
	zip2 v1.16b, v18.16b, v1.16b
	zip2 v2.16b, v6.16b, v0.16b
	st2 { v16.8b, v17.8b }, [x19]
	add x19, x12, x9
	zip2 v0.16b, v3.16b, v4.16b
	str q5, [x13, x9]
	st2 { v3.8b, v4.8b }, [x19]
	add x19, x10, x9
	zip2 v3.16b, v1.16b, v2.16b
	str q0, [x11, x9]
	st2 { v1.8b, v2.8b }, [x19]
	str q3, [x17, x9]
	add x9, x9, #16
	cmp x0, x9
	b.ne .LBB8_3
.LBB8_4:
	ldp x29, x30, [sp, #48]
	ldr x19, [sp, #64]
	ldp d9, d8, [sp, #32]
	ldp d11, d10, [sp, #16]
	ldp d13, d12, [sp], #80
	ret
.LBB8_5:
	adrp x0, .Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	add x0, x0, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	adrp x2, .Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	add x2, x2, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	mov w1, #48
	bl core::panicking::panic
