probe_byte_shuffle_encode_16:
	str d12, [sp, #-80]!
	stp d11, d10, [sp, #8]
	stp d9, d8, [sp, #24]
	stp x29, x30, [sp, #40]
	str x21, [sp, #56]
	stp x20, x19, [sp, #64]
	add x29, sp, #40
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
	ldp q0, q1, [x8]
	add x19, x8, #32
	ld2 { v2.16b, v3.16b }, [x19]
	sub x19, x8, #128
	sub x20, x8, #64
	add x21, x8, #64
	add x8, x8, #256
	ld4 { v16.16b, v17.16b, v18.16b, v19.16b }, [x19]
	uzp1 v4.16b, v0.16b, v1.16b
	uzp2 v0.16b, v0.16b, v1.16b
	ld4 { v20.16b, v21.16b, v22.16b, v23.16b }, [x20]
	ld4 { v24.16b, v25.16b, v26.16b, v27.16b }, [x21]
	uzp1 v1.16b, v4.16b, v2.16b
	uzp1 v5.16b, v0.16b, v3.16b
	uzp2 v4.16b, v4.16b, v2.16b
	uzp2 v0.16b, v0.16b, v3.16b
	uzp1 v2.16b, v16.16b, v20.16b
	uzp1 v6.16b, v17.16b, v21.16b
	uzp1 v28.16b, v18.16b, v22.16b
	uzp1 v31.16b, v19.16b, v23.16b
	uzp2 v30.16b, v16.16b, v20.16b
	uzp2 v9.16b, v17.16b, v21.16b
	uzp2 v11.16b, v18.16b, v22.16b
	uzp2 v16.16b, v19.16b, v23.16b
	uzp1 v3.16b, v1.16b, v24.16b
	uzp1 v7.16b, v5.16b, v25.16b
	uzp1 v29.16b, v4.16b, v26.16b
	uzp1 v8.16b, v0.16b, v27.16b
	uzp2 v1.16b, v1.16b, v24.16b
	uzp2 v5.16b, v5.16b, v25.16b
	uzp2 v4.16b, v4.16b, v26.16b
	uzp2 v0.16b, v0.16b, v27.16b
	uzp1 v10.16b, v2.16b, v3.16b
	uzp1 v12.16b, v6.16b, v7.16b
	uzp1 v17.16b, v28.16b, v29.16b
	uzp1 v18.16b, v31.16b, v8.16b
	uzp1 v19.16b, v30.16b, v1.16b
	uzp1 v20.16b, v9.16b, v5.16b
	uzp2 v2.16b, v2.16b, v3.16b
	uzp2 v3.16b, v6.16b, v7.16b
	uzp2 v6.16b, v28.16b, v29.16b
	str q10, [x2, x9]
	uzp2 v7.16b, v31.16b, v8.16b
	uzp2 v1.16b, v30.16b, v1.16b
	str q12, [x16, x9]
	str q17, [x7, x9]
	uzp1 v17.16b, v11.16b, v4.16b
	str q18, [x6, x9]
	uzp1 v18.16b, v16.16b, v0.16b
	uzp2 v0.16b, v16.16b, v0.16b
	str q19, [x5, x9]
	str q20, [x4, x9]
	str q17, [x15, x9]
	str q18, [x3, x9]
	str q2, [x1, x9]
	uzp2 v2.16b, v9.16b, v5.16b
	str q3, [x18, x9]
	uzp2 v3.16b, v11.16b, v4.16b
	str q6, [x14, x9]
	str q7, [x13, x9]
	str q1, [x12, x9]
	str q2, [x11, x9]
	str q3, [x10, x9]
	str q0, [x17, x9]
	add x9, x9, #16
	cmp x0, x9
	b.ne .LBB8_3
.LBB8_4:
	ldp x20, x19, [sp, #64]
	ldr x21, [sp, #56]
	ldp x29, x30, [sp, #40]
	ldp d9, d8, [sp, #24]
	ldp d11, d10, [sp, #8]
	ldr d12, [sp], #80
	ret
.LBB8_5:
	adrp x0, .Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	add x0, x0, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	adrp x2, .Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	add x2, x2, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	mov w1, #48
	bl core::panicking::panic
