probe_byte_shuffle_encode_8:
	stp x29, x30, [sp, #-32]!
	str x19, [sp, #16]
	mov x29, sp
	and x8, x1, #0x7ffffffffffffff8
	cmp x3, x8
	b.lo .LBB11_5
	lsr x12, x1, #8
	lsr x8, x1, #3
	cbz x12, .LBB11_4
	mov w10, #6
	add x11, x2, x8, lsl #2
	add x14, x2, x8, lsl #1
	madd x10, x8, x10, x2
	sub x15, x2, x8
	lsl x12, x12, #5
	add x11, x11, #16
	add x14, x14, #16
	add x17, x8, x2
	add x15, x15, x8, lsl #3
	mov x9, xzr
	add x13, x11, x8
	add x16, x14, x8
	add x17, x17, #16
	add x18, x0, #128
.LBB11_3:
	sub x0, x18, #128
	sub x1, x18, #64
	add x3, x14, x9
	ldp q16, q17, [x18]
	ld4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0]
	add x0, x18, #32
	add x4, x16, x9
	add x5, x11, x9
	add x6, x13, x9
	add x7, x10, x9
	add x19, x15, x9
	ld4 { v4.16b, v5.16b, v6.16b, v7.16b }, [x1]
	uzp1 v18.16b, v16.16b, v17.16b
	uzp2 v16.16b, v16.16b, v17.16b
	add x1, x2, x9
	ld2 { v22.16b, v23.16b }, [x0]
	add x0, x18, #64
	add x18, x18, #256
	uzp1 v19.16b, v0.16b, v4.16b
	uzp2 v20.16b, v0.16b, v4.16b
	uzp1 v21.16b, v1.16b, v5.16b
	uzp2 v24.16b, v1.16b, v5.16b
	uzp1 v17.16b, v2.16b, v6.16b
	uzp2 v25.16b, v2.16b, v6.16b
	uzp1 v26.16b, v3.16b, v7.16b
	uzp2 v0.16b, v3.16b, v7.16b
	ld4 { v1.16b, v2.16b, v3.16b, v4.16b }, [x0]
	uzp1 v5.16b, v18.16b, v22.16b
	uzp1 v6.16b, v16.16b, v23.16b
	uzp2 v7.16b, v18.16b, v22.16b
	uzp2 v16.16b, v16.16b, v23.16b
	add x0, x17, x9
	str q19, [x1]
	stur q21, [x0, #-16]
	add x9, x9, #32
	stur q17, [x3, #-16]
	cmp x12, x9
	stur q26, [x4, #-16]
	uzp1 v17.16b, v5.16b, v1.16b
	uzp1 v18.16b, v6.16b, v2.16b
	stur q20, [x5, #-16]
	uzp1 v19.16b, v7.16b, v3.16b
	uzp1 v20.16b, v16.16b, v4.16b
	stur q24, [x6, #-16]
	str q25, [x7]
	str q0, [x19]
	uzp2 v0.16b, v5.16b, v1.16b
	uzp2 v5.16b, v6.16b, v2.16b
	uzp2 v6.16b, v7.16b, v3.16b
	uzp2 v1.16b, v16.16b, v4.16b
	str q17, [x1, #16]
	str q18, [x0]
	str q19, [x3]
	str q20, [x4]
	str q0, [x5]
	str q5, [x6]
	str q6, [x7, #16]
	str q1, [x19, #16]
	b.ne .LBB11_3
.LBB11_4:
	and x0, x8, #0xfffffffffffffe0
	ldr x19, [sp, #16]
	ldp x29, x30, [sp], #32
	ret
.LBB11_5:
	adrp x0, .Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	add x0, x0, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	adrp x2, .Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	add x2, x2, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	mov w1, #48
	bl core::panicking::panic
