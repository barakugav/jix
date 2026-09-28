probe_byte_shuffle_encode_4:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffc
	cmp x3, x8
	b.lo .LBB10_5
	lsr x9, x1, #8
	lsr x8, x1, #2
	cbz x9, .LBB10_4
	add x10, x2, x8, lsl #1
	add x11, x8, x2
	add x12, x2, #48
	add x11, x11, #32
	add x13, x0, #128
	add x10, x10, #32
	add x14, x10, x8
.LBB10_3:
	sub x15, x13, #128
	ldp q4, q5, [x13]
	ld4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x15]
	sub x15, x13, #64
	subs x9, x9, #1
	ld4 { v16.16b, v17.16b, v18.16b, v19.16b }, [x15]
	add x15, x13, #32
	uzp1 v6.16b, v4.16b, v5.16b
	uzp2 v4.16b, v4.16b, v5.16b
	ld2 { v20.16b, v21.16b }, [x15]
	add x15, x13, #64
	add x13, x13, #256
	stur q0, [x12, #-48]
	stur q1, [x11, #-32]
	stur q2, [x10, #-32]
	stur q3, [x14, #-32]
	uzp1 v0.16b, v6.16b, v20.16b
	uzp1 v1.16b, v4.16b, v21.16b
	uzp2 v2.16b, v6.16b, v20.16b
	uzp2 v3.16b, v4.16b, v21.16b
	stur q16, [x12, #-32]
	ld4 { v4.16b, v5.16b, v6.16b, v7.16b }, [x15]
	stur q17, [x11, #-16]
	stur q18, [x10, #-16]
	stur q19, [x14, #-16]
	stur q0, [x12, #-16]
	str q1, [x11]
	str q2, [x10]
	str q3, [x14]
	str q4, [x12], #64
	str q5, [x11, #16]
	add x11, x11, #64
	str q6, [x10, #16]
	add x10, x10, #64
	str q7, [x14, #16]
	add x14, x14, #64
	b.ne .LBB10_3
.LBB10_4:
	and x0, x8, #0x1fffffffffffffc0
	ldp x29, x30, [sp], #16
	ret
.LBB10_5:
	adrp x0, .Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	add x0, x0, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	adrp x2, .Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	add x2, x2, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	mov w1, #48
	bl core::panicking::panic
