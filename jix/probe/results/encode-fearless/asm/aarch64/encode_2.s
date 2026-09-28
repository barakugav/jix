probe_byte_shuffle_encode_2:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffe
	cmp x3, x8
	b.lo .LBB9_5
	lsr x9, x1, #8
	lsr x8, x1, #1
	cbz x9, .LBB9_4
	add x12, x8, x2
	add x10, x2, #112
	add x11, x0, #128
	add x12, x12, #64
.LBB9_3:
	sub x13, x11, #128
	ldp q4, q5, [x11]
	ld2 { v0.16b, v1.16b }, [x13]
	sub x13, x11, #96
	subs x9, x9, #1
	ld2 { v2.16b, v3.16b }, [x13]
	sub x13, x11, #64
	stur q0, [x10, #-112]
	stur q1, [x12, #-64]
	ld2 { v0.16b, v1.16b }, [x13]
	sub x13, x11, #32
	stur q2, [x10, #-96]
	ld2 { v6.16b, v7.16b }, [x13]
	add x13, x11, #32
	stur q3, [x12, #-48]
	uzp1 v2.16b, v4.16b, v5.16b
	stur q0, [x10, #-80]
	stur q1, [x12, #-32]
	uzp2 v0.16b, v4.16b, v5.16b
	ld2 { v3.16b, v4.16b }, [x13]
	add x13, x11, #64
	stur q6, [x10, #-64]
	stur q7, [x12, #-16]
	stur q2, [x10, #-48]
	ld2 { v1.16b, v2.16b }, [x13]
	add x13, x11, #96
	add x11, x11, #256
	str q0, [x12]
	ld2 { v5.16b, v6.16b }, [x13]
	stur q3, [x10, #-32]
	str q4, [x12, #16]
	stur q1, [x10, #-16]
	str q2, [x12, #32]
	str q5, [x10], #128
	str q6, [x12, #48]
	add x12, x12, #128
	b.ne .LBB9_3
.LBB9_4:
	and x0, x8, #0x3fffffffffffff80
	ldp x29, x30, [sp], #16
	ret
.LBB9_5:
	adrp x0, .Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	add x0, x0, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.0
	adrp x2, .Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	add x2, x2, :lo12:.Lanon.e7c1b649830da2e0ce998c14243d99dc.3
	mov w1, #48
	bl core::panicking::panic
