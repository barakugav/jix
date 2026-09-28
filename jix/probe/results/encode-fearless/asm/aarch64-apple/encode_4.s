_probe_byte_shuffle_encode_4:
Lfunc_begin10:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffc
	cmp x3, x8
	b.lo LBB10_5
	lsr x8, x1, #2
	lsr x9, x1, #8
	cbz x9, LBB10_4
	add x10, x2, x8, lsl #1
	add x10, x10, #32
	add x11, x8, x2
	add x11, x11, #32
	add x12, x2, #48
	add x13, x0, #128
	add x14, x10, x8
LBB10_3:
	sub x15, x13, #128
	ld4.16b { v0, v1, v2, v3 }, [x15]
	sub x15, x13, #64
	ld4.16b { v4, v5, v6, v7 }, [x15]
	stur q0, [x12, #-48]
	stur q1, [x11, #-32]
	stur q2, [x10, #-32]
	stur q3, [x14, #-32]
	stur q4, [x12, #-32]
	stur q5, [x11, #-16]
	stur q6, [x10, #-16]
	add x15, x13, #32
	ld2.16b { v0, v1 }, [x15]
	stur q7, [x14, #-16]
	ldp q2, q3, [x13]
	uzp1.16b v4, v2, v3
	uzp2.16b v2, v2, v3
	uzp1.16b v3, v4, v0
	uzp2.16b v4, v4, v0
	uzp1.16b v5, v2, v1
	uzp2.16b v0, v2, v1
	stur q3, [x12, #-16]
	str q5, [x11]
	add x15, x13, #64
	ld4.16b { v16, v17, v18, v19 }, [x15]
	str q4, [x10]
	str q0, [x14]
	str q16, [x12], #64
	str q17, [x11, #16]
	str q18, [x10, #16]
	add x10, x10, #64
	add x11, x11, #64
	add x13, x13, #256
	str q19, [x14, #16]
	add x14, x14, #64
	subs x9, x9, #1
	b.ne LBB10_3
LBB10_4:
	and x0, x8, #0x1fffffffffffffc0
	ldp x29, x30, [sp], #16
	ret
LBB10_5:
Lloh32:
	adrp x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGE
Lloh33:
	add x0, x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGEOFF
Lloh34:
	adrp x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGE
Lloh35:
	add x2, x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
