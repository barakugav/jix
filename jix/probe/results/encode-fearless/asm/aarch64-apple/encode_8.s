_probe_byte_shuffle_encode_8:
Lfunc_begin11:
	stp x22, x21, [sp, #-48]!
	stp x20, x19, [sp, #16]
	stp x29, x30, [sp, #32]
	add x29, sp, #32
	and x8, x1, #0x7ffffffffffffff8
	cmp x3, x8
	b.lo LBB11_5
	lsr x8, x1, #3
	lsr x10, x1, #8
	cbz x10, LBB11_4
	mov x9, #0
	mov w11, #6
	lsl x10, x10, #5
	madd x11, x8, x11, x2
	add x12, x2, x8, lsl #2
	add x12, x12, #16
	add x13, x12, x8
	add x14, x2, x8, lsl #1
	add x14, x14, #16
	add x15, x14, x8
	add x16, x8, x2
	add x16, x16, #16
	sub x1, x2, x8
	add x17, x0, #128
	add x0, x1, x8, lsl #3
LBB11_3:
	sub x1, x17, #128
	ld4.16b { v0, v1, v2, v3 }, [x1]
	sub x1, x17, #64
	ld4.16b { v4, v5, v6, v7 }, [x1]
	uzp1.16b v16, v0, v4
	uzp2.16b v17, v0, v4
	uzp1.16b v18, v1, v5
	uzp2.16b v19, v1, v5
	uzp1.16b v20, v2, v6
	uzp2.16b v21, v2, v6
	uzp1.16b v22, v3, v7
	uzp2.16b v0, v3, v7
	add x1, x2, x9
	str q16, [x1]
	add x3, x16, x9
	stur q18, [x3, #-16]
	add x4, x14, x9
	stur q20, [x4, #-16]
	add x5, x15, x9
	stur q22, [x5, #-16]
	add x6, x12, x9
	stur q17, [x6, #-16]
	add x7, x13, x9
	stur q19, [x7, #-16]
	add x19, x11, x9
	str q21, [x19]
	add x20, x0, x9
	str q0, [x20]
	ldp q0, q1, [x17]
	add x21, x17, #64
	ld4.16b { v2, v3, v4, v5 }, [x21]
	add x21, x17, #32
	ld2.16b { v6, v7 }, [x21]
	uzp1.16b v16, v0, v1
	uzp2.16b v0, v0, v1
	uzp1.16b v1, v16, v6
	uzp1.16b v17, v0, v7
	uzp1.16b v18, v1, v2
	str q18, [x1, #16]
	uzp1.16b v18, v17, v3
	str q18, [x3]
	uzp2.16b v16, v16, v6
	uzp1.16b v18, v16, v4
	str q18, [x4]
	uzp2.16b v0, v0, v7
	uzp1.16b v6, v0, v5
	str q6, [x5]
	uzp2.16b v1, v1, v2
	str q1, [x6]
	uzp2.16b v1, v17, v3
	str q1, [x7]
	uzp2.16b v1, v16, v4
	str q1, [x19, #16]
	add x9, x9, #32
	add x17, x17, #256
	uzp2.16b v0, v0, v5
	str q0, [x20, #16]
	cmp x10, x9
	b.ne LBB11_3
LBB11_4:
	and x0, x8, #0xfffffffffffffe0
	ldp x29, x30, [sp, #32]
	ldp x20, x19, [sp, #16]
	ldp x22, x21, [sp], #48
	ret
LBB11_5:
Lloh36:
	adrp x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGE
Lloh37:
	add x0, x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGEOFF
Lloh38:
	adrp x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGE
Lloh39:
	add x2, x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
