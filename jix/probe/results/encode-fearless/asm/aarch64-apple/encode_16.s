_probe_byte_shuffle_encode_16:
Lfunc_begin8:
	stp d11, d10, [sp, #-64]!
	stp d9, d8, [sp, #16]
	stp x20, x19, [sp, #32]
	stp x29, x30, [sp, #48]
	add x29, sp, #48
	and x9, x1, #0x7ffffffffffffff0
	cmp x3, x9
	b.lo LBB8_5
	mov x8, x0
	lsr x9, x1, #8
	lsl x0, x9, #4
	cbz x9, LBB8_4
	mov x9, #0
	lsr x4, x1, #4
	lsl x19, x4, #1
	lsl x6, x4, #2
	mov w5, #6
	lsl x3, x4, #3
	mov w15, #10
	mov w14, #11
	mov w13, #12
	mov w10, #14
	madd x10, x4, x10, x2
	mov w11, #13
	sub x12, x2, x4
	madd x11, x4, x11, x2
	add x12, x12, x4, lsl #4
	madd x13, x4, x13, x2
	madd x14, x4, x14, x2
	madd x15, x4, x15, x2
	add x16, x2, x4
	add x17, x16, x3
	add x1, x2, x3
	sub x3, x3, x4
	add x3, x2, x3
	madd x4, x4, x5, x2
	add x5, x16, x6
	add x6, x2, x6
	add x7, x16, x19
	add x19, x2, x19
	add x8, x8, #128
LBB8_3:
	sub x20, x8, #128
	ld4.16b { v0, v1, v2, v3 }, [x20]
	sub x20, x8, #64
	ld4.16b { v4, v5, v6, v7 }, [x20]
	ldp q16, q17, [x8]
	add x20, x8, #64
	ld4.16b { v18, v19, v20, v21 }, [x20]
	add x20, x8, #32
	ld2.16b { v22, v23 }, [x20]
	uzp1.16b v24, v16, v17
	uzp2.16b v16, v16, v17
	uzp1.16b v17, v24, v22
	uzp2.16b v24, v24, v22
	uzp1.16b v25, v16, v23
	uzp2.16b v16, v16, v23
	uzp1.16b v22, v0, v4
	uzp2.16b v23, v0, v4
	uzp1.16b v26, v17, v18
	uzp1.16b v27, v1, v5
	uzp1.16b v28, v25, v19
	uzp1.16b v29, v2, v6
	uzp1.16b v30, v24, v20
	uzp1.16b v31, v3, v7
	uzp1.16b v8, v16, v21
	uzp1.16b v9, v22, v26
	uzp1.16b v10, v27, v28
	str q9, [x2, x9]
	uzp1.16b v9, v29, v30
	str q10, [x16, x9]
	uzp1.16b v10, v31, v8
	str q9, [x19, x9]
	str q10, [x7, x9]
	uzp2.16b v17, v17, v18
	uzp1.16b v9, v23, v17
	str q9, [x6, x9]
	uzp2.16b v9, v1, v5
	uzp2.16b v25, v25, v19
	uzp1.16b v10, v9, v25
	str q10, [x5, x9]
	uzp2.16b v10, v2, v6
	uzp2.16b v0, v3, v7
	uzp2.16b v1, v24, v20
	uzp1.16b v2, v10, v1
	str q2, [x4, x9]
	uzp2.16b v2, v16, v21
	uzp1.16b v3, v0, v2
	str q3, [x3, x9]
	uzp2.16b v3, v22, v26
	str q3, [x1, x9]
	uzp2.16b v3, v27, v28
	str q3, [x17, x9]
	uzp2.16b v3, v29, v30
	str q3, [x15, x9]
	uzp2.16b v3, v31, v8
	str q3, [x14, x9]
	uzp2.16b v3, v23, v17
	str q3, [x13, x9]
	uzp2.16b v3, v9, v25
	str q3, [x11, x9]
	uzp2.16b v1, v10, v1
	str q1, [x10, x9]
	uzp2.16b v0, v0, v2
	str q0, [x12, x9]
	add x9, x9, #16
	add x8, x8, #256
	cmp x0, x9
	b.ne LBB8_3
LBB8_4:
	ldp x29, x30, [sp, #48]
	ldp x20, x19, [sp, #32]
	ldp d9, d8, [sp, #16]
	ldp d11, d10, [sp], #64
	ret
LBB8_5:
Lloh24:
	adrp x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGE
Lloh25:
	add x0, x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGEOFF
Lloh26:
	adrp x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGE
Lloh27:
	add x2, x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
