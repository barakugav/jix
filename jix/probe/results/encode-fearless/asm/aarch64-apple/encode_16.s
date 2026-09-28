_probe_byte_shuffle_encode_16:
Lfunc_begin8:
	stp d9, d8, [sp, #-48]!
	stp x20, x19, [sp, #16]
	stp x29, x30, [sp, #32]
	add x29, sp, #32
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
	ldp q0, q1, [x8, #-128]
	ldp q2, q3, [x8]
	zip1.16b v16, v0, v2
	zip2.16b v0, v0, v2
	zip1.16b v17, v1, v3
	zip2.16b v1, v1, v3
	ldp q2, q3, [x8, #-96]
	ldp q4, q5, [x8, #32]
	zip1.16b v18, v2, v4
	zip2.16b v2, v2, v4
	zip1.16b v19, v3, v5
	zip2.16b v3, v3, v5
	ldp q4, q5, [x8, #-64]
	ldp q6, q7, [x8, #64]
	zip1.16b v20, v4, v6
	zip2.16b v4, v4, v6
	zip1.16b v21, v5, v7
	zip2.16b v5, v5, v7
	ldp q6, q7, [x8, #-32]
	ldp q22, q23, [x8, #96]
	zip1.16b v24, v6, v22
	zip2.16b v6, v6, v22
	zip1.16b v22, v7, v23
	zip2.16b v7, v7, v23
	zip1.16b v23, v16, v20
	zip2.16b v16, v16, v20
	zip1.16b v20, v0, v4
	zip1.16b v25, v17, v21
	zip2.16b v17, v17, v21
	zip1.16b v21, v1, v5
	zip1.16b v26, v18, v24
	zip2.16b v18, v18, v24
	zip1.16b v24, v2, v6
	zip1.16b v27, v19, v22
	zip2.16b v19, v19, v22
	zip1.16b v22, v3, v7
	zip1.16b v28, v23, v26
	zip2.16b v30, v23, v26
	zip1.16b v8, v16, v18
	zip1.16b v29, v25, v27
	zip2.16b v31, v25, v27
	zip1.16b v9, v17, v19
	zip2.16b v23, v28, v29
	add x20, x2, x9
	st2.8b { v28, v29 }, [x20]
	str q23, [x16, x9]
	zip2.16b v23, v30, v31
	add x20, x19, x9
	st2.8b { v30, v31 }, [x20]
	str q23, [x7, x9]
	zip2.16b v23, v8, v9
	add x20, x6, x9
	st2.8b { v8, v9 }, [x20]
	str q23, [x5, x9]
	zip2.16b v25, v16, v18
	add x20, x4, x9
	zip1.16b v27, v20, v24
	zip2.16b v23, v20, v24
	zip2.16b v26, v17, v19
	zip1.16b v28, v21, v22
	zip2.16b v24, v21, v22
	zip2.16b v16, v25, v26
	st2.8b { v25, v26 }, [x20]
	str q16, [x3, x9]
	zip2.16b v16, v27, v28
	add x20, x1, x9
	st2.8b { v27, v28 }, [x20]
	str q16, [x17, x9]
	zip2.16b v16, v23, v24
	add x20, x15, x9
	st2.8b { v23, v24 }, [x20]
	str q16, [x14, x9]
	zip2.16b v0, v0, v4
	add x20, x13, x9
	zip2.16b v1, v1, v5
	zip2.16b v2, v2, v6
	zip2.16b v3, v3, v7
	zip1.16b v4, v0, v2
	zip1.16b v5, v1, v3
	zip2.16b v6, v4, v5
	st2.8b { v4, v5 }, [x20]
	str q6, [x11, x9]
	add x20, x10, x9
	zip2.16b v4, v0, v2
	zip2.16b v5, v1, v3
	st2.8b { v4, v5 }, [x20]
	zip2.16b v0, v4, v5
	str q0, [x12, x9]
	add x9, x9, #16
	add x8, x8, #256
	cmp x0, x9
	b.ne LBB8_3
LBB8_4:
	ldp x29, x30, [sp, #32]
	ldp x20, x19, [sp, #16]
	ldp d9, d8, [sp], #48
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
