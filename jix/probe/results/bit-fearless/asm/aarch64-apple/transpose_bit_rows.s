_probe_bit_shuffle_transpose_bit_rows:
Lfunc_begin5:
	and x8, x1, #0x7ffffffffffffff8
	cmp x3, x8
	b.lo LBB5_5
	lsr x9, x1, #7
	lsl x8, x9, #4
	cbz x9, LBB5_4
	mov x9, #0
	lsr x7, x1, #3
	lsl x6, x7, #1
	lsl x17, x7, #2
	add x5, x6, x7
	add x10, x2, x5
	lsl x3, x5, #1
	lsl x11, x7, #3
	sub x1, x11, x7
	add x11, x2, x1
	add x12, x2, x3
	add x4, x17, x7
	add x13, x2, x4
	add x14, x2, x17
	add x15, x2, x6
	add x16, x2, x7
	add x17, x0, x17
	add x1, x0, x1
	add x3, x0, x3
	add x4, x0, x4
	movi.16b v0, #15
	add x5, x0, x5
	movi.16b v1, #51
	add x6, x0, x6
	movi.16b v2, #85
	add x7, x0, x7
LBB5_3:
	ldr q3, [x7, x9]
	ldr q4, [x6, x9]
	ldr q5, [x5, x9]
	ldr q6, [x4, x9]
	ldr q7, [x3, x9]
	ldr q16, [x1, x9]
	ldr q17, [x17, x9]
	ldr q18, [x0, x9]
	ushr.2d v19, v18, #4
	eor.16b v19, v17, v19
	and.16b v19, v19, v0
	eor.16b v20, v19, v17
	shl.2d v21, v19, #4
	eor.16b v22, v21, v18
	ushr.2d v23, v3, #4
	eor.16b v23, v6, v23
	and.16b v23, v23, v0
	eor.16b v6, v23, v6
	shl.2d v23, v23, #4
	eor.16b v3, v23, v3
	ushr.2d v23, v4, #4
	eor.16b v23, v7, v23
	and.16b v23, v23, v0
	eor.16b v24, v23, v7
	shl.2d v25, v23, #4
	eor.16b v26, v25, v4
	ushr.2d v27, v5, #4
	eor.16b v27, v16, v27
	and.16b v27, v27, v0
	eor.16b v28, v27, v16
	shl.2d v29, v27, #4
	eor.16b v30, v29, v5
	ushr.2d v31, v22, #2
	eor3.16b v31, v25, v4, v31
	and.16b v31, v31, v1
	eor3.16b v4, v25, v4, v31
	shl.2d v25, v31, #2
	eor3.16b v18, v21, v18, v25
	ushr.2d v21, v3, #2
	eor3.16b v5, v29, v5, v21
	and.16b v5, v5, v1
	shl.2d v21, v5, #2
	ushr.2d v29, v20, #2
	eor3.16b v29, v23, v7, v29
	and.16b v29, v29, v1
	eor3.16b v7, v23, v7, v29
	shl.2d v23, v29, #2
	eor3.16b v17, v19, v17, v23
	ushr.2d v19, v6, #2
	eor3.16b v16, v27, v16, v19
	and.16b v16, v16, v1
	shl.2d v19, v16, #2
	ushr.2d v18, v18, #1
	eor3.16b v18, v21, v3, v18
	and.16b v18, v18, v2
	eor3.16b v3, v21, v3, v18
	add.2d v18, v18, v18
	eor3.16b v18, v25, v22, v18
	ushr.2d v4, v4, #1
	eor3.16b v4, v5, v30, v4
	and.16b v4, v4, v2
	eor3.16b v5, v5, v30, v4
	add.2d v4, v4, v4
	eor3.16b v4, v31, v26, v4
	ushr.2d v17, v17, #1
	eor3.16b v17, v19, v6, v17
	and.16b v17, v17, v2
	eor3.16b v6, v19, v6, v17
	add.2d v17, v17, v17
	eor3.16b v17, v23, v20, v17
	ushr.2d v7, v7, #1
	eor3.16b v7, v16, v28, v7
	and.16b v7, v7, v2
	eor3.16b v16, v16, v28, v7
	str q18, [x2, x9]
	str q3, [x16, x9]
	str q4, [x15, x9]
	str q5, [x10, x9]
	add.2d v3, v7, v7
	eor3.16b v3, v29, v24, v3
	str q17, [x14, x9]
	str q6, [x13, x9]
	str q3, [x12, x9]
	str q16, [x11, x9]
	add x9, x9, #16
	cmp x8, x9
	b.ne LBB5_3
LBB5_4:
	mov x0, x8
	ret
LBB5_5:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
Lloh48:
	adrp x0, l_anon.fc929a10041e28f5b0c52c40f9d7b60a.1@PAGE
Lloh49:
	add x0, x0, l_anon.fc929a10041e28f5b0c52c40f9d7b60a.1@PAGEOFF
Lloh50:
	adrp x2, l_anon.fc929a10041e28f5b0c52c40f9d7b60a.2@PAGE
Lloh51:
	add x2, x2, l_anon.fc929a10041e28f5b0c52c40f9d7b60a.2@PAGEOFF
	mov w1, #36
	bl core::panicking::panic
