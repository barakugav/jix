probe_bit_shuffle_transpose_bit_rows:
	and x8, x1, #0x7ffffffffffffff8
	cmp x3, x8
	b.lo .LBB5_5
	lsr x9, x1, #7
	lsl x8, x9, #4
	cbz x9, .LBB5_4
	lsr x6, x1, #3
	movi v0.16b, #15
	movi v1.16b, #51
	movi v2.16b, #85
	mov x9, xzr
	lsl x5, x6, #1
	lsl x13, x6, #2
	lsl x11, x6, #3
	add x10, x2, x6
	add x4, x5, x6
	add x18, x13, x6
	sub x17, x11, x6
	lsl x3, x4, #1
	add x11, x2, x13
	add x12, x2, x5
	add x13, x0, x13
	add x14, x2, x17
	add x15, x2, x18
	add x16, x2, x4
	add x17, x0, x17
	add x18, x0, x18
	add x1, x2, x3
	add x3, x0, x3
	add x4, x0, x4
	add x5, x0, x5
	add x6, x0, x6
.LBB5_3:
	ldr q3, [x0, x9]
	ldr q4, [x6, x9]
	ldr q6, [x5, x9]
	ldr q7, [x13, x9]
	ldr q17, [x18, x9]
	ldr q19, [x4, x9]
	ushr v5.2d, v3.2d, #4
	ushr v16.2d, v4.2d, #4
	ushr v18.2d, v6.2d, #4
	ldr q20, [x3, x9]
	ushr v21.2d, v19.2d, #4
	ldr q22, [x17, x9]
	eor v5.16b, v7.16b, v5.16b
	eor v16.16b, v17.16b, v16.16b
	eor v18.16b, v20.16b, v18.16b
	eor v21.16b, v22.16b, v21.16b
	and v5.16b, v5.16b, v0.16b
	and v16.16b, v16.16b, v0.16b
	and v18.16b, v18.16b, v0.16b
	and v21.16b, v21.16b, v0.16b
	shl v23.2d, v5.2d, #4
	shl v24.2d, v16.2d, #4
	eor v5.16b, v5.16b, v7.16b
	eor v16.16b, v16.16b, v17.16b
	eor v17.16b, v18.16b, v20.16b
	eor v3.16b, v23.16b, v3.16b
	shl v23.2d, v18.2d, #4
	eor v4.16b, v24.16b, v4.16b
	shl v24.2d, v21.2d, #4
	ushr v18.2d, v5.2d, #2
	eor v21.16b, v21.16b, v22.16b
	ushr v22.2d, v16.2d, #2
	eor v6.16b, v23.16b, v6.16b
	ushr v23.2d, v3.2d, #2
	eor v7.16b, v24.16b, v19.16b
	ushr v19.2d, v4.2d, #2
	eor v18.16b, v17.16b, v18.16b
	eor v22.16b, v21.16b, v22.16b
	eor v23.16b, v6.16b, v23.16b
	eor v19.16b, v7.16b, v19.16b
	and v18.16b, v18.16b, v1.16b
	and v22.16b, v22.16b, v1.16b
	and v20.16b, v23.16b, v1.16b
	and v19.16b, v19.16b, v1.16b
	eor v17.16b, v18.16b, v17.16b
	eor v21.16b, v22.16b, v21.16b
	shl v23.2d, v20.2d, #2
	eor v6.16b, v20.16b, v6.16b
	shl v24.2d, v19.2d, #2
	eor v7.16b, v19.16b, v7.16b
	eor v3.16b, v23.16b, v3.16b
	shl v23.2d, v18.2d, #2
	ushr v19.2d, v6.2d, #1
	eor v4.16b, v24.16b, v4.16b
	shl v24.2d, v22.2d, #2
	ushr v22.2d, v17.2d, #1
	ushr v20.2d, v3.2d, #1
	eor v5.16b, v23.16b, v5.16b
	eor v19.16b, v7.16b, v19.16b
	eor v16.16b, v24.16b, v16.16b
	eor v22.16b, v21.16b, v22.16b
	eor v18.16b, v4.16b, v20.16b
	ushr v20.2d, v5.2d, #1
	and v19.16b, v19.16b, v2.16b
	and v22.16b, v22.16b, v2.16b
	and v18.16b, v18.16b, v2.16b
	eor v20.16b, v16.16b, v20.16b
	add v24.2d, v19.2d, v19.2d
	add v23.2d, v18.2d, v18.2d
	and v20.16b, v20.16b, v2.16b
	eor v4.16b, v18.16b, v4.16b
	eor v6.16b, v24.16b, v6.16b
	eor v3.16b, v23.16b, v3.16b
	add v18.2d, v20.2d, v20.2d
	str q3, [x2, x9]
	eor v3.16b, v19.16b, v7.16b
	str q4, [x10, x9]
	add v4.2d, v22.2d, v22.2d
	eor v5.16b, v18.16b, v5.16b
	str q6, [x12, x9]
	eor v6.16b, v20.16b, v16.16b
	str q3, [x16, x9]
	eor v3.16b, v4.16b, v17.16b
	eor v4.16b, v22.16b, v21.16b
	str q5, [x11, x9]
	str q6, [x15, x9]
	str q3, [x1, x9]
	str q4, [x14, x9]
	add x9, x9, #16
	cmp x8, x9
	b.ne .LBB5_3
.LBB5_4:
	mov x0, x8
	ret
.LBB5_5:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	adrp x0, .Lanon.cbe91f0acd525c5bdf949cb18d023737.1
	add x0, x0, :lo12:.Lanon.cbe91f0acd525c5bdf949cb18d023737.1
	adrp x2, .Lanon.cbe91f0acd525c5bdf949cb18d023737.2
	add x2, x2, :lo12:.Lanon.cbe91f0acd525c5bdf949cb18d023737.2
	mov w1, #36
	bl core::panicking::panic
