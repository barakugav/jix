jix_probe::byte_shuffle::decode_impl_generic:
Lfunc_begin8:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	cbz x4, LBB8_12
	udiv x8, x1, x4
	cmp x5, x8
	b.hs LBB8_11
	and x9, x4, #0xfffffffffffffffc
	madd x10, x5, x4, x2
	add x11, x10, #1
	b LBB8_4
LBB8_3:
	add x5, x5, #1
	add x11, x11, x4
	add x10, x10, x4
	cmp x5, x8
	b.hs LBB8_11
LBB8_4:
	add x12, x0, x5
	cmp x4, #4
	b.hs LBB8_6
	mov x13, #0
	b LBB8_9
LBB8_6:
	mov x13, #0
	mov x14, x11
	and x15, x4, #0xfffffffffffffffc
LBB8_7:
	orr x16, x13, #0x1
	orr x17, x13, #0x2
	orr x1, x13, #0x3
	mul x2, x13, x8
	mul x16, x16, x8
	mul x17, x17, x8
	ldrb w2, [x12, x2]
	ldrb w16, [x12, x16]
	mul x1, x1, x8
	ldrb w17, [x12, x17]
	ldrb w1, [x12, x1]
	sturb w2, [x14, #-1]
	strb w16, [x14]
	strb w17, [x14, #1]
	strb w1, [x14, #2]
	add x13, x13, #4
	add x14, x14, #4
	subs x15, x15, #4
	b.ne LBB8_7
	and x13, x4, #0xfffffffffffffffc
	cmp x4, x9
	b.eq LBB8_3
LBB8_9:
	add x14, x10, x13
	sub x15, x4, x13
LBB8_10:
	mul x16, x13, x8
	add x13, x13, #1
	ldrb w16, [x12, x16]
	strb w16, [x14], #1
	subs x15, x15, #1
	b.ne LBB8_10
	b LBB8_3
LBB8_11:
	ldp x29, x30, [sp], #16
	ret
LBB8_12:
Lloh12:
	adrp x0, l_anon.bc1f2f7c2f312fc2f015440ea3f2cec5.1@PAGE
Lloh13:
	add x0, x0, l_anon.bc1f2f7c2f312fc2f015440ea3f2cec5.1@PAGEOFF
	bl core::panicking::panic_const::panic_const_div_by_zero
