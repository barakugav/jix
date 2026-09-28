jix_probe::byte_shuffle::decode_impl_generic:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	cbz x4, .LBB8_12
	udiv x8, x1, x4
	cmp x5, x8
	b.hs .LBB8_11
	madd x9, x5, x4, x2
	and x10, x4, #0xfffffffffffffffc
	add x11, x9, #1
	b .LBB8_4
.LBB8_3:
	add x5, x5, #1
	add x11, x11, x4
	add x9, x9, x4
	cmp x5, x8
	b.hs .LBB8_11
.LBB8_4:
	cmp x4, #4
	add x12, x0, x5
	b.hs .LBB8_6
	mov x15, xzr
	b .LBB8_9
.LBB8_6:
	mov x13, xzr
	mov x14, x11
	and x15, x4, #0xfffffffffffffffc
.LBB8_7:
	mul x16, x13, x8
	orr x17, x13, #0x1
	orr x18, x13, #0x2
	orr x1, x13, #0x3
	subs x15, x15, #4
	add x13, x13, #4
	mul x17, x17, x8
	mul x18, x18, x8
	ldrb w16, [x12, x16]
	mul x1, x1, x8
	ldrb w17, [x12, x17]
	sturb w16, [x14, #-1]
	ldrb w18, [x12, x18]
	strb w17, [x14]
	ldrb w16, [x12, x1]
	strb w18, [x14, #1]
	strb w16, [x14, #2]
	add x14, x14, #4
	b.ne .LBB8_7
	cmp x4, x10
	and x15, x4, #0xfffffffffffffffc
	b.eq .LBB8_3
.LBB8_9:
	add x13, x9, x15
	sub x14, x4, x15
.LBB8_10:
	mul x16, x15, x8
	subs x14, x14, #1
	add x15, x15, #1
	ldrb w16, [x12, x16]
	strb w16, [x13], #1
	b.ne .LBB8_10
	b .LBB8_3
.LBB8_11:
	ldp x29, x30, [sp], #16
	ret
.LBB8_12:
	adrp x0, .Lanon.4a3613c5db160acb30c4e33b3e9c0513.1
	add x0, x0, :lo12:.Lanon.4a3613c5db160acb30c4e33b3e9c0513.1
	bl core::panicking::panic_const::panic_const_div_by_zero
