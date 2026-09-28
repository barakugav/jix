jix_probe::byte_shuffle::encode_impl_generic:
Lfunc_begin9:
	stp x20, x19, [sp, #-32]!
	stp x29, x30, [sp, #16]
	add x29, sp, #16
	cbz x4, LBB9_17
	udiv x8, x1, x4
	cmp x5, x8
	b.hs LBB9_16
	mov x9, #0
	sub x10, x1, x5
	cmp x10, #7
	ccmp x4, #1, #0, hi
	cset w11, eq
	and x12, x10, #0x38
	and x13, x10, #0xffffffffffffffc0
	and x14, x10, #0xfffffffffffffff8
	add x15, x0, x5
	add x16, x15, #32
	neg x17, x14
	b LBB9_4
LBB9_3:
	add x9, x9, #1
	add x16, x16, #1
	add x15, x15, #1
	add x0, x0, #1
	cmp x9, x4
	b.eq LBB9_16
LBB9_4:
	madd x1, x9, x8, x2
	mov x6, x5
	cbz w11, LBB9_14
	cmp x10, #64
	b.hs LBB9_7
	mov x3, #0
	b LBB9_11
LBB9_7:
	mov x3, #0
	mov x6, x16
	and x7, x10, #0xffffffffffffffc0
LBB9_8:
	ldp q0, q1, [x6, #-32]
	ldp q2, q3, [x6], #64
	add x19, x5, x3
	add x19, x1, x19
	stp q0, q1, [x19]
	stp q2, q3, [x19, #32]
	add x3, x3, #64
	subs x7, x7, #64
	b.ne LBB9_8
	cmp x10, x13
	b.eq LBB9_3
	and x3, x10, #0xffffffffffffffc0
	add x6, x5, x13
	cbz x12, LBB9_14
LBB9_11:
	add x6, x17, x3
	add x7, x15, x3
LBB9_12:
	add x19, x5, x3
	ldr d0, [x7], #8
	str d0, [x1, x19]
	add x3, x3, #8
	adds x6, x6, #8
	b.ne LBB9_12
	add x6, x5, x14
	cmp x10, x14
	b.eq LBB9_3
LBB9_14:
	madd x3, x4, x6, x0
LBB9_15:
	ldrb w7, [x3]
	strb w7, [x1, x6]
	add x6, x6, #1
	add x3, x3, x4
	cmp x6, x8
	b.lo LBB9_15
	b LBB9_3
LBB9_16:
	ldp x29, x30, [sp, #16]
	ldp x20, x19, [sp], #32
	ret
LBB9_17:
Lloh14:
	adrp x0, l_anon.bc1f2f7c2f312fc2f015440ea3f2cec5.2@PAGE
Lloh15:
	add x0, x0, l_anon.bc1f2f7c2f312fc2f015440ea3f2cec5.2@PAGEOFF
	bl core::panicking::panic_const::panic_const_div_by_zero
