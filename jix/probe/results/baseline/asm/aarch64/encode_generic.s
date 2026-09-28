jix_probe::byte_shuffle::encode_impl_generic:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	cbz x4, .LBB9_17
	udiv x8, x1, x4
	cmp x5, x8
	b.hs .LBB9_16
	sub x10, x1, x5
	add x11, x0, x5
	mov x9, xzr
	cmp x10, #7
	and x13, x10, #0xfffffffffffffff8
	and x12, x10, #0x18
	ccmp x4, #1, #0, hi
	and x15, x10, #0xffffffffffffffe0
	add x16, x11, #16
	cset w14, eq
	neg x17, x13
	b .LBB9_4
.LBB9_3:
	add x9, x9, #1
	add x16, x16, #1
	add x11, x11, #1
	cmp x9, x4
	add x0, x0, #1
	b.eq .LBB9_16
.LBB9_4:
	madd x18, x9, x8, x2
	mov x3, x5
	cbz w14, .LBB9_14
	cmp x10, #32
	b.hs .LBB9_7
	mov x1, xzr
	b .LBB9_11
.LBB9_7:
	mov x1, xzr
	mov x3, x16
	and x6, x10, #0xffffffffffffffe0
.LBB9_8:
	ldp q0, q1, [x3, #-16]
	add x7, x5, x1
	add x7, x18, x7
	subs x6, x6, #32
	add x1, x1, #32
	add x3, x3, #32
	stp q0, q1, [x7]
	b.ne .LBB9_8
	cmp x10, x15
	b.eq .LBB9_3
	and x1, x10, #0xffffffffffffffe0
	add x3, x5, x15
	cbz x12, .LBB9_14
.LBB9_11:
	add x3, x17, x1
	add x6, x11, x1
.LBB9_12:
	add x7, x5, x1
	ldr d0, [x6], #8
	adds x3, x3, #8
	add x1, x1, #8
	str d0, [x18, x7]
	b.ne .LBB9_12
	cmp x10, x13
	add x3, x5, x13
	b.eq .LBB9_3
.LBB9_14:
	madd x1, x4, x3, x0
.LBB9_15:
	mov x6, x3
	add x3, x3, #1
	ldrb w7, [x1]
	cmp x3, x8
	add x1, x1, x4
	strb w7, [x18, x6]
	b.lo .LBB9_15
	b .LBB9_3
.LBB9_16:
	ldp x29, x30, [sp], #16
	ret
.LBB9_17:
	adrp x0, .Lanon.557518dcc8a715771bee216ade4f749e.2
	add x0, x0, :lo12:.Lanon.557518dcc8a715771bee216ade4f749e.2
	bl core::panicking::panic_const::panic_const_div_by_zero
