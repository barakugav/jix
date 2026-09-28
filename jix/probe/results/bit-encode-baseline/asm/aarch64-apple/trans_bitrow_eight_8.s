jix_probe::bit_shuffle::trans_bitrow_eight:
Lfunc_begin2:
	sub sp, sp, #128
	stp x28, x27, [sp, #32]
	stp x26, x25, [sp, #48]
	stp x24, x23, [sp, #64]
	stp x22, x21, [sp, #80]
	stp x20, x19, [sp, #96]
	stp x29, x30, [sp, #112]
	add x29, sp, #112
	str x2, [sp, #24]
	cbz x5, LBB2_48
	mov x19, x3
	mov x20, x1
	mov x24, x0
	mov x21, #0
	mov x26, #0
	lsr x25, x4, #3
	lsl x22, x25, #3
	str x5, [sp, #8]
	mov x23, x5
LBB2_2:
	adds x8, x25, x26
	b.hs LBB2_49
	cmp x8, x19
	b.hi LBB2_49
	adds x8, x25, x21
	b.hs LBB2_50
	cmp x8, x20
	b.hi LBB2_50
	ldr x8, [sp, #24]
	add x0, x8, x26
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x26, x26, x22
	add x21, x21, x25
	subs x23, x23, #1
	b.ne LBB2_2
	mov x28, #0
	ldr x8, [sp, #24]
	add x8, x8, x25
	str x8, [sp, #16]
	lsl x26, x25, #1
	ldr x23, [sp, #8]
	mul x21, x23, x25
	add x27, x25, x21
	str x21, [sp]
LBB2_8:
	add x8, x25, x28
	add x9, x26, x28
	cmp x9, x8
	ccmp x9, x19, #2, hs
	b.hi LBB2_52
	cmp x27, x21
	b.lo LBB2_51
	cmp x27, x20
	b.hi LBB2_51
	ldr x8, [sp, #16]
	add x0, x8, x28
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x28, x28, x22
	add x21, x21, x25
	add x27, x27, x25
	subs x23, x23, #1
	b.ne LBB2_8
	add x28, x26, x25
	ldp x8, x23, [sp]
	lsl x21, x8, #1
	mov w8, #1
	orr x8, x8, x23, lsl #1
	mul x27, x25, x8
LBB2_13:
	cmp x28, x26
	b.lo LBB2_53
	cmp x28, x19
	b.hi LBB2_53
	cmp x27, x21
	b.lo LBB2_51
	cmp x27, x20
	b.hi LBB2_51
	ldr x8, [sp, #24]
	add x0, x8, x26
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x26, x26, x22
	add x28, x28, x22
	add x21, x21, x25
	add x27, x27, x25
	subs x23, x23, #1
	b.ne LBB2_13
	add x26, x25, x25, lsl #1
	ldp x8, x23, [sp]
	add x21, x8, x8, lsl #1
	lsl x28, x25, #2
	add x27, x25, x21
	str x28, [sp, #16]
LBB2_19:
	cmp x28, x26
	b.lo LBB2_53
	cmp x28, x19
	b.hi LBB2_53
	cmp x27, x21
	b.lo LBB2_51
	cmp x27, x20
	b.hi LBB2_51
	ldr x8, [sp, #24]
	add x0, x8, x26
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x26, x26, x22
	add x28, x28, x22
	add x21, x21, x25
	add x27, x27, x25
	subs x23, x23, #1
	b.ne LBB2_19
	ldp x23, x26, [sp, #8]
	add x28, x26, x25
	ldr x8, [sp]
	lsl x21, x8, #2
	mov w8, #1
	orr x8, x8, x23, lsl #2
	mul x27, x25, x8
LBB2_25:
	cmp x28, x26
	b.lo LBB2_53
	cmp x28, x19
	b.hi LBB2_53
	cmp x27, x21
	b.lo LBB2_51
	cmp x27, x20
	b.hi LBB2_51
	ldr x8, [sp, #24]
	add x0, x8, x26
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x26, x26, x22
	add x28, x28, x22
	add x21, x21, x25
	add x27, x27, x25
	subs x23, x23, #1
	b.ne LBB2_25
	add x26, x25, x25, lsl #2
	add x8, x25, x25, lsl #1
	lsl x28, x8, #1
	ldp x8, x23, [sp]
	add x21, x8, x8, lsl #2
	add x27, x25, x21
	str x28, [sp, #16]
LBB2_31:
	cmp x28, x26
	b.lo LBB2_53
	cmp x28, x19
	b.hi LBB2_53
	cmp x27, x21
	b.lo LBB2_51
	cmp x27, x20
	b.hi LBB2_51
	ldr x8, [sp, #24]
	add x0, x8, x26
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x26, x26, x22
	add x28, x28, x22
	add x21, x21, x25
	add x27, x27, x25
	subs x23, x23, #1
	b.ne LBB2_31
	ldp x23, x26, [sp, #8]
	add x8, x23, x23, lsl #1
	lsl x9, x8, #1
	mul x21, x9, x25
	mov w9, #1
	bfi x9, x8, #1, #63
	mul x27, x25, x9
	sub x28, x22, x25
LBB2_37:
	cmp x28, x26
	b.lo LBB2_53
	cmp x28, x19
	b.hi LBB2_53
	cmp x27, x21
	b.lo LBB2_51
	cmp x27, x20
	b.hi LBB2_51
	ldr x8, [sp, #24]
	add x0, x8, x26
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x26, x26, x22
	add x28, x28, x22
	add x21, x21, x25
	add x27, x27, x25
	subs x23, x23, #1
	b.ne LBB2_37
	ldp x9, x23, [sp]
	lsl x8, x9, #3
	sub x21, x8, x9
	add x27, x25, x21
	mov x28, x22
	sub x26, x22, x25
LBB2_43:
	cmp x28, x26
	b.lo LBB2_53
	cmp x28, x19
	b.hi LBB2_53
	cmp x27, x21
	b.lo LBB2_51
	cmp x27, x20
	b.hi LBB2_51
	ldr x8, [sp, #24]
	add x0, x8, x26
	add x1, x24, x21
	mov x2, x25
	bl _memcpy
	add x26, x26, x22
	add x28, x28, x22
	add x21, x21, x25
	add x27, x27, x25
	subs x23, x23, #1
	b.ne LBB2_43
LBB2_48:
	ldp x29, x30, [sp, #112]
	ldp x20, x19, [sp, #96]
	ldp x22, x21, [sp, #80]
	ldp x24, x23, [sp, #64]
	ldp x26, x25, [sp, #48]
	ldp x28, x27, [sp, #32]
	add sp, sp, #128
	ret
LBB2_49:
	add x28, x25, x26
Lloh24:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.10@PAGE
Lloh25:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.10@PAGEOFF
	mov x0, x26
	mov x1, x28
	mov x2, x19
	bl core::slice::index::slice_index_fail
LBB2_50:
	add x27, x25, x21
LBB2_51:
Lloh26:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.9@PAGE
Lloh27:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.9@PAGEOFF
	mov x0, x21
	mov x1, x27
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB2_52:
	add x28, x28, x25, lsl #1
	mov x26, x8
LBB2_53:
Lloh28:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.10@PAGE
Lloh29:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.10@PAGEOFF
	mov x0, x26
	mov x1, x28
	mov x2, x19
	bl core::slice::index::slice_index_fail
