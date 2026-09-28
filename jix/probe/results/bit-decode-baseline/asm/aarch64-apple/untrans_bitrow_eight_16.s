jix_probe::bit_shuffle::untrans_bitrow_eight:
Lfunc_begin3:
	sub sp, sp, #384
	stp x28, x27, [sp, #288]
	stp x26, x25, [sp, #304]
	stp x24, x23, [sp, #320]
	stp x22, x21, [sp, #336]
	stp x20, x19, [sp, #352]
	stp x29, x30, [sp, #368]
	add x29, sp, #368
	stp x2, x0, [x29, #-152]
	str x5, [sp, #48]
	cbz x5, LBB3_35
	mov x20, x3
	mov x19, x1
	mov x21, #0
	mov x22, #0
	mov x24, #0
	lsr x23, x4, #3
	ldr x26, [sp, #48]
	lsl x8, x26, #1
	str x8, [sp, #40]
	add x10, x8, x26
	lsl x8, x26, #2
	str x8, [sp, #24]
	add x11, x8, x26
	lsl x8, x10, #1
	str x8, [sp, #8]
	lsl x8, x26, #3
	sub x8, x8, x26
	mul x28, x26, x23
	lsl x9, x28, #3
	sub x9, x9, x28
	madd x8, x23, x8, x23
	stp x8, x9, [sp, #80]
	lsl x8, x28, #1
	stur x8, [x29, #-160]
	add x8, x8, x28
	str x8, [sp, #184]
	lsl x8, x8, #1
	str x8, [sp, #112]
	mov w8, #1
	mov w9, #1
	bfi x9, x10, #1, #63
	mul x9, x23, x9
	str x9, [sp, #104]
	lsl x9, x28, #2
	str x9, [sp, #160]
	add x9, x9, x28
	str x9, [sp, #136]
	str x11, [sp, #16]
	madd x9, x23, x11, x23
	str x9, [sp, #128]
	mov w9, #1
	bfi x9, x26, #2, #62
	mul x9, x23, x9
	str x9, [sp, #152]
	str x10, [sp, #32]
	madd x9, x23, x10, x23
	str x9, [sp, #176]
	bfi x8, x26, #1, #63
	mul x8, x23, x8
	stur x8, [x29, #-168]
	lsl x25, x23, #3
	sub x10, x25, x23
	ldur x8, [x29, #-144]
	sub x9, x25, x23
	stur x9, [x29, #-136]
	add x9, x8, x10
	str x9, [sp, #56]
	lsl x9, x23, #1
	add x11, x9, x23
	add x10, x8, x11
	str x10, [sp, #144]
	add x10, x9, x23
	stur x10, [x29, #-104]
	lsl x10, x11, #1
	stur x10, [x29, #-128]
	add x10, x8, x10
	str x10, [sp, #72]
	lsl x10, x23, #2
	add x12, x10, x23
	add x11, x10, x23
	stur x11, [x29, #-120]
	add x11, x8, x12
	str x11, [sp, #96]
	stur x10, [x29, #-112]
	add x10, x8, x10
	str x10, [sp, #120]
	stur x9, [x29, #-96]
	add x9, x8, x9
	str x9, [sp, #168]
	add x8, x8, x23
	stur x8, [x29, #-176]
	str x25, [sp, #64]
LBB3_2:
	madd x1, x24, x23, x23
	adds x8, x23, x22
	b.hs LBB3_49
	cmp x8, x20
	b.hi LBB3_49
	lsl x25, x24, #3
	mul x8, x25, x23
	add x1, x8, x23
	adds x8, x23, x21
	b.hs LBB3_51
	cmp x8, x19
	b.hi LBB3_51
	ldp x8, x9, [x29, #-152]
	add x27, x8, x22
	add x1, x9, x22, lsl #3
	add x0, x8, x22
	mov x2, x23
	bl _memcpy
	add x9, x28, x22
	add x8, x23, x28
	add x8, x8, x22
	cmp x8, x9
	b.lo LBB3_36
	cmp x8, x20
	b.hi LBB3_36
	add x9, x23, x21
	ldur x8, [x29, #-96]
	add x8, x8, x21
	cmp x8, x9
	b.lo LBB3_37
	cmp x8, x19
	b.hi LBB3_37
	ldur x8, [x29, #-176]
	add x1, x8, x22, lsl #3
	add x27, x27, x28
	mov x0, x27
	mov x2, x23
	bl _memcpy
	ldp x8, x10, [x29, #-168]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo LBB3_38
	cmp x8, x20
	b.hi LBB3_38
	ldp x8, x10, [x29, #-104]
	add x9, x10, x21
	add x8, x8, x21
	cmp x8, x9
	b.lo LBB3_39
	cmp x8, x19
	b.hi LBB3_39
	ldr x8, [sp, #168]
	add x1, x8, x22, lsl #3
	add x27, x27, x28
	mov x0, x27
	mov x2, x23
	bl _memcpy
	ldp x8, x10, [sp, #176]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo LBB3_40
	cmp x8, x20
	b.hi LBB3_40
	ldp x8, x10, [x29, #-112]
	add x9, x10, x21
	add x8, x8, x21
	cmp x8, x9
	b.lo LBB3_41
	cmp x8, x19
	b.hi LBB3_41
	ldr x8, [sp, #144]
	add x1, x8, x22, lsl #3
	add x27, x27, x28
	mov x0, x27
	mov x2, x23
	bl _memcpy
	ldp x8, x10, [sp, #152]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo LBB3_42
	cmp x8, x20
	b.hi LBB3_42
	ldp x8, x10, [x29, #-120]
	add x9, x10, x21
	add x8, x8, x21
	cmp x8, x9
	b.lo LBB3_43
	cmp x8, x19
	b.hi LBB3_43
	ldr x8, [sp, #120]
	add x1, x8, x22, lsl #3
	add x27, x27, x28
	mov x0, x27
	mov x2, x23
	bl _memcpy
	ldp x8, x10, [sp, #128]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo LBB3_44
	cmp x8, x20
	b.hi LBB3_44
	ldp x8, x10, [x29, #-128]
	add x9, x10, x21
	add x8, x8, x21
	cmp x8, x9
	b.lo LBB3_45
	cmp x8, x19
	b.hi LBB3_45
	ldr x8, [sp, #96]
	add x1, x8, x22, lsl #3
	add x27, x27, x28
	mov x0, x27
	mov x2, x23
	bl _memcpy
	ldp x8, x10, [sp, #104]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo LBB3_46
	cmp x8, x20
	b.hi LBB3_46
	ldp x8, x10, [x29, #-136]
	add x9, x10, x21
	add x8, x8, x21
	cmp x8, x9
	b.lo LBB3_47
	cmp x8, x19
	b.hi LBB3_47
	ldr x8, [sp, #72]
	add x1, x8, x22, lsl #3
	add x27, x27, x28
	mov x0, x27
	mov x2, x23
	bl _memcpy
	ldp x9, x8, [sp, #80]
	add x8, x8, x22
	add x1, x9, x22
	cmp x1, x8
	b.lo LBB3_48
	cmp x1, x20
	b.hi LBB3_48
	ldur x8, [x29, #-136]
	add x8, x8, x21
	ldr x25, [sp, #64]
	add x1, x25, x21
	cmp x1, x8
	b.lo LBB3_50
	cmp x1, x19
	b.hi LBB3_50
	ldr x8, [sp, #56]
	add x1, x8, x22, lsl #3
	add x0, x27, x28
	mov x2, x23
	bl _memcpy
	add x22, x22, x23
	add x21, x21, x25
	add x24, x24, #1
	subs x26, x26, #1
	b.ne LBB3_2
LBB3_35:
	ldp x29, x30, [sp, #368]
	ldp x20, x19, [sp, #352]
	ldp x22, x21, [sp, #336]
	ldp x24, x23, [sp, #320]
	ldp x26, x25, [sp, #304]
	ldp x28, x27, [sp, #288]
	add sp, sp, #384
	ret
LBB3_36:
	ldr x9, [sp, #48]
	add x8, x9, x24
	madd x1, x8, x23, x23
	madd x22, x9, x23, x22
Lloh30:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGE
Lloh31:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGEOFF
	mov x0, x22
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB3_37:
	orr x8, x25, #0x1
	mul x8, x8, x23
	add x1, x8, x23
	add x21, x23, x21
Lloh32:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGE
Lloh33:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGEOFF
	mov x0, x21
	mov x2, x19
	bl core::slice::index::slice_index_fail
LBB3_38:
	ldr x8, [sp, #40]
	add x8, x8, x24
	madd x1, x8, x23, x23
	add x22, x22, x28, lsl #1
Lloh34:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGE
Lloh35:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGEOFF
	mov x0, x22
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB3_39:
	orr x8, x25, #0x2
	mul x8, x8, x23
	add x1, x8, x23
	add x21, x21, x23, lsl #1
Lloh36:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGE
Lloh37:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGEOFF
	mov x0, x21
	mov x2, x19
	bl core::slice::index::slice_index_fail
LBB3_40:
	ldr x8, [sp, #32]
	add x8, x8, x24
	madd x1, x8, x23, x23
	add x8, x28, x28, lsl #1
	add x22, x8, x22
Lloh38:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGE
Lloh39:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGEOFF
	mov x0, x22
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB3_41:
	orr x8, x25, #0x3
	mul x8, x8, x23
	add x1, x8, x23
	add x8, x23, x23, lsl #1
	add x21, x8, x21
Lloh40:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGE
Lloh41:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGEOFF
	mov x0, x21
	mov x2, x19
	bl core::slice::index::slice_index_fail
LBB3_42:
	ldr x8, [sp, #24]
	add x8, x8, x24
	madd x1, x8, x23, x23
	add x22, x22, x28, lsl #2
Lloh42:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGE
Lloh43:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGEOFF
	mov x0, x22
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB3_43:
	orr x8, x25, #0x4
	mul x8, x8, x23
	add x1, x8, x23
	add x21, x21, x23, lsl #2
Lloh44:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGE
Lloh45:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGEOFF
	mov x0, x21
	mov x2, x19
	bl core::slice::index::slice_index_fail
LBB3_44:
	ldr x8, [sp, #16]
	add x8, x8, x24
	madd x1, x8, x23, x23
	add x8, x28, x28, lsl #2
	add x22, x8, x22
Lloh46:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGE
Lloh47:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGEOFF
	mov x0, x22
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB3_45:
	mov w8, #5
	orr x8, x25, x8
	mul x8, x8, x23
	add x1, x8, x23
	add x8, x23, x23, lsl #2
	add x21, x8, x21
Lloh48:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGE
Lloh49:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGEOFF
	mov x0, x21
	mov x2, x19
	bl core::slice::index::slice_index_fail
LBB3_46:
	ldr x8, [sp, #8]
	add x8, x8, x24
	madd x1, x8, x23, x23
	mov w8, #6
	madd x22, x28, x8, x22
Lloh50:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGE
Lloh51:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGEOFF
	mov x0, x22
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB3_47:
	orr x8, x25, #0x6
	mul x8, x8, x23
	add x1, x8, x23
	mov w8, #6
	madd x21, x23, x8, x21
Lloh52:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGE
Lloh53:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGEOFF
	mov x0, x21
	mov x2, x19
	bl core::slice::index::slice_index_fail
LBB3_48:
	sub x8, x22, x28
	add x22, x8, x28, lsl #3
LBB3_49:
Lloh54:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGE
Lloh55:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.12@PAGEOFF
	mov x0, x22
	mov x2, x20
	bl core::slice::index::slice_index_fail
LBB3_50:
	sub x8, x21, x23
	add x21, x8, x23, lsl #3
LBB3_51:
Lloh56:
	adrp x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGE
Lloh57:
	add x3, x3, l_anon.86fcf0a1111718708cdca7b99afbb18a.11@PAGEOFF
	mov x0, x21
	mov x2, x19
	bl core::slice::index::slice_index_fail
