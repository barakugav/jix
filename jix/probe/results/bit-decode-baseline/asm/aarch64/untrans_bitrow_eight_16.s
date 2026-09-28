jix_probe::bit_shuffle::untrans_bitrow_eight:
	sub sp, sp, #384
	stp x29, x30, [sp, #288]
	stp x28, x27, [sp, #304]
	stp x26, x25, [sp, #320]
	stp x24, x23, [sp, #336]
	stp x22, x21, [sp, #352]
	stp x20, x19, [sp, #368]
	add x29, sp, #288
	stp x2, x0, [x29, #-72]
	str x5, [sp, #48]
	cbz x5, .LBB3_35
	ldr x26, [sp, #48]
	lsr x23, x4, #3
	mov x21, x3
	mov x19, x1
	mov x20, xzr
	mov x22, xzr
	lsl x8, x26, #3
	lsl x9, x26, #1
	lsl x10, x26, #2
	mul x28, x26, x23
	lsl x13, x23, #1
	lsl x25, x23, #3
	sub x8, x8, x26
	str x9, [sp, #40]
	add x11, x9, x26
	madd x8, x23, x8, x23
	mov w9, #1
	stp x10, x11, [sp, #24]
	bfi x9, x11, #1, #63
	add x10, x10, x26
	sub x14, x25, x23
	str x10, [sp, #16]
	madd x10, x23, x10, x23
	mov x24, xzr
	lsl x12, x28, #1
	stur x13, [x29, #-16]
	str x8, [sp, #88]
	mul x8, x23, x9
	mov w9, #1
	stur x12, [x29, #-80]
	add x12, x12, x28
	bfi x9, x26, #1, #63
	str x10, [sp, #136]
	lsl x10, x28, #3
	stur x12, [x29, #-104]
	sub x10, x10, x28
	str x25, [sp, #64]
	str x8, [sp, #112]
	mov w8, #1
	bfi x8, x26, #2, #62
	str x10, [sp, #80]
	lsl x10, x12, #1
	mul x8, x23, x8
	str x10, [sp, #104]
	madd x10, x23, x11, x23
	stur x8, [x29, #-128]
	lsl x8, x11, #1
	lsl x11, x23, #2
	str x8, [sp, #8]
	lsl x8, x28, #2
	add x15, x11, x23
	stur x10, [x29, #-112]
	add x10, x13, x23
	stur x8, [x29, #-136]
	add x8, x8, x28
	lsl x12, x10, #1
	str x8, [sp, #128]
	mul x8, x23, x9
	sub x9, x25, x23
	stp x9, x12, [x29, #-56]
	stur x11, [x29, #-32]
	stur x8, [x29, #-88]
	ldur x8, [x29, #-64]
	add x9, x8, x14
	str x9, [sp, #56]
	add x9, x8, x12
	str x9, [sp, #72]
	add x9, x11, x23
	stur x9, [x29, #-40]
	add x9, x8, x15
	str x9, [sp, #96]
	add x9, x8, x11
	str x9, [sp, #120]
	add x9, x13, x23
	stur x9, [x29, #-24]
	add x9, x8, x10
	str x9, [sp, #144]
	add x9, x8, x13
	add x8, x8, x23
	stur x9, [x29, #-120]
	stur x8, [x29, #-96]
.LBB3_2:
	madd x1, x24, x23, x23
	adds x8, x23, x22
	b.hs .LBB3_53
	cmp x8, x21
	b.hi .LBB3_53
	lsl x25, x24, #3
	adds x8, x23, x20
	mul x9, x25, x23
	add x1, x9, x23
	b.hs .LBB3_49
	cmp x8, x19
	b.hi .LBB3_49
	ldp x8, x9, [x29, #-72]
	mov x2, x23
	add x1, x9, x22, lsl #3
	add x0, x8, x22
	add x27, x8, x22
	bl memcpy
	add x8, x23, x28
	add x9, x28, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo .LBB3_36
	cmp x8, x21
	b.hi .LBB3_36
	ldur x8, [x29, #-16]
	add x9, x23, x20
	add x8, x8, x20
	cmp x8, x9
	b.lo .LBB3_37
	cmp x8, x19
	b.hi .LBB3_37
	ldur x8, [x29, #-96]
	add x27, x27, x28
	mov x2, x23
	mov x0, x27
	add x1, x8, x22, lsl #3
	bl memcpy
	ldp x8, x10, [x29, #-88]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo .LBB3_38
	cmp x8, x21
	b.hi .LBB3_38
	ldp x8, x10, [x29, #-24]
	add x9, x10, x20
	add x8, x8, x20
	cmp x8, x9
	b.lo .LBB3_39
	cmp x8, x19
	b.hi .LBB3_39
	ldur x8, [x29, #-120]
	add x27, x27, x28
	mov x2, x23
	mov x0, x27
	add x1, x8, x22, lsl #3
	bl memcpy
	ldp x8, x10, [x29, #-112]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo .LBB3_40
	cmp x8, x21
	b.hi .LBB3_40
	ldp x8, x10, [x29, #-32]
	add x9, x10, x20
	add x8, x8, x20
	cmp x8, x9
	b.lo .LBB3_41
	cmp x8, x19
	b.hi .LBB3_41
	ldr x8, [sp, #144]
	add x27, x27, x28
	mov x2, x23
	mov x0, x27
	add x1, x8, x22, lsl #3
	bl memcpy
	ldp x10, x8, [x29, #-136]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo .LBB3_42
	cmp x8, x21
	b.hi .LBB3_42
	ldp x8, x10, [x29, #-40]
	add x9, x10, x20
	add x8, x8, x20
	cmp x8, x9
	b.lo .LBB3_44
	cmp x8, x19
	b.hi .LBB3_44
	ldr x8, [sp, #120]
	add x27, x27, x28
	mov x2, x23
	mov x0, x27
	add x1, x8, x22, lsl #3
	bl memcpy
	ldp x10, x8, [sp, #128]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo .LBB3_46
	cmp x8, x21
	b.hi .LBB3_46
	ldp x8, x10, [x29, #-48]
	add x9, x10, x20
	add x8, x8, x20
	cmp x8, x9
	b.lo .LBB3_47
	cmp x8, x19
	b.hi .LBB3_47
	ldr x8, [sp, #96]
	add x27, x27, x28
	mov x2, x23
	mov x0, x27
	add x1, x8, x22, lsl #3
	bl memcpy
	ldp x10, x8, [sp, #104]
	add x9, x10, x22
	add x8, x8, x22
	cmp x8, x9
	b.lo .LBB3_50
	cmp x8, x21
	b.hi .LBB3_50
	ldp x8, x10, [x29, #-56]
	add x9, x10, x20
	add x8, x8, x20
	cmp x8, x9
	b.lo .LBB3_51
	cmp x8, x19
	b.hi .LBB3_51
	ldr x8, [sp, #72]
	add x27, x27, x28
	mov x2, x23
	mov x0, x27
	add x1, x8, x22, lsl #3
	bl memcpy
	ldp x8, x9, [sp, #80]
	add x8, x8, x22
	add x1, x9, x22
	cmp x1, x8
	b.lo .LBB3_52
	cmp x1, x21
	b.hi .LBB3_52
	ldur x8, [x29, #-56]
	ldr x25, [sp, #64]
	add x8, x8, x20
	add x1, x25, x20
	cmp x1, x8
	b.lo .LBB3_54
	cmp x1, x19
	b.hi .LBB3_54
	ldr x8, [sp, #56]
	add x0, x27, x28
	mov x2, x23
	add x1, x8, x22, lsl #3
	bl memcpy
	subs x26, x26, #1
	add x22, x22, x23
	add x20, x20, x25
	add x24, x24, #1
	b.ne .LBB3_2
.LBB3_35:
	ldp x20, x19, [sp, #368]
	ldp x22, x21, [sp, #352]
	ldp x24, x23, [sp, #336]
	ldp x26, x25, [sp, #320]
	ldp x28, x27, [sp, #304]
	ldp x29, x30, [sp, #288]
	add sp, sp, #384
	ret
.LBB3_36:
	ldr x9, [sp, #48]
	add x8, x9, x24
	madd x22, x9, x23, x22
	madd x1, x8, x23, x23
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	mov x0, x22
	mov x2, x21
	bl core::slice::index::slice_index_fail
.LBB3_37:
	orr x8, x25, #0x1
	add x20, x23, x20
	b .LBB3_45
.LBB3_38:
	ldr x8, [sp, #40]
	add x22, x22, x28, lsl #1
	b .LBB3_43
.LBB3_39:
	orr x8, x25, #0x2
	add x20, x20, x23, lsl #1
	b .LBB3_45
.LBB3_40:
	ldr x8, [sp, #32]
	add x8, x8, x24
	madd x1, x8, x23, x23
	add x8, x28, x28, lsl #1
	add x22, x8, x22
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	mov x0, x22
	mov x2, x21
	bl core::slice::index::slice_index_fail
.LBB3_41:
	orr x8, x25, #0x3
	add x9, x23, x23, lsl #1
	b .LBB3_48
.LBB3_42:
	ldr x8, [sp, #24]
	add x22, x22, x28, lsl #2
.LBB3_43:
	add x8, x8, x24
	madd x1, x8, x23, x23
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	mov x0, x22
	mov x2, x21
	bl core::slice::index::slice_index_fail
.LBB3_44:
	orr x8, x25, #0x4
	add x20, x20, x23, lsl #2
.LBB3_45:
	mul x8, x8, x23
	add x1, x8, x23
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	mov x0, x20
	mov x2, x19
	bl core::slice::index::slice_index_fail
.LBB3_46:
	ldr x8, [sp, #16]
	add x8, x8, x24
	madd x1, x8, x23, x23
	add x8, x28, x28, lsl #2
	add x22, x8, x22
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	mov x0, x22
	mov x2, x21
	bl core::slice::index::slice_index_fail
.LBB3_47:
	mov w8, #5
	add x9, x23, x23, lsl #2
	orr x8, x25, x8
.LBB3_48:
	mul x8, x8, x23
	add x1, x8, x23
	add x20, x9, x20
.LBB3_49:
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	mov x0, x20
	mov x2, x19
	bl core::slice::index::slice_index_fail
.LBB3_50:
	ldr x8, [sp, #8]
	add x8, x8, x24
	madd x1, x8, x23, x23
	mov w8, #6
	madd x22, x28, x8, x22
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	mov x0, x22
	mov x2, x21
	bl core::slice::index::slice_index_fail
.LBB3_51:
	orr x8, x25, #0x6
	mov w9, #6
	mul x8, x8, x23
	madd x20, x23, x9, x20
	add x1, x8, x23
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	mov x0, x20
	mov x2, x19
	bl core::slice::index::slice_index_fail
.LBB3_52:
	sub x8, x22, x28
	add x22, x8, x28, lsl #3
.LBB3_53:
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.10
	mov x0, x22
	mov x2, x21
	bl core::slice::index::slice_index_fail
.LBB3_54:
	sub x8, x20, x23
	add x20, x8, x23, lsl #3
	adrp x3, .Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	add x3, x3, :lo12:.Lanon.bd4fc3f0c3cd03cd6429fc05320c1acf.9
	mov x0, x20
	mov x2, x19
	bl core::slice::index::slice_index_fail
