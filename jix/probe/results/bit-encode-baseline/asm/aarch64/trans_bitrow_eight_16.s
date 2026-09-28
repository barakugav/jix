jix_probe::bit_shuffle::trans_bitrow_eight:
	sub sp, sp, #144
	stp x29, x30, [sp, #48]
	stp x28, x27, [sp, #64]
	stp x26, x25, [sp, #80]
	stp x24, x23, [sp, #96]
	stp x22, x21, [sp, #112]
	stp x20, x19, [sp, #128]
	add x29, sp, #48
	stur x2, [x29, #-8]
	cbz x5, .LBB2_49
	lsr x25, x4, #3
	mov x19, x3
	mov x20, x1
	mov x24, x0
	mov x23, xzr
	mov x27, xzr
	lsl x21, x25, #3
	mov x22, x5
	str x5, [sp, #16]
.LBB2_2:
	adds x8, x25, x27
	b.hs .LBB2_50
	cmp x8, x19
	b.hi .LBB2_50
	adds x8, x25, x23
	b.hs .LBB2_51
	cmp x8, x20
	b.hi .LBB2_51
	ldur x8, [x29, #-8]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x27
	bl memcpy
	subs x22, x22, #1
	add x27, x27, x21
	add x23, x23, x25
	b.ne .LBB2_2
	ldr x22, [sp, #16]
	ldur x8, [x29, #-8]
	mov x28, xzr
	mul x23, x22, x25
	add x8, x8, x25
	str x8, [sp, #24]
	lsl x8, x25, #1
	stur x8, [x29, #-16]
	add x26, x25, x23
	str x23, [sp, #8]
.LBB2_8:
	ldur x8, [x29, #-16]
	add x27, x25, x28
	add x8, x8, x28
	cmp x8, x27
	b.lo .LBB2_53
	cmp x8, x19
	b.hi .LBB2_53
	cmp x26, x23
	b.lo .LBB2_52
	cmp x26, x20
	b.hi .LBB2_52
	ldr x8, [sp, #24]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x28
	bl memcpy
	subs x22, x22, #1
	add x28, x28, x21
	add x23, x23, x25
	add x26, x26, x25
	b.ne .LBB2_8
	ldr x22, [sp, #16]
	mov w8, #1
	ldur x27, [x29, #-16]
	orr x8, x8, x22, lsl #1
	add x28, x27, x25
	mul x26, x25, x8
	ldr x8, [sp, #8]
	lsl x23, x8, #1
.LBB2_14:
	cmp x28, x27
	b.lo .LBB2_54
	cmp x28, x19
	b.hi .LBB2_54
	cmp x26, x23
	b.lo .LBB2_52
	cmp x26, x20
	b.hi .LBB2_52
	ldur x8, [x29, #-8]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x27
	bl memcpy
	add x27, x27, x21
	subs x22, x22, #1
	add x28, x28, x21
	add x23, x23, x25
	add x26, x26, x25
	b.ne .LBB2_14
	ldp x8, x22, [sp, #8]
	add x27, x25, x25, lsl #1
	lsl x28, x25, #2
	add x23, x8, x8, lsl #1
	stur x28, [x29, #-16]
	add x26, x25, x23
.LBB2_20:
	cmp x28, x27
	b.lo .LBB2_54
	cmp x28, x19
	b.hi .LBB2_54
	cmp x26, x23
	b.lo .LBB2_52
	cmp x26, x20
	b.hi .LBB2_52
	ldur x8, [x29, #-8]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x27
	bl memcpy
	add x27, x27, x21
	subs x22, x22, #1
	add x28, x28, x21
	add x23, x23, x25
	add x26, x26, x25
	b.ne .LBB2_20
	ldr x22, [sp, #16]
	mov w8, #1
	ldur x27, [x29, #-16]
	orr x8, x8, x22, lsl #2
	add x28, x27, x25
	mul x26, x25, x8
	ldr x8, [sp, #8]
	lsl x23, x8, #2
.LBB2_26:
	cmp x28, x27
	b.lo .LBB2_54
	cmp x28, x19
	b.hi .LBB2_54
	cmp x26, x23
	b.lo .LBB2_52
	cmp x26, x20
	b.hi .LBB2_52
	ldur x8, [x29, #-8]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x27
	bl memcpy
	add x27, x27, x21
	subs x22, x22, #1
	add x28, x28, x21
	add x23, x23, x25
	add x26, x26, x25
	b.ne .LBB2_26
	add x8, x25, x25, lsl #1
	add x27, x25, x25, lsl #2
	lsl x28, x8, #1
	ldp x8, x22, [sp, #8]
	stur x28, [x29, #-16]
	add x23, x8, x8, lsl #2
	add x26, x25, x23
.LBB2_32:
	cmp x28, x27
	b.lo .LBB2_54
	cmp x28, x19
	b.hi .LBB2_54
	cmp x26, x23
	b.lo .LBB2_52
	cmp x26, x20
	b.hi .LBB2_52
	ldur x8, [x29, #-8]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x27
	bl memcpy
	add x27, x27, x21
	subs x22, x22, #1
	add x28, x28, x21
	add x23, x23, x25
	add x26, x26, x25
	b.ne .LBB2_32
	ldr x22, [sp, #16]
	mov w10, #1
	ldur x27, [x29, #-16]
	sub x28, x21, x25
	add x8, x22, x22, lsl #1
	lsl x9, x8, #1
	bfi x10, x8, #1, #63
	mul x23, x9, x25
	mul x26, x25, x10
.LBB2_38:
	cmp x28, x27
	b.lo .LBB2_54
	cmp x28, x19
	b.hi .LBB2_54
	cmp x26, x23
	b.lo .LBB2_52
	cmp x26, x20
	b.hi .LBB2_52
	ldur x8, [x29, #-8]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x27
	bl memcpy
	add x27, x27, x21
	subs x22, x22, #1
	add x28, x28, x21
	add x23, x23, x25
	add x26, x26, x25
	b.ne .LBB2_38
	ldp x9, x22, [sp, #8]
	mov x28, x21
	sub x27, x21, x25
	lsl x8, x9, #3
	sub x23, x8, x9
	add x26, x25, x23
.LBB2_44:
	cmp x28, x27
	b.lo .LBB2_54
	cmp x28, x19
	b.hi .LBB2_54
	cmp x26, x23
	b.lo .LBB2_52
	cmp x26, x20
	b.hi .LBB2_52
	ldur x8, [x29, #-8]
	add x1, x24, x23
	mov x2, x25
	add x0, x8, x27
	bl memcpy
	add x27, x27, x21
	subs x22, x22, #1
	add x28, x28, x21
	add x23, x23, x25
	add x26, x26, x25
	b.ne .LBB2_44
.LBB2_49:
	ldp x20, x19, [sp, #128]
	ldp x22, x21, [sp, #112]
	ldp x24, x23, [sp, #96]
	ldp x26, x25, [sp, #80]
	ldp x28, x27, [sp, #64]
	ldp x29, x30, [sp, #48]
	add sp, sp, #144
	ret
.LBB2_50:
	add x28, x25, x27
	adrp x3, .Lanon.5e183fe0a34aeb9c6619c58bb96c331f.10
	add x3, x3, :lo12:.Lanon.5e183fe0a34aeb9c6619c58bb96c331f.10
	mov x0, x27
	mov x1, x28
	mov x2, x19
	bl core::slice::index::slice_index_fail
.LBB2_51:
	add x26, x25, x23
.LBB2_52:
	adrp x3, .Lanon.5e183fe0a34aeb9c6619c58bb96c331f.9
	add x3, x3, :lo12:.Lanon.5e183fe0a34aeb9c6619c58bb96c331f.9
	mov x0, x23
	mov x1, x26
	mov x2, x20
	bl core::slice::index::slice_index_fail
.LBB2_53:
	add x28, x28, x25, lsl #1
.LBB2_54:
	adrp x3, .Lanon.5e183fe0a34aeb9c6619c58bb96c331f.10
	add x3, x3, :lo12:.Lanon.5e183fe0a34aeb9c6619c58bb96c331f.10
	mov x0, x27
	mov x1, x28
	mov x2, x19
	bl core::slice::index::slice_index_fail
