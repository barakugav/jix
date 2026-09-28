_probe_byte_shuffle_encode_2:
Lfunc_begin9:
	stp x29, x30, [sp, #-16]!
	mov x29, sp
	and x8, x1, #0x7ffffffffffffffe
	cmp x3, x8
	b.lo LBB9_5
	lsr x8, x1, #1
	lsr x9, x1, #8
	cbz x9, LBB9_4
	add x10, x2, #112
	add x11, x0, #128
	add x12, x8, x2
	add x12, x12, #64
LBB9_3:
	sub x13, x11, #128
	ld2.16b { v0, v1 }, [x13]
	stur q0, [x10, #-112]
	sub x13, x11, #96
	ld2.16b { v2, v3 }, [x13]
	stur q1, [x12, #-64]
	sub x13, x11, #64
	ld2.16b { v0, v1 }, [x13]
	stur q2, [x10, #-96]
	sub x13, x11, #32
	ld2.16b { v4, v5 }, [x13]
	stur q3, [x12, #-48]
	stur q0, [x10, #-80]
	stur q1, [x12, #-32]
	stur q4, [x10, #-64]
	stur q5, [x12, #-16]
	ldp q0, q1, [x11]
	uzp1.16b v2, v0, v1
	uzp2.16b v0, v0, v1
	stur q2, [x10, #-48]
	str q0, [x12]
	add x13, x11, #32
	ld2.16b { v0, v1 }, [x13]
	stur q0, [x10, #-32]
	str q1, [x12, #16]
	add x13, x11, #64
	ld2.16b { v0, v1 }, [x13]
	stur q0, [x10, #-16]
	str q1, [x12, #32]
	add x13, x11, #96
	ld2.16b { v0, v1 }, [x13]
	str q0, [x10], #128
	add x11, x11, #256
	str q1, [x12, #48]
	add x12, x12, #128
	subs x9, x9, #1
	b.ne LBB9_3
LBB9_4:
	and x0, x8, #0x3fffffffffffff80
	ldp x29, x30, [sp], #16
	ret
LBB9_5:
Lloh28:
	adrp x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGE
Lloh29:
	add x0, x0, l_anon.62923e1c64a8998ec37396e8f14ab088.0@PAGEOFF
Lloh30:
	adrp x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGE
Lloh31:
	add x2, x2, l_anon.62923e1c64a8998ec37396e8f14ab088.3@PAGEOFF
	mov w1, #48
	bl core::panicking::panic
