jix_probe::byte_shuffle::decode_impl_generic:
.Lfunc_gep8:
	addis 2, 12, .TOC.-.Lfunc_gep8@ha
	addi 2, 2, .TOC.-.Lfunc_gep8@l
.Lfunc_lep8:
	mflr 0
	std 16, -128(1)
	std 17, -120(1)
	std 18, -112(1)
	std 19, -104(1)
	std 20, -96(1)
	std 21, -88(1)
	std 22, -80(1)
	std 23, -72(1)
	std 24, -64(1)
	std 25, -56(1)
	std 26, -48(1)
	std 27, -40(1)
	std 28, -32(1)
	std 29, -24(1)
	std 30, -16(1)
	stdu 1, -160(1)
	cmpldi 7, 0
	std 0, 176(1)
	beq- 0, .LBB8_15
	divdu 4, 4, 7
	cmpld 8, 4
	bge 0, .LBB8_14
	mulld 30, 8, 7
	rldicr 6, 7, 0, 60
	addi 29, 7, -8
	li 28, 0
	cmpldi 1, 7, 8
	andi. 9, 7, 7
	li 27, -8
	add 10, 5, 30
	addi 11, 30, -1
	addi 0, 30, -8
	addi 30, 6, -8
	cmpd 5, 7, 6
	rldicl 30, 30, 61, 3
	addi 12, 10, 3
	addi 30, 30, 1
	b .LBB8_4
.LBB8_3:
	addi 8, 8, 1
	add 12, 12, 7
	add 10, 10, 7
	addi 28, 28, 1
	cmpld 6, 8, 4
	bge 6, .LBB8_14
.LBB8_4:
	add 26, 3, 8
	li 24, 0
	bc 12, 4, .LBB8_8
	mtctr 30
	li 25, 0
	mr 24, 12
.LBB8_6:
	ori 23, 25, 1
	ori 22, 25, 2
	ori 21, 25, 3
	ori 20, 25, 4
	ori 19, 25, 5
	ori 18, 25, 6
	ori 17, 25, 7
	mulld 16, 25, 4
	addi 25, 25, 8
	mulld 23, 23, 4
	mulld 22, 22, 4
	mulld 21, 21, 4
	mulld 20, 20, 4
	mulld 19, 19, 4
	mulld 18, 18, 4
	mulld 17, 17, 4
	lbzx 16, 26, 16
	stb 16, -3(24)
	lbzx 23, 26, 23
	lbzx 22, 26, 22
	stb 23, -2(24)
	lbzx 21, 26, 21
	lbzx 20, 26, 20
	lbzx 19, 26, 19
	lbzx 18, 26, 18
	stb 22, -1(24)
	stb 21, 0(24)
	stb 20, 1(24)
	stb 19, 2(24)
	stb 18, 3(24)
	lbzx 23, 26, 17
	stb 23, 4(24)
	addi 24, 24, 8
	bdnz .LBB8_6
	mr 24, 6
	bc 12, 22, .LBB8_3
.LBB8_8:
	mulld 23, 7, 28
	mr 25, 24
	bc 12, 2, .LBB8_11
	add 25, 11, 23
	mtctr 9
	add 25, 5, 25
	add 22, 25, 24
	mr 25, 24
.LBB8_10:
	mulld 21, 25, 4
	addi 25, 25, 1
	lbzx 21, 26, 21
	stbu 21, 1(22)
	bdnz .LBB8_10
.LBB8_11:
	sub 24, 24, 7
	cmpld 6, 24, 27
	bgt 6, .LBB8_3
	add 24, 0, 23
	sub 23, 29, 25
	rldicl 23, 23, 61, 3
	add 24, 5, 24
	addi 23, 23, 1
	add 24, 24, 25
	mtctr 23
.LBB8_13:
	mulld 23, 25, 4
	addi 22, 25, 1
	mulld 22, 22, 4
	lbzx 23, 26, 23
	lbzx 22, 26, 22
	stbu 23, 8(24)
	addi 23, 25, 2
	mulld 23, 23, 4
	stb 22, 1(24)
	addi 22, 25, 3
	mulld 22, 22, 4
	lbzx 23, 26, 23
	lbzx 22, 26, 22
	stb 23, 2(24)
	addi 23, 25, 4
	mulld 23, 23, 4
	stb 22, 3(24)
	addi 22, 25, 5
	mulld 22, 22, 4
	lbzx 23, 26, 23
	lbzx 22, 26, 22
	stb 23, 4(24)
	addi 23, 25, 6
	mulld 23, 23, 4
	stb 22, 5(24)
	addi 22, 25, 7
	addi 25, 25, 8
	lbzx 23, 26, 23
	stb 23, 6(24)
	mulld 23, 22, 4
	lbzx 23, 26, 23
	stb 23, 7(24)
	bdnz .LBB8_13
	b .LBB8_3
.LBB8_14:
	addi 1, 1, 160
	ld 0, 16(1)
	ld 30, -16(1)
	ld 29, -24(1)
	ld 28, -32(1)
	ld 27, -40(1)
	ld 26, -48(1)
	ld 25, -56(1)
	ld 24, -64(1)
	ld 23, -72(1)
	ld 22, -80(1)
	ld 21, -88(1)
	ld 20, -96(1)
	mtlr 0
	ld 19, -104(1)
	ld 18, -112(1)
	ld 17, -120(1)
	ld 16, -128(1)
	blr
.LBB8_15:
	addis 3, 2, .L_MergedGlobals@toc@ha
	addi 3, 3, .L_MergedGlobals@toc@l
	bl core::panicking::panic_const::panic_const_div_by_zero
	nop
