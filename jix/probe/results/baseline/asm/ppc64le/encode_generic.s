jix_probe::byte_shuffle::encode_impl_generic:
.Lfunc_gep9:
	addis 2, 12, .TOC.-.Lfunc_gep9@ha
	addi 2, 2, .TOC.-.Lfunc_gep9@l
.Lfunc_lep9:
	mflr 0
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
	stdu 1, -144(1)
	cmpldi 7, 0
	std 0, 160(1)
	beq- 0, .LBB9_16
	divdu 6, 4, 7
	cmpld 8, 6
	bge 0, .LBB9_15
	sub 11, 4, 8
	cmpdi 7, 1
	add 9, 3, 8
	li 10, 0
	li 28, 32
	li 27, 48
	li 26, 64
	li 25, 80
	rldicr 4, 11, 0, 56
	cmpldi 6, 11, 15
	rldicr 30, 11, 0, 59
	cmpldi 1, 11, 128
	li 24, 96
	li 23, 112
	addi 12, 4, -128
	cmpd 5, 11, 4
	addi 0, 30, -16
	rldicl 29, 12, 57, 7
	add 12, 8, 30
	crand 20, 25, 2
	cmpd 6, 11, 30
	andi. 11, 11, 112
	add 11, 8, 4
	addi 30, 29, 1
	li 29, 16
	b .LBB9_4
.LBB9_3:
	addi 10, 10, 1
	addi 9, 9, 1
	addi 3, 3, 1
	cmpld 7, 10, 7
	beq 7, .LBB9_15
.LBB9_4:
	mulld 22, 10, 6
	mr 20, 8
	add 22, 5, 22
	bc 4, 20, .LBB9_13
	li 21, 0
	bc 12, 4, .LBB9_10
	mtctr 30
	mr 20, 9
.LBB9_7:
	lxvd2x 0, 0, 20
	add 19, 8, 21
	lxvd2x 1, 20, 29
	addi 21, 21, 128
	lxvd2x 2, 20, 28
	lxvd2x 3, 20, 27
	add 18, 22, 19
	lxvd2x 4, 20, 26
	lxvd2x 5, 20, 25
	stxvd2x 0, 22, 19
	lxvd2x 6, 20, 24
	lxvd2x 0, 20, 23
	addi 20, 20, 128
	stxvd2x 1, 18, 29
	stxvd2x 2, 18, 28
	stxvd2x 3, 18, 27
	stxvd2x 4, 18, 26
	stxvd2x 5, 18, 25
	stxvd2x 6, 18, 24
	stxvd2x 0, 18, 23
	bdnz .LBB9_7
	bc 12, 22, .LBB9_3
	mr 21, 4
	mr 20, 11
	bc 12, 2, .LBB9_13
.LBB9_10:
	sub 20, 0, 21
	rldicl 20, 20, 60, 4
	addi 20, 20, 1
	mtctr 20
.LBB9_11:
	lxvd2x 0, 9, 21
	add 20, 8, 21
	addi 21, 21, 16
	stxvd2x 0, 22, 20
	bdnz .LBB9_11
	mr 20, 12
	bc 12, 26, .LBB9_3
.LBB9_13:
	mulld 21, 7, 20
	add 21, 3, 21
.LBB9_14:
	lbz 19, 0(21)
	add 21, 21, 7
	stbx 19, 22, 20
	addi 20, 20, 1
	cmpld 7, 20, 6
	blt 7, .LBB9_14
	b .LBB9_3
.LBB9_15:
	addi 1, 1, 144
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
	blr
.LBB9_16:
	addis 3, 2, .L_MergedGlobals@toc@ha
	addi 3, 3, .L_MergedGlobals@toc@l
	addi 3, 3, 24
	bl core::panicking::panic_const::panic_const_div_by_zero
	nop
