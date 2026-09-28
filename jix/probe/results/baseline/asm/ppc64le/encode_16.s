jix_probe::byte_shuffle::encode_impl::<16, 8>:
.Lfunc_gep4:
	addis 2, 12, .TOC.-.Lfunc_gep4@ha
	addi 2, 2, .TOC.-.Lfunc_gep4@l
.Lfunc_lep4:
	mflr 0
	stdu 1, -304(1)
	std 0, 320(1)
	rotldi 6, 4, 57
	std 14, 160(1)
	std 15, 168(1)
	rldic. 6, 6, 3, 5
	std 16, 176(1)
	std 17, 184(1)
	std 18, 192(1)
	std 19, 200(1)
	std 20, 208(1)
	std 21, 216(1)
	std 22, 224(1)
	std 23, 232(1)
	std 24, 240(1)
	std 25, 248(1)
	std 26, 256(1)
	std 27, 264(1)
	std 28, 272(1)
	std 29, 280(1)
	std 30, 288(1)
	beq 0, .LBB4_5
	rldicl 7, 4, 60, 4
	li 8, 0
	addi 9, 1, 30
	li 10, 112
	addi 11, 1, 32
	li 0, 96
	li 30, 80
	li 29, 64
	sldi 12, 7, 1
	li 28, 48
	li 27, 32
	li 26, 16
	li 25, 8
	mr 24, 5
.LBB4_2:
	sldi 23, 8, 4
	add 22, 3, 23
	lxvd2x 0, 22, 10
	stxvd2x 0, 11, 10
	lxvd2x 0, 22, 0
	stxvd2x 0, 11, 0
	lxvd2x 0, 22, 30
	stxvd2x 0, 11, 30
	lxvd2x 0, 22, 29
	stxvd2x 0, 11, 29
	lxvd2x 0, 22, 28
	stxvd2x 0, 11, 28
	lxvd2x 0, 22, 27
	stxvd2x 0, 11, 27
	lxvd2x 0, 22, 26
	mr 22, 9
	stxvd2x 0, 11, 26
	lxvd2x 0, 3, 23
	mr 23, 24
	stxvd2x 0, 0, 11
	mtctr 25
.LBB4_3:
	lbzu 21, 2(22)
	lbz 20, 16(22)
	lbz 19, 32(22)
	lbz 18, 48(22)
	lbz 17, 64(22)
	lbz 16, 80(22)
	lbz 15, 96(22)
	lbz 14, 112(22)
	rlwimi 21, 20, 8, 16, 23
	lbz 20, 1(22)
	rlwimi 21, 19, 16, 8, 15
	lbz 19, 17(22)
	rlwimi 21, 18, 24, 0, 7
	lbz 18, 33(22)
	rldimi 21, 17, 32, 24
	lbz 17, 49(22)
	rlwimi 20, 19, 8, 16, 23
	lbz 19, 81(22)
	rldimi 21, 16, 40, 16
	lbz 16, 65(22)
	rlwimi 20, 18, 16, 8, 15
	lbz 18, 113(22)
	rlwimi 20, 17, 24, 0, 7
	rldimi 21, 15, 48, 8
	lbz 15, 97(22)
	rldimi 20, 16, 32, 24
	rldimi 21, 14, 56, 0
	rldimi 20, 19, 40, 16
	std 21, 0(23)
	rldimi 20, 15, 48, 8
	rldimi 20, 18, 56, 0
	stdx 20, 23, 7
	add 23, 23, 12
	bdnz .LBB4_3
	addi 8, 8, 8
	addi 24, 24, 8
	cmpld 8, 6
	blt 0, .LBB4_2
	b .LBB4_6
.LBB4_5:
	li 8, 0
.LBB4_6:
	li 7, 16
	bl jix_probe::byte_shuffle::encode_impl_generic
	nop
	ld 30, 288(1)
	ld 29, 280(1)
	ld 28, 272(1)
	ld 27, 264(1)
	ld 26, 256(1)
	ld 25, 248(1)
	ld 24, 240(1)
	ld 23, 232(1)
	ld 22, 224(1)
	ld 21, 216(1)
	ld 20, 208(1)
	ld 19, 200(1)
	ld 18, 192(1)
	ld 17, 184(1)
	ld 16, 176(1)
	ld 15, 168(1)
	ld 14, 160(1)
	addi 1, 1, 304
	ld 0, 16(1)
	mtlr 0
	blr
