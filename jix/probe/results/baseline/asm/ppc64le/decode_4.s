jix_probe::byte_shuffle::decode_impl::<4, 32>:
.Lfunc_gep2:
	addis 2, 12, .TOC.-.Lfunc_gep2@ha
	addi 2, 2, .TOC.-.Lfunc_gep2@l
.Lfunc_lep2:
	mflr 0
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
	rotldi 6, 4, 57
	std 0, 160(1)
	rldic. 6, 6, 5, 3
	beq 0, .LBB2_3
	addis 8, 2, .LCPI2_0@toc@ha
	addis 9, 2, .LCPI2_1@toc@ha
	addis 10, 2, .LCPI2_2@toc@ha
	addis 11, 2, .LCPI2_3@toc@ha
	addis 12, 2, .LCPI2_4@toc@ha
	addis 30, 2, .LCPI2_5@toc@ha
	addi 7, 6, -1
	rldicl 0, 4, 62, 2
	addi 8, 8, .LCPI2_0@toc@l
	addi 9, 9, .LCPI2_1@toc@l
	addi 10, 10, .LCPI2_2@toc@l
	addi 11, 11, .LCPI2_3@toc@l
	addi 12, 12, .LCPI2_4@toc@l
	addi 30, 30, .LCPI2_5@toc@l
	rldicl 29, 7, 59, 5
	addi 7, 3, -4
	lxvd2x 0, 0, 8
	lxvd2x 1, 0, 9
	lxvd2x 2, 0, 10
	lxvd2x 3, 0, 11
	lxvd2x 4, 0, 12
	lxvd2x 5, 0, 30
	addi 29, 29, 1
	sldi 28, 0, 1
	add 11, 3, 0
	add 12, 7, 0
	add 0, 0, 28
	mtctr 29
	addi 6, 5, 80
	li 8, 0
	li 9, -80
	li 10, 4
	add 30, 3, 28
	add 29, 7, 28
	add 28, 3, 0
	li 0, -48
	xxswapd 34, 0
	xxswapd 35, 1
	xxswapd 36, 2
	xxswapd 37, 3
	xxswapd 32, 4
	xxswapd 33, 5
	li 27, -64
	li 26, 12
	li 25, -16
	li 24, -32
	li 23, 20
	li 22, 16
.LBB2_2:
	add 20, 3, 8
	add 21, 11, 8
	lfiwzx 6, 3, 8
	lfiwzx 7, 11, 8
	lxsiwzx 48, 30, 8
	ld 19, 4(20)
	mtfprd 0, 19
	ld 19, 4(21)
	mtfprd 1, 19
	ld 19, 12(20)
	ld 20, 20(20)
	mtfprd 2, 19
	add 19, 30, 8
	mtfprd 4, 20
	lbzu 20, 32(29)
	lxsdx 39, 19, 10
	lxsdx 38, 19, 26
	lxsdx 40, 19, 23
	ld 19, 12(21)
	ld 21, 20(21)
	xxmrghw 44, 7, 6
	stb 20, 34(6)
	mtfprd 5, 21
	lbzu 21, 32(7)
	mtfprd 3, 19
	stb 21, 32(6)
	lbzu 21, 32(12)
	stb 21, 33(6)
	add 21, 28, 8
	vperm 12, 16, 12, 2
	lbz 20, 28(21)
	lxsdx 41, 21, 10
	lxsdx 42, 21, 26
	lxsdx 43, 21, 23
	stb 20, 35(6)
	lbz 20, 1(7)
	stb 20, 36(6)
	lbz 20, 1(12)
	stb 20, 37(6)
	lbz 20, 1(29)
	stb 20, 38(6)
	lbz 20, 29(21)
	stb 20, 39(6)
	lbz 20, 2(7)
	stb 20, 40(6)
	lbz 20, 2(12)
	stb 20, 41(6)
	lbz 20, 2(29)
	stb 20, 42(6)
	lbz 20, 30(21)
	lbz 21, 31(21)
	stb 20, 43(6)
	xxmrghd 45, 1, 0
	lbz 20, 3(7)
	stb 21, 47(6)
	stb 20, 44(6)
	lbz 20, 3(12)
	stb 20, 45(6)
	lbz 20, 3(29)
	stb 20, 46(6)
	vperm 16, 7, 13, 4
	vperm 7, 7, 13, 0
	xxmrghd 46, 3, 2
	xxmrghd 47, 5, 4
	vperm 13, 6, 14, 4
	vperm 6, 6, 14, 0
	vperm 14, 8, 15, 4
	vperm 8, 8, 15, 0
	lxsiwzx 47, 28, 8
	addi 8, 8, 32
	vperm 12, 15, 12, 3
	vperm 15, 9, 16, 5
	vperm 7, 9, 7, 1
	vperm 9, 10, 13, 5
	vperm 6, 10, 6, 1
	vperm 10, 11, 14, 5
	vperm 8, 11, 8, 1
	xxswapd 0, 44
	xxswapd 1, 39
	xxswapd 2, 47
	xxswapd 4, 41
	xxswapd 3, 38
	xxswapd 5, 40
	xxswapd 6, 42
	stxvd2x 0, 6, 9
	stxvd2x 1, 6, 0
	stxvd2x 2, 6, 27
	stxvd2x 3, 6, 25
	stxvd2x 4, 6, 24
	stxvd2x 5, 6, 22
	stxvd2x 6, 0, 6
	addi 6, 6, 128
	bdnz .LBB2_2
	b .LBB2_4
.LBB2_3:
	li 8, 0
.LBB2_4:
	li 7, 4
	bl jix_probe::byte_shuffle::decode_impl_generic
	nop
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
	blr
