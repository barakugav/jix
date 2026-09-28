jix_probe::byte_shuffle::decode_impl::<8, 16>:
.Lfunc_gep3:
	addis 2, 12, .TOC.-.Lfunc_gep3@ha
	addi 2, 2, .TOC.-.Lfunc_gep3@l
.Lfunc_lep3:
	mflr 0
	stdu 1, -192(1)
	std 0, 208(1)
	li 6, 48
	std 18, 80(1)
	std 19, 88(1)
	std 20, 96(1)
	std 21, 104(1)
	std 22, 112(1)
	std 23, 120(1)
	std 24, 128(1)
	stvx 30, 1, 6
	li 6, 64
	std 25, 136(1)
	std 26, 144(1)
	std 27, 152(1)
	std 28, 160(1)
	std 29, 168(1)
	std 30, 176(1)
	stvx 31, 1, 6
	rotldi 6, 4, 57
	rldic. 8, 6, 4, 4
	beq 0, .LBB3_3
	addis 12, 2, .LCPI3_1@toc@ha
	addis 11, 2, .LCPI3_0@toc@ha
	addis 30, 2, .LCPI3_2@toc@ha
	addis 28, 2, .LCPI3_4@toc@ha
	addis 27, 2, .LCPI3_5@toc@ha
	addis 26, 2, .LCPI3_6@toc@ha
	addis 29, 2, .LCPI3_3@toc@ha
	addi 8, 8, -1
	addi 12, 12, .LCPI3_1@toc@l
	addi 11, 11, .LCPI3_0@toc@l
	addi 0, 29, .LCPI3_3@toc@l
	rldicl 8, 8, 60, 4
	rldicl 6, 4, 61, 3
	addi 7, 3, -12
	sldi 9, 6, 1
	sldi 10, 6, 2
	lxvd2x 1, 0, 12
	addi 12, 30, .LCPI3_2@toc@l
	addi 30, 28, .LCPI3_4@toc@l
	addi 28, 27, .LCPI3_5@toc@l
	addi 27, 26, .LCPI3_6@toc@l
	lxvd2x 0, 0, 11
	lxvd2x 3, 0, 0
	addi 8, 8, 1
	lxvd2x 6, 0, 27
	addis 27, 2, .LCPI3_7@toc@ha
	lxvd2x 2, 0, 12
	lxvd2x 4, 0, 30
	lxvd2x 5, 0, 28
	mulli 11, 6, 6
	sldi 25, 6, 3
	mtctr 8
	addi 27, 27, .LCPI3_7@toc@l
	add 12, 7, 6
	add 0, 6, 9
	add 30, 6, 10
	sub 29, 25, 6
	add 28, 7, 9
	xxswapd 35, 1
	li 8, 0
	xxswapd 34, 0
	lxvd2x 0, 0, 27
	addis 27, 2, .LCPI3_8@toc@ha
	xxswapd 37, 3
	xxswapd 36, 2
	xxswapd 32, 4
	addi 27, 27, .LCPI3_8@toc@l
	xxswapd 33, 5
	xxswapd 38, 6
	xxswapd 39, 0
	lxvd2x 0, 0, 27
	addis 27, 2, .LCPI3_9@toc@ha
	addi 27, 27, .LCPI3_9@toc@l
	xxswapd 40, 0
	lxvd2x 0, 0, 27
	addis 27, 2, .LCPI3_10@toc@ha
	addi 27, 27, .LCPI3_10@toc@l
	xxswapd 41, 0
	lxvd2x 0, 0, 27
	addis 27, 2, .LCPI3_11@toc@ha
	addi 27, 27, .LCPI3_11@toc@l
	xxswapd 42, 0
	lxvd2x 0, 0, 27
	li 27, 16
	xxswapd 43, 0
.LBB3_2:
	sldi 26, 8, 3
	lbzu 24, 16(7)
	add 20, 3, 8
	lxsiwzx 51, 3, 8
	lbzu 23, 16(12)
	lbzu 22, 16(28)
	addi 8, 8, 16
	add 25, 5, 26
	lxsiwzx 44, 20, 6
	lxsiwzx 45, 20, 9
	lxsiwzx 46, 20, 0
	lxsiwzx 47, 20, 10
	lxsiwzx 48, 20, 30
	lxsiwzx 49, 20, 11
	lxsiwzx 50, 20, 29
	stb 24, 32(25)
	add 24, 20, 0
	stb 23, 33(25)
	add 23, 20, 10
	lbz 21, 4(24)
	lbz 19, 4(23)
	stb 22, 34(25)
	add 22, 20, 30
	lbz 18, 4(22)
	stb 21, 35(25)
	add 21, 20, 11
	stb 19, 36(25)
	add 20, 20, 29
	vperm 31, 12, 19, 7
	vperm 30, 14, 13, 8
	vperm 12, 12, 19, 2
	vperm 13, 14, 13, 3
	vperm 14, 16, 15, 5
	stb 18, 37(25)
	lbz 19, 4(21)
	stb 19, 38(25)
	lbz 19, 4(20)
	vperm 31, 30, 31, 4
	vperm 30, 16, 15, 9
	vperm 12, 13, 12, 4
	vperm 13, 17, 14, 0
	stb 19, 39(25)
	lbz 19, 1(7)
	stb 19, 40(25)
	lbz 19, 1(12)
	vperm 30, 17, 30, 10
	stb 19, 41(25)
	lbz 19, 1(28)
	vperm 12, 13, 12, 1
	stb 19, 42(25)
	lbz 19, 5(24)
	stb 19, 43(25)
	lbz 19, 5(23)
	vperm 31, 30, 31, 1
	stb 19, 44(25)
	lbz 19, 5(22)
	vperm 12, 18, 12, 6
	stb 19, 45(25)
	lbz 19, 5(21)
	stb 19, 46(25)
	lbz 19, 5(20)
	vperm 31, 18, 31, 11
	stb 19, 47(25)
	lbz 19, 2(7)
	stb 19, 48(25)
	lbz 19, 2(12)
	stb 19, 49(25)
	lbz 19, 2(28)
	xxswapd 0, 63
	stxvd2x 0, 25, 27
	stb 19, 50(25)
	lbz 19, 6(24)
	stb 19, 51(25)
	lbz 19, 6(23)
	stb 19, 52(25)
	lbz 19, 6(22)
	xxswapd 0, 44
	stxvd2x 0, 5, 26
	stb 19, 53(25)
	lbz 19, 6(21)
	stb 19, 54(25)
	lbz 19, 6(20)
	stb 19, 55(25)
	lbz 19, 3(7)
	stb 19, 56(25)
	lbz 19, 3(12)
	stb 19, 57(25)
	lbz 19, 3(28)
	stb 19, 58(25)
	lbz 19, 7(24)
	stb 19, 59(25)
	lbz 19, 7(23)
	stb 19, 60(25)
	lbz 19, 7(22)
	stb 19, 61(25)
	lbz 19, 7(21)
	stb 19, 62(25)
	lbz 19, 7(20)
	stb 19, 63(25)
	lbz 19, 4(7)
	stb 19, 64(25)
	lbz 19, 4(12)
	stb 19, 65(25)
	lbz 19, 4(28)
	stb 19, 66(25)
	lbz 19, 8(24)
	stb 19, 67(25)
	lbz 19, 8(23)
	stb 19, 68(25)
	lbz 19, 8(22)
	stb 19, 69(25)
	lbz 19, 8(21)
	stb 19, 70(25)
	lbz 19, 8(20)
	stb 19, 71(25)
	lbz 19, 5(7)
	stb 19, 72(25)
	lbz 19, 5(12)
	stb 19, 73(25)
	lbz 19, 5(28)
	stb 19, 74(25)
	lbz 19, 9(24)
	stb 19, 75(25)
	lbz 19, 9(23)
	stb 19, 76(25)
	lbz 19, 9(22)
	stb 19, 77(25)
	lbz 19, 9(21)
	stb 19, 78(25)
	lbz 19, 9(20)
	stb 19, 79(25)
	lbz 19, 6(7)
	stb 19, 80(25)
	lbz 19, 6(12)
	stb 19, 81(25)
	lbz 19, 6(28)
	stb 19, 82(25)
	lbz 19, 10(24)
	stb 19, 83(25)
	lbz 19, 10(23)
	stb 19, 84(25)
	lbz 19, 10(22)
	stb 19, 85(25)
	lbz 19, 10(21)
	stb 19, 86(25)
	lbz 19, 10(20)
	stb 19, 87(25)
	lbz 19, 7(7)
	stb 19, 88(25)
	lbz 19, 7(12)
	stb 19, 89(25)
	lbz 19, 7(28)
	stb 19, 90(25)
	lbz 19, 11(24)
	stb 19, 91(25)
	lbz 19, 11(23)
	stb 19, 92(25)
	lbz 19, 11(22)
	stb 19, 93(25)
	lbz 19, 11(21)
	stb 19, 94(25)
	lbz 19, 11(20)
	stb 19, 95(25)
	lbz 19, 8(7)
	stb 19, 96(25)
	lbz 19, 8(12)
	stb 19, 97(25)
	lbz 19, 8(28)
	stb 19, 98(25)
	lbz 19, 12(24)
	stb 19, 99(25)
	lbz 19, 12(23)
	stb 19, 100(25)
	lbz 19, 12(22)
	stb 19, 101(25)
	lbz 19, 12(21)
	stb 19, 102(25)
	lbz 19, 12(20)
	stb 19, 103(25)
	lbz 19, 9(7)
	stb 19, 104(25)
	lbz 19, 9(12)
	stb 19, 105(25)
	lbz 19, 9(28)
	stb 19, 106(25)
	lbz 19, 13(24)
	stb 19, 107(25)
	lbz 19, 13(23)
	stb 19, 108(25)
	lbz 19, 13(22)
	stb 19, 109(25)
	lbz 19, 13(21)
	stb 19, 110(25)
	lbz 19, 13(20)
	stb 19, 111(25)
	lbz 19, 10(7)
	stb 19, 112(25)
	lbz 19, 10(12)
	stb 19, 113(25)
	lbz 19, 10(28)
	stb 19, 114(25)
	lbz 19, 14(24)
	lbz 24, 15(24)
	stb 19, 115(25)
	lbz 19, 14(23)
	stb 24, 123(25)
	lbz 24, 15(23)
	stb 19, 116(25)
	lbz 19, 14(22)
	stb 24, 124(25)
	lbz 24, 15(22)
	stb 19, 117(25)
	lbz 19, 14(21)
	stb 24, 125(25)
	lbz 24, 15(21)
	stb 19, 118(25)
	lbz 19, 14(20)
	stb 24, 126(25)
	lbz 24, 15(20)
	stb 19, 119(25)
	lbz 19, 11(7)
	stb 24, 127(25)
	stb 19, 120(25)
	lbz 19, 11(12)
	stb 19, 121(25)
	lbz 19, 11(28)
	stb 19, 122(25)
	bdnz .LBB3_2
	b .LBB3_4
.LBB3_3:
	li 8, 0
.LBB3_4:
	li 7, 8
	bl jix_probe::byte_shuffle::decode_impl_generic
	nop
	li 3, 64
	ld 30, 176(1)
	ld 29, 168(1)
	lvx 31, 1, 3
	li 3, 48
	ld 28, 160(1)
	ld 27, 152(1)
	ld 26, 144(1)
	ld 25, 136(1)
	lvx 30, 1, 3
	ld 24, 128(1)
	ld 23, 120(1)
	ld 22, 112(1)
	ld 21, 104(1)
	ld 20, 96(1)
	ld 19, 88(1)
	ld 18, 80(1)
	addi 1, 1, 192
	ld 0, 16(1)
	mtlr 0
	blr
