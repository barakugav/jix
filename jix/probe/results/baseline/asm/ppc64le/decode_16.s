jix_probe::byte_shuffle::decode_impl::<16, 8>:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	mflr 0
	std 14, -144(1)
	std 15, -136(1)
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
	std 31, -8(1)
	stdu 1, -256(1)
	rotldi 6, 4, 57
	std 0, 272(1)
	std 5, 56(1)
	rldic. 19, 6, 3, 5
	std 4, 48(1)
	std 3, 40(1)
	beq 0, .LBB0_3
	rldicl 4, 4, 60, 4
	addi 22, 3, -8
	addi 23, 5, 63
	sldi 6, 4, 2
	sldi 20, 4, 1
	sldi 8, 4, 3
	mulli 7, 4, 6
	mulli 9, 4, 10
	add 21, 22, 4
	add 29, 4, 6
	add 6, 3, 6
	add 30, 4, 20
	mulli 10, 4, 11
	mulli 11, 4, 12
	mulli 0, 4, 13
	mulli 25, 4, 14
	sub 28, 8, 4
	add 27, 4, 8
	add 20, 22, 20
	std 6, 104(1)
	add 6, 3, 8
	sldi 12, 4, 4
	sub 24, 12, 4
	add 24, 3, 24
	li 8, 0
	add 26, 3, 0
	add 25, 3, 25
	std 6, 96(1)
	add 6, 3, 30
	add 30, 3, 27
	add 27, 3, 11
	std 6, 88(1)
	add 6, 3, 29
	add 29, 3, 9
	std 6, 80(1)
	add 6, 3, 28
	add 28, 3, 10
	std 6, 72(1)
	add 6, 3, 7
	addi 3, 19, -1
	rldicl 3, 3, 61, 3
	std 6, 64(1)
	addi 3, 3, 1
	mtctr 3
.LBB0_2:
	ld 0, 8(22)
	ld 19, 8(21)
	addi 21, 21, 8
	addi 22, 22, 8
	ld 18, 8(20)
	ldx 5, 30, 8
	addi 20, 20, 8
	stb 5, -54(23)
	rldicl 3, 0, 56, 8
	ldx 4, 29, 8
	stb 4, -53(23)
	stb 0, -63(23)
	stb 19, -62(23)
	stb 18, -61(23)
	stb 3, -47(23)
	rldicl 3, 19, 56, 8
	stb 3, -46(23)
	rldicl 3, 18, 56, 8
	stb 3, -45(23)
	ld 3, 88(1)
	ldx 17, 3, 8
	rldicl 3, 17, 56, 8
	stb 17, -60(23)
	stb 3, -44(23)
	ld 3, 104(1)
	ldx 16, 3, 8
	rldicl 3, 16, 56, 8
	stb 16, -59(23)
	stb 3, -43(23)
	ld 3, 80(1)
	ldx 15, 3, 8
	rldicl 3, 15, 56, 8
	stb 15, -58(23)
	stb 3, -42(23)
	ld 3, 64(1)
	ldx 14, 3, 8
	rldicl 3, 14, 56, 8
	stb 14, -57(23)
	stb 3, -41(23)
	ld 3, 72(1)
	ldx 31, 3, 8
	rldicl 3, 31, 56, 8
	stb 31, -56(23)
	stb 3, -40(23)
	ld 3, 96(1)
	ldx 6, 3, 8
	rldicl 3, 6, 56, 8
	stb 6, -55(23)
	stb 3, -39(23)
	rldicl 3, 5, 56, 8
	stb 3, -38(23)
	rldicl 3, 4, 56, 8
	stb 3, -37(23)
	ldx 3, 28, 8
	rldicl 7, 3, 56, 8
	stb 3, -52(23)
	stb 7, -36(23)
	ldx 7, 27, 8
	rldicl 9, 7, 56, 8
	stb 7, -51(23)
	stb 9, -35(23)
	ldx 9, 26, 8
	rldicl 10, 9, 56, 8
	stb 9, -50(23)
	stb 10, -34(23)
	ldx 10, 25, 8
	rldicl 11, 10, 56, 8
	stb 10, -49(23)
	stb 11, -33(23)
	ldx 11, 24, 8
	addi 8, 8, 8
	rldicl 12, 11, 56, 8
	stb 11, -48(23)
	stb 12, -32(23)
	rldicl 12, 0, 48, 16
	stb 12, -31(23)
	rldicl 12, 19, 48, 16
	stb 12, -30(23)
	rldicl 12, 18, 48, 16
	stb 12, -29(23)
	rldicl 12, 17, 48, 16
	stb 12, -28(23)
	rldicl 12, 16, 48, 16
	stb 12, -27(23)
	rldicl 12, 15, 48, 16
	stb 12, -26(23)
	rldicl 12, 14, 48, 16
	stb 12, -25(23)
	rldicl 12, 31, 48, 16
	stb 12, -24(23)
	rldicl 12, 6, 48, 16
	stb 12, -23(23)
	rldicl 12, 5, 48, 16
	stb 12, -22(23)
	rldicl 12, 4, 48, 16
	stb 12, -21(23)
	rldicl 12, 3, 48, 16
	stb 12, -20(23)
	rldicl 12, 7, 48, 16
	stb 12, -19(23)
	rldicl 12, 9, 48, 16
	stb 12, -18(23)
	rldicl 12, 10, 48, 16
	stb 12, -17(23)
	rldicl 12, 11, 48, 16
	stb 12, -16(23)
	rldicl 12, 0, 40, 24
	stb 12, -15(23)
	rldicl 12, 19, 40, 24
	stb 12, -14(23)
	rldicl 12, 18, 40, 24
	stb 12, -13(23)
	rldicl 12, 17, 40, 24
	stb 12, -12(23)
	rldicl 12, 16, 40, 24
	stb 12, -11(23)
	rldicl 12, 15, 40, 24
	stb 12, -10(23)
	rldicl 12, 14, 40, 24
	stb 12, -9(23)
	rldicl 12, 31, 40, 24
	stb 12, -8(23)
	rldicl 12, 6, 40, 24
	stb 12, -7(23)
	rldicl 12, 5, 40, 24
	stb 12, -6(23)
	rldicl 12, 4, 40, 24
	stb 12, -5(23)
	rldicl 12, 3, 40, 24
	stb 12, -4(23)
	rldicl 12, 7, 40, 24
	stb 12, -3(23)
	rldicl 12, 9, 40, 24
	stb 12, -2(23)
	rldicl 12, 10, 40, 24
	stb 12, -1(23)
	rldicl 12, 11, 40, 24
	stb 12, 0(23)
	rldicl 12, 0, 32, 32
	stb 12, 1(23)
	rldicl 12, 19, 32, 32
	stb 12, 2(23)
	rldicl 12, 18, 32, 32
	stb 12, 3(23)
	rldicl 12, 17, 32, 32
	stb 12, 4(23)
	rldicl 12, 16, 32, 32
	stb 12, 5(23)
	rldicl 12, 15, 32, 32
	stb 12, 6(23)
	rldicl 12, 14, 32, 32
	stb 12, 7(23)
	rldicl 12, 31, 32, 32
	stb 12, 8(23)
	rldicl 12, 6, 32, 32
	stb 12, 9(23)
	rldicl 12, 5, 32, 32
	stb 12, 10(23)
	rldicl 12, 4, 32, 32
	stb 12, 11(23)
	rldicl 12, 3, 32, 32
	stb 12, 12(23)
	rldicl 12, 7, 32, 32
	stb 12, 13(23)
	rldicl 12, 9, 32, 32
	stb 12, 14(23)
	rldicl 12, 10, 32, 32
	stb 12, 15(23)
	rldicl 12, 11, 32, 32
	stb 12, 16(23)
	rldicl 12, 0, 24, 40
	stb 12, 17(23)
	rldicl 12, 19, 24, 40
	stb 12, 18(23)
	rldicl 12, 18, 24, 40
	stb 12, 19(23)
	rldicl 12, 17, 24, 40
	stb 12, 20(23)
	rldicl 12, 16, 24, 40
	stb 12, 21(23)
	rldicl 12, 15, 24, 40
	stb 12, 22(23)
	rldicl 12, 14, 24, 40
	stb 12, 23(23)
	rldicl 12, 31, 24, 40
	stb 12, 24(23)
	rldicl 12, 6, 24, 40
	stb 12, 25(23)
	rldicl 12, 5, 24, 40
	stb 12, 26(23)
	rldicl 12, 4, 24, 40
	stb 12, 27(23)
	rldicl 12, 3, 24, 40
	stb 12, 28(23)
	rldicl 12, 7, 24, 40
	stb 12, 29(23)
	rldicl 12, 9, 24, 40
	stb 12, 30(23)
	rldicl 12, 10, 24, 40
	stb 12, 31(23)
	rldicl 12, 11, 24, 40
	stb 12, 32(23)
	rldicl 12, 0, 16, 48
	stb 12, 33(23)
	rldicl 12, 19, 16, 48
	stb 12, 34(23)
	rldicl 12, 18, 16, 48
	stb 12, 35(23)
	rldicl 12, 17, 16, 48
	stb 12, 36(23)
	rldicl 12, 16, 16, 48
	stb 12, 37(23)
	rldicl 12, 15, 16, 48
	stb 12, 38(23)
	rldicl 12, 14, 16, 48
	stb 12, 39(23)
	rldicl 12, 31, 16, 48
	stb 12, 40(23)
	rldicl 12, 6, 16, 48
	rldicl 6, 6, 8, 56
	stb 12, 41(23)
	rldicl 12, 5, 16, 48
	rldicl 5, 5, 8, 56
	stb 6, 57(23)
	stb 12, 42(23)
	rldicl 12, 4, 16, 48
	rldicl 4, 4, 8, 56
	stb 5, 58(23)
	stb 12, 43(23)
	rldicl 12, 3, 16, 48
	rldicl 3, 3, 8, 56
	stb 4, 59(23)
	stb 12, 44(23)
	rldicl 12, 7, 16, 48
	stb 3, 60(23)
	rldicl 3, 7, 8, 56
	stb 12, 45(23)
	rldicl 12, 9, 16, 48
	stb 3, 61(23)
	rldicl 3, 9, 8, 56
	stb 12, 46(23)
	rldicl 12, 10, 16, 48
	stb 3, 62(23)
	rldicl 3, 10, 8, 56
	stb 12, 47(23)
	rldicl 12, 11, 16, 48
	stb 3, 63(23)
	rldicl 3, 11, 8, 56
	stb 12, 48(23)
	rldicl 12, 0, 8, 56
	stb 3, 64(23)
	stb 12, 49(23)
	rldicl 12, 19, 8, 56
	stb 12, 50(23)
	rldicl 12, 18, 8, 56
	stb 12, 51(23)
	rldicl 12, 17, 8, 56
	stb 12, 52(23)
	rldicl 12, 16, 8, 56
	stb 12, 53(23)
	rldicl 12, 15, 8, 56
	stb 12, 54(23)
	rldicl 12, 14, 8, 56
	stb 12, 55(23)
	rldicl 12, 31, 8, 56
	stb 12, 56(23)
	addi 23, 23, 128
	bdnz .LBB0_2
	b .LBB0_4
.LBB0_3:
	li 8, 0
.LBB0_4:
	ld 3, 40(1)
	ld 4, 48(1)
	li 7, 16
	ld 5, 56(1)
	bl jix_probe::byte_shuffle::decode_impl_generic
	nop
	addi 1, 1, 256
	ld 0, 16(1)
	ld 31, -8(1)
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
	mtlr 0
	ld 20, -96(1)
	ld 19, -104(1)
	ld 18, -112(1)
	ld 17, -120(1)
	ld 16, -128(1)
	ld 15, -136(1)
	ld 14, -144(1)
	blr
