jix_probe::byte_shuffle::decode_impl::<2, 64>:
.Lfunc_gep1:
	addis 2, 12, .TOC.-.Lfunc_gep1@ha
	addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
	mflr 0
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
	stdu 1, -128(1)
	rotldi 6, 4, 57
	std 0, 144(1)
	rldic. 8, 6, 6, 2
	beq 0, .LBB1_3
	addi 8, 8, -1
	rldicl 7, 4, 63, 1
	addi 6, 5, 112
	li 9, -112
	li 10, 8
	li 11, -96
	li 12, 16
	li 0, -80
	rldicl 8, 8, 58, 6
	add 7, 3, 7
	li 30, 24
	li 29, -64
	li 28, 32
	li 27, -48
	li 26, 40
	li 25, -32
	addi 8, 8, 1
	li 24, 48
	li 23, -16
	li 22, 56
	mtctr 8
	li 8, 0
.LBB1_2:
	lxsdx 34, 3, 8
	lxsdx 35, 7, 8
	add 21, 3, 8
	add 20, 7, 8
	addi 8, 8, 64
	lxsdx 36, 21, 10
	lxsdx 37, 21, 12
	lxsdx 32, 20, 12
	lxsdx 33, 20, 28
	lxsdx 38, 21, 26
	lxsdx 39, 21, 24
	lxsdx 40, 20, 24
	vmrghb 2, 3, 2
	lxsdx 35, 20, 10
	xxswapd 0, 34
	stxvd2x 0, 6, 9
	vmrghb 3, 3, 4
	vmrghb 4, 0, 5
	lxsdx 37, 21, 30
	lxsdx 32, 20, 30
	xxswapd 1, 35
	xxswapd 2, 36
	stxvd2x 1, 6, 11
	stxvd2x 2, 6, 0
	vmrghb 5, 0, 5
	lxsdx 32, 21, 28
	xxswapd 3, 37
	stxvd2x 3, 6, 29
	vmrghb 0, 1, 0
	lxsdx 33, 20, 26
	xxswapd 4, 32
	stxvd2x 4, 6, 27
	vmrghb 1, 1, 6
	vmrghb 6, 8, 7
	lxsdx 39, 21, 22
	lxsdx 40, 20, 22
	xxswapd 5, 33
	xxswapd 6, 38
	stxvd2x 5, 6, 25
	stxvd2x 6, 6, 23
	vmrghb 7, 8, 7
	xxswapd 7, 39
	stxvd2x 7, 0, 6
	addi 6, 6, 128
	bdnz .LBB1_2
	b .LBB1_4
.LBB1_3:
	li 8, 0
.LBB1_4:
	li 7, 2
	bl jix_probe::byte_shuffle::decode_impl_generic
	nop
	addi 1, 1, 128
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
	blr
