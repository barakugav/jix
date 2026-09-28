probe_byte_shuffle_encode_2:
	movabsq $9223372036854775806, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB19_5
	movq %rsi, %rax
	shrq %rax
	shrq $8, %rsi
	je .LBB19_4
	leaq (%rax,%rdx), %rcx
	addq $96, %rcx
	shlq $7, %rsi
	xorl %r8d, %r8d
	vmovdqa .LCPI19_0(%rip), %ymm0
.LBB19_3:
	vmovdqu (%rdi,%r8,2), %ymm1
	vmovdqu 32(%rdi,%r8,2), %ymm2
	vpshufb %ymm0, %ymm1, %ymm1
	vpshufb %ymm0, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm1, %ymm3
	vperm2i128 $49, %ymm2, %ymm1, %ymm1
	vpunpcklqdq %ymm1, %ymm3, %ymm2
	vpunpckhqdq %ymm1, %ymm3, %ymm1
	vmovdqu %ymm2, (%rdx,%r8)
	vmovdqu %ymm1, -96(%rcx,%r8)
	vmovdqu 64(%rdi,%r8,2), %ymm1
	vmovdqu 96(%rdi,%r8,2), %ymm2
	vpshufb %ymm0, %ymm1, %ymm1
	vpshufb %ymm0, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm1, %ymm3
	vperm2i128 $49, %ymm2, %ymm1, %ymm1
	vpunpcklqdq %ymm1, %ymm3, %ymm2
	vpunpckhqdq %ymm1, %ymm3, %ymm1
	vmovdqu %ymm2, 32(%rdx,%r8)
	vmovdqu %ymm1, -64(%rcx,%r8)
	vmovdqu 128(%rdi,%r8,2), %ymm1
	vmovdqu 160(%rdi,%r8,2), %ymm2
	vpshufb %ymm0, %ymm1, %ymm1
	vpshufb %ymm0, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm1, %ymm3
	vperm2i128 $49, %ymm2, %ymm1, %ymm1
	vpunpcklqdq %ymm1, %ymm3, %ymm2
	vpunpckhqdq %ymm1, %ymm3, %ymm1
	vmovdqu %ymm2, 64(%rdx,%r8)
	vmovdqu %ymm1, -32(%rcx,%r8)
	vmovdqu 192(%rdi,%r8,2), %ymm1
	vmovdqu 224(%rdi,%r8,2), %ymm2
	vpshufb %ymm0, %ymm1, %ymm1
	vpshufb %ymm0, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm1, %ymm3
	vperm2i128 $49, %ymm2, %ymm1, %ymm1
	vpunpcklqdq %ymm1, %ymm3, %ymm2
	vpunpckhqdq %ymm1, %ymm3, %ymm1
	vmovdqu %ymm2, 96(%rdx,%r8)
	vmovdqu %ymm1, (%rcx,%r8)
	subq $-128, %r8
	cmpq %r8, %rsi
	jne .LBB19_3
.LBB19_4:
	movabsq $4611686018427387776, %rcx
	andq %rcx, %rax
	vzeroupper
	retq
.LBB19_5:
	pushq %rax
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
