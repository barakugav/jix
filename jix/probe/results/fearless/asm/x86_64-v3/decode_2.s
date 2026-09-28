probe_byte_shuffle_decode_2:
	movabsq $9223372036854775806, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB15_5
	movq %rsi, %rax
	shrq %rax
	shrq $8, %rsi
	je .LBB15_4
	leaq (%rax,%rdi), %rcx
	addq $96, %rcx
	shlq $7, %rsi
	xorl %r8d, %r8d
.LBB15_3:
	vmovdqu (%rdi,%r8), %ymm0
	vmovdqu -96(%rcx,%r8), %ymm1
	vpunpcklbw %ymm1, %ymm0, %ymm2
	vpunpckhbw %ymm1, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm2, %ymm1
	vperm2i128 $49, %ymm0, %ymm2, %ymm0
	vmovdqu %ymm1, (%rdx,%r8,2)
	vmovdqu %ymm0, 32(%rdx,%r8,2)
	vmovdqu 32(%rdi,%r8), %ymm0
	vmovdqu -64(%rcx,%r8), %ymm1
	vpunpcklbw %ymm1, %ymm0, %ymm2
	vpunpckhbw %ymm1, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm2, %ymm1
	vperm2i128 $49, %ymm0, %ymm2, %ymm0
	vmovdqu %ymm1, 64(%rdx,%r8,2)
	vmovdqu %ymm0, 96(%rdx,%r8,2)
	vmovdqu 64(%rdi,%r8), %ymm0
	vmovdqu -32(%rcx,%r8), %ymm1
	vpunpcklbw %ymm1, %ymm0, %ymm2
	vpunpckhbw %ymm1, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm2, %ymm1
	vperm2i128 $49, %ymm0, %ymm2, %ymm0
	vmovdqu %ymm1, 128(%rdx,%r8,2)
	vmovdqu %ymm0, 160(%rdx,%r8,2)
	vmovdqu 96(%rdi,%r8), %ymm0
	vmovdqu (%rcx,%r8), %ymm1
	vpunpcklbw %ymm1, %ymm0, %ymm2
	vpunpckhbw %ymm1, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm2, %ymm1
	vperm2i128 $49, %ymm0, %ymm2, %ymm0
	vmovdqu %ymm1, 192(%rdx,%r8,2)
	vmovdqu %ymm0, 224(%rdx,%r8,2)
	subq $-128, %r8
	cmpq %r8, %rsi
	jne .LBB15_3
.LBB15_4:
	movabsq $4611686018427387776, %rcx
	andq %rcx, %rax
	vzeroupper
	retq
.LBB15_5:
	pushq %rax
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
