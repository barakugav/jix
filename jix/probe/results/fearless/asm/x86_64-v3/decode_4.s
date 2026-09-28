probe_byte_shuffle_decode_4:
	movabsq $9223372036854775804, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB16_5
	movq %rsi, %rax
	shrq $2, %rax
	shrq $8, %rsi
	je .LBB16_4
	leaq (%rax,%rax,2), %rcx
	addq %rdi, %rcx
	addq $32, %rcx
	leaq (%rdi,%rax,2), %r8
	addq $32, %r8
	leaq (%rax,%rdi), %r9
	addq $32, %r9
	shlq $6, %rsi
	xorl %r10d, %r10d
.LBB16_3:
	vmovdqu (%rdi,%r10), %ymm0
	vmovdqu -32(%r9,%r10), %ymm1
	vmovdqu -32(%r8,%r10), %ymm2
	vmovdqu -32(%rcx,%r10), %ymm3
	vpunpcklbw %ymm2, %ymm0, %ymm4
	vpunpckhbw %ymm2, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm4, %ymm2
	vpunpcklbw %ymm3, %ymm1, %ymm5
	vpunpckhbw %ymm3, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm5, %ymm3
	vperm2i128 $49, %ymm0, %ymm4, %ymm0
	vperm2i128 $49, %ymm1, %ymm5, %ymm1
	vpunpcklbw %ymm3, %ymm2, %ymm4
	vpunpckhbw %ymm3, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm4, %ymm3
	vperm2i128 $49, %ymm2, %ymm4, %ymm2
	vpunpcklbw %ymm1, %ymm0, %ymm4
	vpunpckhbw %ymm1, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm4, %ymm1
	vperm2i128 $49, %ymm0, %ymm4, %ymm0
	vmovdqu %ymm3, (%rdx,%r10,4)
	vmovdqu %ymm2, 32(%rdx,%r10,4)
	vmovdqu %ymm1, 64(%rdx,%r10,4)
	vmovdqu %ymm0, 96(%rdx,%r10,4)
	vmovdqu 32(%rdi,%r10), %ymm0
	vmovdqu (%r9,%r10), %ymm1
	vmovdqu (%r8,%r10), %ymm2
	vmovdqu (%rcx,%r10), %ymm3
	vpunpcklbw %ymm2, %ymm0, %ymm4
	vpunpckhbw %ymm2, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm4, %ymm2
	vpunpcklbw %ymm3, %ymm1, %ymm5
	vpunpckhbw %ymm3, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm5, %ymm3
	vperm2i128 $49, %ymm0, %ymm4, %ymm0
	vperm2i128 $49, %ymm1, %ymm5, %ymm1
	vpunpcklbw %ymm3, %ymm2, %ymm4
	vpunpckhbw %ymm3, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm4, %ymm3
	vperm2i128 $49, %ymm2, %ymm4, %ymm2
	vpunpcklbw %ymm1, %ymm0, %ymm4
	vpunpckhbw %ymm1, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm4, %ymm1
	vperm2i128 $49, %ymm0, %ymm4, %ymm0
	vmovdqu %ymm3, 128(%rdx,%r10,4)
	vmovdqu %ymm2, 160(%rdx,%r10,4)
	vmovdqu %ymm1, 192(%rdx,%r10,4)
	vmovdqu %ymm0, 224(%rdx,%r10,4)
	addq $64, %r10
	cmpq %r10, %rsi
	jne .LBB16_3
.LBB16_4:
	movabsq $2305843009213693888, %rcx
	andq %rcx, %rax
	vzeroupper
	retq
.LBB16_5:
	pushq %rax
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
