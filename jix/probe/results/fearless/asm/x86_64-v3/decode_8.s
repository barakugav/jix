probe_byte_shuffle_decode_8:
	pushq %r15
	pushq %r14
	pushq %rbx
	movabsq $9223372036854775800, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB17_5
	movq %rsi, %rax
	shrq $8, %rax
	je .LBB17_4
	shrq $3, %rsi
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rdi,%rsi,8), %r9
	subq %rsi, %r9
	leaq (%rdi,%rcx,2), %r10
	addq %rdi, %r8
	leaq (%rdi,%rsi,4), %r11
	addq %rdi, %rcx
	leaq (%rdi,%rsi,2), %rbx
	addq %rdi, %rsi
	movq %rax, %r14
	shlq $5, %r14
	xorl %r15d, %r15d
.LBB17_3:
	vmovdqu (%rdi,%r15), %ymm0
	vmovdqu (%rsi,%r15), %ymm1
	vmovdqu (%rbx,%r15), %ymm2
	vmovdqu (%rcx,%r15), %ymm3
	vmovdqu (%r11,%r15), %ymm4
	vmovdqu (%r8,%r15), %ymm5
	vmovdqu (%r10,%r15), %ymm6
	vmovdqu (%r9,%r15), %ymm7
	vpunpcklbw %ymm4, %ymm0, %ymm8
	vpunpckhbw %ymm4, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm8, %ymm4
	vpunpcklbw %ymm5, %ymm1, %ymm9
	vpunpckhbw %ymm5, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm9, %ymm5
	vpunpcklbw %ymm6, %ymm2, %ymm10
	vpunpckhbw %ymm6, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm10, %ymm6
	vpunpcklbw %ymm7, %ymm3, %ymm11
	vpunpckhbw %ymm7, %ymm3, %ymm3
	vinserti128 $1, %xmm3, %ymm11, %ymm7
	vperm2i128 $49, %ymm0, %ymm8, %ymm0
	vperm2i128 $49, %ymm1, %ymm9, %ymm1
	vperm2i128 $49, %ymm2, %ymm10, %ymm2
	vperm2i128 $49, %ymm3, %ymm11, %ymm3
	vpunpcklbw %ymm6, %ymm4, %ymm8
	vpunpckhbw %ymm6, %ymm4, %ymm4
	vinserti128 $1, %xmm4, %ymm8, %ymm6
	vpunpcklbw %ymm2, %ymm0, %ymm9
	vpunpckhbw %ymm2, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm9, %ymm2
	vpunpcklbw %ymm7, %ymm5, %ymm10
	vpunpckhbw %ymm7, %ymm5, %ymm5
	vinserti128 $1, %xmm5, %ymm10, %ymm7
	vpunpcklbw %ymm3, %ymm1, %ymm11
	vpunpckhbw %ymm3, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm11, %ymm3
	vperm2i128 $49, %ymm4, %ymm8, %ymm4
	vperm2i128 $49, %ymm0, %ymm9, %ymm0
	vperm2i128 $49, %ymm5, %ymm10, %ymm5
	vperm2i128 $49, %ymm1, %ymm11, %ymm1
	vpunpcklbw %ymm7, %ymm6, %ymm8
	vpunpckhbw %ymm7, %ymm6, %ymm6
	vinserti128 $1, %xmm6, %ymm8, %ymm7
	vperm2i128 $49, %ymm6, %ymm8, %ymm6
	vpunpcklbw %ymm5, %ymm4, %ymm8
	vpunpckhbw %ymm5, %ymm4, %ymm4
	vinserti128 $1, %xmm4, %ymm8, %ymm5
	vperm2i128 $49, %ymm4, %ymm8, %ymm4
	vpunpcklbw %ymm3, %ymm2, %ymm8
	vpunpckhbw %ymm3, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm8, %ymm3
	vperm2i128 $49, %ymm2, %ymm8, %ymm2
	vpunpcklbw %ymm1, %ymm0, %ymm8
	vpunpckhbw %ymm1, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm8, %ymm1
	vperm2i128 $49, %ymm0, %ymm8, %ymm0
	vmovdqu %ymm7, (%rdx,%r15,8)
	vmovdqu %ymm6, 32(%rdx,%r15,8)
	vmovdqu %ymm5, 64(%rdx,%r15,8)
	vmovdqu %ymm4, 96(%rdx,%r15,8)
	vmovdqu %ymm3, 128(%rdx,%r15,8)
	vmovdqu %ymm2, 160(%rdx,%r15,8)
	vmovdqu %ymm1, 192(%rdx,%r15,8)
	vmovdqu %ymm0, 224(%rdx,%r15,8)
	addq $32, %r15
	cmpq %r15, %r14
	jne .LBB17_3
.LBB17_4:
	shlq $5, %rax
	popq %rbx
	popq %r14
	popq %r15
	vzeroupper
	retq
.LBB17_5:
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
