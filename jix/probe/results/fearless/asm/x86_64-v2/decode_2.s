probe_byte_shuffle_decode_2:
	movabsq $9223372036854775806, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB19_5
	movq %rsi, %rax
	shrq %rax
	shrq $8, %rsi
	je .LBB19_4
	leaq (%rax,%rdi), %rcx
	addq $112, %rcx
	shlq $7, %rsi
	xorl %r8d, %r8d
.LBB19_3:
	movdqu (%rdi,%r8), %xmm0
	movdqu -112(%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, (%rdx,%r8,2)
	movdqu %xmm0, 16(%rdx,%r8,2)
	movdqu 16(%rdi,%r8), %xmm0
	movdqu -96(%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, 32(%rdx,%r8,2)
	movdqu %xmm0, 48(%rdx,%r8,2)
	movdqu 32(%rdi,%r8), %xmm0
	movdqu -80(%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, 64(%rdx,%r8,2)
	movdqu %xmm0, 80(%rdx,%r8,2)
	movdqu 48(%rdi,%r8), %xmm0
	movdqu -64(%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, 96(%rdx,%r8,2)
	movdqu %xmm0, 112(%rdx,%r8,2)
	movdqu 64(%rdi,%r8), %xmm0
	movdqu -48(%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, 128(%rdx,%r8,2)
	movdqu %xmm0, 144(%rdx,%r8,2)
	movdqu 80(%rdi,%r8), %xmm0
	movdqu -32(%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, 160(%rdx,%r8,2)
	movdqu %xmm0, 176(%rdx,%r8,2)
	movdqu 96(%rdi,%r8), %xmm0
	movdqu -16(%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, 192(%rdx,%r8,2)
	movdqu %xmm0, 208(%rdx,%r8,2)
	movdqu 112(%rdi,%r8), %xmm0
	movdqu (%rcx,%r8), %xmm1
	movdqa %xmm0, %xmm2
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm0
	movdqu %xmm2, 224(%rdx,%r8,2)
	movdqu %xmm0, 240(%rdx,%r8,2)
	subq $-128, %r8
	cmpq %r8, %rsi
	jne .LBB19_3
.LBB19_4:
	movabsq $4611686018427387776, %rcx
	andq %rcx, %rax
	retq
.LBB19_5:
	pushq %rax
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
