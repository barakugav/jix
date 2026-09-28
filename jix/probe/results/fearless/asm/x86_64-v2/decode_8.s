probe_byte_shuffle_decode_8:
	pushq %r15
	pushq %r14
	pushq %rbx
	movabsq $9223372036854775800, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB21_5
	movq %rsi, %rax
	shrq $3, %rax
	shrq $8, %rsi
	je .LBB21_4
	leaq (%rax,%rax,2), %r11
	leaq (%rax,%rax,4), %r9
	leaq (%rdi,%rax,8), %rcx
	subq %rax, %rcx
	addq $16, %rcx
	leaq (%rdi,%r11,2), %r8
	addq $16, %r8
	leaq 16(%rdi,%r9), %r9
	leaq 16(%rdi,%rax,4), %r10
	addq %rdi, %r11
	addq $16, %r11
	leaq (%rdi,%rax,2), %rbx
	addq $16, %rbx
	leaq (%rax,%rdi), %r14
	addq $16, %r14
	shlq $5, %rsi
	xorl %r15d, %r15d
.LBB21_3:
	movdqu (%rdi,%r15), %xmm0
	movdqu -16(%r14,%r15), %xmm1
	movdqu -16(%rbx,%r15), %xmm7
	movdqu -16(%r11,%r15), %xmm3
	movdqu -16(%r10,%r15), %xmm4
	movdqu -16(%r9,%r15), %xmm5
	movdqu -16(%r8,%r15), %xmm6
	movdqu -16(%rcx,%r15), %xmm8
	movdqa %xmm0, %xmm2
	punpcklbw %xmm4, %xmm2
	punpckhbw %xmm4, %xmm0
	movdqa %xmm1, %xmm4
	punpcklbw %xmm5, %xmm4
	punpckhbw %xmm5, %xmm1
	movdqa %xmm7, %xmm9
	punpcklbw %xmm6, %xmm9
	punpckhbw %xmm6, %xmm7
	movdqa %xmm3, %xmm10
	punpcklbw %xmm8, %xmm10
	punpckhbw %xmm8, %xmm3
	movdqa %xmm2, %xmm5
	punpcklbw %xmm9, %xmm5
	punpckhbw %xmm9, %xmm2
	movdqa %xmm0, %xmm6
	punpcklbw %xmm7, %xmm6
	punpckhbw %xmm7, %xmm0
	movdqa %xmm4, %xmm7
	punpcklbw %xmm10, %xmm7
	punpckhbw %xmm10, %xmm4
	movdqa %xmm1, %xmm8
	punpcklbw %xmm3, %xmm8
	punpckhbw %xmm3, %xmm1
	movdqa %xmm5, %xmm3
	punpcklbw %xmm7, %xmm3
	punpckhbw %xmm7, %xmm5
	movdqa %xmm2, %xmm7
	punpcklbw %xmm4, %xmm7
	punpckhbw %xmm4, %xmm2
	movdqa %xmm6, %xmm4
	punpcklbw %xmm8, %xmm4
	punpckhbw %xmm8, %xmm6
	movdqa %xmm0, %xmm8
	punpcklbw %xmm1, %xmm8
	punpckhbw %xmm1, %xmm0
	movdqu %xmm3, (%rdx,%r15,8)
	movdqu %xmm5, 16(%rdx,%r15,8)
	movdqu %xmm7, 32(%rdx,%r15,8)
	movdqu %xmm2, 48(%rdx,%r15,8)
	movdqu %xmm4, 64(%rdx,%r15,8)
	movdqu %xmm6, 80(%rdx,%r15,8)
	movdqu %xmm8, 96(%rdx,%r15,8)
	movdqu %xmm0, 112(%rdx,%r15,8)
	movdqu 16(%rdi,%r15), %xmm0
	movdqu (%r14,%r15), %xmm1
	movdqu (%rbx,%r15), %xmm7
	movdqu (%r11,%r15), %xmm3
	movdqu (%r10,%r15), %xmm4
	movdqu (%r9,%r15), %xmm5
	movdqu (%r8,%r15), %xmm6
	movdqu (%rcx,%r15), %xmm8
	movdqa %xmm0, %xmm2
	punpcklbw %xmm4, %xmm2
	punpckhbw %xmm4, %xmm0
	movdqa %xmm1, %xmm4
	punpcklbw %xmm5, %xmm4
	punpckhbw %xmm5, %xmm1
	movdqa %xmm7, %xmm9
	punpcklbw %xmm6, %xmm9
	punpckhbw %xmm6, %xmm7
	movdqa %xmm3, %xmm10
	punpcklbw %xmm8, %xmm10
	punpckhbw %xmm8, %xmm3
	movdqa %xmm2, %xmm5
	punpcklbw %xmm9, %xmm5
	punpckhbw %xmm9, %xmm2
	movdqa %xmm0, %xmm6
	punpcklbw %xmm7, %xmm6
	punpckhbw %xmm7, %xmm0
	movdqa %xmm4, %xmm7
	punpcklbw %xmm10, %xmm7
	punpckhbw %xmm10, %xmm4
	movdqa %xmm1, %xmm8
	punpcklbw %xmm3, %xmm8
	punpckhbw %xmm3, %xmm1
	movdqa %xmm5, %xmm3
	punpcklbw %xmm7, %xmm3
	punpckhbw %xmm7, %xmm5
	movdqa %xmm2, %xmm7
	punpcklbw %xmm4, %xmm7
	punpckhbw %xmm4, %xmm2
	movdqa %xmm6, %xmm4
	punpcklbw %xmm8, %xmm4
	punpckhbw %xmm8, %xmm6
	movdqa %xmm0, %xmm8
	punpcklbw %xmm1, %xmm8
	punpckhbw %xmm1, %xmm0
	movdqu %xmm3, 128(%rdx,%r15,8)
	movdqu %xmm5, 144(%rdx,%r15,8)
	movdqu %xmm7, 160(%rdx,%r15,8)
	movdqu %xmm2, 176(%rdx,%r15,8)
	movdqu %xmm4, 192(%rdx,%r15,8)
	movdqu %xmm6, 208(%rdx,%r15,8)
	movdqu %xmm8, 224(%rdx,%r15,8)
	movdqu %xmm0, 240(%rdx,%r15,8)
	addq $32, %r15
	cmpq %r15, %rsi
	jne .LBB21_3
.LBB21_4:
	movabsq $1152921504606846944, %rcx
	andq %rcx, %rax
	popq %rbx
	popq %r14
	popq %r15
	retq
.LBB21_5:
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
