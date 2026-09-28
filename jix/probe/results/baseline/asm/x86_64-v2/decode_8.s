jix_probe::byte_shuffle::decode_impl::<8, 16>:
	pushq %r15
	pushq %r14
	pushq %rbx
	movq %rsi, %r9
	shrq $3, %r9
	movabsq $1152921504606846960, %rax
	andq %r9, %rax
	je .LBB3_1
	leaq (%r9,%r9,2), %rbx
	leaq (%r9,%r9,4), %r10
	leaq (%rdi,%r9,8), %rcx
	subq %r9, %rcx
	addq $12, %rcx
	leaq (%rdi,%rbx,2), %r8
	addq $12, %r8
	leaq 12(%rdi,%r10), %r10
	leaq 12(%rdi,%r9,4), %r11
	addq %rdi, %rbx
	addq $12, %rbx
	leaq (%rdi,%r9,2), %r14
	addq $12, %r14
	leaq (%r9,%rdi), %r15
	addq $12, %r15
	xorl %r9d, %r9d
	movq .LCPI3_2(%rip), %xmm0
	movq .LCPI3_3(%rip), %xmm1
.LBB3_4:
	movd (%rdi,%r9), %xmm2
	pinsrd $1, -12(%r15,%r9), %xmm2
	pinsrd $2, -12(%r14,%r9), %xmm2
	pinsrd $3, -12(%rbx,%r9), %xmm2
	movd -12(%r11,%r9), %xmm3
	pinsrd $1, -12(%r10,%r9), %xmm3
	pinsrd $2, -12(%r8,%r9), %xmm3
	pinsrd $3, -12(%rcx,%r9), %xmm3
	movdqa %xmm3, %xmm4
	pshufb %xmm0, %xmm4
	movdqa %xmm2, %xmm5
	pshufb %xmm0, %xmm5
	punpckldq %xmm4, %xmm5
	pshufb %xmm1, %xmm3
	pshufb %xmm1, %xmm2
	punpckldq %xmm3, %xmm2
	movdqu %xmm2, 16(%rdx,%r9,8)
	movd 4(%rdi,%r9), %xmm2
	pinsrd $1, -8(%r15,%r9), %xmm2
	pinsrd $2, -8(%r14,%r9), %xmm2
	pinsrd $3, -8(%rbx,%r9), %xmm2
	movdqu %xmm5, (%rdx,%r9,8)
	movd -8(%r11,%r9), %xmm3
	pinsrd $1, -8(%r10,%r9), %xmm3
	pinsrd $2, -8(%r8,%r9), %xmm3
	pinsrd $3, -8(%rcx,%r9), %xmm3
	movdqa %xmm3, %xmm4
	pshufb %xmm0, %xmm4
	movdqa %xmm2, %xmm5
	pshufb %xmm0, %xmm5
	punpckldq %xmm4, %xmm5
	pshufb %xmm1, %xmm3
	pshufb %xmm1, %xmm2
	punpckldq %xmm3, %xmm2
	movdqu %xmm2, 48(%rdx,%r9,8)
	movd 8(%rdi,%r9), %xmm2
	pinsrd $1, -4(%r15,%r9), %xmm2
	pinsrd $2, -4(%r14,%r9), %xmm2
	pinsrd $3, -4(%rbx,%r9), %xmm2
	movdqu %xmm5, 32(%rdx,%r9,8)
	movd -4(%r11,%r9), %xmm3
	pinsrd $1, -4(%r10,%r9), %xmm3
	pinsrd $2, -4(%r8,%r9), %xmm3
	pinsrd $3, -4(%rcx,%r9), %xmm3
	movdqa %xmm3, %xmm4
	pshufb %xmm0, %xmm4
	movdqa %xmm2, %xmm5
	pshufb %xmm0, %xmm5
	punpckldq %xmm4, %xmm5
	pshufb %xmm1, %xmm3
	pshufb %xmm1, %xmm2
	punpckldq %xmm3, %xmm2
	movdqu %xmm2, 80(%rdx,%r9,8)
	movd 12(%rdi,%r9), %xmm2
	pinsrd $1, (%r15,%r9), %xmm2
	pinsrd $2, (%r14,%r9), %xmm2
	pinsrd $3, (%rbx,%r9), %xmm2
	movdqu %xmm5, 64(%rdx,%r9,8)
	movd (%r11,%r9), %xmm3
	pinsrd $1, (%r10,%r9), %xmm3
	pinsrd $2, (%r8,%r9), %xmm3
	pinsrd $3, (%rcx,%r9), %xmm3
	movdqa %xmm3, %xmm4
	pshufb %xmm0, %xmm4
	movdqa %xmm2, %xmm5
	pshufb %xmm0, %xmm5
	punpckldq %xmm4, %xmm5
	pshufb %xmm1, %xmm3
	pshufb %xmm1, %xmm2
	punpckldq %xmm3, %xmm2
	movdqu %xmm2, 112(%rdx,%r9,8)
	movdqu %xmm5, 96(%rdx,%r9,8)
	addq $16, %r9
	cmpq %rax, %r9
	jb .LBB3_4
	jmp .LBB3_2
.LBB3_1:
	xorl %r9d, %r9d
.LBB3_2:
	movl $8, %r8d
	popq %rbx
	popq %r14
	popq %r15
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
