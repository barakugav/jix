jix_probe::byte_shuffle::decode_impl::<4, 32>:
	movq %rsi, %rax
	shrq $2, %rax
	movabsq $2305843009213693920, %rcx
	andq %rax, %rcx
	je .LBB2_1
	leaq (%rax,%rax,2), %r8
	xorl %r9d, %r9d
	movdqa .LCPI2_0(%rip), %xmm0
	movdqa .LCPI2_1(%rip), %xmm1
.LBB2_4:
	leaq (%rdi,%r9), %r10
	movd (%r8,%r10), %xmm2
	movd (%rdi,%r9), %xmm3
	pinsrd $1, (%rax,%r10), %xmm3
	pinsrd $2, (%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, (%rdx,%r9,4)
	movd 4(%r8,%r10), %xmm2
	movd 4(%rdi,%r9), %xmm3
	pinsrd $1, 4(%rax,%r10), %xmm3
	pinsrd $2, 4(%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, 16(%rdx,%r9,4)
	movd 8(%r8,%r10), %xmm2
	movd 8(%rdi,%r9), %xmm3
	pinsrd $1, 8(%rax,%r10), %xmm3
	pinsrd $2, 8(%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, 32(%rdx,%r9,4)
	movd 12(%r8,%r10), %xmm2
	movd 12(%rdi,%r9), %xmm3
	pinsrd $1, 12(%rax,%r10), %xmm3
	pinsrd $2, 12(%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, 48(%rdx,%r9,4)
	movd 16(%r8,%r10), %xmm2
	movd 16(%rdi,%r9), %xmm3
	pinsrd $1, 16(%rax,%r10), %xmm3
	pinsrd $2, 16(%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, 64(%rdx,%r9,4)
	movd 20(%r8,%r10), %xmm2
	movd 20(%rdi,%r9), %xmm3
	pinsrd $1, 20(%rax,%r10), %xmm3
	pinsrd $2, 20(%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, 80(%rdx,%r9,4)
	movd 24(%r8,%r10), %xmm2
	movd 24(%rdi,%r9), %xmm3
	pinsrd $1, 24(%rax,%r10), %xmm3
	pinsrd $2, 24(%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, 96(%rdx,%r9,4)
	movd 28(%r8,%r10), %xmm2
	movd 28(%rdi,%r9), %xmm3
	pinsrd $1, 28(%rax,%r10), %xmm3
	pinsrd $2, 28(%r10,%rax,2), %xmm3
	pshufb %xmm0, %xmm3
	pshufb %xmm1, %xmm2
	por %xmm3, %xmm2
	movdqu %xmm2, 112(%rdx,%r9,4)
	addq $32, %r9
	cmpq %rcx, %r9
	jb .LBB2_4
	movl $4, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
.LBB2_1:
	xorl %r9d, %r9d
	movl $4, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
