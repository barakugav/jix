jix_probe::byte_shuffle::decode_impl::<2, 64>:
	movq %rsi, %r8
	shrq %r8
	movabsq $4611686018427387840, %rax
	andq %r8, %rax
	je .LBB1_1
	leaq 56(%rdi), %rcx
	addq %rdi, %r8
	addq $56, %r8
	xorl %r9d, %r9d
.LBB1_4:
	movq -56(%rcx,%r9), %xmm0
	movq -56(%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, (%rdx,%r9,2)
	movq -48(%rcx,%r9), %xmm0
	movq -48(%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, 16(%rdx,%r9,2)
	movq -40(%rcx,%r9), %xmm0
	movq -40(%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, 32(%rdx,%r9,2)
	movq -32(%rcx,%r9), %xmm0
	movq -32(%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, 48(%rdx,%r9,2)
	movq -24(%rcx,%r9), %xmm0
	movq -24(%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, 64(%rdx,%r9,2)
	movq -16(%rcx,%r9), %xmm0
	movq -16(%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, 80(%rdx,%r9,2)
	movq -8(%rcx,%r9), %xmm0
	movq -8(%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, 96(%rdx,%r9,2)
	movq (%rcx,%r9), %xmm0
	movq (%r8,%r9), %xmm1
	punpcklbw %xmm1, %xmm0
	movdqu %xmm0, 112(%rdx,%r9,2)
	addq $64, %r9
	cmpq %rax, %r9
	jb .LBB1_4
	movl $2, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
.LBB1_1:
	xorl %r9d, %r9d
	movl $2, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
