jix_probe::byte_shuffle::decode_impl::<4, 32>:
	movq %rsi, %rax
	shrq $2, %rax
	movabsq $2305843009213693920, %rcx
	andq %rax, %rcx
	je .LBB2_1
	leaq (%rax,%rax,2), %r8
	xorl %r9d, %r9d
.LBB2_4:
	leaq (%rdi,%r9), %r10
	vmovq (%rdi,%r9), %xmm0
	vmovq (%rax,%r10), %xmm1
	vpunpcklbw %xmm1, %xmm0, %xmm0
	vmovq (%r10,%rax,2), %xmm1
	vmovq (%r8,%r10), %xmm2
	vpunpcklbw %xmm2, %xmm1, %xmm1
	vpunpcklwd %xmm1, %xmm0, %xmm2
	vpunpckhwd %xmm1, %xmm0, %xmm0
	vmovdqu %xmm0, 16(%rdx,%r9,4)
	vmovdqu %xmm2, (%rdx,%r9,4)
	vmovq 8(%rdi,%r9), %xmm0
	vmovq 8(%rax,%r10), %xmm1
	vpunpcklbw %xmm1, %xmm0, %xmm0
	vmovq 8(%r10,%rax,2), %xmm1
	vmovq 8(%r8,%r10), %xmm2
	vpunpcklbw %xmm2, %xmm1, %xmm1
	vpunpcklwd %xmm1, %xmm0, %xmm2
	vpunpckhwd %xmm1, %xmm0, %xmm0
	vmovdqu %xmm0, 48(%rdx,%r9,4)
	vmovdqu %xmm2, 32(%rdx,%r9,4)
	vmovq 16(%rdi,%r9), %xmm0
	vmovq 16(%rax,%r10), %xmm1
	vpunpcklbw %xmm1, %xmm0, %xmm0
	vmovq 16(%r10,%rax,2), %xmm1
	vmovq 16(%r8,%r10), %xmm2
	vpunpcklbw %xmm2, %xmm1, %xmm1
	vpunpcklwd %xmm1, %xmm0, %xmm2
	vpunpckhwd %xmm1, %xmm0, %xmm0
	vmovdqu %xmm0, 80(%rdx,%r9,4)
	vmovdqu %xmm2, 64(%rdx,%r9,4)
	vmovq 24(%rdi,%r9), %xmm0
	vmovq 24(%rax,%r10), %xmm1
	vpunpcklbw %xmm1, %xmm0, %xmm0
	vmovq 24(%r10,%rax,2), %xmm1
	vmovq 24(%r8,%r10), %xmm2
	vpunpcklbw %xmm2, %xmm1, %xmm1
	vpunpcklwd %xmm1, %xmm0, %xmm2
	vpunpckhwd %xmm1, %xmm0, %xmm0
	vmovdqu %xmm0, 112(%rdx,%r9,4)
	vmovdqu %xmm2, 96(%rdx,%r9,4)
	addq $32, %r9
	cmpq %rcx, %r9
	jb .LBB2_4
	movl $4, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
.LBB2_1:
	xorl %r9d, %r9d
	movl $4, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
