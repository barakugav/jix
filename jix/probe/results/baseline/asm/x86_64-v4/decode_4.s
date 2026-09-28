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
	vmovdqu (%rdi,%r9), %xmm0
	vmovdqu (%rax,%r10), %xmm1
	vmovdqu (%r10,%rax,2), %xmm2
	vmovdqu (%r8,%r10), %xmm3
	vpunpcklbw %xmm1, %xmm0, %xmm4
	vpunpckhbw %xmm1, %xmm0, %xmm0
	vpunpcklbw %xmm3, %xmm2, %xmm1
	vpunpckhbw %xmm3, %xmm2, %xmm2
	vpunpcklwd %xmm1, %xmm4, %xmm3
	vpunpckhwd %xmm1, %xmm4, %xmm1
	vpunpcklwd %xmm2, %xmm0, %xmm4
	vpunpckhwd %xmm2, %xmm0, %xmm0
	vinserti128 $1, %xmm0, %ymm4, %ymm0
	vinserti128 $1, %xmm1, %ymm3, %ymm1
	vinserti64x4 $1, %ymm0, %zmm1, %zmm0
	vmovdqu64 %zmm0, (%rdx,%r9,4)
	vmovdqu 16(%rdi,%r9), %xmm0
	vmovdqu 16(%rax,%r10), %xmm1
	vmovdqu 16(%r10,%rax,2), %xmm2
	vmovdqu 16(%r8,%r10), %xmm3
	vpunpcklbw %xmm1, %xmm0, %xmm4
	vpunpckhbw %xmm1, %xmm0, %xmm0
	vpunpcklbw %xmm3, %xmm2, %xmm1
	vpunpckhbw %xmm3, %xmm2, %xmm2
	vpunpcklwd %xmm1, %xmm4, %xmm3
	vpunpckhwd %xmm1, %xmm4, %xmm1
	vpunpcklwd %xmm2, %xmm0, %xmm4
	vpunpckhwd %xmm2, %xmm0, %xmm0
	vinserti128 $1, %xmm0, %ymm4, %ymm0
	vinserti128 $1, %xmm1, %ymm3, %ymm1
	vinserti64x4 $1, %ymm0, %zmm1, %zmm0
	vmovdqu64 %zmm0, 64(%rdx,%r9,4)
	addq $32, %r9
	cmpq %rcx, %r9
	jb .LBB2_4
	movl $4, %r8d
	vzeroupper
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
.LBB2_1:
	xorl %r9d, %r9d
	movl $4, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
