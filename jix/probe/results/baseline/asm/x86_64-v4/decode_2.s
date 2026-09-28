jix_probe::byte_shuffle::decode_impl::<2, 64>:
	movq %rsi, %r8
	shrq %r8
	movabsq $4611686018427387840, %rax
	andq %r8, %rax
	je .LBB1_1
	leaq 32(%rdi), %rcx
	addq %rdi, %r8
	addq $32, %r8
	xorl %r9d, %r9d
.LBB1_4:
	vbroadcasti64x4 -32(%r8,%r9), %zmm0
	vextracti32x4 $3, %zmm0, %xmm1
	vbroadcasti64x4 -32(%rcx,%r9), %zmm2
	vextracti32x4 $3, %zmm2, %xmm3
	vpunpckhbw %xmm1, %xmm3, %xmm4
	vpunpcklbw %xmm1, %xmm3, %xmm1
	vinserti128 $1, %xmm4, %ymm1, %ymm1
	vpunpckhbw %xmm0, %xmm2, %xmm3
	vpunpcklbw %xmm0, %xmm2, %xmm0
	vinserti128 $1, %xmm3, %ymm0, %ymm0
	vinserti64x4 $1, %ymm1, %zmm0, %zmm0
	vmovdqu64 %zmm0, (%rdx,%r9,2)
	vbroadcasti64x4 (%r8,%r9), %zmm0
	vbroadcasti64x4 (%rcx,%r9), %zmm1
	vextracti32x4 $3, %zmm0, %xmm2
	vextracti32x4 $3, %zmm1, %xmm3
	vpunpckhbw %xmm2, %xmm3, %xmm4
	vpunpcklbw %xmm2, %xmm3, %xmm2
	vinserti128 $1, %xmm4, %ymm2, %ymm2
	vpunpckhbw %xmm0, %xmm1, %xmm3
	vpunpcklbw %xmm0, %xmm1, %xmm0
	vinserti128 $1, %xmm3, %ymm0, %ymm0
	vinserti64x4 $1, %ymm2, %zmm0, %zmm0
	vmovdqu64 %zmm0, 64(%rdx,%r9,2)
	addq $64, %r9
	cmpq %rax, %r9
	jb .LBB1_4
	movl $2, %r8d
	vzeroupper
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
.LBB1_1:
	xorl %r9d, %r9d
	movl $2, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
