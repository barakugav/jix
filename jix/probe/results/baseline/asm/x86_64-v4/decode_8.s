jix_probe::byte_shuffle::decode_impl::<8, 16>:
	pushq %r15
	pushq %r14
	pushq %r12
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
	addq $8, %rcx
	leaq (%rdi,%rbx,2), %r8
	addq $8, %r8
	leaq 8(%rdi,%r10), %r10
	leaq 8(%rdi,%r9,4), %r11
	leaq 8(%rdi,%rbx), %rbx
	leaq (%rdi,%r9,2), %r14
	addq $8, %r14
	leaq (%r9,%rdi), %r15
	addq $8, %r15
	vbroadcasti64x4 .LCPI3_0(%rip), %zmm0
	xorl %r9d, %r9d
	vmovdqa64 .LCPI3_1(%rip), %zmm1
	vmovdqa64 .LCPI3_2(%rip), %zmm2
	vmovdqa64 .LCPI3_3(%rip), %zmm3
	vmovdqa64 .LCPI3_4(%rip), %zmm4
	vmovdqa64 .LCPI3_5(%rip), %zmm5
	movabsq $-9187201950435737472, %r12
	kmovq %r12, %k1
.LBB3_4:
	vmovq (%rdi,%r9), %xmm6
	vmovq -8(%r15,%r9), %xmm7
	vmovq -8(%r14,%r9), %xmm8
	vmovq -8(%rbx,%r9), %xmm9
	vmovq -8(%r11,%r9), %xmm10
	vmovq -8(%r10,%r9), %xmm11
	vinserti128 $1, %xmm9, %ymm8, %ymm8
	vinserti128 $1, %xmm7, %ymm6, %ymm6
	vpunpcklqdq %ymm8, %ymm6, %ymm6
	vinserti32x4 $1, %xmm11, %zmm10, %zmm7
	vmovq -8(%r8,%r9), %xmm8
	vpermt2q %zmm8, %zmm0, %zmm7
	vinserti64x4 $1, %ymm6, %zmm6, %zmm6
	vpermq $216, %zmm6, %zmm8
	vpshufb %zmm1, %zmm8, %zmm8
	vpermq $141, %zmm6, %zmm6
	vpshufb %zmm2, %zmm6, %zmm6
	vporq %zmm8, %zmm6, %zmm6
	vshufi64x2 $238, %zmm7, %zmm7, %zmm8
	vpshufb %zmm3, %zmm8, %zmm8
	vshufi64x2 $187, %zmm7, %zmm7, %zmm7
	vpshufb %zmm4, %zmm7, %zmm7
	vpbroadcastq -8(%rcx,%r9), %zmm9
	vpternlogq $254, %zmm8, %zmm6, %zmm7
	vpshufb %zmm5, %zmm9, %zmm7 {%k1}
	vmovdqu64 %zmm7, (%rdx,%r9,8)
	vmovq 8(%rdi,%r9), %xmm6
	vmovq (%r15,%r9), %xmm7
	vmovq (%r14,%r9), %xmm8
	vmovq (%rbx,%r9), %xmm9
	vmovq (%r11,%r9), %xmm10
	vmovq (%r10,%r9), %xmm11
	vinserti128 $1, %xmm9, %ymm8, %ymm8
	vinserti128 $1, %xmm7, %ymm6, %ymm6
	vpunpcklqdq %ymm8, %ymm6, %ymm6
	vinserti32x4 $1, %xmm11, %zmm10, %zmm7
	vmovq (%r8,%r9), %xmm8
	vpermt2q %zmm8, %zmm0, %zmm7
	vinserti64x4 $1, %ymm6, %zmm6, %zmm6
	vpermq $216, %zmm6, %zmm8
	vpshufb %zmm1, %zmm8, %zmm8
	vpermq $141, %zmm6, %zmm6
	vpshufb %zmm2, %zmm6, %zmm6
	vporq %zmm8, %zmm6, %zmm6
	vshufi64x2 $238, %zmm7, %zmm7, %zmm8
	vpshufb %zmm3, %zmm8, %zmm8
	vshufi64x2 $187, %zmm7, %zmm7, %zmm7
	vpshufb %zmm4, %zmm7, %zmm7
	vpternlogq $254, %zmm8, %zmm6, %zmm7
	vpbroadcastq (%rcx,%r9), %zmm6
	vpshufb %zmm5, %zmm6, %zmm7 {%k1}
	vmovdqu64 %zmm7, 64(%rdx,%r9,8)
	addq $16, %r9
	cmpq %rax, %r9
	jb .LBB3_4
	jmp .LBB3_2
.LBB3_1:
	xorl %r9d, %r9d
.LBB3_2:
	movl $8, %r8d
	popq %rbx
	popq %r12
	popq %r14
	popq %r15
	vzeroupper
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
