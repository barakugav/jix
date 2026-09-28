jix_probe::byte_shuffle::encode_impl::<8, 16>:
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	movq %rsi, %rax
	shrq $3, %rax
	movabsq $1152921504606846960, %rcx
	andq %rax, %rcx
	je .LBB7_1
	leaq (%rax,%rax,2), %r8
	leaq (%rax,%rax,4), %r10
	leaq (%rdx,%rax,8), %r11
	subq %rax, %r11
	leaq (%rdx,%r8,2), %rbx
	addq %rdx, %r10
	leaq (%rdx,%rax,4), %r14
	addq %rdx, %r8
	leaq (%rdx,%rax,2), %r15
	addq %rdx, %rax
	leaq 7(%rdi), %r12
	xorl %r9d, %r9d
	movabsq $72340172838076673, %r13
	kmovq %r13, %k1
	vmovq .LCPI7_2(%rip), %xmm0
	vmovq .LCPI7_3(%rip), %xmm1
.LBB7_4:
	vmovdqu8 -7(%r12,%r9,8), %zmm2 {%k1} {z}
	vmovdqu8 57(%r12,%r9,8), %zmm3 {%k1} {z}
	vextracti64x4 $1, %zmm3, %ymm4
	vextracti128 $1, %ymm4, %xmm5
	vpshufb %xmm0, %xmm5, %xmm5
	vpshufb %xmm0, %xmm4, %xmm4
	vpunpcklwd %xmm5, %xmm4, %xmm4
	vextracti128 $1, %ymm3, %xmm5
	vpshufb %xmm1, %xmm5, %xmm5
	vpshufb %xmm1, %xmm3, %xmm3
	vpunpcklwd %xmm5, %xmm3, %xmm3
	vpblendd $8, %xmm4, %xmm3, %xmm3
	vpmovqb %zmm2, %xmm2
	vpblendd $12, %xmm3, %xmm2, %xmm2
	vmovdqu %xmm2, (%rdx,%r9)
	vmovdqu8 -6(%r12,%r9,8), %zmm2 {%k1} {z}
	vmovdqu8 58(%r12,%r9,8), %zmm3 {%k1} {z}
	vextracti64x4 $1, %zmm3, %ymm4
	vextracti128 $1, %ymm4, %xmm5
	vpshufb %xmm0, %xmm5, %xmm5
	vpshufb %xmm0, %xmm4, %xmm4
	vpunpcklwd %xmm5, %xmm4, %xmm4
	vextracti128 $1, %ymm3, %xmm5
	vpshufb %xmm1, %xmm5, %xmm5
	vpshufb %xmm1, %xmm3, %xmm3
	vpunpcklwd %xmm5, %xmm3, %xmm3
	vpblendd $8, %xmm4, %xmm3, %xmm3
	vpmovqb %zmm2, %xmm2
	vpblendd $12, %xmm3, %xmm2, %xmm2
	vmovdqu %xmm2, (%rax,%r9)
	vmovdqu8 (%r12,%r9,8), %zmm2 {%k1} {z}
	vmovdqu8 64(%r12,%r9,8), %zmm3 {%k1} {z}
	vextracti64x4 $1, %zmm3, %ymm4
	vextracti128 $1, %ymm4, %xmm5
	vpshufb %xmm0, %xmm5, %xmm5
	vpshufb %xmm0, %xmm4, %xmm4
	vpunpcklwd %xmm5, %xmm4, %xmm4
	vextracti128 $1, %ymm3, %xmm5
	vpshufb %xmm1, %xmm5, %xmm5
	vpshufb %xmm1, %xmm3, %xmm3
	vpunpcklwd %xmm5, %xmm3, %xmm3
	vpblendd $8, %xmm4, %xmm3, %xmm3
	vpmovqb %zmm2, %xmm2
	vpblendd $12, %xmm3, %xmm2, %xmm2
	vmovdqu8 -5(%r12,%r9,8), %zmm3 {%k1} {z}
	vmovdqu8 59(%r12,%r9,8), %zmm4 {%k1} {z}
	vextracti64x4 $1, %zmm4, %ymm5
	vextracti128 $1, %ymm5, %xmm6
	vpshufb %xmm0, %xmm6, %xmm6
	vpshufb %xmm0, %xmm5, %xmm5
	vpunpcklwd %xmm6, %xmm5, %xmm5
	vextracti128 $1, %ymm4, %xmm6
	vpshufb %xmm1, %xmm6, %xmm6
	vpshufb %xmm1, %xmm4, %xmm4
	vpunpcklwd %xmm6, %xmm4, %xmm4
	vpblendd $8, %xmm5, %xmm4, %xmm4
	vpmovqb %zmm3, %xmm3
	vpblendd $12, %xmm4, %xmm3, %xmm3
	vmovdqu %xmm3, (%r15,%r9)
	vmovdqu8 -1(%r12,%r9,8), %zmm3 {%k1} {z}
	vmovdqu8 63(%r12,%r9,8), %zmm4 {%k1} {z}
	vextracti64x4 $1, %zmm4, %ymm5
	vextracti128 $1, %ymm5, %xmm6
	vpshufb %xmm0, %xmm6, %xmm6
	vpshufb %xmm0, %xmm5, %xmm5
	vpunpcklwd %xmm6, %xmm5, %xmm5
	vextracti128 $1, %ymm4, %xmm6
	vpshufb %xmm1, %xmm6, %xmm6
	vpshufb %xmm1, %xmm4, %xmm4
	vpunpcklwd %xmm6, %xmm4, %xmm4
	vpblendd $8, %xmm5, %xmm4, %xmm4
	vpmovqb %zmm3, %xmm3
	vpblendd $12, %xmm4, %xmm3, %xmm3
	vmovdqu8 -4(%r12,%r9,8), %zmm4 {%k1} {z}
	vmovdqu8 60(%r12,%r9,8), %zmm5 {%k1} {z}
	vextracti64x4 $1, %zmm5, %ymm6
	vextracti128 $1, %ymm6, %xmm7
	vpshufb %xmm0, %xmm7, %xmm7
	vpshufb %xmm0, %xmm6, %xmm6
	vpunpcklwd %xmm7, %xmm6, %xmm6
	vextracti128 $1, %ymm5, %xmm7
	vpshufb %xmm1, %xmm7, %xmm7
	vpshufb %xmm1, %xmm5, %xmm5
	vpunpcklwd %xmm7, %xmm5, %xmm5
	vpblendd $8, %xmm6, %xmm5, %xmm5
	vpmovqb %zmm4, %xmm4
	vpblendd $12, %xmm5, %xmm4, %xmm4
	vmovdqu %xmm4, (%r8,%r9)
	vmovdqu8 -2(%r12,%r9,8), %zmm4 {%k1} {z}
	vmovdqu8 62(%r12,%r9,8), %zmm5 {%k1} {z}
	vextracti64x4 $1, %zmm5, %ymm6
	vextracti128 $1, %ymm6, %xmm7
	vpshufb %xmm0, %xmm7, %xmm7
	vpshufb %xmm0, %xmm6, %xmm6
	vpunpcklwd %xmm7, %xmm6, %xmm6
	vextracti128 $1, %ymm5, %xmm7
	vpshufb %xmm1, %xmm7, %xmm7
	vpshufb %xmm1, %xmm5, %xmm5
	vpunpcklwd %xmm7, %xmm5, %xmm5
	vpblendd $8, %xmm6, %xmm5, %xmm5
	vpmovqb %zmm4, %xmm4
	vpblendd $12, %xmm5, %xmm4, %xmm4
	vmovdqu8 -3(%r12,%r9,8), %zmm5 {%k1} {z}
	vmovdqu8 61(%r12,%r9,8), %zmm6 {%k1} {z}
	vextracti64x4 $1, %zmm6, %ymm7
	vextracti128 $1, %ymm7, %xmm8
	vpshufb %xmm0, %xmm8, %xmm8
	vpshufb %xmm0, %xmm7, %xmm7
	vpunpcklwd %xmm8, %xmm7, %xmm7
	vextracti128 $1, %ymm6, %xmm8
	vpshufb %xmm1, %xmm8, %xmm8
	vpshufb %xmm1, %xmm6, %xmm6
	vpunpcklwd %xmm8, %xmm6, %xmm6
	vpblendd $8, %xmm7, %xmm6, %xmm6
	vpmovqb %zmm5, %xmm5
	vpblendd $12, %xmm6, %xmm5, %xmm5
	vmovdqu %xmm5, (%r14,%r9)
	vmovdqu %xmm4, (%r10,%r9)
	vmovdqu %xmm3, (%rbx,%r9)
	vmovdqu %xmm2, (%r11,%r9)
	addq $16, %r9
	cmpq %rcx, %r9
	jb .LBB7_4
	jmp .LBB7_2
.LBB7_1:
	xorl %r9d, %r9d
.LBB7_2:
	movl $8, %r8d
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	vzeroupper
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
