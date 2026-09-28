probe_bit_shuffle_trans_bit_byte:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $392, %rsp
	vpbroadcastq %r8, %ymm0
	vpsrlvq .LCPI10_0(%rip), %ymm0, %ymm0
	vpextrq $1, %xmm0, %r10
	testq %r9, %r9
	sete %al
	testq %r10, %r10
	sete %r11b
	orb %al, %r11b
	jne .LBB10_29
	movq %r9, %r11
	imulq %r10, %r11
	leaq (%r11,%r11), %rbx
	leaq (%r11,%r11,2), %r15
	leaq (,%r11,4), %rax
	leaq (%rbx,%rbx,2), %r12
	leaq (,%r11,8), %r13
	subq %r11, %r13
	vpbroadcastq %r9, %xmm1
	vpshufd $238, %xmm0, %xmm2
	vpmullq %xmm2, %xmm1, %xmm6
	vpermq $80, %ymm6, %ymm1
	vpbroadcastq %xmm6, %xmm2
	vpmullq .LCPI10_1(%rip), %xmm2, %xmm4
	vpermq $64, %ymm6, %ymm3
	vpmullq .LCPI10_2(%rip), %ymm1, %ymm1
	vpmullq .LCPI10_3(%rip), %xmm6, %xmm2
	vpmullq .LCPI10_4(%rip), %ymm3, %ymm5
	movq %rax, 96(%rsp)
	vmovq %rax, %xmm3
	vmovq %r15, %xmm7
	vpunpcklqdq %xmm3, %xmm7, %xmm3
	vmovq %r11, %xmm7
	vmovq %rbx, %xmm8
	vpunpcklqdq %xmm8, %xmm7, %xmm7
	vinserti128 $1, %xmm3, %ymm7, %ymm7
	vpbroadcastq %rsi, %zmm3
	vshufi64x2 $64, %zmm5, %zmm4, %zmm4
	vpbroadcastq %rcx, %ymm5
	vpxor %xmm8, %xmm8, %xmm8
	vinserti128 $1, %xmm6, %ymm8, %ymm6
	vpcmpeqd %ymm8, %ymm8, %ymm8
	vpaddq %ymm7, %ymm8, %ymm7
	vpcmpltuq .LCPI10_5(%rip){1to4}, %ymm7, %k0
	leaq (%r11,%r11,4), %rax
	decq %rax
	cmpq $31, %rax
	setb %r14b
	movq %rbx, 120(%rsp)
	leaq (%rbx,%rbx,2), %rbx
	decq %rbx
	cmpq $31, %rbx
	setb %bpl
	leaq -1(%r13), %rbx
	cmpq $31, %rbx
	setb %al
	kortestb %k0, %k0
	setne %bl
	orb %bpl, %al
	orb %r14b, %al
	orb %bl, %al
	movb %al, 15(%rsp)
	movq %r13, 48(%rsp)
	leaq (%rdx,%r13), %rax
	movq %rax, 32(%rsp)
	movq %r12, 64(%rsp)
	leaq (%rdx,%r12), %rax
	movq %rax, 40(%rsp)
	leaq (%r11,%r11,4), %rax
	movq %rax, 80(%rsp)
	leaq (%r11,%r11,4), %rax
	leaq (%rdx,%rax), %rax
	movq %rax, 56(%rsp)
	leaq (%rdx,%r11,4), %rax
	movq %rax, 72(%rsp)
	movq %r15, 112(%rsp)
	leaq (%rdx,%r15), %rax
	movq %rax, 88(%rsp)
	leaq (%rdx,%r11,2), %rax
	movq %rax, 104(%rsp)
	vmovdqa64 .LCPI10_6(%rip), %zmm7
	vmovdqa64 .LCPI10_7(%rip), %zmm8
	vmovdqa64 .LCPI10_8(%rip), %zmm9
	vmovdqa .LCPI10_9(%rip), %xmm10
	vmovdqa .LCPI10_10(%rip), %xmm11
	vmovdqa .LCPI10_11(%rip), %xmm12
	vmovdqa .LCPI10_12(%rip), %xmm13
	vmovdqa .LCPI10_13(%rip), %xmm14
	vmovdqa .LCPI10_14(%rip), %xmm15
	vmovdqa64 .LCPI10_15(%rip), %xmm16
	vmovdqa64 .LCPI10_16(%rip), %xmm17
	vpbroadcastq .LCPI10_17(%rip), %zmm18
	vpbroadcastq .LCPI10_18(%rip), %zmm19
	vpbroadcastq .LCPI10_19(%rip), %zmm20
	leaq (%rdx,%r11), %r13
	movq $0, 16(%rsp)
	movq %r11, %rbp
	xorl %ebx, %ebx
	xorl %eax, %eax
	movq %r9, 136(%rsp)
	movq %r8, 128(%rsp)
	vmovdqu %ymm1, 224(%rsp)
	vmovdqa %xmm2, 144(%rsp)
	vmovdqu64 %zmm3, 320(%rsp)
	vmovdqu64 %zmm4, 256(%rsp)
	vmovdqu %ymm5, 192(%rsp)
	vmovdqu %ymm6, 160(%rsp)
.LBB10_3:
	movq %r8, %r15
	movq %r10, %r14
	imulq %rax, %r14
	movq %rax, %r12
	addq %r9, %rax
	imulq %r10, %rax
	cmpq %rax, %rcx
	cmovaq %rcx, %rax
	cmpq %r14, %rcx
	movq %r14, %r8
	cmovaq %rcx, %r8
	vpbroadcastq %r14, %ymm21
	movq %r15, %r9
	imulq %r12, %r9
	movq %r12, 24(%rsp)
	vpbroadcastq %r12, %ymm22
	vpmullq %ymm22, %ymm0, %ymm22
	vpaddq %ymm22, %ymm6, %ymm22
	vpbroadcastq %r9, %zmm23
	vpaddq %zmm7, %zmm23, %zmm23
	vpmaxuq %zmm23, %zmm3, %zmm23
	vpbroadcastq %xmm22, %zmm24
	vpsubq %zmm24, %zmm23, %zmm23
	vpaddq %zmm9, %zmm23, %zmm23
	vpermq %zmm22, %zmm8, %zmm22
	vpaddq %ymm21, %ymm1, %ymm24
	vpmaxuq %ymm24, %ymm5, %ymm24
	vpaddq %xmm21, %xmm2, %xmm21
	vpmaxuq %xmm21, %xmm5, %xmm21
	vmovq %r8, %xmm25
	vmovq %rax, %xmm26
	vpunpcklqdq %xmm25, %xmm26, %xmm25
	valignq $2, %zmm4, %zmm25, %zmm25
	vpsubq %zmm22, %zmm25, %zmm22
	vinserti32x4 $2, %xmm21, %zmm24, %zmm21
	vpaddq %zmm21, %zmm22, %zmm21
	vpsrlq $3, %zmm23, %zmm22
	vpminuq %zmm21, %zmm22, %zmm21
	vextracti64x4 $1, %zmm21, %ymm22
	vpminuq %ymm22, %ymm21, %ymm21
	vextracti32x4 $1, %ymm21, %xmm22
	vpminuq %xmm22, %xmm21, %xmm21
	vpshufd $238, %xmm21, %xmm22
	vpminuq %xmm22, %xmm21, %xmm21
	vmovq %xmm21, %r14
	leaq -1(%r10), %rax
	cmpq %rax, %r14
	cmovaeq %rax, %r14
	cmpq $32, %r14
	setb %al
	orb 15(%rsp), %al
	testb $1, %al
	je .LBB10_5
	xorl %r14d, %r14d
	jmp .LBB10_8
.LBB10_5:
	vmovdqa %ymm0, %ymm5
	incq %r14
	movl %r14d, %eax
	andl $31, %eax
	movl $32, %r8d
	cmoveq %r8, %rax
	subq %rax, %r14
	xorl %eax, %eax
.LBB10_6:
	vmovdqu64 (%rdi,%rax,8), %zmm8
	vmovdqu64 64(%rdi,%rax,8), %zmm24
	vmovdqu64 128(%rdi,%rax,8), %zmm1
	vmovdqu64 192(%rdi,%rax,8), %zmm22
	vmovdqa64 %zmm8, %zmm25
	vpermt2b %zmm24, %zmm10, %zmm25
	vpmovzxbw %xmm25, %ymm25
	vextracti32x4 $1, %ymm25, %xmm26
	vpmovzxwq %xmm26, %zmm26
	vpmovzxwq %xmm25, %zmm25
	vmovdqa64 %zmm1, %zmm27
	vpermt2b %zmm22, %zmm10, %zmm27
	vpmovzxbw %xmm27, %ymm27
	vextracti32x4 $1, %ymm27, %xmm28
	vpmovzxwq %xmm28, %zmm28
	vpmovzxwq %xmm27, %zmm27
	vpsllq $56, %zmm27, %zmm27
	vpsllq $56, %zmm28, %zmm28
	vpsllq $56, %zmm25, %zmm25
	vpsllq $56, %zmm26, %zmm26
	vmovdqa64 %zmm8, %zmm29
	vpermt2b %zmm24, %zmm11, %zmm29
	vpmovzxbw %xmm29, %ymm29
	vextracti32x4 $1, %ymm29, %xmm30
	vpmovzxwq %xmm30, %zmm30
	vpmovzxwq %xmm29, %zmm29
	vmovdqa64 %zmm1, %zmm31
	vpermt2b %zmm22, %zmm11, %zmm31
	vpmovzxbw %xmm31, %ymm31
	vextracti32x4 $1, %ymm31, %xmm7
	vpmovzxwq %xmm7, %zmm7
	vpmovzxwq %xmm31, %zmm31
	vpsllq $48, %zmm31, %zmm31
	vpsllq $48, %zmm7, %zmm7
	vpsllq $48, %zmm29, %zmm29
	vpsllq $48, %zmm30, %zmm30
	vmovdqa64 %zmm8, %zmm23
	vpermt2b %zmm24, %zmm12, %zmm23
	vpmovzxbw %xmm23, %ymm23
	vextracti32x4 $1, %ymm23, %xmm9
	vpmovzxwq %xmm9, %zmm9
	vpmovzxwq %xmm23, %zmm23
	vmovdqa64 %zmm1, %zmm21
	vpermt2b %zmm22, %zmm12, %zmm21
	vpmovzxbw %xmm21, %ymm21
	vextracti32x4 $1, %ymm21, %xmm2
	vpmovzxwq %xmm2, %zmm2
	vpmovzxwq %xmm21, %zmm21
	vpsllq $40, %zmm21, %zmm21
	vpternlogq $254, %zmm31, %zmm27, %zmm21
	vpsllq $40, %zmm2, %zmm2
	vpternlogq $254, %zmm7, %zmm28, %zmm2
	vpsllq $40, %zmm23, %zmm7
	vpternlogq $254, %zmm29, %zmm25, %zmm7
	vpsllq $40, %zmm9, %zmm9
	vpternlogq $254, %zmm30, %zmm26, %zmm9
	vmovdqa64 %zmm1, %zmm23
	vpermt2b %zmm22, %zmm13, %zmm23
	vpmovzxbw %xmm23, %ymm23
	vpmovzxwq %xmm23, %zmm25
	vextracti32x4 $1, %ymm23, %xmm23
	vpmovzxwq %xmm23, %zmm23
	vmovdqa64 %zmm8, %zmm26
	vpermt2b %zmm24, %zmm13, %zmm26
	vpmovzxbw %xmm26, %ymm26
	vpmovzxwq %xmm26, %zmm27
	vextracti32x4 $1, %ymm26, %xmm26
	vpmovzxwq %xmm26, %zmm26
	vpsllq $32, %zmm26, %zmm26
	vpsllq $32, %zmm27, %zmm27
	vpsllq $32, %zmm23, %zmm23
	vpsllq $32, %zmm25, %zmm25
	vmovdqa64 %zmm1, %zmm28
	vpermt2b %zmm22, %zmm14, %zmm28
	vpmovzxbw %xmm28, %ymm28
	vpmovzxwq %xmm28, %zmm29
	vextracti32x4 $1, %ymm28, %xmm28
	vpmovzxwq %xmm28, %zmm28
	vmovdqa64 %zmm8, %zmm30
	vpermt2b %zmm24, %zmm14, %zmm30
	vpmovzxbw %xmm30, %ymm30
	vpmovzxwq %xmm30, %zmm31
	vextracti32x4 $1, %ymm30, %xmm30
	vpmovzxwq %xmm30, %zmm30
	vpsllq $24, %zmm30, %zmm30
	vpternlogq $254, %zmm26, %zmm9, %zmm30
	vpsllq $24, %zmm31, %zmm9
	vpternlogq $254, %zmm27, %zmm7, %zmm9
	vpsllq $24, %zmm28, %zmm7
	vpternlogq $254, %zmm23, %zmm2, %zmm7
	vpsllq $24, %zmm29, %zmm2
	vpternlogq $254, %zmm25, %zmm21, %zmm2
	vmovdqa64 %zmm8, %zmm21
	vpermt2b %zmm24, %zmm15, %zmm21
	vpmovzxbw %xmm21, %ymm21
	vextracti32x4 $1, %ymm21, %xmm23
	vpmovzxwq %xmm23, %zmm23
	vpmovzxwq %xmm21, %zmm21
	vmovdqa64 %zmm1, %zmm25
	vpermt2b %zmm22, %zmm15, %zmm25
	vpmovzxbw %xmm25, %ymm25
	vextracti32x4 $1, %ymm25, %xmm26
	vpmovzxwq %xmm26, %zmm26
	vpmovzxwq %xmm25, %zmm25
	vpsllq $16, %zmm25, %zmm25
	vpsllq $16, %zmm26, %zmm26
	vpsllq $16, %zmm21, %zmm21
	vpsllq $16, %zmm23, %zmm23
	vmovdqa64 %zmm8, %zmm27
	vpermt2b %zmm24, %zmm16, %zmm27
	vpmovzxbw %xmm27, %ymm27
	vextracti32x4 $1, %ymm27, %xmm28
	vpmovzxwq %xmm28, %zmm28
	vpmovzxwq %xmm27, %zmm27
	vmovdqa64 %zmm1, %zmm29
	vpermt2b %zmm22, %zmm16, %zmm29
	vpmovzxbw %xmm29, %ymm29
	vextracti32x4 $1, %ymm29, %xmm31
	vpmovzxwq %xmm31, %zmm31
	vpmovzxwq %xmm29, %zmm29
	vpsllq $8, %zmm29, %zmm29
	vpternlogq $254, %zmm25, %zmm2, %zmm29
	vpsllq $8, %zmm31, %zmm2
	vpternlogq $254, %zmm26, %zmm7, %zmm2
	vpsllq $8, %zmm27, %zmm7
	vpternlogq $254, %zmm21, %zmm9, %zmm7
	vpsllq $8, %zmm28, %zmm9
	vpternlogq $254, %zmm23, %zmm30, %zmm9
	vpermt2b %zmm24, %zmm17, %zmm8
	vpmovzxbw %xmm8, %ymm8
	vextracti32x4 $1, %ymm8, %xmm21
	vpmovzxwq %xmm21, %zmm21
	vpmovzxwq %xmm8, %zmm8
	vpermt2b %zmm22, %zmm17, %zmm1
	vpmovzxbw %xmm1, %ymm1
	vextracti32x4 $1, %ymm1, %xmm22
	vpmovzxwq %xmm22, %zmm22
	vpmovzxwq %xmm1, %zmm1
	vporq %zmm1, %zmm29, %zmm1
	vporq %zmm22, %zmm2, %zmm22
	vporq %zmm8, %zmm7, %zmm8
	vporq %zmm21, %zmm9, %zmm21
	vpsrlq $7, %zmm29, %zmm23
	vpsrlq $7, %zmm2, %zmm2
	vpsrlq $7, %zmm7, %zmm7
	vpsrlq $7, %zmm9, %zmm9
	vpternlogq $72, %zmm21, %zmm18, %zmm9
	vpternlogq $72, %zmm8, %zmm18, %zmm7
	vpternlogq $72, %zmm22, %zmm18, %zmm2
	vpternlogq $72, %zmm1, %zmm18, %zmm23
	vpsllq $7, %zmm23, %zmm24
	vpsllq $7, %zmm2, %zmm25
	vpsllq $7, %zmm7, %zmm26
	vpsllq $7, %zmm9, %zmm27
	vpternlogq $54, %zmm9, %zmm21, %zmm27
	vpternlogq $54, %zmm7, %zmm8, %zmm26
	vpternlogq $54, %zmm2, %zmm22, %zmm25
	vpternlogq $54, %zmm23, %zmm1, %zmm24
	vpsrlq $14, %zmm24, %zmm1
	vpsrlq $14, %zmm25, %zmm2
	vpsrlq $14, %zmm26, %zmm7
	vpsrlq $14, %zmm27, %zmm8
	vpternlogq $72, %zmm27, %zmm19, %zmm8
	vpternlogq $72, %zmm26, %zmm19, %zmm7
	vpternlogq $72, %zmm25, %zmm19, %zmm2
	vpternlogq $72, %zmm24, %zmm19, %zmm1
	vpsllq $14, %zmm1, %zmm9
	vpsllq $14, %zmm2, %zmm28
	vpsllq $14, %zmm7, %zmm29
	vpsllq $14, %zmm8, %zmm30
	vpternlogq $54, %zmm8, %zmm27, %zmm30
	vpternlogq $54, %zmm7, %zmm26, %zmm29
	vpternlogq $54, %zmm2, %zmm25, %zmm28
	vpternlogq $54, %zmm1, %zmm24, %zmm9
	vpsrlq $28, %zmm9, %zmm1
	vpsrlq $28, %zmm28, %zmm2
	vpsrlq $28, %zmm29, %zmm7
	vpsrlq $28, %zmm30, %zmm8
	vpternlogq $72, %zmm30, %zmm20, %zmm8
	vpternlogq $72, %zmm29, %zmm20, %zmm7
	vpternlogq $72, %zmm28, %zmm20, %zmm2
	vpternlogq $72, %zmm9, %zmm20, %zmm1
	vpsllq $28, %zmm1, %zmm21
	vpsllq $28, %zmm2, %zmm22
	vpsllq $28, %zmm7, %zmm23
	vpsllq $28, %zmm8, %zmm24
	vpternlogq $54, %zmm8, %zmm30, %zmm24
	vpternlogq $54, %zmm7, %zmm29, %zmm23
	vpternlogq $54, %zmm2, %zmm28, %zmm22
	vpternlogq $54, %zmm1, %zmm9, %zmm21
	vpsrlq $8, %zmm21, %zmm1
	vpsrlq $8, %zmm22, %zmm2
	vpsrlq $8, %zmm23, %zmm7
	vpsrlq $8, %zmm24, %zmm8
	vpmovqb %zmm8, %xmm8
	vpmovqb %zmm7, %xmm7
	vpunpcklqdq %xmm8, %xmm7, %xmm25
	vpmovqb %zmm2, %xmm2
	vpmovqb %zmm1, %xmm1
	vpunpcklqdq %xmm2, %xmm1, %xmm26
	vpsrlq $16, %zmm21, %zmm1
	vpsrlq $16, %zmm22, %zmm2
	vpsrlq $16, %zmm23, %zmm7
	vpsrlq $16, %zmm24, %zmm8
	vpmovqb %zmm8, %xmm8
	vpmovqb %zmm7, %xmm7
	vpunpcklqdq %xmm8, %xmm7, %xmm27
	vpmovqb %zmm2, %xmm2
	vpmovqb %zmm1, %xmm1
	vpunpcklqdq %xmm2, %xmm1, %xmm1
	vpsrlq $24, %zmm21, %zmm2
	vpsrlq $24, %zmm22, %zmm7
	vpsrlq $24, %zmm23, %zmm8
	vpsrlq $24, %zmm24, %zmm9
	vpmovqb %zmm9, %xmm9
	vpmovqb %zmm8, %xmm8
	vpunpcklqdq %xmm9, %xmm8, %xmm8
	vpmovqb %zmm7, %xmm7
	vpmovqb %zmm2, %xmm2
	vpunpcklqdq %xmm7, %xmm2, %xmm2
	vpsrlq $32, %zmm21, %zmm7
	vpsrlq $32, %zmm22, %zmm9
	vpsrlq $32, %zmm23, %zmm28
	vpsrlq $32, %zmm24, %zmm29
	vpmovqb %zmm29, %xmm29
	vpmovqb %zmm28, %xmm28
	vpunpcklqdq %xmm29, %xmm28, %xmm28
	vpmovqb %zmm9, %xmm9
	vpmovqb %zmm7, %xmm7
	vpunpcklqdq %xmm9, %xmm7, %xmm7
	vpsrlq $40, %zmm21, %zmm9
	vpsrlq $40, %zmm22, %zmm29
	vpsrlq $40, %zmm23, %zmm30
	vpsrlq $40, %zmm24, %zmm31
	vpmovqb %zmm31, %xmm31
	vpmovqb %zmm30, %xmm30
	vpunpcklqdq %xmm31, %xmm30, %xmm30
	vpmovqb %zmm29, %xmm29
	vpmovqb %zmm9, %xmm9
	vpunpcklqdq %xmm29, %xmm9, %xmm9
	vpsrlq $48, %zmm21, %zmm29
	vpsrlq $48, %zmm22, %zmm31
	vpsrlq $48, %zmm23, %zmm3
	vpsrlq $48, %zmm24, %zmm4
	vpmovqb %zmm4, %xmm4
	vpmovqb %zmm3, %xmm3
	vpunpcklqdq %xmm4, %xmm3, %xmm3
	vpmovqb %zmm31, %xmm4
	vpmovqb %zmm29, %xmm29
	vpunpcklqdq %xmm4, %xmm29, %xmm4
	vpsrlq $56, %zmm21, %zmm29
	vpsrlq $56, %zmm22, %zmm31
	vpsrlq $56, %zmm23, %zmm6
	vpsrlq $56, %zmm24, %zmm0
	vpmovqb %zmm0, %xmm0
	vpmovqb %zmm6, %xmm6
	vpunpcklqdq %xmm0, %xmm6, %xmm0
	vpmovqb %zmm31, %xmm6
	vpmovqb %zmm29, %xmm29
	vpunpcklqdq %xmm6, %xmm29, %xmm6
	vpmovqb %zmm24, %xmm24
	vpmovqb %zmm23, %xmm23
	vpunpcklqdq %xmm24, %xmm23, %xmm23
	vpmovqb %zmm22, %xmm22
	vpmovqb %zmm21, %xmm21
	vpunpcklqdq %xmm22, %xmm21, %xmm21
	leaq (%rdx,%rax), %r8
	vmovdqu64 %xmm21, 16(%rdx,%rax)
	vmovdqu64 %xmm23, (%rdx,%rax)
	leaq (%r8,%r11), %r9
	vmovdqu64 %xmm26, 16(%r11,%r8)
	vmovdqu64 %xmm25, (%r11,%r8)
	leaq (%r9,%r11), %r8
	vmovdqu %xmm1, 16(%r11,%r9)
	vmovdqu64 %xmm27, (%r11,%r9)
	leaq (%r8,%r11), %r9
	vmovdqu %xmm2, 16(%r11,%r8)
	vmovdqu %xmm8, (%r11,%r8)
	leaq (%r9,%r11), %r8
	vmovdqu %xmm7, 16(%r11,%r9)
	vmovdqu64 %xmm28, (%r11,%r9)
	leaq (%r8,%r11), %r9
	vmovdqu %xmm9, 16(%r11,%r8)
	vmovdqu64 %xmm30, (%r11,%r8)
	leaq (%r9,%r11), %r8
	vmovdqu %xmm4, 16(%r11,%r9)
	vmovdqu %xmm3, (%r11,%r9)
	vmovdqu %xmm6, 16(%r11,%r8)
	vmovdqu %xmm0, (%r11,%r8)
	addq $32, %rax
	cmpq %rax, %r14
	jne .LBB10_6
	vmovdqa %ymm5, %ymm0
	vmovdqu 224(%rsp), %ymm1
	vmovdqa 144(%rsp), %xmm2
	vmovdqu64 320(%rsp), %zmm3
	vmovdqu64 256(%rsp), %zmm4
	vmovdqu 192(%rsp), %ymm5
	vmovdqu 160(%rsp), %ymm6
	vmovdqa64 .LCPI10_6(%rip), %zmm7
	vmovdqa64 .LCPI10_7(%rip), %zmm8
	vmovdqa64 .LCPI10_8(%rip), %zmm9
.LBB10_8:
	incq 24(%rsp)
	movq 16(%rsp), %rax
	leaq (%rax,%r14,8), %r8
.LBB10_9:
	cmpq %rsi, %r8
	jae .LBB10_10
	leaq 1(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB10_11
	leaq 2(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB10_11
	leaq 3(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB10_11
	leaq 4(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB10_11
	leaq 5(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB10_11
	leaq 6(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB10_11
	leaq 7(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB10_11
	leaq (%rbx,%r14), %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	leaq 1(%r14), %r9
	movzbl (%rdi,%r14,8), %eax
	movzbl -2(%rdi,%r9,8), %r15d
	movzbl -1(%rdi,%r9,8), %r12d
	shlq $56, %r12
	shlq $48, %r15
	orq %r12, %r15
	movzbl -3(%rdi,%r9,8), %r12d
	shlq $40, %r12
	orq %r15, %r12
	movzbl -4(%rdi,%r9,8), %r15d
	shlq $32, %r15
	orq %r12, %r15
	movzbl -5(%rdi,%r9,8), %r12d
	shll $24, %r12d
	orq %r15, %r12
	movzbl -6(%rdi,%r9,8), %r15d
	shll $16, %r15d
	orq %r12, %r15
	movzbl -7(%rdi,%r9,8), %r12d
	shll $8, %r12d
	orq %r15, %r12
	orq %r12, %rax
	shrq $7, %r12
	xorq %rax, %r12
	movabsq $47851476196393130, %r15
	andq %r15, %r12
	movq %r12, %r15
	shlq $7, %r15
	orq %r12, %r15
	xorq %rax, %r15
	movq %r15, %rax
	shrq $14, %rax
	xorq %r15, %rax
	movabsq $225176545447116, %r12
	andq %r12, %rax
	movq %rax, %r12
	shlq $14, %r12
	orq %rax, %r12
	xorq %r15, %r12
	movq %r12, %rax
	shrq $28, %rax
	xorl %r12d, %eax
	andl $-252645136, %eax
	movq %rax, %r15
	shlq $28, %r15
	orq %rax, %r15
	xorq %r12, %r15
	movb %r15b, (%rdx,%r14)
	leaq (%r14,%rbp), %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	movl %r15d, %eax
	shrl $8, %eax
	movb %al, (%r13,%r14)
	movq 120(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	movl %r15d, %eax
	shrl $16, %eax
	movq 104(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 112(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	movl %r15d, %eax
	shrl $24, %eax
	movq 88(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 96(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	movq %r15, %rax
	shrq $32, %rax
	movq 72(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 80(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	movq %r15, %rax
	shrq $40, %rax
	movq 56(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 64(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	movq %r15, %rax
	shrq $48, %rax
	movq 40(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 48(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB10_28
	shrq $56, %r15
	movq 32(%rsp), %rax
	movb %r15b, (%rax,%r14)
	addq $8, %r8
	movq %r9, %r14
	cmpq %r9, %r10
	jne .LBB10_9
	addq %r10, %rdx
	movq 128(%rsp), %r8
	addq %r8, %rdi
	addq %r10, %rbx
	addq %r10, 48(%rsp)
	addq %r10, 32(%rsp)
	addq %r10, 64(%rsp)
	addq %r10, 40(%rsp)
	addq %r10, 80(%rsp)
	addq %r10, 56(%rsp)
	addq %r10, 96(%rsp)
	addq %r10, 72(%rsp)
	addq %r10, 112(%rsp)
	addq %r10, 88(%rsp)
	addq %r10, 120(%rsp)
	addq %r10, 104(%rsp)
	addq %r10, %rbp
	addq %r10, %r13
	addq %r8, 16(%rsp)
	movq 136(%rsp), %r9
	movq 24(%rsp), %rax
	cmpq %r9, %rax
	jne .LBB10_3
.LBB10_29:
	addq $392, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB10_28:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.11(%rip), %rdx
	movq %rax, %rdi
	movq %rcx, %rsi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
.LBB10_10:
	movq %r8, %rax
.LBB10_11:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.12(%rip), %rdx
	movq %rax, %rdi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
