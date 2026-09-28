probe_bit_shuffle_trans_bit_byte:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $136, %rsp
	vpbroadcastq %r8, %ymm0
	vpsrlvq .LCPI22_0(%rip), %ymm0, %ymm0
	vpextrq $1, %xmm0, %r10
	testq %r9, %r9
	sete %al
	testq %r10, %r10
	sete %r11b
	orb %al, %r11b
	jne .LBB22_28
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
	vpmullq .LCPI22_1(%rip), %xmm2, %xmm4
	vpermq $64, %ymm6, %ymm3
	vpmullq .LCPI22_2(%rip), %ymm1, %ymm1
	vpmullq .LCPI22_3(%rip), %xmm6, %xmm2
	vpmullq .LCPI22_4(%rip), %ymm3, %ymm5
	movq %rax, 88(%rsp)
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
	vpcmpltuq .LCPI22_5(%rip){1to4}, %ymm7, %k0
	leaq (%r11,%r11,4), %rax
	decq %rax
	cmpq $7, %rax
	setb %r14b
	movq %rbx, 112(%rsp)
	leaq (%rbx,%rbx,2), %rbx
	decq %rbx
	cmpq $7, %rbx
	setb %bpl
	leaq -1(%r13), %rbx
	cmpq $7, %rbx
	setb %al
	kortestb %k0, %k0
	setne %bl
	orb %bpl, %al
	orb %r14b, %al
	orb %bl, %al
	movb %al, 7(%rsp)
	movq %r13, 40(%rsp)
	leaq (%rdx,%r13), %rax
	movq %rax, 24(%rsp)
	movq %r12, 56(%rsp)
	leaq (%rdx,%r12), %rax
	movq %rax, 32(%rsp)
	leaq (%r11,%r11,4), %rax
	movq %rax, 72(%rsp)
	leaq (%r11,%r11,4), %rax
	leaq (%rdx,%rax), %rax
	movq %rax, 48(%rsp)
	leaq (%rdx,%r11,4), %rax
	movq %rax, 64(%rsp)
	movq %r15, 104(%rsp)
	leaq (%rdx,%r15), %rax
	movq %rax, 80(%rsp)
	leaq (%rdx,%r11,2), %rax
	movq %rax, 96(%rsp)
	vmovdqa64 .LCPI22_6(%rip), %zmm7
	vmovdqa64 .LCPI22_7(%rip), %zmm8
	vmovdqa64 .LCPI22_8(%rip), %zmm9
	vpbroadcastq .LCPI22_9(%rip), %zmm10
	vpbroadcastq .LCPI22_10(%rip), %zmm11
	vpbroadcastq .LCPI22_11(%rip), %zmm12
	leaq (%rdx,%r11), %r13
	movq $0, 8(%rsp)
	movq %r11, %rbp
	xorl %ebx, %ebx
	xorl %eax, %eax
	movq %r9, 128(%rsp)
	movq %r8, 120(%rsp)
.LBB22_3:
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
	vpbroadcastq %r14, %ymm13
	movq %r15, %r9
	imulq %r12, %r9
	movq %r12, 16(%rsp)
	vpbroadcastq %r12, %ymm14
	vpmullq %ymm14, %ymm0, %ymm14
	vpaddq %ymm6, %ymm14, %ymm14
	vpbroadcastq %r9, %zmm15
	vpaddq %zmm7, %zmm15, %zmm15
	vpmaxuq %zmm15, %zmm3, %zmm15
	vpbroadcastq %xmm14, %zmm16
	vpsubq %zmm16, %zmm15, %zmm15
	vpaddq %zmm9, %zmm15, %zmm15
	vpermq %zmm14, %zmm8, %zmm14
	vpaddq %ymm13, %ymm1, %ymm16
	vpmaxuq %ymm16, %ymm5, %ymm16
	vpaddq %xmm2, %xmm13, %xmm13
	vpmaxuq %xmm13, %xmm5, %xmm13
	vmovq %r8, %xmm17
	vmovq %rax, %xmm18
	vpunpcklqdq %xmm17, %xmm18, %xmm17
	valignq $2, %zmm4, %zmm17, %zmm17
	vpsubq %zmm14, %zmm17, %zmm14
	vinserti32x4 $2, %xmm13, %zmm16, %zmm13
	vpaddq %zmm13, %zmm14, %zmm13
	vpsrlq $3, %zmm15, %zmm14
	vpminuq %zmm13, %zmm14, %zmm13
	vextracti64x4 $1, %zmm13, %ymm14
	vpminuq %ymm14, %ymm13, %ymm13
	vextracti128 $1, %ymm13, %xmm14
	vpminuq %xmm14, %xmm13, %xmm13
	vpshufd $238, %xmm13, %xmm14
	vpminuq %xmm14, %xmm13, %xmm13
	vmovq %xmm13, %r14
	leaq -1(%r10), %rax
	cmpq %rax, %r14
	cmovaeq %rax, %r14
	cmpq $8, %r14
	setb %al
	orb 7(%rsp), %al
	testb $1, %al
	je .LBB22_5
	xorl %r14d, %r14d
	jmp .LBB22_7
.LBB22_5:
	incq %r14
	movl %r14d, %eax
	andl $7, %eax
	movl $8, %r8d
	cmoveq %r8, %rax
	subq %rax, %r14
	xorl %eax, %eax
.LBB22_6:
	vmovdqu64 (%rdi,%rax,8), %zmm13
	vpmovqb %zmm13, %xmm14
	vpsrlq $8, %zmm13, %zmm15
	vpmovqb %zmm15, %xmm15
	vpsrlq $16, %zmm13, %zmm16
	vpmovqb %zmm16, %xmm16
	vpsrlq $24, %zmm13, %zmm17
	vpmovqb %zmm17, %xmm17
	vpsrlq $32, %zmm13, %zmm18
	vpmovqb %zmm18, %xmm18
	vpsrlq $40, %zmm13, %zmm19
	vpmovqb %zmm19, %xmm19
	vpsrlq $48, %zmm13, %zmm20
	vpmovqb %zmm20, %xmm20
	vpsrlq $56, %zmm13, %zmm13
	vpmovqb %zmm13, %xmm13
	vpmovzxbq %xmm13, %zmm13
	vpsllq $56, %zmm13, %zmm13
	vpmovzxbq %xmm20, %zmm20
	vpsllq $48, %zmm20, %zmm20
	vpmovzxbq %xmm19, %zmm19
	vpsllq $40, %zmm19, %zmm19
	vpternlogq $254, %zmm20, %zmm13, %zmm19
	vpmovzxbq %xmm18, %zmm13
	vpsllq $32, %zmm13, %zmm13
	vpmovzxbq %xmm17, %zmm17
	vpsllq $24, %zmm17, %zmm17
	vpternlogq $254, %zmm13, %zmm19, %zmm17
	vpmovzxbq %xmm16, %zmm13
	vpsllq $16, %zmm13, %zmm13
	vpmovzxbq %xmm15, %zmm15
	vpsllq $8, %zmm15, %zmm15
	vpternlogq $254, %zmm13, %zmm17, %zmm15
	vpmovzxbq %xmm14, %zmm13
	vporq %zmm13, %zmm15, %zmm13
	vpsrlq $7, %zmm15, %zmm14
	vpternlogq $40, %zmm10, %zmm13, %zmm14
	vpsllq $7, %zmm14, %zmm15
	vpternlogq $54, %zmm14, %zmm13, %zmm15
	vpsrlq $14, %zmm15, %zmm13
	vpternlogq $40, %zmm11, %zmm15, %zmm13
	vpsllq $14, %zmm13, %zmm14
	vpternlogq $54, %zmm13, %zmm15, %zmm14
	vpsrlq $28, %zmm14, %zmm13
	vpternlogq $40, %zmm12, %zmm14, %zmm13
	vpsllq $28, %zmm13, %zmm15
	vpternlogq $54, %zmm13, %zmm14, %zmm15
	vpsrlq $8, %zmm15, %zmm13
	vpsrlq $16, %zmm15, %zmm14
	vpsrlq $24, %zmm15, %zmm16
	leaq (%rdx,%rax), %r8
	vpmovqb %zmm15, (%rdx,%rax)
	vpmovqb %zmm13, (%r11,%r8)
	vpsrlq $32, %zmm15, %zmm13
	addq %r11, %r8
	vpmovqb %zmm14, (%r11,%r8)
	vpsrlq $40, %zmm15, %zmm14
	addq %r11, %r8
	vpmovqb %zmm16, (%r11,%r8)
	vpsrlq $48, %zmm15, %zmm16
	addq %r11, %r8
	vpmovqb %zmm13, (%r11,%r8)
	vpsrlq $56, %zmm15, %zmm13
	addq %r11, %r8
	vpmovqb %zmm14, (%r11,%r8)
	addq %r11, %r8
	vpmovqb %zmm16, (%r11,%r8)
	addq %r11, %r8
	vpmovqb %zmm13, (%r11,%r8)
	addq $8, %rax
	cmpq %rax, %r14
	jne .LBB22_6
.LBB22_7:
	incq 16(%rsp)
	movq 8(%rsp), %rax
	leaq (%rax,%r14,8), %r8
.LBB22_8:
	cmpq %rsi, %r8
	jae .LBB22_9
	leaq 1(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 2(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 3(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 4(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 5(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 6(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 7(%r8), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq (%rbx,%r14), %rax
	cmpq %rcx, %rax
	jae .LBB22_27
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
	jae .LBB22_27
	movl %r15d, %eax
	shrl $8, %eax
	movb %al, (%r13,%r14)
	movq 112(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB22_27
	movl %r15d, %eax
	shrl $16, %eax
	movq 96(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 104(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB22_27
	movl %r15d, %eax
	shrl $24, %eax
	movq 80(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 88(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB22_27
	movq %r15, %rax
	shrq $32, %rax
	movq 64(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 72(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB22_27
	movq %r15, %rax
	shrq $40, %rax
	movq 48(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 56(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB22_27
	movq %r15, %rax
	shrq $48, %rax
	movq 32(%rsp), %r12
	movb %al, (%r12,%r14)
	movq 40(%rsp), %rax
	addq %r14, %rax
	cmpq %rcx, %rax
	jae .LBB22_27
	shrq $56, %r15
	movq 24(%rsp), %rax
	movb %r15b, (%rax,%r14)
	addq $8, %r8
	movq %r9, %r14
	cmpq %r9, %r10
	jne .LBB22_8
	addq %r10, %rdx
	movq 120(%rsp), %r8
	addq %r8, %rdi
	addq %r10, %rbx
	addq %r10, 40(%rsp)
	addq %r10, 24(%rsp)
	addq %r10, 56(%rsp)
	addq %r10, 32(%rsp)
	addq %r10, 72(%rsp)
	addq %r10, 48(%rsp)
	addq %r10, 88(%rsp)
	addq %r10, 64(%rsp)
	addq %r10, 104(%rsp)
	addq %r10, 80(%rsp)
	addq %r10, 112(%rsp)
	addq %r10, 96(%rsp)
	addq %r10, %rbp
	addq %r10, %r13
	addq %r8, 8(%rsp)
	movq 128(%rsp), %r9
	movq 16(%rsp), %rax
	cmpq %r9, %rax
	jne .LBB22_3
.LBB22_28:
	addq $136, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB22_27:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.11(%rip), %rdx
	movq %rax, %rdi
	movq %rcx, %rsi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
.LBB22_9:
	movq %r8, %rax
.LBB22_10:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.12(%rip), %rdx
	movq %rax, %rdi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
