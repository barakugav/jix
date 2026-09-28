probe_bit_shuffle_untrans_bit_byte:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $184, %rsp
	movq %rcx, 16(%rsp)
	movq %r8, %r11
	shrq $3, %r11
	movq %r9, 24(%rsp)
	testq %r9, %r9
	sete %al
	testq %r11, %r11
	sete %cl
	orb %al, %cl
	jne .LBB23_19
	movq 24(%rsp), %r12
	vmovq %r12, %xmm0
	vmovq %r11, %xmm3
	vpmullq %xmm3, %xmm0, %xmm5
	vpbroadcastq %xmm5, %ymm4
	vpmullq .LCPI23_0(%rip), %ymm4, %ymm6
	vpmullq .LCPI23_1(%rip), %ymm4, %ymm0
	movq 16(%rsp), %rax
	incq %rax
	movq %rax, 160(%rsp)
	vpmullq .LCPI23_2(%rip), %xmm4, %xmm1
	vpbroadcastq %rsi, %zmm2
	vpblendd $252, .LCPI23_3(%rip), %ymm3, %ymm3
	vmovq %xmm5, 152(%rsp)
	vpmullq .LCPI23_4(%rip), %xmm4, %xmm4
	vinserti32x4 $2, %xmm4, %zmm6, %zmm6
	vmovdqa64 .LCPI23_5(%rip), %zmm4
	vpermi2q %zmm5, %zmm6, %zmm4
	imulq %r11, %r12
	leaq (,%r12,8), %rax
	subq %r12, %rax
	movq %rax, 96(%rsp)
	addq %rdi, %rax
	movq %rax, 88(%rsp)
	leaq (%r12,%r12), %rax
	movq %rax, 136(%rsp)
	leaq (%rax,%rax,2), %rax
	movq %rax, 104(%rsp)
	leaq (%rdi,%rax), %rax
	movq %rax, 80(%rsp)
	leaq (%r12,%r12,4), %rax
	movq %rax, 112(%rsp)
	leaq (%rdi,%rax), %rax
	movq %rax, 72(%rsp)
	leaq (,%r12,4), %rax
	movq %rax, 120(%rsp)
	leaq (%rdi,%r12,4), %rax
	movq %rax, 64(%rsp)
	leaq (%r12,%r12,2), %rcx
	movb $-64, %al
	kmovd %eax, %k1
	movq %rcx, 128(%rsp)
	leaq (%rdi,%rcx), %rax
	movq %rax, 56(%rsp)
	vmovdqa .LCPI23_6(%rip), %ymm5
	vbroadcasti32x4 .LCPI23_8(%rip), %zmm6
	vpbroadcastq .LCPI23_9(%rip), %zmm7
	vpbroadcastq .LCPI23_10(%rip), %zmm8
	vmovdqa64 .LCPI23_7(%rip), %zmm9
	vpbroadcastq .LCPI23_11(%rip), %zmm10
	leaq (%rdi,%r12,2), %r9
	leaq (%rdi,%r12), %r14
	movq $-1, 40(%rsp)
	movl $7, %eax
	movq %rax, 32(%rsp)
	movl $0, %r13d
	movq %r12, %rbx
	xorl %ecx, %ecx
	movl $8, %r10d
	xorl %r15d, %r15d
	movq %r8, 144(%rsp)
.LBB23_3:
	movq %r15, 8(%rsp)
	movq 160(%rsp), %rbp
	cmpq %rbp, %r10
	movq %rbp, %rax
	movq %r10, 176(%rsp)
	cmovaq %r10, %rax
	cmpq $-7, %rcx
	movq $-8, %r10
	movq %rcx, 48(%rsp)
	cmovaeq %rcx, %r10
	movq %r10, 168(%rsp)
	movq %r8, %rcx
	imulq %r15, %rcx
	movq $-8, %r8
	subq %rcx, %r8
	movq %r8, %r10
	movl $0, %r15d
	cmovbq %r15, %r10
	movq 8(%rsp), %r15
	negq %r8
	cmpq %rbp, %r8
	cmovbeq %rbp, %r8
	notq %rcx
	addq %rcx, %r8
	shrq $3, %r8
	addq $7, %r10
	shrq $3, %r10
	movq %r11, %rcx
	imulq %r15, %rcx
	vmovq %r15, %xmm11
	vpblendd $252, %ymm5, %ymm11, %ymm11
	vpmullq %ymm11, %ymm3, %ymm11
	vpermq %zmm11, %zmm9, %zmm12
	movq 152(%rsp), %rbp
	addq %rcx, %rbp
	vpbroadcastq %rcx, %zmm13
	vpaddq %zmm4, %zmm13, %zmm13
	vpmaxuq %zmm13, %zmm2, %zmm13
	vpbroadcastq %rbp, %zmm14
	vpermt2q %zmm11, %zmm6, %zmm14
	vshufi64x2 $228, %zmm14, %zmm0, %zmm11
	vinserti32x4 $2, %xmm1, %zmm11, %zmm11
	vpsubq %zmm12, %zmm11, %zmm11
	vpaddq %zmm11, %zmm13, %zmm12
	vpsubq %zmm11, %zmm13, %zmm12 {%k1}
	vextracti64x4 $1, %zmm12, %ymm11
	vpminuq %ymm11, %ymm12, %ymm11
	vextracti128 $1, %ymm11, %xmm12
	vpminuq %xmm12, %xmm11, %xmm11
	vpshufd $238, %xmm11, %xmm12
	vpminuq %xmm12, %xmm11, %xmm11
	vmovq %xmm11, %rbp
	cmpq %r8, %rbp
	cmovbq %rbp, %r8
	leaq -1(%r11), %rcx
	cmpq %rcx, %r10
	cmovaeq %rcx, %r10
	cmpq %r10, %r8
	cmovbq %r8, %r10
	movl $0, %r8d
	cmpq $8, %r10
	jb .LBB23_7
	addq 40(%rsp), %rax
	shrq $3, %rax
	movq %rbp, %r15
	movq 168(%rsp), %rbp
	addq 32(%rsp), %rbp
	shrq $3, %rbp
	incq %r10
	movl %r10d, %ecx
	andl $7, %ecx
	movl $8, %r8d
	cmoveq %r8, %rcx
	movq %r10, %r8
	subq %rcx, %r8
	cmpq %rax, %r15
	cmovbq %r15, %rax
	cmpq %rbp, %rax
	cmovaeq %rbp, %rax
	leaq -1(%r11), %r10
	cmpq %r10, %rax
	cmovaeq %r10, %rax
	subq %rcx, %rax
	incq %rax
	xorl %r10d, %r10d
.LBB23_5:
	leaq (%rdi,%r10), %rcx
	vpmovzxbq (%rdi,%r10), %zmm11
	vpmovzxbq (%r12,%rcx), %zmm12
	addq %r12, %rcx
	vpmovzxbq (%r12,%rcx), %zmm13
	addq %r12, %rcx
	vpmovzxbq (%r12,%rcx), %zmm14
	addq %r12, %rcx
	vpmovzxbq (%r12,%rcx), %zmm15
	addq %r12, %rcx
	vpmovzxbq (%r12,%rcx), %zmm16
	addq %r12, %rcx
	vpmovzxbq (%r12,%rcx), %zmm17
	addq %r12, %rcx
	vpmovzxbq (%r12,%rcx), %zmm18
	vpsllq $56, %zmm18, %zmm18
	vpsllq $48, %zmm17, %zmm17
	vpsllq $40, %zmm16, %zmm16
	vpternlogq $254, %zmm17, %zmm18, %zmm16
	vpsllq $32, %zmm15, %zmm15
	vpsllq $24, %zmm14, %zmm14
	vpternlogq $254, %zmm15, %zmm16, %zmm14
	vpsllq $16, %zmm13, %zmm13
	vpsllq $8, %zmm12, %zmm12
	vpternlogq $254, %zmm13, %zmm14, %zmm12
	vporq %zmm11, %zmm12, %zmm11
	vpsrlq $7, %zmm12, %zmm12
	vpternlogq $40, %zmm7, %zmm11, %zmm12
	vpsllq $7, %zmm12, %zmm13
	vpternlogq $54, %zmm12, %zmm11, %zmm13
	vpsrlq $14, %zmm13, %zmm11
	vpternlogq $40, %zmm8, %zmm13, %zmm11
	vpsllq $14, %zmm11, %zmm12
	vpternlogq $54, %zmm11, %zmm13, %zmm12
	vpsrlq $28, %zmm12, %zmm11
	vpternlogq $40, %zmm10, %zmm12, %zmm11
	vpsllq $28, %zmm11, %zmm13
	vpternlogq $54, %zmm11, %zmm12, %zmm13
	vmovdqu64 %zmm13, (%rdx,%r10,8)
	addq $8, %r10
	cmpq %r10, %rax
	jne .LBB23_5
	movq 8(%rsp), %r15
.LBB23_7:
	incq %r15
	movq %r15, 8(%rsp)
	movq 48(%rsp), %rax
	leaq (%rax,%r8,8), %rax
	movabsq $47851476196393130, %r15
.LBB23_8:
	leaq (%r8,%r13), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	leaq (%rbx,%r8), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 136(%rsp), %rcx
	leaq (%rcx,%r8), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 128(%rsp), %rcx
	leaq (%rcx,%r8), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 120(%rsp), %rcx
	leaq (%rcx,%r8), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 112(%rsp), %rcx
	leaq (%rcx,%r8), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 104(%rsp), %rcx
	leaq (%rcx,%r8), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 96(%rsp), %rcx
	leaq (%rcx,%r8), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	cmpq $-9, %rax
	ja .LBB23_20
	leaq 8(%rax), %r10
	cmpq 16(%rsp), %r10
	ja .LBB23_20
	movzbl (%rdi,%r8), %eax
	movq 80(%rsp), %rcx
	movzbl (%rcx,%r8), %ecx
	movq 88(%rsp), %rbp
	movzbl (%rbp,%r8), %ebp
	shlq $56, %rbp
	shlq $48, %rcx
	orq %rbp, %rcx
	movq 72(%rsp), %rbp
	movzbl (%rbp,%r8), %ebp
	shlq $40, %rbp
	orq %rcx, %rbp
	movq 64(%rsp), %rcx
	movzbl (%rcx,%r8), %ecx
	shlq $32, %rcx
	orq %rbp, %rcx
	movq 56(%rsp), %rbp
	movzbl (%rbp,%r8), %ebp
	shll $24, %ebp
	orq %rcx, %rbp
	movzbl (%r9,%r8), %ecx
	shll $16, %ecx
	orq %rbp, %rcx
	movzbl (%r14,%r8), %ebp
	shll $8, %ebp
	orq %rcx, %rbp
	orq %rbp, %rax
	shrq $7, %rbp
	xorq %rax, %rbp
	andq %r15, %rbp
	movq %rbp, %rcx
	shlq $7, %rcx
	orq %rbp, %rcx
	xorq %rax, %rcx
	movq %rcx, %rax
	shrq $14, %rax
	xorq %rcx, %rax
	movabsq $225176545447116, %rbp
	andq %rbp, %rax
	movq %rax, %rbp
	shlq $14, %rbp
	orq %rax, %rbp
	xorq %rcx, %rbp
	movq %rbp, %rax
	shrq $28, %rax
	xorl %ebp, %eax
	andl $-252645136, %eax
	movq %rax, %rcx
	shlq $28, %rcx
	orq %rax, %rcx
	xorq %rbp, %rcx
	movq %rcx, (%rdx,%r8,8)
	incq %r8
	movq %r10, %rax
	cmpq %r8, %r11
	jne .LBB23_8
	movq 144(%rsp), %r8
	movq 176(%rsp), %r10
	addq %r8, %r10
	subq %r8, 40(%rsp)
	movq 48(%rsp), %rcx
	addq %r8, %rcx
	subq %r8, 32(%rsp)
	addq %r8, %rdx
	addq %r11, %rdi
	addq %r11, 96(%rsp)
	addq %r11, 88(%rsp)
	addq %r11, 80(%rsp)
	addq %r11, 104(%rsp)
	addq %r11, 72(%rsp)
	addq %r11, 112(%rsp)
	addq %r11, 64(%rsp)
	addq %r11, 120(%rsp)
	addq %r11, 56(%rsp)
	addq %r11, 128(%rsp)
	addq %r11, %r9
	addq %r11, 136(%rsp)
	addq %r11, %r14
	addq %r11, %rbx
	addq %r11, %r13
	movq 8(%rsp), %r15
	cmpq 24(%rsp), %r15
	jne .LBB23_3
.LBB23_19:
	addq $184, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB23_20:
	leaq 8(%rax), %rsi
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.15(%rip), %rcx
	movq %rax, %rdi
	movq 16(%rsp), %rdx
	vzeroupper
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB23_21:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.16(%rip), %rdx
	movq %r10, %rdi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
