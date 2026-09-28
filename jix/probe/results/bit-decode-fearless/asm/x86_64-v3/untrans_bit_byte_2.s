probe_bit_shuffle_untrans_bit_byte:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $296, %rsp
	movq %r8, %r12
	movq %rdx, 56(%rsp)
	movq %rdi, %rdx
	movq %r8, 136(%rsp)
	shrq $3, %r12
	movq %r9, 64(%rsp)
	testq %r9, %r9
	sete %al
	testq %r12, %r12
	sete %dil
	orb %al, %dil
	jne .LBB23_18
	leaq 1(%rcx), %rax
	movq %rax, 240(%rsp)
	movq 64(%rsp), %rax
	imulq %r12, %rax
	vmovq %rax, %xmm0
	vpbroadcastq %xmm0, %ymm1
	vmovdqa .LCPI23_0(%rip), %ymm0
	vpmuludq %ymm0, %ymm1, %ymm2
	vpsrlq $32, %ymm1, %ymm3
	vpmuludq %ymm0, %ymm3, %ymm0
	vpsllq $32, %ymm0, %ymm0
	vpaddq %ymm0, %ymm2, %ymm0
	vpbroadcastq .LCPI23_1(%rip), %ymm2
	vpmuludq %ymm2, %ymm1, %ymm2
	vmovdqa .LCPI23_2(%rip), %ymm4
	vpmuludq %ymm4, %ymm3, %ymm3
	vpaddq %ymm3, %ymm2, %ymm2
	vpsllq $32, %ymm2, %ymm2
	vpmuludq %ymm4, %ymm1, %ymm1
	vpaddq %ymm2, %ymm1, %ymm1
	leaq (%rax,%rax), %r13
	leaq (,%r13,2), %r8
	addq %r13, %r8
	movq %r8, %r11
	negq %r11
	leaq (,%rax,8), %rbx
	movq %rax, %rdi
	subq %rbx, %rdi
	subq %rax, %rbx
	leaq -1(%r12), %r9
	movq %r9, 120(%rsp)
	vmovq %rsi, %xmm2
	vpbroadcastq %xmm2, %ymm2
	movq %rax, %r9
	negq %r9
	movq %r9, 112(%rsp)
	leaq (%rdx,%rbx), %r9
	movq %r9, 48(%rsp)
	leaq (%rdx,%r8), %r9
	movq %r9, 40(%rsp)
	leaq (%rax,%rax,4), %r9
	movq %r9, 144(%rsp)
	addq %rdx, %r9
	movq %r9, 32(%rsp)
	leaq (,%rax,4), %r9
	movq %r9, 152(%rsp)
	leaq (%rdx,%rax,4), %r14
	vpbroadcastq .LCPI23_3(%rip), %ymm3
	vpxor %ymm3, %ymm2, %ymm4
	vpbroadcastq .LCPI23_4(%rip), %ymm5
	vpbroadcastq .LCPI23_5(%rip), %ymm6
	leaq (%rax,%rax,2), %r9
	movq %r9, 160(%rsp)
	leaq (%rdx,%r9), %r15
	leaq (%rdx,%rax,2), %rbp
	leaq (%rdx,%rax), %r9
	movq $-1, 104(%rsp)
	movl $7, %r10d
	movq %r10, 96(%rsp)
	movq $0, 88(%rsp)
	xorl %r10d, %r10d
	movq %rdi, 200(%rsp)
	movq %rdi, 80(%rsp)
	movq %rbx, 208(%rsp)
	movq %r11, 216(%rsp)
	movq %r11, 72(%rsp)
	movl $8, %edi
	movq %rdi, 128(%rsp)
	movq %r8, 224(%rsp)
	movq %r8, %rdi
	movq %rax, 232(%rsp)
	movq %rax, %r8
	xorl %eax, %eax
	movq $0, (%rsp)
	movq %rcx, 256(%rsp)
	movq %r12, 248(%rsp)
.LBB23_3:
	movq %r13, 288(%rsp)
	movq 240(%rsp), %r11
	movq 128(%rsp), %r13
	cmpq %r11, %r13
	movq %rax, %rcx
	movq %r11, %rax
	cmovaq %r13, %rax
	cmpq $-7, %rcx
	movq $-8, %r13
	movq %rcx, 24(%rsp)
	cmovaeq %rcx, %r13
	movq %r13, 264(%rsp)
	cmpq %r8, %rsi
	movq %r8, 184(%rsp)
	cmovaq %rsi, %r8
	movq %r8, 8(%rsp)
	cmpq %rdi, %rsi
	movq %rdi, 176(%rsp)
	cmovaq %rsi, %rdi
	movq %rdi, 280(%rsp)
	cmpq %rbx, %rsi
	movq %rbx, 168(%rsp)
	cmovaq %rsi, %rbx
	movq %rbx, 272(%rsp)
	cmpq %r10, %rsi
	movq %r10, 192(%rsp)
	cmovaq %rsi, %r10
	movq %r10, 16(%rsp)
	movq 136(%rsp), %rcx
	movq (%rsp), %rdi
	imulq %rdi, %rcx
	movq $-8, %rbx
	subq %rcx, %rbx
	movq %rbx, %r10
	movl $0, %r8d
	cmovbq %r8, %r10
	xorl %r13d, %r13d
	negq %rbx
	cmpq %r11, %rbx
	notq %rcx
	cmovbeq %r11, %rbx
	addq %rcx, %rbx
	shrq $3, %rbx
	addq $7, %r10
	shrq $3, %r10
	imulq %rdi, %r12
	movq 232(%rsp), %rcx
	leaq (%rcx,%r12), %rdi
	movq %rsi, %r8
	subq %rdi, %r8
	cmovbq %r13, %r8
	vmovq %r12, %xmm7
	vpbroadcastq %xmm7, %ymm7
	vpaddq %ymm7, %ymm0, %ymm8
	vpxor %ymm3, %ymm8, %ymm9
	vpcmpgtq %ymm9, %ymm4, %ymm9
	vblendvpd %ymm9, %ymm2, %ymm8, %ymm8
	vpsubq %ymm7, %ymm1, %ymm7
	movq 224(%rsp), %rcx
	leaq (%rcx,%r12), %rdi
	cmpq %rdi, %rsi
	cmovaq %rsi, %rdi
	movq %rdx, %rcx
	movq 216(%rsp), %r11
	subq %r12, %r11
	addq %rdi, %r11
	movq 208(%rsp), %rdx
	leaq (%rdx,%r12), %rdi
	cmpq %rdi, %rsi
	cmovaq %rsi, %rdi
	vpaddq %ymm7, %ymm8, %ymm7
	movq 200(%rsp), %r13
	subq %r12, %r13
	addq %rdi, %r13
	movq %rsi, %rdi
	subq %r12, %rdi
	movl $0, %edx
	cmovbq %rdx, %rdi
	vpxor %xmm3, %xmm7, %xmm8
	vextracti128 $1, %ymm7, %xmm9
	vpxor %xmm3, %xmm9, %xmm10
	vpcmpgtq %xmm8, %xmm10, %xmm8
	vblendvpd %xmm8, %xmm7, %xmm9, %xmm7
	vpextrq $1, %xmm7, %r12
	vmovq %xmm7, %rdx
	cmpq %r12, %rdx
	cmovbq %rdx, %r12
	cmpq %r11, %r12
	cmovbq %r12, %r11
	cmpq %r8, %r13
	cmovaeq %r8, %r13
	cmpq %rbx, %rdi
	cmovaeq %rbx, %rdi
	movq 120(%rsp), %rdx
	cmpq %rdx, %r10
	cmovaeq %rdx, %r10
	cmpq %r13, %r11
	cmovbq %r11, %r13
	movq %rcx, %rdx
	cmpq %r10, %rdi
	cmovbq %rdi, %r10
	cmpq %r10, %r13
	cmovbq %r13, %r10
	movl $0, %ebx
	cmpq $4, %r10
	jb .LBB23_6
	addq 104(%rsp), %rax
	shrq $3, %rax
	movq 264(%rsp), %rdi
	addq 96(%rsp), %rdi
	shrq $3, %rdi
	movq 8(%rsp), %r11
	addq 112(%rsp), %r11
	movq %r12, 8(%rsp)
	movq 280(%rsp), %r12
	addq 72(%rsp), %r12
	movq 272(%rsp), %r13
	addq 80(%rsp), %r13
	movq 16(%rsp), %rcx
	addq 88(%rsp), %rcx
	movq %rcx, 16(%rsp)
	incq %r10
	movl %r10d, %r8d
	andl $3, %r8d
	movl $4, %ecx
	cmoveq %rcx, %r8
	movq %r10, %rbx
	subq %r8, %rbx
	movq 8(%rsp), %rcx
	cmpq %rax, %rcx
	cmovbq %rcx, %rax
	cmpq %rdi, %rax
	cmovaeq %rdi, %rax
	cmpq %r11, %rax
	cmovaeq %r11, %rax
	cmpq %r12, %rax
	cmovaeq %r12, %rax
	cmpq %r13, %rax
	cmovaeq %r13, %rax
	movq 16(%rsp), %rcx
	cmpq %rcx, %rax
	cmovaeq %rcx, %rax
	movq 120(%rsp), %rcx
	cmpq %rcx, %rax
	cmovaeq %rcx, %rax
	subq %r8, %rax
	incq %rax
	xorl %r8d, %r8d
	movq 48(%rsp), %rcx
	movq 40(%rsp), %rdi
	movq 32(%rsp), %r10
.LBB23_5:
	vpmovzxbq (%rdx,%r8), %ymm7
	vpmovzxbq (%r9,%r8), %ymm8
	vpmovzxbq (%rbp,%r8), %ymm9
	vpmovzxbq (%r15,%r8), %ymm10
	vpmovzxbq (%r14,%r8), %ymm11
	vpmovzxbq (%r10,%r8), %ymm12
	vpmovzxbq (%rdi,%r8), %ymm13
	vpmovzxbq (%rcx,%r8), %ymm14
	vpsllq $56, %ymm14, %ymm14
	vpsllq $48, %ymm13, %ymm13
	vpor %ymm13, %ymm14, %ymm13
	vpsllq $40, %ymm12, %ymm12
	vpsllq $32, %ymm11, %ymm11
	vpor %ymm11, %ymm12, %ymm11
	vpor %ymm11, %ymm13, %ymm11
	vpsllq $24, %ymm10, %ymm10
	vpsllq $16, %ymm9, %ymm9
	vpor %ymm9, %ymm10, %ymm9
	vpsllq $8, %ymm8, %ymm8
	vpor %ymm8, %ymm9, %ymm8
	vpor %ymm8, %ymm11, %ymm8
	vpor %ymm7, %ymm8, %ymm7
	vpsrlq $7, %ymm8, %ymm8
	vpxor %ymm7, %ymm8, %ymm8
	vpand %ymm5, %ymm8, %ymm8
	vpsllq $7, %ymm8, %ymm9
	vpor %ymm8, %ymm9, %ymm8
	vpxor %ymm7, %ymm8, %ymm7
	vpsrlq $14, %ymm7, %ymm8
	vpxor %ymm7, %ymm8, %ymm8
	vpand %ymm6, %ymm8, %ymm8
	vpsllq $14, %ymm8, %ymm9
	vpor %ymm8, %ymm9, %ymm8
	vpxor %ymm7, %ymm8, %ymm7
	vpsrlq $28, %ymm7, %ymm8
	vpbroadcastq .LCPI23_6(%rip), %ymm9
	vpxor %ymm7, %ymm8, %ymm8
	vpand %ymm9, %ymm8, %ymm8
	vpsllq $28, %ymm8, %ymm9
	vpor %ymm8, %ymm9, %ymm8
	vpxor %ymm7, %ymm8, %ymm7
	movq 56(%rsp), %r11
	vmovdqu %ymm7, (%r11,%r8,8)
	addq $4, %r8
	cmpq %r8, %rax
	jne .LBB23_5
.LBB23_6:
	movq %r9, %r11
	movq %rbp, %r9
	movq %r15, %rbp
	movq %r14, %r15
	movq (%rsp), %r14
	incq %r14
	movq 24(%rsp), %rax
	leaq (%rax,%rbx,8), %rax
	movq 256(%rsp), %rcx
	movq 248(%rsp), %r12
	movq 288(%rsp), %r13
.LBB23_7:
	movq 192(%rsp), %rdi
	leaq (%rdi,%rbx), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	movq 184(%rsp), %rdi
	leaq (%rdi,%rbx), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	leaq (%rbx,%r13), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	movq 160(%rsp), %rdi
	leaq (%rdi,%rbx), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	movq 152(%rsp), %rdi
	leaq (%rdi,%rbx), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	movq 144(%rsp), %rdi
	leaq (%rdi,%rbx), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	movq 176(%rsp), %rdi
	leaq (%rdi,%rbx), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	movq 168(%rsp), %rdi
	leaq (%rdi,%rbx), %r10
	cmpq %rsi, %r10
	jae .LBB23_20
	cmpq $-9, %rax
	ja .LBB23_19
	leaq 8(%rax), %r10
	cmpq %rcx, %r10
	ja .LBB23_19
	movzbl (%rdx,%rbx), %eax
	movq 40(%rsp), %rdi
	movzbl (%rdi,%rbx), %r8d
	movq 48(%rsp), %rdi
	movzbl (%rdi,%rbx), %edi
	shlq $56, %rdi
	shlq $48, %r8
	orq %rdi, %r8
	movq 32(%rsp), %rdi
	movzbl (%rdi,%rbx), %edi
	shlq $40, %rdi
	orq %r8, %rdi
	movzbl (%r15,%rbx), %r8d
	shlq $32, %r8
	orq %rdi, %r8
	movzbl (%rbp,%rbx), %edi
	shll $24, %edi
	orq %r8, %rdi
	movzbl (%r9,%rbx), %r8d
	shll $16, %r8d
	orq %rdi, %r8
	movzbl (%r11,%rbx), %edi
	shll $8, %edi
	orq %r8, %rdi
	orq %rdi, %rax
	shrq $7, %rdi
	xorq %rax, %rdi
	movabsq $47851476196393130, %r8
	andq %r8, %rdi
	movq %rdi, %r8
	shlq $7, %r8
	orq %rdi, %r8
	xorq %rax, %r8
	movq %r8, %rax
	shrq $14, %rax
	xorq %r8, %rax
	movabsq $225176545447116, %rdi
	andq %rdi, %rax
	movq %rax, %rdi
	shlq $14, %rdi
	orq %rax, %rdi
	xorq %r8, %rdi
	movq %rdi, %rax
	shrq $28, %rax
	xorl %edi, %eax
	andl $-252645136, %eax
	movq %rax, %r8
	shlq $28, %r8
	orq %rax, %r8
	xorq %rdi, %r8
	movq 56(%rsp), %rax
	movq %r8, (%rax,%rbx,8)
	incq %rbx
	movq %r10, %rax
	cmpq %rbx, %r12
	jne .LBB23_7
	movq 136(%rsp), %rax
	addq %rax, 128(%rsp)
	subq %rax, 104(%rsp)
	addq %rax, 24(%rsp)
	subq %rax, 96(%rsp)
	movq %r14, (%rsp)
	movq 184(%rsp), %r8
	addq %r12, %r8
	subq %r12, 112(%rsp)
	movq 176(%rsp), %rdi
	addq %r12, %rdi
	subq %r12, 72(%rsp)
	movq 168(%rsp), %rbx
	addq %r12, %rbx
	subq %r12, 80(%rsp)
	movq 192(%rsp), %r10
	addq %r12, %r10
	subq %r12, 88(%rsp)
	addq %r12, 48(%rsp)
	addq %rax, 56(%rsp)
	addq %r12, 40(%rsp)
	addq %r12, 32(%rsp)
	movq %r15, %r14
	addq %r12, %r14
	movq %rbp, %r15
	addq %r12, %r15
	movq %r9, %rbp
	addq %r12, %rbp
	movq %r11, %r9
	addq %r12, %r9
	addq %r12, %rdx
	addq %r12, 144(%rsp)
	addq %r12, 152(%rsp)
	addq %r12, 160(%rsp)
	addq %r12, %r13
	movq (%rsp), %r11
	movq 24(%rsp), %rax
	cmpq 64(%rsp), %r11
	jne .LBB23_3
.LBB23_18:
	addq $296, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB23_19:
	leaq 8(%rax), %rsi
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.15(%rip), %r8
	movq %rax, %rdi
	movq %rcx, %rdx
	movq %r8, %rcx
	vzeroupper
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB23_20:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.16(%rip), %rdx
	movq %r10, %rdi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
