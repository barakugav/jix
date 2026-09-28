jix_probe::bit_shuffle::trans_bit_byte:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $536, %rsp
	movq %rcx, 304(%rsp)
	movq %rsi, 288(%rsp)
	movq %r8, %r12
	shrq $3, %r12
	movq %r9, 200(%rsp)
	testq %r9, %r9
	sete %al
	testq %r12, %r12
	sete %r9b
	orb %al, %r9b
	jne .LBB22_28
	movq 200(%rsp), %rsi
	imulq %r12, %rsi
	leaq (%rsi,%rsi), %rcx
	leaq (,%rsi,4), %r9
	leaq (,%rsi,8), %r14
	movq %rcx, %rax
	negq %rax
	movq %rax, 192(%rsp)
	leaq (%rsi,%rsi,2), %rax
	movq %rax, %r11
	negq %r11
	movq %r11, 184(%rsp)
	movq %r9, %rbp
	negq %rbp
	leaq (%rsi,%rsi,4), %r15
	negq %r15
	leaq (%rcx,%rcx,2), %rbx
	movq %rbx, 176(%rsp)
	negq %rbx
	movq %rsi, %r11
	subq %r14, %r11
	subq %rsi, %r14
	movq %r9, %xmm0
	movq %rax, %xmm1
	punpcklqdq %xmm0, %xmm1
	movq %rsi, %xmm0
	movq %rcx, %xmm2
	punpcklqdq %xmm2, %xmm0
	pcmpeqd %xmm2, %xmm2
	paddq %xmm2, %xmm0
	paddq %xmm2, %xmm1
	leaq (%rsi,%rsi,4), %rax
	decq %rax
	cmpq $15, %rax
	setb %al
	leaq (%rcx,%rcx,2), %r9
	decq %r9
	cmpq $15, %r9
	setb %r10b
	leaq -1(%r14), %r9
	cmpq $15, %r9
	setb %r13b
	movdqa .LCPI22_0(%rip), %xmm7
	pxor %xmm7, %xmm1
	movdqa .LCPI22_1(%rip), %xmm2
	movdqa %xmm2, %xmm3
	pcmpgtq %xmm1, %xmm3
	pxor %xmm7, %xmm0
	pcmpgtq %xmm0, %xmm2
	packssdw %xmm3, %xmm2
	movmskps %xmm2, %r9d
	testl %r9d, %r9d
	movq 288(%rsp), %xmm0
	pshufd $68, %xmm0, %xmm8
	setne %r9b
	orb %r10b, %r13b
	orb %al, %r13b
	orb %r9b, %r13b
	movb %r13b, 15(%rsp)
	movq %rsi, %rax
	negq %rax
	movq %rax, 264(%rsp)
	leaq 127(%rdi), %r13
	addq $3, %rdi
	movq %rdi, 296(%rsp)
	movdqa .LCPI22_2(%rip), %xmm9
	movdqa .LCPI22_3(%rip), %xmm10
	movdqa .LCPI22_4(%rip), %xmm11
	movdqa .LCPI22_5(%rip), %xmm12
	movdqa %xmm8, %xmm13
	pxor %xmm7, %xmm13
	movdqa .LCPI22_6(%rip), %xmm14
	movdqa .LCPI22_7(%rip), %xmm15
	leaq -1(%r12), %rax
	movq %rax, 272(%rsp)
	leaq (%rdx,%r14), %rax
	movq 176(%rsp), %r10
	leaq (%rdx,%r10), %rdi
	movq %rdi, 56(%rsp)
	leaq (%rsi,%rsi,4), %rcx
	leaq (%rdx,%rcx), %rdi
	movq %rdi, 120(%rsp)
	leaq (%rdx,%rsi,4), %rdi
	movq %rdi, 96(%rsp)
	leaq (%rsi,%rsi,2), %rdi
	leaq (%rdx,%rdi), %rdi
	movq %rdi, 104(%rsp)
	leaq (%rdx,%rsi,2), %rdi
	movq %rdi, 112(%rsp)
	leaq (%rdx,%rsi), %rdi
	movl $0, %r9d
	movq %r9, 280(%rsp)
	movl $0, %r9d
	movq %r9, 256(%rsp)
	movl $0, %r9d
	movq %r11, 368(%rsp)
	movq %r11, 248(%rsp)
	movq %r14, 416(%rsp)
	movq %rbx, 376(%rsp)
	movq %rbx, 240(%rsp)
	movq %r15, 384(%rsp)
	movq %r15, 232(%rsp)
	movq %rcx, 392(%rsp)
	movq %rcx, %r11
	movq %rbp, 400(%rsp)
	movq %rbp, 224(%rsp)
	movq %r10, %rbp
	leaq (,%rsi,4), %rcx
	movq %rcx, 424(%rsp)
	leaq (,%rsi,4), %r10
	movq 184(%rsp), %rbx
	movq %rbx, 216(%rsp)
	leaq (%rsi,%rsi,2), %rcx
	movq %rcx, 408(%rsp)
	leaq (%rsi,%rsi,2), %rbx
	movq 192(%rsp), %r15
	movq %r15, 208(%rsp)
	leaq (%rsi,%rsi), %rcx
	movq %rcx, 432(%rsp)
	leaq (%rsi,%rsi), %r15
	movq %rsi, 440(%rsp)
	xorl %ecx, %ecx
	movq %r8, 360(%rsp)
	movdqa %xmm8, 512(%rsp)
	movdqa %xmm13, 496(%rsp)
	movq %r12, 480(%rsp)
.LBB22_3:
	movq %rdi, 128(%rsp)
	movq %rax, 312(%rsp)
	movq %rdx, 320(%rsp)
	movq %rcx, %rdx
	movq 304(%rsp), %rcx
	cmpq %rsi, %rcx
	movq %rsi, 152(%rsp)
	cmovaq %rcx, %rsi
	movq %rsi, 16(%rsp)
	cmpq %r15, %rcx
	movq %r15, 144(%rsp)
	cmovaq %rcx, %r15
	movq %r15, 32(%rsp)
	cmpq %rbx, %rcx
	movq %rbx, 136(%rsp)
	cmovaq %rcx, %rbx
	movq %rbx, 64(%rsp)
	cmpq %r10, %rcx
	movq %r10, 344(%rsp)
	cmovaq %rcx, %r10
	movq %r10, 160(%rsp)
	cmpq %r11, %rcx
	movq %r11, 336(%rsp)
	cmovaq %rcx, %r11
	movq %r11, 88(%rsp)
	cmpq %rbp, %rcx
	movq %rbp, 488(%rsp)
	cmovaq %rcx, %rbp
	movq %rbp, 464(%rsp)
	cmpq %r14, %rcx
	movq %r14, 328(%rsp)
	cmovaq %rcx, %r14
	movq %r14, 456(%rsp)
	cmpq %r9, %rcx
	movq %r9, 352(%rsp)
	cmovaq %rcx, %r9
	movq %r9, 448(%rsp)
	movq %r8, %rax
	imulq %rdx, %rax
	movq %rax, %xmm0
	movq %r12, %r10
	movq %rdx, 472(%rsp)
	imulq %rdx, %r10
	movq 440(%rsp), %rax
	addq %r10, %rax
	movq %rcx, %r8
	subq %rax, %r8
	movl $0, %edi
	cmovbq %rdi, %r8
	movq 432(%rsp), %rax
	leaq (%rax,%r10), %rdx
	cmpq %rdx, %rcx
	cmovaq %rcx, %rdx
	movq 192(%rsp), %rax
	subq %r10, %rax
	addq %rdx, %rax
	movq 408(%rsp), %rdx
	addq %r10, %rdx
	cmpq %rdx, %rcx
	cmovaq %rcx, %rdx
	movq 184(%rsp), %r15
	subq %r10, %r15
	addq %rdx, %r15
	movq 424(%rsp), %rdx
	leaq (%rdx,%r10), %r9
	cmpq %r9, %rcx
	cmovaq %rcx, %r9
	movq 400(%rsp), %rdx
	subq %r10, %rdx
	addq %r9, %rdx
	movq 392(%rsp), %rsi
	leaq (%rsi,%r10), %r9
	cmpq %r9, %rcx
	cmovaq %rcx, %r9
	movq 384(%rsp), %rbp
	subq %r10, %rbp
	addq %r9, %rbp
	movq 176(%rsp), %rsi
	leaq (%rsi,%r10), %rbx
	cmpq %rbx, %rcx
	cmovaq %rcx, %rbx
	movq 376(%rsp), %r9
	subq %r10, %r9
	addq %rbx, %r9
	movq 416(%rsp), %rsi
	leaq (%rsi,%r10), %rbx
	cmpq %rbx, %rcx
	cmovaq %rcx, %rbx
	movq 368(%rsp), %r14
	subq %r10, %r14
	addq %rbx, %r14
	pshufd $68, %xmm0, %xmm4
	movdqa %xmm4, %xmm3
	paddq %xmm9, %xmm3
	movdqa %xmm4, %xmm1
	movdqa %xmm4, %xmm5
	movdqa %xmm4, %xmm2
	paddq %xmm12, %xmm2
	movdqa %xmm2, %xmm6
	pxor %xmm7, %xmm6
	movdqa %xmm13, %xmm0
	pcmpgtq %xmm6, %xmm0
	blendvpd %xmm0, %xmm8, %xmm2
	paddq %xmm11, %xmm5
	movdqa %xmm5, %xmm6
	pxor %xmm7, %xmm6
	movdqa %xmm13, %xmm0
	pcmpgtq %xmm6, %xmm0
	blendvpd %xmm0, %xmm8, %xmm5
	paddq %xmm10, %xmm1
	movdqa %xmm1, %xmm6
	pxor %xmm7, %xmm6
	movdqa %xmm13, %xmm0
	pcmpgtq %xmm6, %xmm0
	blendvpd %xmm0, %xmm8, %xmm1
	movdqa %xmm3, %xmm6
	pxor %xmm7, %xmm6
	movdqa %xmm13, %xmm0
	pcmpgtq %xmm6, %xmm0
	blendvpd %xmm0, %xmm8, %xmm3
	movq %rcx, %rbx
	subq %r10, %rbx
	psubq %xmm4, %xmm5
	paddq %xmm15, %xmm5
	psubq %xmm4, %xmm2
	paddq %xmm14, %xmm2
	psrlq $3, %xmm5
	psrlq $3, %xmm2
	movdqa %xmm2, %xmm0
	pcmpgtq %xmm5, %xmm0
	blendvpd %xmm0, %xmm5, %xmm2
	psubq %xmm4, %xmm3
	paddq .LCPI22_9(%rip), %xmm3
	psubq %xmm4, %xmm1
	paddq .LCPI22_8(%rip), %xmm1
	psrlq $3, %xmm3
	psrlq $3, %xmm1
	movdqa %xmm1, %xmm0
	pcmpgtq %xmm3, %xmm0
	blendvpd %xmm0, %xmm3, %xmm1
	movapd %xmm1, %xmm0
	pcmpgtq %xmm2, %xmm0
	blendvpd %xmm0, %xmm2, %xmm1
	cmovbq %rdi, %rbx
	pextrq $1, %xmm1, %r10
	movq %xmm1, %r11
	cmpq %r10, %r11
	cmovbq %r11, %r10
	cmpq %rax, %r10
	cmovbq %r10, %rax
	cmpq %rdx, %r15
	cmovbq %r15, %rdx
	cmpq %r9, %rbp
	cmovbq %rbp, %r9
	cmpq %r8, %r14
	cmovaeq %r8, %r14
	movq 272(%rsp), %rsi
	cmpq %rsi, %rbx
	cmovaeq %rsi, %rbx
	cmpq %rdx, %rax
	cmovbq %rax, %rdx
	cmpq %r14, %r9
	cmovbq %r9, %r14
	cmpq %r14, %rdx
	cmovbq %rdx, %r14
	cmpq %rbx, %r14
	cmovaeq %rbx, %r14
	cmpq $16, %r14
	setb %al
	orb 15(%rsp), %al
	movl $0, %ecx
	testb $1, %al
	movq 56(%rsp), %r9
	movq 120(%rsp), %rbx
	jne .LBB22_7
	movq 16(%rsp), %rcx
	addq 264(%rsp), %rcx
	movq 32(%rsp), %rdi
	addq 208(%rsp), %rdi
	movq %r9, 56(%rsp)
	movq 64(%rsp), %r11
	addq 216(%rsp), %r11
	movq 160(%rsp), %r15
	addq 224(%rsp), %r15
	movq 88(%rsp), %rbp
	addq 232(%rsp), %rbp
	movq 464(%rsp), %rsi
	addq 240(%rsp), %rsi
	movq 456(%rsp), %r8
	addq 248(%rsp), %r8
	movq 448(%rsp), %r9
	addq 256(%rsp), %r9
	incq %r14
	movl %r14d, %eax
	andl $15, %eax
	movl $16, %edx
	cmoveq %rdx, %rax
	subq %rax, %r14
	movq %r14, 88(%rsp)
	cmpq %rcx, %r10
	cmovbq %r10, %rcx
	cmpq %rdi, %rcx
	cmovaeq %rdi, %rcx
	movq 320(%rsp), %rdx
	movq 312(%rsp), %r14
	cmpq %r11, %rcx
	cmovaeq %r11, %rcx
	movq %rbx, %r11
	movq 56(%rsp), %rdi
	cmpq %r15, %rcx
	cmovaeq %r15, %rcx
	cmpq %rbp, %rcx
	cmovaeq %rbp, %rcx
	cmpq %rsi, %rcx
	cmovaeq %rsi, %rcx
	movq %rdx, %rsi
	cmpq %r8, %rcx
	cmovaeq %r8, %rcx
	cmpq %r9, %rcx
	cmovaeq %r9, %rcx
	movq 272(%rsp), %rdx
	cmpq %rdx, %rcx
	cmovaeq %rdx, %rcx
	subq %rax, %rcx
	incq %rcx
	xorl %eax, %eax
	movq 128(%rsp), %r10
	movq 96(%rsp), %r9
	movq 104(%rsp), %rbx
	movq 112(%rsp), %r15
.LBB22_5:
	movzbl -120(%r13,%rax,8), %edx
	movd %edx, %xmm11
	movzbl -104(%r13,%rax,8), %edx
	movd %edx, %xmm5
	movzbl -88(%r13,%rax,8), %edx
	movd %edx, %xmm8
	movzbl -72(%r13,%rax,8), %edx
	movd %edx, %xmm9
	movzbl -56(%r13,%rax,8), %edx
	movd %edx, %xmm10
	movzbl -40(%r13,%rax,8), %edx
	movd %edx, %xmm1
	movzbl -24(%r13,%rax,8), %edx
	movd %edx, %xmm14
	movzbl -8(%r13,%rax,8), %edx
	movd %edx, %xmm0
	movzbl -121(%r13,%rax,8), %edx
	movd %edx, %xmm2
	movdqa %xmm2, 16(%rsp)
	movzbl -105(%r13,%rax,8), %edx
	movd %edx, %xmm7
	movzbl -89(%r13,%rax,8), %edx
	movd %edx, %xmm12
	movzbl -73(%r13,%rax,8), %edx
	movd %edx, %xmm4
	movzbl -57(%r13,%rax,8), %edx
	movd %edx, %xmm13
	movzbl -41(%r13,%rax,8), %edx
	movd %edx, %xmm2
	movzbl -25(%r13,%rax,8), %edx
	movd %edx, %xmm3
	movzbl -9(%r13,%rax,8), %edx
	pinsrb $8, (%r13,%rax,8), %xmm0
	movd %edx, %xmm6
	movzbl -10(%r13,%rax,8), %edx
	psllq $56, %xmm0
	pinsrb $8, -1(%r13,%rax,8), %xmm6
	psllq $48, %xmm6
	por %xmm0, %xmm6
	movd %edx, %xmm0
	movzbl -26(%r13,%rax,8), %edx
	pinsrb $8, -16(%r13,%rax,8), %xmm14
	pinsrb $8, -17(%r13,%rax,8), %xmm3
	psllq $56, %xmm14
	psllq $48, %xmm3
	por %xmm14, %xmm3
	movd %edx, %xmm14
	movzbl -42(%r13,%rax,8), %edx
	pinsrb $8, -32(%r13,%rax,8), %xmm1
	psllq $56, %xmm1
	pinsrb $8, -33(%r13,%rax,8), %xmm2
	psllq $48, %xmm2
	por %xmm1, %xmm2
	movd %edx, %xmm15
	movzbl -58(%r13,%rax,8), %edx
	pinsrb $8, -64(%r13,%rax,8), %xmm9
	pinsrb $8, -48(%r13,%rax,8), %xmm10
	pinsrb $8, -49(%r13,%rax,8), %xmm13
	psllq $56, %xmm10
	psllq $48, %xmm13
	por %xmm10, %xmm13
	movd %edx, %xmm1
	movzbl -74(%r13,%rax,8), %edx
	psllq $56, %xmm9
	pinsrb $8, -65(%r13,%rax,8), %xmm4
	psllq $48, %xmm4
	por %xmm9, %xmm4
	movd %edx, %xmm9
	movzbl -90(%r13,%rax,8), %edx
	pinsrb $8, -80(%r13,%rax,8), %xmm8
	pinsrb $8, -81(%r13,%rax,8), %xmm12
	psllq $56, %xmm8
	psllq $48, %xmm12
	por %xmm8, %xmm12
	movd %edx, %xmm8
	movzbl -106(%r13,%rax,8), %edx
	pinsrb $8, -96(%r13,%rax,8), %xmm5
	psllq $56, %xmm5
	pinsrb $8, -97(%r13,%rax,8), %xmm7
	psllq $48, %xmm7
	por %xmm5, %xmm7
	movd %edx, %xmm5
	movzbl -11(%r13,%rax,8), %edx
	pinsrb $8, -112(%r13,%rax,8), %xmm11
	psllq $56, %xmm11
	movdqa 16(%rsp), %xmm10
	pinsrb $8, -113(%r13,%rax,8), %xmm10
	psllq $48, %xmm10
	por %xmm11, %xmm10
	movdqa %xmm10, 16(%rsp)
	pinsrb $8, -2(%r13,%rax,8), %xmm0
	movd %edx, %xmm11
	movzbl -122(%r13,%rax,8), %edx
	psllq $40, %xmm0
	pinsrb $8, -3(%r13,%rax,8), %xmm11
	psllq $32, %xmm11
	por %xmm0, %xmm11
	movd %edx, %xmm10
	movzbl -27(%r13,%rax,8), %edx
	pinsrb $8, -18(%r13,%rax,8), %xmm14
	por %xmm6, %xmm11
	movdqa %xmm11, 32(%rsp)
	movd %edx, %xmm0
	movzbl -123(%r13,%rax,8), %edx
	psllq $40, %xmm14
	pinsrb $8, -19(%r13,%rax,8), %xmm0
	psllq $32, %xmm0
	por %xmm14, %xmm0
	movd %edx, %xmm14
	movzbl -43(%r13,%rax,8), %edx
	pinsrb $8, -34(%r13,%rax,8), %xmm15
	por %xmm3, %xmm0
	movdqa %xmm0, 64(%rsp)
	movd %edx, %xmm0
	movzbl -107(%r13,%rax,8), %edx
	psllq $40, %xmm15
	pinsrb $8, -35(%r13,%rax,8), %xmm0
	psllq $32, %xmm0
	por %xmm15, %xmm0
	movd %edx, %xmm15
	movzbl -59(%r13,%rax,8), %edx
	pinsrb $8, -50(%r13,%rax,8), %xmm1
	por %xmm2, %xmm0
	movdqa %xmm0, 160(%rsp)
	movd %edx, %xmm11
	movzbl -91(%r13,%rax,8), %edx
	psllq $40, %xmm1
	pinsrb $8, -51(%r13,%rax,8), %xmm11
	psllq $32, %xmm11
	por %xmm1, %xmm11
	movd %edx, %xmm3
	movzbl -75(%r13,%rax,8), %edx
	pinsrb $8, -66(%r13,%rax,8), %xmm9
	por %xmm13, %xmm11
	movd %edx, %xmm13
	movzbl -12(%r13,%rax,8), %edx
	psllq $40, %xmm9
	pinsrb $8, -67(%r13,%rax,8), %xmm13
	psllq $32, %xmm13
	por %xmm9, %xmm13
	movd %edx, %xmm0
	movzbl -28(%r13,%rax,8), %edx
	pinsrb $8, -82(%r13,%rax,8), %xmm8
	por %xmm4, %xmm13
	movd %edx, %xmm6
	movzbl -44(%r13,%rax,8), %edx
	psllq $40, %xmm8
	pinsrb $8, -83(%r13,%rax,8), %xmm3
	psllq $32, %xmm3
	por %xmm8, %xmm3
	movd %edx, %xmm9
	movzbl -60(%r13,%rax,8), %edx
	pinsrb $8, -98(%r13,%rax,8), %xmm5
	por %xmm12, %xmm3
	movd %edx, %xmm8
	movzbl -76(%r13,%rax,8), %edx
	psllq $40, %xmm5
	pinsrb $8, -99(%r13,%rax,8), %xmm15
	psllq $32, %xmm15
	por %xmm5, %xmm15
	movd %edx, %xmm4
	movzbl -92(%r13,%rax,8), %edx
	pinsrb $8, -114(%r13,%rax,8), %xmm10
	por %xmm7, %xmm15
	movd %edx, %xmm2
	movzbl -108(%r13,%rax,8), %edx
	psllq $40, %xmm10
	pinsrb $8, -115(%r13,%rax,8), %xmm14
	psllq $32, %xmm14
	por %xmm10, %xmm14
	movd %edx, %xmm1
	movzbl -13(%r13,%rax,8), %edx
	por 16(%rsp), %xmm14
	pinsrb $8, -4(%r13,%rax,8), %xmm0
	movd %edx, %xmm5
	movzbl -29(%r13,%rax,8), %edx
	psllq $24, %xmm0
	pinsrb $8, -5(%r13,%rax,8), %xmm5
	psllq $16, %xmm5
	por %xmm0, %xmm5
	movdqa %xmm5, 16(%rsp)
	movd %edx, %xmm5
	movzbl -45(%r13,%rax,8), %edx
	pinsrb $8, -20(%r13,%rax,8), %xmm6
	pinsrb $8, -21(%r13,%rax,8), %xmm5
	psllq $24, %xmm6
	psllq $16, %xmm5
	por %xmm6, %xmm5
	movd %edx, %xmm6
	movzbl -61(%r13,%rax,8), %edx
	pinsrb $8, -36(%r13,%rax,8), %xmm9
	psllq $24, %xmm9
	pinsrb $8, -37(%r13,%rax,8), %xmm6
	psllq $16, %xmm6
	por %xmm9, %xmm6
	movd %edx, %xmm7
	movzbl -77(%r13,%rax,8), %edx
	pinsrb $8, -52(%r13,%rax,8), %xmm8
	pinsrb $8, -68(%r13,%rax,8), %xmm4
	pinsrb $8, -53(%r13,%rax,8), %xmm7
	psllq $24, %xmm8
	psllq $16, %xmm7
	por %xmm8, %xmm7
	movd %edx, %xmm0
	movzbl -93(%r13,%rax,8), %edx
	psllq $24, %xmm4
	pinsrb $8, -69(%r13,%rax,8), %xmm0
	psllq $16, %xmm0
	por %xmm4, %xmm0
	movd %edx, %xmm8
	movzbl -109(%r13,%rax,8), %edx
	pinsrb $8, -84(%r13,%rax,8), %xmm2
	pinsrb $8, -85(%r13,%rax,8), %xmm8
	psllq $24, %xmm2
	psllq $16, %xmm8
	por %xmm2, %xmm8
	movd %edx, %xmm9
	movzbl -124(%r13,%rax,8), %edx
	pinsrb $8, -100(%r13,%rax,8), %xmm1
	psllq $24, %xmm1
	pinsrb $8, -101(%r13,%rax,8), %xmm9
	psllq $16, %xmm9
	por %xmm1, %xmm9
	movd %edx, %xmm2
	movzbl -125(%r13,%rax,8), %edx
	pinsrb $8, -116(%r13,%rax,8), %xmm2
	psllq $24, %xmm2
	movd %edx, %xmm1
	pinsrb $8, -117(%r13,%rax,8), %xmm1
	psllq $16, %xmm1
	por %xmm2, %xmm1
	movzbl -126(%r13,%rax,8), %edx
	movd %edx, %xmm2
	pinsrb $8, -118(%r13,%rax,8), %xmm2
	psllq $8, %xmm2
	por %xmm1, %xmm2
	movzbl -110(%r13,%rax,8), %edx
	por %xmm14, %xmm2
	movd %edx, %xmm4
	movzbl -14(%r13,%rax,8), %edx
	pinsrb $8, -102(%r13,%rax,8), %xmm4
	psllq $8, %xmm4
	por %xmm9, %xmm4
	movd %edx, %xmm14
	movzbl -94(%r13,%rax,8), %edx
	por %xmm15, %xmm4
	movd %edx, %xmm12
	movzbl -30(%r13,%rax,8), %edx
	pinsrb $8, -86(%r13,%rax,8), %xmm12
	psllq $8, %xmm12
	por %xmm8, %xmm12
	movd %edx, %xmm15
	movzbl -78(%r13,%rax,8), %edx
	por %xmm3, %xmm12
	movd %edx, %xmm3
	movzbl -46(%r13,%rax,8), %edx
	pinsrb $8, -70(%r13,%rax,8), %xmm3
	psllq $8, %xmm3
	por %xmm0, %xmm3
	movd %edx, %xmm9
	movzbl -62(%r13,%rax,8), %edx
	por %xmm13, %xmm3
	movd %edx, %xmm8
	pinsrb $8, -54(%r13,%rax,8), %xmm8
	movzbl -15(%r13,%rax,8), %edx
	psllq $8, %xmm8
	por %xmm7, %xmm8
	movd %edx, %xmm13
	movzbl -31(%r13,%rax,8), %edx
	por %xmm11, %xmm8
	movd %edx, %xmm0
	movzbl -47(%r13,%rax,8), %edx
	pinsrb $8, -38(%r13,%rax,8), %xmm9
	psllq $8, %xmm9
	por %xmm6, %xmm9
	movd %edx, %xmm10
	movzbl -63(%r13,%rax,8), %edx
	por 160(%rsp), %xmm9
	movd %edx, %xmm11
	movzbl -79(%r13,%rax,8), %edx
	pinsrb $8, -22(%r13,%rax,8), %xmm15
	psllq $8, %xmm15
	por %xmm5, %xmm15
	movd %edx, %xmm7
	movzbl -95(%r13,%rax,8), %edx
	por 64(%rsp), %xmm15
	movd %edx, %xmm6
	movzbl -111(%r13,%rax,8), %edx
	pinsrb $8, -6(%r13,%rax,8), %xmm14
	psllq $8, %xmm14
	por 16(%rsp), %xmm14
	movd %edx, %xmm1
	movzbl -127(%r13,%rax,8), %edx
	pinsrb $8, -7(%r13,%rax,8), %xmm13
	pinsrb $8, -23(%r13,%rax,8), %xmm0
	pinsrb $8, -39(%r13,%rax,8), %xmm10
	pinsrb $8, -55(%r13,%rax,8), %xmm11
	pinsrb $8, -71(%r13,%rax,8), %xmm7
	pinsrb $8, -87(%r13,%rax,8), %xmm6
	por 32(%rsp), %xmm14
	pinsrb $8, -103(%r13,%rax,8), %xmm1
	por %xmm4, %xmm1
	por %xmm12, %xmm6
	por %xmm3, %xmm7
	por %xmm8, %xmm11
	por %xmm9, %xmm10
	movdqa %xmm10, 32(%rsp)
	por %xmm15, %xmm0
	movdqa %xmm0, 16(%rsp)
	por %xmm14, %xmm13
	psrlq $7, %xmm4
	psrlq $7, %xmm12
	psrlq $7, %xmm3
	psrlq $7, %xmm8
	psrlq $7, %xmm9
	psrlq $7, %xmm15
	psrlq $7, %xmm14
	pxor %xmm13, %xmm14
	pxor %xmm0, %xmm15
	pxor %xmm10, %xmm9
	pxor %xmm11, %xmm8
	pxor %xmm7, %xmm3
	pxor %xmm6, %xmm12
	pxor %xmm1, %xmm4
	movdqa .LCPI22_10(%rip), %xmm0
	pand %xmm0, %xmm4
	pand %xmm0, %xmm12
	pand %xmm0, %xmm3
	pand %xmm0, %xmm8
	pand %xmm0, %xmm9
	pand %xmm0, %xmm15
	pand %xmm0, %xmm14
	movdqa %xmm14, %xmm5
	psllq $7, %xmm5
	por %xmm14, %xmm5
	movdqa %xmm5, %xmm14
	movdqa %xmm15, %xmm5
	psllq $7, %xmm5
	por %xmm15, %xmm5
	movdqa %xmm9, %xmm10
	psllq $7, %xmm10
	por %xmm9, %xmm10
	movdqa %xmm8, %xmm9
	psllq $7, %xmm9
	por %xmm8, %xmm9
	movdqa %xmm3, %xmm8
	psllq $7, %xmm8
	por %xmm3, %xmm8
	movdqa %xmm12, %xmm15
	psllq $7, %xmm15
	por %xmm12, %xmm15
	movdqa %xmm4, %xmm12
	psllq $7, %xmm12
	por %xmm4, %xmm12
	movd %edx, %xmm3
	pinsrb $8, -119(%r13,%rax,8), %xmm3
	por %xmm2, %xmm3
	psrlq $7, %xmm2
	pxor %xmm3, %xmm2
	pand %xmm0, %xmm2
	movdqa %xmm2, %xmm0
	psllq $7, %xmm0
	por %xmm2, %xmm0
	pxor %xmm3, %xmm0
	pxor %xmm1, %xmm12
	pxor %xmm6, %xmm15
	pxor %xmm7, %xmm8
	pxor %xmm11, %xmm9
	pxor 32(%rsp), %xmm10
	pxor 16(%rsp), %xmm5
	pxor %xmm13, %xmm14
	movdqa %xmm14, 64(%rsp)
	movdqa %xmm5, %xmm1
	psrlq $14, %xmm1
	movdqa %xmm10, %xmm2
	psrlq $14, %xmm2
	movdqa %xmm9, %xmm11
	psrlq $14, %xmm11
	movdqa %xmm8, %xmm7
	psrlq $14, %xmm7
	movdqa %xmm15, %xmm14
	psrlq $14, %xmm14
	movdqa %xmm12, %xmm6
	psrlq $14, %xmm6
	movdqa %xmm0, %xmm4
	psrlq $14, %xmm4
	pxor %xmm0, %xmm4
	pxor %xmm12, %xmm6
	pxor %xmm15, %xmm14
	pxor %xmm8, %xmm7
	pxor %xmm9, %xmm11
	pxor %xmm10, %xmm2
	pxor %xmm5, %xmm1
	movdqa .LCPI22_11(%rip), %xmm3
	pand %xmm3, %xmm1
	pand %xmm3, %xmm2
	pand %xmm3, %xmm11
	pand %xmm3, %xmm7
	pand %xmm3, %xmm14
	pand %xmm3, %xmm6
	pand %xmm3, %xmm4
	movdqa %xmm4, %xmm13
	psllq $14, %xmm13
	por %xmm4, %xmm13
	movdqa %xmm13, 16(%rsp)
	movdqa %xmm6, %xmm4
	psllq $14, %xmm4
	por %xmm6, %xmm4
	movdqa %xmm4, 32(%rsp)
	movdqa %xmm14, %xmm4
	psllq $14, %xmm4
	por %xmm14, %xmm4
	movdqa %xmm7, %xmm14
	psllq $14, %xmm14
	por %xmm7, %xmm14
	movdqa %xmm11, %xmm7
	psllq $14, %xmm7
	por %xmm11, %xmm7
	movdqa %xmm2, %xmm11
	psllq $14, %xmm11
	por %xmm2, %xmm11
	movdqa %xmm1, %xmm6
	psllq $14, %xmm6
	por %xmm1, %xmm6
	movdqa 64(%rsp), %xmm1
	movdqa %xmm1, %xmm13
	psrlq $14, %xmm13
	pxor %xmm1, %xmm13
	pand %xmm3, %xmm13
	movdqa %xmm13, %xmm2
	psllq $14, %xmm2
	por %xmm13, %xmm2
	pxor %xmm1, %xmm2
	pxor %xmm5, %xmm6
	pxor %xmm10, %xmm11
	pxor %xmm9, %xmm7
	pxor %xmm8, %xmm14
	pxor %xmm15, %xmm4
	movdqa 32(%rsp), %xmm1
	pxor %xmm12, %xmm1
	movdqa 16(%rsp), %xmm3
	pxor %xmm0, %xmm3
	movdqa %xmm3, 16(%rsp)
	movdqa %xmm1, %xmm13
	movdqa %xmm1, 32(%rsp)
	psrlq $28, %xmm13
	movdqa %xmm4, %xmm12
	movdqa %xmm4, 160(%rsp)
	psrlq $28, %xmm12
	movdqa %xmm14, %xmm5
	psrlq $28, %xmm5
	movdqa %xmm7, %xmm9
	psrlq $28, %xmm9
	movdqa %xmm11, %xmm0
	psrlq $28, %xmm0
	movdqa %xmm6, %xmm10
	psrlq $28, %xmm10
	movdqa %xmm2, %xmm15
	psrlq $28, %xmm15
	pxor %xmm2, %xmm15
	pxor %xmm6, %xmm10
	pxor %xmm11, %xmm0
	pxor %xmm7, %xmm9
	pxor %xmm14, %xmm5
	pxor %xmm4, %xmm12
	pxor %xmm1, %xmm13
	movdqa .LCPI22_12(%rip), %xmm4
	pand %xmm4, %xmm13
	pand %xmm4, %xmm12
	pand %xmm4, %xmm5
	pand %xmm4, %xmm9
	pand %xmm4, %xmm0
	pand %xmm4, %xmm10
	pand %xmm4, %xmm15
	movdqa %xmm15, %xmm1
	psllq $28, %xmm1
	por %xmm15, %xmm1
	movdqa %xmm10, %xmm3
	psllq $28, %xmm3
	por %xmm10, %xmm3
	movdqa %xmm3, 64(%rsp)
	movdqa %xmm0, %xmm10
	psllq $28, %xmm10
	por %xmm0, %xmm10
	movdqa %xmm9, %xmm8
	psllq $28, %xmm8
	por %xmm9, %xmm8
	movdqa %xmm5, %xmm3
	psllq $28, %xmm3
	por %xmm5, %xmm3
	movdqa %xmm3, %xmm15
	movdqa %xmm12, %xmm5
	psllq $28, %xmm5
	por %xmm12, %xmm5
	movdqa %xmm13, %xmm9
	psllq $28, %xmm9
	por %xmm13, %xmm9
	movdqa 16(%rsp), %xmm12
	movdqa %xmm12, %xmm3
	psrlq $28, %xmm3
	pxor %xmm12, %xmm3
	pand %xmm4, %xmm3
	movdqa %xmm3, %xmm13
	psllq $28, %xmm13
	por %xmm3, %xmm13
	pxor %xmm12, %xmm13
	pxor 32(%rsp), %xmm9
	pxor 160(%rsp), %xmm5
	pxor %xmm14, %xmm15
	pxor %xmm7, %xmm8
	pxor %xmm11, %xmm10
	movdqa 64(%rsp), %xmm0
	pxor %xmm6, %xmm0
	movdqa %xmm1, %xmm3
	pxor %xmm2, %xmm3
	movdqa %xmm3, %xmm1
	movdqa .LCPI22_13(%rip), %xmm6
	pshufb %xmm6, %xmm1
	movdqa %xmm0, %xmm2
	pshufb %xmm6, %xmm2
	packusdw %xmm1, %xmm2
	movdqa %xmm10, %xmm1
	pshufb %xmm6, %xmm1
	movdqa %xmm8, %xmm4
	pshufb %xmm6, %xmm4
	packusdw %xmm1, %xmm4
	packusdw %xmm2, %xmm4
	movdqa %xmm15, %xmm1
	pshufb %xmm6, %xmm1
	movdqa %xmm5, %xmm2
	pshufb %xmm6, %xmm2
	packusdw %xmm1, %xmm2
	movdqa %xmm9, %xmm1
	pshufb %xmm6, %xmm1
	movdqa %xmm13, %xmm7
	pshufb %xmm6, %xmm7
	packusdw %xmm1, %xmm7
	packusdw %xmm2, %xmm7
	packuswb %xmm4, %xmm7
	movdqa %xmm7, 32(%rsp)
	movdqa %xmm3, %xmm1
	movdqa .LCPI22_14(%rip), %xmm7
	pshufb %xmm7, %xmm1
	movdqa %xmm0, %xmm2
	pshufb %xmm7, %xmm2
	packusdw %xmm1, %xmm2
	movdqa %xmm10, %xmm1
	pshufb %xmm7, %xmm1
	movdqa %xmm8, %xmm6
	pshufb %xmm7, %xmm6
	packusdw %xmm1, %xmm6
	packusdw %xmm2, %xmm6
	movdqa %xmm15, %xmm1
	pshufb %xmm7, %xmm1
	movdqa %xmm5, %xmm2
	pshufb %xmm7, %xmm2
	packusdw %xmm1, %xmm2
	movdqa %xmm9, %xmm1
	pshufb %xmm7, %xmm1
	movdqa %xmm13, %xmm4
	pshufb %xmm7, %xmm4
	packusdw %xmm1, %xmm4
	packusdw %xmm2, %xmm4
	packuswb %xmm6, %xmm4
	movdqa %xmm3, %xmm1
	movdqa .LCPI22_15(%rip), %xmm11
	pshufb %xmm11, %xmm1
	movdqa %xmm0, %xmm2
	pshufb %xmm11, %xmm2
	packusdw %xmm1, %xmm2
	movdqa %xmm10, %xmm1
	pshufb %xmm11, %xmm1
	movdqa %xmm8, %xmm6
	pshufb %xmm11, %xmm6
	packusdw %xmm1, %xmm6
	packusdw %xmm2, %xmm6
	movdqa %xmm15, %xmm1
	pshufb %xmm11, %xmm1
	movdqa %xmm5, %xmm2
	pshufb %xmm11, %xmm2
	packusdw %xmm1, %xmm2
	movdqa %xmm9, %xmm7
	pshufb %xmm11, %xmm7
	movdqa %xmm13, %xmm1
	pshufb %xmm11, %xmm1
	packusdw %xmm7, %xmm1
	packusdw %xmm2, %xmm1
	packuswb %xmm6, %xmm1
	movdqa %xmm3, %xmm2
	movdqa .LCPI22_16(%rip), %xmm12
	pshufb %xmm12, %xmm2
	movdqa %xmm0, %xmm6
	pshufb %xmm12, %xmm6
	packusdw %xmm2, %xmm6
	movdqa %xmm10, %xmm2
	pshufb %xmm12, %xmm2
	movdqa %xmm8, %xmm7
	pshufb %xmm12, %xmm7
	packusdw %xmm2, %xmm7
	packusdw %xmm6, %xmm7
	movdqa %xmm15, %xmm2
	movdqa %xmm15, 16(%rsp)
	pshufb %xmm12, %xmm2
	movdqa %xmm5, %xmm11
	pshufb %xmm12, %xmm11
	packusdw %xmm2, %xmm11
	movdqa %xmm9, %xmm2
	pshufb %xmm12, %xmm2
	movdqa %xmm13, %xmm6
	pshufb %xmm12, %xmm6
	packusdw %xmm2, %xmm6
	packusdw %xmm11, %xmm6
	packuswb %xmm7, %xmm6
	movdqa %xmm3, %xmm2
	movdqa .LCPI22_17(%rip), %xmm12
	pshufb %xmm12, %xmm2
	movdqa %xmm0, %xmm7
	movdqa %xmm0, 64(%rsp)
	pshufb %xmm12, %xmm7
	packusdw %xmm2, %xmm7
	movdqa %xmm10, %xmm2
	pshufb %xmm12, %xmm2
	movdqa %xmm8, %xmm11
	pshufb %xmm12, %xmm11
	packusdw %xmm2, %xmm11
	packusdw %xmm7, %xmm11
	movdqa %xmm15, %xmm2
	pshufb %xmm12, %xmm2
	movdqa %xmm5, %xmm7
	pshufb %xmm12, %xmm7
	packusdw %xmm2, %xmm7
	movdqa %xmm9, %xmm14
	pshufb %xmm12, %xmm14
	movdqa %xmm13, %xmm2
	pshufb %xmm12, %xmm2
	packusdw %xmm14, %xmm2
	packusdw %xmm7, %xmm2
	packuswb %xmm11, %xmm2
	movdqa %xmm3, %xmm7
	movdqa .LCPI22_18(%rip), %xmm15
	pshufb %xmm15, %xmm7
	pshufb %xmm15, %xmm0
	packusdw %xmm7, %xmm0
	movdqa %xmm10, %xmm7
	pshufb %xmm15, %xmm7
	movdqa %xmm8, %xmm14
	pshufb %xmm15, %xmm14
	packusdw %xmm7, %xmm14
	packusdw %xmm0, %xmm14
	movdqa 16(%rsp), %xmm0
	movdqa %xmm0, %xmm7
	pshufb %xmm15, %xmm7
	movdqa %xmm5, %xmm11
	pshufb %xmm15, %xmm11
	packusdw %xmm7, %xmm11
	movdqa %xmm9, %xmm12
	pshufb %xmm15, %xmm12
	movdqa %xmm13, %xmm7
	pshufb %xmm15, %xmm7
	packusdw %xmm12, %xmm7
	packusdw %xmm11, %xmm7
	packuswb %xmm14, %xmm7
	movdqa %xmm9, %xmm12
	psrlq $56, %xmm12
	movdqa %xmm13, %xmm11
	psrlq $56, %xmm11
	packusdw %xmm12, %xmm11
	psrlq $56, %xmm0
	movdqa %xmm5, %xmm14
	psrlq $56, %xmm14
	packusdw %xmm0, %xmm14
	packusdw %xmm14, %xmm11
	movdqa %xmm10, %xmm12
	psrlq $56, %xmm12
	movdqa %xmm8, %xmm14
	psrlq $56, %xmm14
	packusdw %xmm12, %xmm14
	movdqa %xmm3, %xmm12
	psrlq $56, %xmm12
	movdqa 64(%rsp), %xmm0
	movdqa %xmm0, %xmm15
	psrlq $56, %xmm15
	packusdw %xmm12, %xmm15
	packusdw %xmm15, %xmm14
	packuswb %xmm14, %xmm11
	movdqa .LCPI22_19(%rip), %xmm12
	pand %xmm12, %xmm3
	pand %xmm12, %xmm0
	packusdw %xmm3, %xmm0
	pand %xmm12, %xmm10
	pand %xmm12, %xmm8
	packusdw %xmm10, %xmm8
	packusdw %xmm0, %xmm8
	movdqa 16(%rsp), %xmm3
	pand %xmm12, %xmm3
	pand %xmm12, %xmm5
	packusdw %xmm3, %xmm5
	pand %xmm12, %xmm9
	pand %xmm12, %xmm13
	packusdw %xmm9, %xmm13
	packusdw %xmm5, %xmm13
	packuswb %xmm8, %xmm13
	movdqu %xmm13, (%rsi,%rax)
	movdqa 32(%rsp), %xmm0
	movdqu %xmm0, (%r10,%rax)
	movdqu %xmm4, (%r15,%rax)
	movdqu %xmm1, (%rbx,%rax)
	movdqu %xmm6, (%r9,%rax)
	movdqu %xmm2, (%r11,%rax)
	movdqu %xmm7, (%rdi,%rax)
	movdqu %xmm11, (%r14,%rax)
	addq $16, %rax
	cmpq %rax, %rcx
	jne .LBB22_5
	movdqa .LCPI22_0(%rip), %xmm7
	movapd 512(%rsp), %xmm8
	movdqa .LCPI22_2(%rip), %xmm9
	movdqa .LCPI22_3(%rip), %xmm10
	movdqa .LCPI22_4(%rip), %xmm11
	movdqa .LCPI22_5(%rip), %xmm12
	movdqa 496(%rsp), %xmm13
	movdqa .LCPI22_6(%rip), %xmm14
	movdqa .LCPI22_7(%rip), %xmm15
	movq 88(%rsp), %rcx
.LBB22_7:
	movq 472(%rsp), %rdi
	incq %rdi
	movq 280(%rsp), %rax
	leaq (%rax,%rcx,8), %r14
.LBB22_8:
	movq 288(%rsp), %rsi
	cmpq %rsi, %r14
	movq 352(%rsp), %rdx
	jae .LBB22_9
	leaq 1(%r14), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 2(%r14), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 3(%r14), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 4(%r14), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 5(%r14), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 6(%r14), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq 7(%r14), %rax
	cmpq %rsi, %rax
	jae .LBB22_10
	leaq (%rdx,%rcx), %rax
	movq 304(%rsp), %r8
	cmpq %r8, %rax
	jae .LBB22_27
	movq 296(%rsp), %rdx
	movzbl -3(%rdx,%rcx,8), %eax
	movzbl -2(%rdx,%rcx,8), %r12d
	movzbl -1(%rdx,%rcx,8), %r9d
	movzbl (%rdx,%rcx,8), %r10d
	movzbl 1(%rdx,%rcx,8), %r11d
	movzbl 2(%rdx,%rcx,8), %ebx
	movzbl 3(%rdx,%rcx,8), %r15d
	movzbl 4(%rdx,%rcx,8), %ebp
	shlq $56, %rbp
	shlq $48, %r15
	orq %rbp, %r15
	shlq $40, %rbx
	orq %r15, %rbx
	shlq $32, %r11
	orq %rbx, %r11
	shll $24, %r10d
	orq %r11, %r10
	movq 320(%rsp), %rdx
	shll $16, %r9d
	orq %r10, %r9
	shll $8, %r12d
	orq %r9, %r12
	orq %r12, %rax
	shrq $7, %r12
	xorq %rax, %r12
	movabsq $47851476196393130, %rsi
	andq %rsi, %r12
	movq %r12, %r9
	shlq $7, %r9
	orq %r12, %r9
	xorq %rax, %r9
	movq %r9, %rax
	shrq $14, %rax
	xorq %r9, %rax
	movabsq $225176545447116, %rsi
	andq %rsi, %rax
	movq %rax, %r10
	shlq $14, %r10
	orq %rax, %r10
	xorq %r9, %r10
	movq %r10, %rax
	shrq $28, %rax
	xorl %r10d, %eax
	andl $-252645136, %eax
	movq %rax, %r9
	shlq $28, %r9
	orq %rax, %r9
	xorq %r10, %r9
	movb %r9b, (%rdx,%rcx)
	movq 152(%rsp), %rax
	addq %rcx, %rax
	cmpq %r8, %rax
	jae .LBB22_27
	movl %r9d, %eax
	shrl $8, %eax
	movq 128(%rsp), %rsi
	movb %al, (%rsi,%rcx)
	movq 144(%rsp), %rax
	addq %rcx, %rax
	cmpq %r8, %rax
	jae .LBB22_27
	movl %r9d, %eax
	shrl $16, %eax
	movq 112(%rsp), %r15
	movb %al, (%r15,%rcx)
	movq 136(%rsp), %rax
	addq %rcx, %rax
	cmpq %r8, %rax
	jae .LBB22_27
	movl %r9d, %eax
	shrl $24, %eax
	movq 104(%rsp), %rbx
	movb %al, (%rbx,%rcx)
	movq 344(%rsp), %rax
	addq %rcx, %rax
	cmpq %r8, %rax
	jae .LBB22_27
	movq %r9, %rax
	shrq $32, %rax
	movq 96(%rsp), %r11
	movb %al, (%r11,%rcx)
	movq 336(%rsp), %rax
	addq %rcx, %rax
	cmpq %r8, %rax
	jae .LBB22_27
	movq %r9, %rax
	shrq $40, %rax
	movq 120(%rsp), %r10
	movb %al, (%r10,%rcx)
	movq 488(%rsp), %rbp
	leaq (%rcx,%rbp), %rax
	cmpq %r8, %rax
	jae .LBB22_27
	movq %r9, %rax
	shrq $48, %rax
	movq 56(%rsp), %rsi
	movb %al, (%rsi,%rcx)
	movq 328(%rsp), %rax
	addq %rcx, %rax
	cmpq %r8, %rax
	jae .LBB22_27
	shrq $56, %r9
	movq 312(%rsp), %rax
	movb %r9b, (%rax,%rcx)
	incq %rcx
	addq $8, %r14
	movq 480(%rsp), %r12
	cmpq %rcx, %r12
	jne .LBB22_8
	addq %r12, 152(%rsp)
	subq %r12, 264(%rsp)
	addq %r12, 144(%rsp)
	subq %r12, 208(%rsp)
	addq %r12, 136(%rsp)
	subq %r12, 216(%rsp)
	movq %rdi, %rcx
	movq 344(%rsp), %r8
	addq %r12, %r8
	subq %r12, 224(%rsp)
	movq 336(%rsp), %rdi
	addq %r12, %rdi
	subq %r12, 232(%rsp)
	addq %r12, %rbp
	subq %r12, 240(%rsp)
	movq 328(%rsp), %r14
	addq %r12, %r14
	subq %r12, 248(%rsp)
	movq 352(%rsp), %r9
	addq %r12, %r9
	subq %r12, 256(%rsp)
	addq %r12, %rax
	addq %r12, %rsi
	movq %rsi, 56(%rsp)
	movq 152(%rsp), %rsi
	addq %r12, %r10
	movq %r10, 120(%rsp)
	movq %r8, %r10
	addq %r12, %r11
	movq %r11, 96(%rsp)
	movq %rdi, %r11
	addq %r12, %rbx
	movq %rbx, 104(%rsp)
	movq 136(%rsp), %rbx
	addq %r12, %r15
	movq %r15, 112(%rsp)
	movq 144(%rsp), %r15
	movq 128(%rsp), %rdi
	addq %r12, %rdi
	addq %r12, %rdx
	movq 360(%rsp), %r8
	addq %r8, %r13
	addq %r8, 280(%rsp)
	addq %r8, 296(%rsp)
	cmpq 200(%rsp), %rcx
	jne .LBB22_3
.LBB22_28:
	addq $536, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	retq
.LBB22_27:
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.9(%rip), %rdx
	movq %rax, %rdi
	movq %r8, %rsi
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
.LBB22_9:
	movq %r14, %rax
.LBB22_10:
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.10(%rip), %rdx
	movq %rax, %rdi
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
