jix_probe::bit_shuffle::untrans_bit_byte:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $328, %rsp
	movq %r8, %r11
	movq %rcx, 16(%rsp)
	movq %r8, 232(%rsp)
	shrq $3, %r11
	movq %r9, 136(%rsp)
	testq %r9, %r9
	sete %al
	testq %r11, %r11
	sete %cl
	orb %al, %cl
	jne .LBB23_19
	movq %rdi, %r8
	movq 16(%rsp), %rax
	incq %rax
	movq %rax, 224(%rsp)
	movq 136(%rsp), %rax
	imulq %r11, %rax
	leaq (%rax,%rax), %rcx
	movq %rcx, %rdi
	negq %rdi
	movq %rdi, 128(%rsp)
	leaq (%rax,%rax,2), %rdi
	movq %rdi, %r9
	negq %r9
	movq %r9, 120(%rsp)
	leaq (,%rax,4), %r9
	negq %r9
	movq %r9, 112(%rsp)
	leaq (%rax,%rax,4), %r9
	movq %r9, %r10
	movq %r9, %rbx
	negq %r10
	movq %r10, 104(%rsp)
	leaq (%rcx,%rcx,2), %rcx
	movq %rcx, %r12
	movq %rcx, %r14
	movq %rcx, 96(%rsp)
	negq %r12
	leaq (,%rax,8), %rcx
	movq %rax, %r10
	subq %rcx, %r10
	subq %rax, %rcx
	movq %rax, %r9
	negq %r9
	movq %r9, 208(%rsp)
	movdqa .LCPI23_0(%rip), %xmm0
	movdqa .LCPI23_1(%rip), %xmm1
	movdqa .LCPI23_2(%rip), %xmm2
	leaq -1(%r11), %r9
	movq %r9, 216(%rsp)
	leaq (%r8,%rcx), %r9
	movq %r9, 24(%rsp)
	leaq (%r8,%r14), %r15
	leaq (%r8,%rbx), %r9
	leaq (%r8,%rax,4), %rbp
	leaq (%r8,%rdi), %rbx
	leaq (%r8,%rax,2), %r13
	leaq (%r8,%rax), %r14
	movl $7, %edi
	movq %rdi, 192(%rsp)
	movq $0, 184(%rsp)
	movq $0, 72(%rsp)
	movq %r10, 256(%rsp)
	movq %r10, 176(%rsp)
	movq %rcx, 264(%rsp)
	movq %rcx, 40(%rsp)
	movl $8, %ecx
	movq %r12, 272(%rsp)
	movq %r12, 168(%rsp)
	movq $-1, 200(%rsp)
	movq 96(%rsp), %rdi
	movq %rdi, 48(%rsp)
	movq 104(%rsp), %rdi
	movq %rdi, 32(%rsp)
	leaq (%rax,%rax,4), %rdi
	movq %rdi, 280(%rsp)
	leaq (%rax,%rax,4), %rdi
	movq %rdi, 56(%rsp)
	movq 112(%rsp), %rdi
	movq %rdi, 160(%rsp)
	leaq (,%rax,4), %rdi
	movq %rdi, 288(%rsp)
	leaq (,%rax,4), %rdi
	movq %rdi, 64(%rsp)
	movq 120(%rsp), %rdi
	movq %rdi, 152(%rsp)
	leaq (%rax,%rax,2), %rdi
	movq %rdi, 296(%rsp)
	leaq (%rax,%rax,2), %r12
	movq 128(%rsp), %rdi
	movq %rdi, 144(%rsp)
	leaq (%rax,%rax), %rdi
	movq %rdi, 304(%rsp)
	leaq (%rax,%rax), %rdi
	movq %rax, 312(%rsp)
	movq %rax, %r10
	movq $0, (%rsp)
	xorl %eax, %eax
.LBB23_3:
	movq %r12, 88(%rsp)
	movq %rax, 8(%rsp)
	movq %r8, 80(%rsp)
	movq 224(%rsp), %r12
	cmpq %r12, %rcx
	movq (%rsp), %r8
	movq %rcx, 320(%rsp)
	cmovaq %rcx, %r12
	addq 200(%rsp), %r12
	shrq $3, %r12
	cmpq $-7, %r8
	movq $-8, %rcx
	cmovaeq %r8, %rcx
	addq 192(%rsp), %rcx
	shrq $3, %rcx
	cmpq %rcx, %r12
	cmovbq %r12, %rcx
	cmpq %r10, %rsi
	movq %r10, 248(%rsp)
	cmovaq %rsi, %r10
	addq 208(%rsp), %r10
	cmpq %r10, %rcx
	cmovbq %rcx, %r10
	cmpq %rdi, %rsi
	movq %rdi, 240(%rsp)
	cmovaq %rsi, %rdi
	addq 144(%rsp), %rdi
	cmpq %rdi, %r10
	cmovbq %r10, %rdi
	movq 88(%rsp), %rax
	cmpq %rax, %rsi
	cmovaq %rsi, %rax
	addq 152(%rsp), %rax
	cmpq %rax, %rdi
	cmovbq %rdi, %rax
	movq 64(%rsp), %rcx
	cmpq %rcx, %rsi
	cmovaq %rsi, %rcx
	addq 160(%rsp), %rcx
	cmpq %rcx, %rax
	cmovbq %rax, %rcx
	movq 56(%rsp), %rax
	cmpq %rax, %rsi
	cmovaq %rsi, %rax
	addq 32(%rsp), %rax
	cmpq %rax, %rcx
	cmovbq %rcx, %rax
	movq 48(%rsp), %rcx
	cmpq %rcx, %rsi
	cmovaq %rsi, %rcx
	addq 168(%rsp), %rcx
	cmpq %rcx, %rax
	cmovbq %rax, %rcx
	movq 40(%rsp), %r10
	cmpq %r10, %rsi
	cmovaq %rsi, %r10
	addq 176(%rsp), %r10
	cmpq %r10, %rcx
	cmovbq %rcx, %r10
	movq 72(%rsp), %rax
	cmpq %rax, %rsi
	cmovaq %rsi, %rax
	addq 184(%rsp), %rax
	cmpq %rax, %r10
	cmovbq %r10, %rax
	movq 216(%rsp), %rcx
	cmpq %rcx, %rax
	cmovaeq %rcx, %rax
	movq 232(%rsp), %rcx
	movq 8(%rsp), %rdi
	imulq %rdi, %rcx
	movq $-8, %r10
	subq %rcx, %r10
	movq %r10, %r8
	movl $0, %r12d
	cmovbq %r12, %r8
	negq %r10
	movq 224(%rsp), %r12
	cmpq %r12, %r10
	cmovbeq %r12, %r10
	notq %rcx
	addq %rcx, %r10
	shrq $3, %r10
	addq $7, %r8
	shrq $3, %r8
	cmpq %r8, %r10
	cmovbq %r10, %r8
	movq %r11, %r12
	movq %r11, %rcx
	imulq %rdi, %rcx
	movq 312(%rsp), %rdi
	leaq (%rdi,%rcx), %r10
	movq %rdx, %rdi
	movq %rsi, %r11
	subq %r10, %r11
	movl $0, %edx
	cmovbq %rdx, %r11
	cmpq %r11, %r8
	cmovbq %r8, %r11
	movq 304(%rsp), %rdx
	leaq (%rdx,%rcx), %r8
	cmpq %r8, %rsi
	cmovaq %rsi, %r8
	movq 128(%rsp), %r10
	subq %rcx, %r10
	addq %r8, %r10
	cmpq %r10, %r11
	cmovbq %r11, %r10
	movq 296(%rsp), %rdx
	leaq (%rdx,%rcx), %r8
	cmpq %r8, %rsi
	cmovaq %rsi, %r8
	movq 120(%rsp), %r11
	subq %rcx, %r11
	addq %r8, %r11
	cmpq %r11, %r10
	cmovbq %r10, %r11
	movq 288(%rsp), %rdx
	leaq (%rdx,%rcx), %r8
	cmpq %r8, %rsi
	cmovaq %rsi, %r8
	movq 112(%rsp), %r10
	subq %rcx, %r10
	addq %r8, %r10
	cmpq %r10, %r11
	cmovbq %r11, %r10
	movq 280(%rsp), %rdx
	leaq (%rdx,%rcx), %r8
	cmpq %r8, %rsi
	cmovaq %rsi, %r8
	movq 104(%rsp), %r11
	subq %rcx, %r11
	addq %r8, %r11
	cmpq %r11, %r10
	cmovbq %r10, %r11
	movq 96(%rsp), %rdx
	leaq (%rdx,%rcx), %r8
	cmpq %r8, %rsi
	cmovaq %rsi, %r8
	movq 272(%rsp), %r10
	subq %rcx, %r10
	addq %r8, %r10
	cmpq %r10, %r11
	cmovbq %r11, %r10
	movq 264(%rsp), %rdx
	leaq (%rdx,%rcx), %r8
	cmpq %r8, %rsi
	cmovaq %rsi, %r8
	movq 256(%rsp), %r11
	subq %rcx, %r11
	addq %r8, %r11
	cmpq %r11, %r10
	cmovbq %r10, %r11
	movq %rsi, %r10
	subq %rcx, %r10
	movl $0, %ecx
	cmovbq %rcx, %r10
	cmpq %r10, %r11
	cmovbq %r11, %r10
	movq 216(%rsp), %rcx
	cmpq %rcx, %r10
	cmovaeq %rcx, %r10
	movl $0, %ecx
	cmpq $2, %r10
	jb .LBB23_7
	movq %rdi, %rdx
	andq $-2, %rax
	andq $-2, %r10
	xorl %ecx, %ecx
	movq 80(%rsp), %r8
	movq 24(%rsp), %r11
.LBB23_5:
	pmovzxbq (%r8,%rcx), %xmm3
	pmovzxbq (%r14,%rcx), %xmm4
	pmovzxbq (%r13,%rcx), %xmm5
	pmovzxbq (%rbx,%rcx), %xmm6
	pmovzxbq (%rbp,%rcx), %xmm7
	pmovzxbq (%r9,%rcx), %xmm8
	pmovzxbq (%r15,%rcx), %xmm9
	pmovzxbq (%r11,%rcx), %xmm10
	psllq $56, %xmm10
	psllq $48, %xmm9
	por %xmm10, %xmm9
	psllq $40, %xmm8
	psllq $32, %xmm7
	por %xmm8, %xmm7
	por %xmm9, %xmm7
	psllq $24, %xmm6
	psllq $16, %xmm5
	por %xmm6, %xmm5
	psllq $8, %xmm4
	por %xmm5, %xmm4
	por %xmm7, %xmm4
	por %xmm4, %xmm3
	psrlq $7, %xmm4
	pxor %xmm3, %xmm4
	pand %xmm0, %xmm4
	movdqa %xmm4, %xmm5
	psllq $7, %xmm5
	por %xmm4, %xmm5
	pxor %xmm3, %xmm5
	movdqa %xmm5, %xmm3
	psrlq $14, %xmm3
	pxor %xmm5, %xmm3
	pand %xmm1, %xmm3
	movdqa %xmm3, %xmm4
	psllq $14, %xmm4
	por %xmm3, %xmm4
	pxor %xmm5, %xmm4
	movdqa %xmm4, %xmm3
	psrlq $28, %xmm3
	pxor %xmm4, %xmm3
	pand %xmm2, %xmm3
	movdqa %xmm3, %xmm5
	psllq $28, %xmm5
	por %xmm3, %xmm5
	pxor %xmm4, %xmm5
	movdqu %xmm5, (%rdx,%rcx,8)
	addq $2, %rcx
	cmpq %rcx, %rax
	jne .LBB23_5
	movq %r10, %rcx
.LBB23_7:
	incq 8(%rsp)
	movq (%rsp), %rax
	leaq (%rax,%rcx,8), %rax
.LBB23_8:
	movq 72(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 248(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 240(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 88(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 64(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 56(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 48(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	movq 40(%rsp), %rdx
	leaq (%rdx,%rcx), %r10
	cmpq %rsi, %r10
	jae .LBB23_21
	cmpq $-9, %rax
	ja .LBB23_20
	leaq 8(%rax), %r10
	cmpq 16(%rsp), %r10
	ja .LBB23_20
	movq 80(%rsp), %rax
	movzbl (%rax,%rcx), %eax
	movzbl (%r15,%rcx), %r8d
	movq 24(%rsp), %rdx
	movzbl (%rdx,%rcx), %r11d
	shlq $56, %r11
	shlq $48, %r8
	orq %r11, %r8
	movzbl (%r9,%rcx), %r11d
	shlq $40, %r11
	orq %r8, %r11
	movzbl (%rbp,%rcx), %r8d
	shlq $32, %r8
	orq %r11, %r8
	movzbl (%rbx,%rcx), %r11d
	shll $24, %r11d
	orq %r8, %r11
	movzbl (%r13,%rcx), %r8d
	shll $16, %r8d
	orq %r11, %r8
	movzbl (%r14,%rcx), %r11d
	shll $8, %r11d
	orq %r8, %r11
	orq %r11, %rax
	shrq $7, %r11
	xorq %rax, %r11
	movabsq $47851476196393130, %rdx
	andq %rdx, %r11
	movq %r11, %r8
	shlq $7, %r8
	orq %r11, %r8
	xorq %rax, %r8
	movq %r8, %rax
	shrq $14, %rax
	xorq %r8, %rax
	movabsq $225176545447116, %rdx
	andq %rdx, %rax
	movq %rax, %r11
	shlq $14, %r11
	orq %rax, %r11
	xorq %r8, %r11
	movq %r11, %rax
	shrq $28, %rax
	xorl %r11d, %eax
	andl $-252645136, %eax
	movq %rax, %r8
	shlq $28, %r8
	orq %rax, %r8
	xorq %r11, %r8
	movq %r8, (%rdi,%rcx,8)
	movq 32(%rsp), %r8
	incq %rcx
	movq %r10, %rax
	cmpq %rcx, %r12
	jne .LBB23_8
	movq %r12, %r11
	movq %rdi, %rdx
	movq 232(%rsp), %rax
	movq 320(%rsp), %rcx
	addq %rax, %rcx
	subq %rax, 200(%rsp)
	addq %rax, (%rsp)
	subq %rax, 192(%rsp)
	movq 248(%rsp), %r10
	addq %r12, %r10
	subq %r12, 208(%rsp)
	movq 240(%rsp), %rdi
	addq %r12, %rdi
	subq %r12, 144(%rsp)
	movq 88(%rsp), %r12
	addq %r11, %r12
	subq %r11, 152(%rsp)
	addq %r11, 64(%rsp)
	subq %r11, 160(%rsp)
	addq %r11, 56(%rsp)
	subq %r11, %r8
	movq %r8, 32(%rsp)
	addq %r11, 48(%rsp)
	subq %r11, 168(%rsp)
	addq %r11, 40(%rsp)
	subq %r11, 176(%rsp)
	addq %r11, 72(%rsp)
	subq %r11, 184(%rsp)
	addq %r11, 24(%rsp)
	addq %rax, %rdx
	addq %r11, %r15
	addq %r11, %r9
	addq %r11, %rbp
	addq %r11, %rbx
	addq %r11, %r13
	addq %r11, %r14
	movq 80(%rsp), %r8
	addq %r11, %r8
	movq 8(%rsp), %rax
	cmpq 136(%rsp), %rax
	jne .LBB23_3
.LBB23_19:
	addq $328, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	retq
.LBB23_20:
	leaq 8(%rax), %rsi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.11(%rip), %rcx
	movq %rax, %rdi
	movq 16(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB23_21:
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.12(%rip), %rdx
	movq %r10, %rdi
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
