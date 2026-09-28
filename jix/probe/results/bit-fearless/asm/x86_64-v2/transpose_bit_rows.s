probe_bit_shuffle_transpose_bit_rows:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $40, %rsp
	movabsq $9223372036854775800, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB29_5
	movq %rsi, %rax
	shrq $7, %rax
	movq %rax, (%rsp)
	je .LBB29_4
	shrq $3, %rsi
	leaq (%rsi,%rsi), %rax
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rax,%rax,2), %r9
	leaq (,%rsi,8), %r10
	subq %rsi, %r10
	movq (%rsp), %r11
	shlq $4, %r11
	leaq (%rdx,%r10), %rax
	movq %rax, 32(%rsp)
	leaq (%rdx,%r9), %rax
	movq %rax, 24(%rsp)
	leaq (%rdx,%r8), %rax
	movq %rax, 16(%rsp)
	leaq (%rdx,%rsi,4), %rax
	movq %rax, 8(%rsp)
	leaq (%rdx,%rcx), %r13
	leaq (%rdx,%rsi,2), %rbp
	leaq (%rdx,%rsi), %rbx
	leaq (%rdi,%rsi,4), %rax
	addq %rdi, %r10
	addq %rdi, %r9
	addq %rdi, %r8
	addq %rdi, %rcx
	leaq (%rdi,%rsi,2), %r14
	addq %rdi, %rsi
	xorl %r15d, %r15d
	movdqa .LCPI29_0(%rip), %xmm0
	movdqa .LCPI29_1(%rip), %xmm1
	movdqa .LCPI29_2(%rip), %xmm2
.LBB29_3:
	movdqu (%rsi,%r15), %xmm7
	movdqu (%r14,%r15), %xmm5
	movdqu (%rcx,%r15), %xmm11
	movdqu (%r8,%r15), %xmm6
	movdqu (%r9,%r15), %xmm4
	movdqu (%r10,%r15), %xmm3
	movdqu (%rax,%r15), %xmm8
	movdqu (%rdi,%r15), %xmm10
	movdqa %xmm10, %xmm9
	psrlq $4, %xmm9
	pxor %xmm8, %xmm9
	pand %xmm0, %xmm9
	pxor %xmm9, %xmm8
	psllq $4, %xmm9
	pxor %xmm10, %xmm9
	movdqa %xmm7, %xmm10
	psrlq $4, %xmm10
	pxor %xmm6, %xmm10
	pand %xmm0, %xmm10
	pxor %xmm10, %xmm6
	psllq $4, %xmm10
	pxor %xmm7, %xmm10
	movdqa %xmm5, %xmm7
	psrlq $4, %xmm7
	pxor %xmm4, %xmm7
	pand %xmm0, %xmm7
	pxor %xmm7, %xmm4
	psllq $4, %xmm7
	pxor %xmm5, %xmm7
	movdqa %xmm11, %xmm5
	psrlq $4, %xmm5
	pxor %xmm3, %xmm5
	pand %xmm0, %xmm5
	pxor %xmm5, %xmm3
	psllq $4, %xmm5
	pxor %xmm11, %xmm5
	movdqa %xmm9, %xmm11
	psrlq $2, %xmm11
	pxor %xmm7, %xmm11
	pand %xmm1, %xmm11
	pxor %xmm11, %xmm7
	psllq $2, %xmm11
	pxor %xmm9, %xmm11
	movdqa %xmm10, %xmm9
	psrlq $2, %xmm9
	pxor %xmm5, %xmm9
	pand %xmm1, %xmm9
	pxor %xmm9, %xmm5
	psllq $2, %xmm9
	pxor %xmm10, %xmm9
	movdqa %xmm8, %xmm10
	psrlq $2, %xmm10
	pxor %xmm4, %xmm10
	pand %xmm1, %xmm10
	pxor %xmm10, %xmm4
	psllq $2, %xmm10
	pxor %xmm8, %xmm10
	movdqa %xmm6, %xmm8
	psrlq $2, %xmm8
	pxor %xmm3, %xmm8
	pand %xmm1, %xmm8
	pxor %xmm8, %xmm3
	psllq $2, %xmm8
	pxor %xmm6, %xmm8
	movdqa %xmm11, %xmm6
	psrlq $1, %xmm6
	pxor %xmm9, %xmm6
	pand %xmm2, %xmm6
	pxor %xmm6, %xmm9
	paddq %xmm6, %xmm6
	pxor %xmm11, %xmm6
	movdqa %xmm7, %xmm11
	psrlq $1, %xmm11
	pxor %xmm5, %xmm11
	pand %xmm2, %xmm11
	pxor %xmm11, %xmm5
	paddq %xmm11, %xmm11
	pxor %xmm7, %xmm11
	movdqa %xmm10, %xmm7
	psrlq $1, %xmm7
	pxor %xmm8, %xmm7
	pand %xmm2, %xmm7
	pxor %xmm7, %xmm8
	paddq %xmm7, %xmm7
	pxor %xmm10, %xmm7
	movdqa %xmm4, %xmm10
	psrlq $1, %xmm10
	pxor %xmm3, %xmm10
	pand %xmm2, %xmm10
	pxor %xmm10, %xmm3
	paddq %xmm10, %xmm10
	pxor %xmm4, %xmm10
	movdqu %xmm6, (%rdx,%r15)
	movdqu %xmm9, (%rbx,%r15)
	movdqu %xmm11, (%rbp,%r15)
	movdqu %xmm5, (%r13,%r15)
	movq 8(%rsp), %r12
	movdqu %xmm7, (%r12,%r15)
	movq 16(%rsp), %r12
	movdqu %xmm8, (%r12,%r15)
	movq 24(%rsp), %r12
	movdqu %xmm10, (%r12,%r15)
	movq 32(%rsp), %r12
	movdqu %xmm3, (%r12,%r15)
	addq $16, %r15
	cmpq %r15, %r11
	jne .LBB29_3
.LBB29_4:
	movq (%rsp), %rax
	shlq $4, %rax
	addq $40, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	retq
.LBB29_5:
	leaq .Lanon.6931a3aa7c1713de1da1f697d6f7f66a.4(%rip), %rdi
	leaq .Lanon.6931a3aa7c1713de1da1f697d6f7f66a.5(%rip), %rdx
	movl $36, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
