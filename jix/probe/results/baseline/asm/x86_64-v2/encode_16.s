jix_probe::byte_shuffle::encode_impl::<16, 8>:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	movq %rsi, -72(%rsp)
	movq %rsi, %rax
	shrq $4, %rax
	movabsq $576460752303423480, %rcx
	andq %rax, %rcx
	movq %rdi, -80(%rsp)
	je .LBB4_1
	leaq (%rax,%rax,2), %r8
	leaq (%rax,%rax,4), %r10
	leaq (%rax,%rax,8), %r9
	leaq (%rax,%r10,2), %r11
	leaq (%rax,%r8,4), %r14
	movq %rax, %r15
	shlq $4, %r15
	subq %rax, %r15
	subq %rax, %r15
	leaq (%r10,%r10,2), %r12
	addq %rdx, %r12
	addq %rdx, %r15
	addq %rdx, %r14
	leaq (%rdx,%r8,4), %rsi
	movq %rsi, -40(%rsp)
	addq %rdx, %r11
	movq %r11, -32(%rsp)
	leaq (%rdx,%r10,2), %rsi
	movq %rsi, -48(%rsp)
	addq %rdx, %r9
	leaq (%rdx,%rax,8), %rbx
	movq %rbx, %r13
	subq %rax, %r13
	leaq (%rdx,%r8,2), %rsi
	movq %rsi, -56(%rsp)
	addq %rdx, %r10
	leaq (%rdx,%rax,4), %rsi
	movq %rsi, -64(%rsp)
	addq %rdx, %r8
	leaq (%rdx,%rax,2), %rsi
	addq %rdx, %rax
	leaq 126(%rdi), %r11
	movq %r9, %rdi
	xorl %r9d, %r9d
.LBB4_4:
	pmovzxbq -2(%r11), %xmm1
	pmovzxbq -4(%r11), %xmm3
	pmovzxbq -6(%r11), %xmm5
	pmovzxbq -8(%r11), %xmm7
	pmovzxbq -10(%r11), %xmm9
	pmovzxbq -12(%r11), %xmm10
	pmovzxbq -14(%r11), %xmm11
	pmovzxbq -16(%r11), %xmm0
	movdqa %xmm0, -24(%rsp)
	pmovzxbq -18(%r11), %xmm2
	pmovzxbq -20(%r11), %xmm4
	pmovzxbq -22(%r11), %xmm6
	pmovzxbq -24(%r11), %xmm8
	pmovzxbq -30(%r11), %xmm12
	pmovzxbq -46(%r11), %xmm14
	pmovzxbq -62(%r11), %xmm13
	pmovzxbq -78(%r11), %xmm15
	psllq $40, %xmm14
	psllq $32, %xmm13
	por %xmm14, %xmm13
	pmovzxbq -94(%r11), %xmm14
	psllq $48, %xmm12
	por %xmm12, %xmm13
	pmovzxbq -110(%r11), %xmm0
	psllq $24, %xmm15
	psllq $16, %xmm14
	por %xmm15, %xmm14
	pmovzxbq -126(%r11), %xmm12
	psllq $8, %xmm0
	por %xmm12, %xmm0
	pmovzxbq -26(%r11), %xmm12
	por %xmm14, %xmm0
	pmovzxbq -28(%r11), %xmm14
	pmovzxbq -44(%r11), %xmm15
	por %xmm13, %xmm0
	psllq $56, %xmm11
	por %xmm11, %xmm0
	movq %xmm0, (%rdx,%r9)
	pextrq $1, %xmm0, (%rax,%r9)
	pmovzxbq -60(%r11), %xmm0
	pmovzxbq -76(%r11), %xmm11
	psllq $40, %xmm15
	psllq $32, %xmm0
	por %xmm15, %xmm0
	pmovzxbq -92(%r11), %xmm13
	psllq $48, %xmm14
	por %xmm14, %xmm0
	pmovzxbq -108(%r11), %xmm14
	psllq $24, %xmm11
	psllq $16, %xmm13
	por %xmm11, %xmm13
	pmovzxbq -124(%r11), %xmm11
	psllq $8, %xmm14
	por %xmm11, %xmm14
	por %xmm13, %xmm14
	pmovzxbq -42(%r11), %xmm11
	por %xmm0, %xmm14
	psllq $56, %xmm10
	por %xmm10, %xmm14
	movq %xmm14, (%rsi,%r9)
	pextrq $1, %xmm14, (%r8,%r9)
	pmovzxbq -58(%r11), %xmm0
	pmovzxbq -74(%r11), %xmm10
	psllq $40, %xmm11
	psllq $32, %xmm0
	por %xmm11, %xmm0
	pmovzxbq -90(%r11), %xmm11
	psllq $48, %xmm12
	por %xmm12, %xmm0
	pmovzxbq -106(%r11), %xmm12
	psllq $24, %xmm10
	psllq $16, %xmm11
	por %xmm10, %xmm11
	pmovzxbq -122(%r11), %xmm10
	psllq $8, %xmm12
	por %xmm10, %xmm12
	por %xmm11, %xmm12
	pmovzxbq -40(%r11), %xmm10
	por %xmm0, %xmm12
	psllq $56, %xmm9
	por %xmm9, %xmm12
	movq -64(%rsp), %rbp
	movq %xmm12, (%rbp,%r9)
	pextrq $1, %xmm12, (%r10,%r9)
	pmovzxbq -56(%r11), %xmm0
	pmovzxbq -72(%r11), %xmm9
	psllq $40, %xmm10
	psllq $32, %xmm0
	por %xmm10, %xmm0
	pmovzxbq -88(%r11), %xmm10
	psllq $48, %xmm8
	por %xmm8, %xmm0
	pmovzxbq -104(%r11), %xmm8
	psllq $24, %xmm9
	psllq $16, %xmm10
	por %xmm9, %xmm10
	pmovzxbq -120(%r11), %xmm9
	psllq $8, %xmm8
	por %xmm9, %xmm8
	por %xmm10, %xmm8
	pmovzxbq -38(%r11), %xmm9
	por %xmm0, %xmm8
	psllq $56, %xmm7
	por %xmm7, %xmm8
	movq -56(%rsp), %rbp
	movq %xmm8, (%rbp,%r9)
	pextrq $1, %xmm8, (%r13,%r9)
	pmovzxbq -54(%r11), %xmm0
	pmovzxbq -70(%r11), %xmm7
	psllq $40, %xmm9
	psllq $32, %xmm0
	por %xmm9, %xmm0
	pmovzxbq -86(%r11), %xmm8
	psllq $48, %xmm6
	por %xmm6, %xmm0
	pmovzxbq -102(%r11), %xmm6
	psllq $24, %xmm7
	psllq $16, %xmm8
	por %xmm7, %xmm8
	pmovzxbq -118(%r11), %xmm7
	psllq $8, %xmm6
	por %xmm7, %xmm6
	por %xmm8, %xmm6
	pmovzxbq -36(%r11), %xmm7
	por %xmm0, %xmm6
	psllq $56, %xmm5
	por %xmm5, %xmm6
	movq %xmm6, (%rbx,%r9)
	pextrq $1, %xmm6, (%rdi,%r9)
	pmovzxbq -52(%r11), %xmm0
	pmovzxbq -68(%r11), %xmm5
	psllq $40, %xmm7
	psllq $32, %xmm0
	por %xmm7, %xmm0
	pmovzxbq -84(%r11), %xmm6
	psllq $48, %xmm4
	por %xmm4, %xmm0
	pmovzxbq -100(%r11), %xmm4
	psllq $24, %xmm5
	psllq $16, %xmm6
	por %xmm5, %xmm6
	pmovzxbq -116(%r11), %xmm5
	psllq $8, %xmm4
	por %xmm5, %xmm4
	por %xmm6, %xmm4
	pmovzxbq -34(%r11), %xmm5
	por %xmm0, %xmm4
	psllq $56, %xmm3
	por %xmm3, %xmm4
	movq -48(%rsp), %rbp
	movq %xmm4, (%rbp,%r9)
	movq -32(%rsp), %rbp
	pextrq $1, %xmm4, (%rbp,%r9)
	pmovzxbq -50(%r11), %xmm0
	pmovzxbq -66(%r11), %xmm3
	psllq $40, %xmm5
	psllq $32, %xmm0
	por %xmm5, %xmm0
	pmovzxbq -82(%r11), %xmm4
	psllq $48, %xmm2
	por %xmm2, %xmm0
	pmovzxbq -98(%r11), %xmm2
	psllq $24, %xmm3
	psllq $16, %xmm4
	por %xmm3, %xmm4
	pmovzxbq -114(%r11), %xmm3
	psllq $8, %xmm2
	por %xmm3, %xmm2
	por %xmm4, %xmm2
	pmovzxbq -32(%r11), %xmm3
	por %xmm0, %xmm2
	psllq $56, %xmm1
	por %xmm1, %xmm2
	movq -40(%rsp), %rbp
	movq %xmm2, (%rbp,%r9)
	pextrq $1, %xmm2, (%r14,%r9)
	pmovzxbq -48(%r11), %xmm0
	pmovzxbq -64(%r11), %xmm1
	psllq $40, %xmm3
	psllq $32, %xmm0
	por %xmm3, %xmm0
	pmovzxbq -80(%r11), %xmm2
	movdqa -24(%rsp), %xmm3
	psllq $48, %xmm3
	por %xmm3, %xmm0
	pmovzxbq -96(%r11), %xmm3
	psllq $24, %xmm1
	psllq $16, %xmm2
	por %xmm1, %xmm2
	pmovzxbq -112(%r11), %xmm1
	psllq $8, %xmm3
	por %xmm1, %xmm3
	por %xmm2, %xmm3
	pmovzxbq (%r11), %xmm1
	por %xmm0, %xmm3
	psllq $56, %xmm1
	por %xmm1, %xmm3
	movq %xmm3, (%r15,%r9)
	pextrq $1, %xmm3, (%r12,%r9)
	addq $8, %r9
	subq $-128, %r11
	cmpq %rcx, %r9
	jb .LBB4_4
	jmp .LBB4_2
.LBB4_1:
	xorl %r9d, %r9d
.LBB4_2:
	movl $16, %r8d
	movq -80(%rsp), %rdi
	movq -72(%rsp), %rsi
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
