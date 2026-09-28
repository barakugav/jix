jix_probe::byte_shuffle::decode_impl::<16, 8>:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	movq %rsi, %rax
	shrq $4, %rax
	movabsq $576460752303423480, %r11
	xorl %r9d, %r9d
	andq %rax, %r11
	je .LBB0_5
	movq %rdi, %r8
.LBB0_2:
	movq %r8, %r10
	xorl %ebx, %ebx
.LBB0_3:
	movq (%r10), %rcx
	movl %ecx, %ebp
	movq %rcx, %r14
	movq %rcx, %r15
	movq %rcx, %r12
	movq %rcx, %r13
	movb %cl, -128(%rsp,%rbx)
	movb %ch, -112(%rsp,%rbx)
	shrl $16, %ecx
	shrl $24, %ebp
	shrq $32, %r14
	shrq $40, %r15
	shrq $48, %r12
	shrq $56, %r13
	movb %cl, -96(%rsp,%rbx)
	movb %bpl, -80(%rsp,%rbx)
	movb %r14b, -64(%rsp,%rbx)
	movb %r15b, -48(%rsp,%rbx)
	movb %r12b, -32(%rsp,%rbx)
	movb %r13b, -16(%rsp,%rbx)
	incq %rbx
	addq %rax, %r10
	cmpq $16, %rbx
	jne .LBB0_3
	movq %r9, %rcx
	shlq $4, %rcx
	movups -16(%rsp), %xmm0
	movups %xmm0, 112(%rdx,%rcx)
	movups -32(%rsp), %xmm0
	movups %xmm0, 96(%rdx,%rcx)
	movups -48(%rsp), %xmm0
	movups %xmm0, 80(%rdx,%rcx)
	movups -64(%rsp), %xmm0
	movups %xmm0, 64(%rdx,%rcx)
	movups -128(%rsp), %xmm0
	movups -112(%rsp), %xmm1
	movups -96(%rsp), %xmm2
	movups -80(%rsp), %xmm3
	movups %xmm3, 48(%rdx,%rcx)
	movups %xmm2, 32(%rdx,%rcx)
	movups %xmm1, 16(%rdx,%rcx)
	movups %xmm0, (%rdx,%rcx)
	addq $8, %r9
	addq $8, %r8
	cmpq %r11, %r9
	jb .LBB0_2
.LBB0_5:
	movl $16, %r8d
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
