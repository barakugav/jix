jix_probe::byte_shuffle::decode_impl_generic:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r12
	pushq %rbx
	testq %r8, %r8
	je .LBB8_6
	movq %rdx, %rcx
	movq %rsi, %rax
	orq %r8, %rax
	shrq $32, %rax
	je .LBB8_2
	movq %rsi, %rax
	xorl %edx, %edx
	divq %r8
	cmpq %rax, %r9
	jb .LBB8_5
	jmp .LBB8_13
.LBB8_2:
	movl %esi, %eax
	xorl %edx, %edx
	divl %r8d
	cmpq %rax, %r9
	jae .LBB8_13
.LBB8_5:
	movl %r8d, %edx
	andl $3, %edx
	movq %r8, %rsi
	andq $-4, %rsi
	movq %r9, %r10
	imulq %r8, %r10
	leaq (%rcx,%r10), %r11
	addq $3, %r11
	addq %rcx, %r10
	jmp .LBB8_11
.LBB8_10:
	incq %r9
	addq %r8, %r11
	addq %r8, %r10
	cmpq %rax, %r9
	jae .LBB8_13
.LBB8_11:
	leaq (%rdi,%r9), %rcx
	xorl %ebx, %ebx
	cmpq $4, %r8
	jb .LBB8_8
.LBB8_12:
	leaq 1(%rbx), %r14
	movq %rbx, %r15
	imulq %rax, %r15
	movzbl (%rcx,%r15), %ebp
	movb %bpl, -3(%r11,%rbx)
	leaq 2(%rbx), %r15
	imulq %rax, %r14
	movzbl (%rcx,%r14), %ebp
	movb %bpl, -2(%r11,%rbx)
	leaq 3(%rbx), %r14
	imulq %rax, %r15
	movzbl (%rcx,%r15), %ebp
	movb %bpl, -1(%r11,%rbx)
	imulq %rax, %r14
	movzbl (%rcx,%r14), %ebp
	movb %bpl, (%r11,%rbx)
	addq $4, %rbx
	cmpq %rbx, %rsi
	jne .LBB8_12
	testq %rdx, %rdx
	je .LBB8_10
.LBB8_8:
	leaq (%r10,%rbx), %r14
	xorl %r15d, %r15d
.LBB8_9:
	leaq 1(%rbx), %r12
	imulq %rax, %rbx
	movzbl (%rcx,%rbx), %ebx
	movb %bl, (%r14,%r15)
	incq %r15
	movq %r12, %rbx
	cmpq %r15, %rdx
	jne .LBB8_9
	jmp .LBB8_10
.LBB8_13:
	popq %rbx
	popq %r12
	popq %r14
	popq %r15
	popq %rbp
	retq
.LBB8_6:
	leaq .Lanon.bc1a8278342cb0a09c647aa48cb9aec1.1(%rip), %rdi
	callq *core::panicking::panic_const::panic_const_div_by_zero@GOTPCREL(%rip)
