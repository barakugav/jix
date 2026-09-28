jix_probe::byte_shuffle::encode_impl_generic:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	pushq %rax
	movq %rdx, (%rsp)
	testq %r8, %r8
	je .LBB9_6
	movq %r9, %r11
	movq %rsi, %r14
	movq %rsi, %rax
	orq %r8, %rax
	shrq $32, %rax
	je .LBB9_2
	movq %r14, %rax
	xorl %edx, %edx
	divq %r8
	cmpq %rax, %r11
	jb .LBB9_5
	jmp .LBB9_19
.LBB9_2:
	movl %r14d, %eax
	xorl %edx, %edx
	divl %r8d
	cmpq %rax, %r11
	jae .LBB9_19
.LBB9_5:
	subq %r11, %r14
	cmpq $16, %r14
	setae %sil
	cmpq $1, %r8
	sete %dl
	andb %sil, %dl
	movq %r14, %rsi
	andq $-128, %rsi
	movq %r14, %rbx
	andq $-16, %rbx
	leaq (%rdi,%r11), %r15
	addq $96, %r15
	leaq (%rdi,%r11), %r12
	xorl %r13d, %r13d
	jmp .LBB9_8
.LBB9_7:
	incq %r13
	incq %r15
	incq %r12
	incq %rdi
	cmpq %r8, %r13
	je .LBB9_19
.LBB9_8:
	movq %r13, %rbp
	imulq %rax, %rbp
	addq (%rsp), %rbp
	movq %r11, %r9
	testb %dl, %dl
	je .LBB9_17
	cmpq $128, %r14
	jae .LBB9_11
	xorl %r10d, %r10d
	jmp .LBB9_15
.LBB9_11:
	xorl %r9d, %r9d
.LBB9_12:
	leaq (%r11,%r9), %r10
	vmovups -96(%r15,%r9), %ymm0
	vmovups -64(%r15,%r9), %ymm1
	vmovups -32(%r15,%r9), %ymm2
	vmovups (%r15,%r9), %ymm3
	vmovups %ymm0, (%rbp,%r10)
	vmovups %ymm1, 32(%rbp,%r10)
	vmovups %ymm2, 64(%rbp,%r10)
	vmovups %ymm3, 96(%rbp,%r10)
	subq $-128, %r9
	cmpq %r9, %rsi
	jne .LBB9_12
	cmpq %rsi, %r14
	je .LBB9_7
	movq %rsi, %r10
	leaq (%r11,%rsi), %r9
	testb $112, %r14b
	je .LBB9_17
.LBB9_15:
	leaq (%r11,%r10), %r9
	vmovups (%r12,%r10), %xmm0
	vmovups %xmm0, (%rbp,%r9)
	addq $16, %r10
	cmpq %r10, %rbx
	jne .LBB9_15
	leaq (%r11,%rbx), %r9
	cmpq %rbx, %r14
	je .LBB9_7
.LBB9_17:
	movq %r8, %r10
	imulq %r9, %r10
	addq %rdi, %r10
.LBB9_18:
	movzbl (%r10), %ecx
	movb %cl, (%rbp,%r9)
	incq %r9
	addq %r8, %r10
	cmpq %rax, %r9
	jb .LBB9_18
	jmp .LBB9_7
.LBB9_19:
	addq $8, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB9_6:
	leaq .Lanon.bc1a8278342cb0a09c647aa48cb9aec1.2(%rip), %rdi
	callq *core::panicking::panic_const::panic_const_div_by_zero@GOTPCREL(%rip)
