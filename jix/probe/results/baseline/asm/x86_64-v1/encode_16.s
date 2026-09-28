jix_probe::byte_shuffle::encode_impl::<16, 8>:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $24, %rsp
	movq %rsi, -128(%rsp)
	movq %rsi, %rax
	shrq $4, %rax
	movabsq $576460752303423480, %rsi
	andq %rax, %rsi
	movq %rdx, -120(%rsp)
	je .LBB4_1
	xorl %r9d, %r9d
	movq %rdx, %r8
.LBB4_4:
	movq %r9, %r10
	shlq $4, %r10
	movups 112(%rdi,%r10), %xmm0
	movaps %xmm0, (%rsp)
	movups 96(%rdi,%r10), %xmm0
	movaps %xmm0, -16(%rsp)
	movups 80(%rdi,%r10), %xmm0
	movaps %xmm0, -32(%rsp)
	movups 64(%rdi,%r10), %xmm0
	movaps %xmm0, -48(%rsp)
	movups (%rdi,%r10), %xmm0
	movups 16(%rdi,%r10), %xmm1
	movups 32(%rdi,%r10), %xmm2
	movups 48(%rdi,%r10), %xmm3
	movaps %xmm3, -64(%rsp)
	movaps %xmm2, -80(%rsp)
	movaps %xmm1, -96(%rsp)
	movaps %xmm0, -112(%rsp)
	movq $-16, %r10
	movq %r8, %r11
.LBB4_5:
	movzbl -96(%rsp,%r10), %ebp
	movzbl -80(%rsp,%r10), %ebx
	movzbl -64(%rsp,%r10), %ecx
	movzbl -32(%rsp,%r10), %r13d
	movzbl -16(%rsp,%r10), %r12d
	movzbl (%rsp,%r10), %r15d
	movzbl 16(%rsp,%r10), %r14d
	shlq $56, %r14
	shlq $48, %r15
	shlq $40, %r12
	shlq $32, %r13
	movzbl -48(%rsp,%r10), %edx
	shll $24, %edx
	shll $16, %ecx
	shll $8, %ebx
	orq %rbp, %rbx
	orq %rcx, %rbx
	orq %rdx, %rbx
	orq %r13, %rbx
	orq %r12, %rbx
	orq %r15, %rbx
	orq %r14, %rbx
	movq %rbx, (%r11)
	addq %rax, %r11
	incq %r10
	jne .LBB4_5
	addq $8, %r9
	addq $8, %r8
	cmpq %rsi, %r9
	jb .LBB4_4
	jmp .LBB4_2
.LBB4_1:
	xorl %r9d, %r9d
.LBB4_2:
	movl $16, %r8d
	movq -128(%rsp), %rsi
	movq -120(%rsp), %rdx
	addq $24, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
