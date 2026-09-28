probe_byte_shuffle_decode_16:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $56, %rsp
	movabsq $9223372036854775792, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB10_5
	movq %rsi, %rax
	shrq $10, %rax
	movq %rax, 8(%rsp)
	je .LBB10_4
	shrq $4, %rsi
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rsi,%rsi,8), %r12
	leaq (%rsi,%r8,2), %r9
	leaq (%rsi,%rcx,4), %r10
	movq %rsi, %rbx
	shlq $4, %rbx
	subq %rsi, %rbx
	subq %rsi, %rbx
	leaq (%r8,%r8,2), %rax
	addq $960, %rdx
	addq %rdi, %rax
	movq %rax, 32(%rsp)
	addq %rdi, %rbx
	addq %rdi, %r10
	movq %r10, 48(%rsp)
	leaq (%rdi,%rcx,4), %rax
	movq %rax, 24(%rsp)
	addq %rdi, %r9
	movq %r9, 40(%rsp)
	leaq (%rdi,%r8,2), %rax
	movq %rax, 16(%rsp)
	addq %rdi, %r12
	leaq (%rdi,%rsi,8), %r13
	movq %r13, %rbp
	subq %rsi, %rbp
	leaq (%rdi,%rcx,2), %r14
	addq %rdi, %r8
	leaq (%rdi,%rsi,4), %rax
	addq %rdi, %rcx
	leaq (%rdi,%rsi,2), %r10
	addq %rdi, %rsi
	movq 8(%rsp), %r11
	shlq $6, %r11
	vmovdqa64 .LCPI10_0(%rip), %zmm0
	vmovdqa64 .LCPI10_1(%rip), %zmm1
	xorl %r15d, %r15d
.LBB10_3:
	vmovdqu64 (%rdi,%r15), %zmm2
	vmovdqu64 (%rsi,%r15), %zmm3
	vmovdqu64 (%r10,%r15), %zmm9
	vmovdqu64 (%rcx,%r15), %zmm5
	vmovdqu64 (%rax,%r15), %zmm11
	vmovdqu64 (%r8,%r15), %zmm14
	vmovdqu64 (%r14,%r15), %zmm13
	vmovdqu64 (%rbp,%r15), %zmm12
	vmovdqu64 (%r13,%r15), %zmm6
	vmovdqu64 (%r12,%r15), %zmm7
	movq 16(%rsp), %r9
	vmovdqu64 (%r9,%r15), %zmm8
	movq 40(%rsp), %r9
	vmovdqu64 (%r9,%r15), %zmm15
	movq 24(%rsp), %r9
	vmovdqu64 (%r9,%r15), %zmm16
	movq 48(%rsp), %r9
	vmovdqu64 (%r9,%r15), %zmm17
	vmovdqu64 (%rbx,%r15), %zmm18
	movq 32(%rsp), %r9
	vmovdqu64 (%r9,%r15), %zmm19
	vmovdqa64 %zmm2, %zmm4
	vpermt2b %zmm6, %zmm0, %zmm4
	vpermt2b %zmm6, %zmm1, %zmm2
	vmovdqa64 %zmm3, %zmm6
	vpermt2b %zmm7, %zmm0, %zmm6
	vpermt2b %zmm7, %zmm1, %zmm3
	vmovdqa64 %zmm9, %zmm20
	vpermt2b %zmm8, %zmm0, %zmm20
	vpermt2b %zmm8, %zmm1, %zmm9
	vmovdqa64 %zmm5, %zmm10
	vpermt2b %zmm15, %zmm0, %zmm10
	vpermt2b %zmm15, %zmm1, %zmm5
	vmovdqa64 %zmm11, %zmm7
	vpermt2b %zmm16, %zmm0, %zmm7
	vpermt2b %zmm16, %zmm1, %zmm11
	vmovdqa64 %zmm14, %zmm15
	vpermt2b %zmm17, %zmm0, %zmm15
	vpermt2b %zmm17, %zmm1, %zmm14
	vmovdqa64 %zmm13, %zmm16
	vpermt2b %zmm18, %zmm0, %zmm16
	vpermt2b %zmm18, %zmm1, %zmm13
	vmovdqa64 %zmm12, %zmm17
	vpermt2b %zmm19, %zmm0, %zmm17
	vpermt2b %zmm19, %zmm1, %zmm12
	vmovdqa64 %zmm4, %zmm8
	vpermt2b %zmm7, %zmm0, %zmm8
	vpermt2b %zmm7, %zmm1, %zmm4
	vmovdqa64 %zmm2, %zmm7
	vpermt2b %zmm11, %zmm0, %zmm7
	vpermt2b %zmm11, %zmm1, %zmm2
	vmovdqa64 %zmm6, %zmm18
	vpermt2b %zmm15, %zmm0, %zmm18
	vpermt2b %zmm15, %zmm1, %zmm6
	vmovdqa64 %zmm3, %zmm11
	vpermt2b %zmm14, %zmm0, %zmm11
	vpermt2b %zmm14, %zmm1, %zmm3
	vmovdqa64 %zmm20, %zmm14
	vpermt2b %zmm16, %zmm0, %zmm14
	vpermt2b %zmm16, %zmm1, %zmm20
	vmovdqa64 %zmm9, %zmm15
	vpermt2b %zmm13, %zmm0, %zmm15
	vpermt2b %zmm13, %zmm1, %zmm9
	vmovdqa64 %zmm10, %zmm16
	vpermt2b %zmm17, %zmm0, %zmm16
	vpermt2b %zmm17, %zmm1, %zmm10
	vmovdqa64 %zmm5, %zmm17
	vpermt2b %zmm12, %zmm0, %zmm17
	vpermt2b %zmm12, %zmm1, %zmm5
	vmovdqa64 %zmm8, %zmm12
	vpermt2b %zmm14, %zmm0, %zmm12
	vpermt2b %zmm14, %zmm1, %zmm8
	vmovdqa64 %zmm4, %zmm13
	vpermt2b %zmm20, %zmm0, %zmm13
	vpermt2b %zmm20, %zmm1, %zmm4
	vmovdqa64 %zmm7, %zmm14
	vpermt2b %zmm15, %zmm0, %zmm14
	vpermt2b %zmm15, %zmm1, %zmm7
	vmovdqa64 %zmm2, %zmm15
	vpermt2b %zmm9, %zmm0, %zmm15
	vpermt2b %zmm9, %zmm1, %zmm2
	vmovdqa64 %zmm18, %zmm9
	vpermt2b %zmm16, %zmm0, %zmm9
	vpermt2b %zmm16, %zmm1, %zmm18
	vmovdqa64 %zmm6, %zmm16
	vpermt2b %zmm10, %zmm0, %zmm16
	vpermt2b %zmm10, %zmm1, %zmm6
	vmovdqa64 %zmm11, %zmm10
	vpermt2b %zmm17, %zmm0, %zmm10
	vpermt2b %zmm17, %zmm1, %zmm11
	vmovdqa64 %zmm3, %zmm17
	vpermt2b %zmm5, %zmm0, %zmm17
	vpermt2b %zmm5, %zmm1, %zmm3
	vmovdqa64 %zmm12, %zmm5
	vpermt2b %zmm9, %zmm0, %zmm5
	vpermt2b %zmm9, %zmm1, %zmm12
	vmovdqa64 %zmm8, %zmm9
	vpermt2b %zmm18, %zmm0, %zmm9
	vpermt2b %zmm18, %zmm1, %zmm8
	vmovdqa64 %zmm13, %zmm18
	vpermt2b %zmm16, %zmm0, %zmm18
	vpermt2b %zmm16, %zmm1, %zmm13
	vmovdqa64 %zmm4, %zmm16
	vpermt2b %zmm6, %zmm0, %zmm16
	vpermt2b %zmm6, %zmm1, %zmm4
	vmovdqa64 %zmm14, %zmm6
	vpermt2b %zmm10, %zmm0, %zmm6
	vpermt2b %zmm10, %zmm1, %zmm14
	vmovdqa64 %zmm7, %zmm10
	vpermt2b %zmm11, %zmm0, %zmm10
	vpermt2b %zmm11, %zmm1, %zmm7
	vmovdqa64 %zmm15, %zmm11
	vpermt2b %zmm17, %zmm0, %zmm11
	vpermt2b %zmm17, %zmm1, %zmm15
	vmovdqa64 %zmm2, %zmm17
	vpermt2b %zmm3, %zmm0, %zmm17
	vpermt2b %zmm3, %zmm1, %zmm2
	vmovdqu64 %zmm5, -960(%rdx)
	vmovdqu64 %zmm12, -896(%rdx)
	vmovdqu64 %zmm9, -832(%rdx)
	vmovdqu64 %zmm8, -768(%rdx)
	vmovdqu64 %zmm18, -704(%rdx)
	vmovdqu64 %zmm13, -640(%rdx)
	vmovdqu64 %zmm16, -576(%rdx)
	vmovdqu64 %zmm4, -512(%rdx)
	vmovdqu64 %zmm6, -448(%rdx)
	vmovdqu64 %zmm14, -384(%rdx)
	vmovdqu64 %zmm10, -320(%rdx)
	vmovdqu64 %zmm7, -256(%rdx)
	vmovdqu64 %zmm11, -192(%rdx)
	vmovdqu64 %zmm15, -128(%rdx)
	vmovdqu64 %zmm17, -64(%rdx)
	vmovdqu64 %zmm2, (%rdx)
	addq $1024, %rdx
	addq $64, %r15
	cmpq %r15, %r11
	jne .LBB10_3
.LBB10_4:
	movq 8(%rsp), %rax
	shlq $6, %rax
	addq $56, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB10_5:
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
