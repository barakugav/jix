probe_byte_shuffle_encode_16:
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
	addq %rdx, %rax
	movq %rax, 32(%rsp)
	addq %rdx, %rbx
	addq %rdx, %r10
	movq %r10, 40(%rsp)
	leaq (%rdx,%rcx,4), %rax
	movq %rax, 24(%rsp)
	addq %rdx, %r9
	movq %r9, 48(%rsp)
	leaq (%rdx,%r8,2), %rax
	movq %rax, 16(%rsp)
	addq %rdx, %r12
	leaq (%rdx,%rsi,8), %r13
	movq %r13, %rbp
	subq %rsi, %rbp
	leaq (%rdx,%rcx,2), %r9
	addq %rdx, %r8
	leaq (%rdx,%rsi,4), %rax
	addq %rdx, %rcx
	leaq (%rdx,%rsi,2), %r10
	addq %rdx, %rsi
	addq $960, %rdi
	movq 8(%rsp), %r15
	shlq $6, %r15
	vmovdqa64 .LCPI10_0(%rip), %zmm0
	vmovdqa64 .LCPI10_1(%rip), %zmm1
	xorl %r11d, %r11d
.LBB10_3:
	vmovdqu64 -960(%rdi), %zmm2
	vmovdqu64 -896(%rdi), %zmm5
	vmovdqu64 -832(%rdi), %zmm12
	vmovdqu64 -768(%rdi), %zmm6
	vmovdqu64 -704(%rdi), %zmm7
	vmovdqu64 -640(%rdi), %zmm9
	vmovdqu64 -576(%rdi), %zmm13
	vmovdqu64 -512(%rdi), %zmm11
	vmovdqu64 -448(%rdi), %zmm4
	vmovdqu64 -384(%rdi), %zmm16
	vmovdqu64 -320(%rdi), %zmm14
	vmovdqu64 -256(%rdi), %zmm17
	vmovdqu64 -192(%rdi), %zmm8
	vmovdqu64 -128(%rdi), %zmm18
	vmovdqu64 -64(%rdi), %zmm15
	vmovdqu64 (%rdi), %zmm19
	vmovdqa64 %zmm2, %zmm3
	vpermt2b %zmm5, %zmm0, %zmm3
	vpermt2b %zmm5, %zmm1, %zmm2
	vmovdqa64 %zmm12, %zmm20
	vpermt2b %zmm6, %zmm0, %zmm20
	vpermt2b %zmm6, %zmm1, %zmm12
	vmovdqa64 %zmm7, %zmm10
	vpermt2b %zmm9, %zmm0, %zmm10
	vpermt2b %zmm9, %zmm1, %zmm7
	vmovdqa64 %zmm13, %zmm9
	vpermt2b %zmm11, %zmm0, %zmm9
	vpermt2b %zmm11, %zmm1, %zmm13
	vmovdqa64 %zmm4, %zmm6
	vpermt2b %zmm16, %zmm0, %zmm6
	vpermt2b %zmm16, %zmm1, %zmm4
	vmovdqa64 %zmm14, %zmm16
	vpermt2b %zmm17, %zmm0, %zmm16
	vpermt2b %zmm17, %zmm1, %zmm14
	vmovdqa64 %zmm8, %zmm17
	vpermt2b %zmm18, %zmm0, %zmm17
	vpermt2b %zmm18, %zmm1, %zmm8
	vmovdqa64 %zmm15, %zmm18
	vpermt2b %zmm19, %zmm0, %zmm18
	vpermt2b %zmm19, %zmm1, %zmm15
	vmovdqa64 %zmm3, %zmm5
	vpermt2b %zmm20, %zmm0, %zmm5
	vpermt2b %zmm20, %zmm1, %zmm3
	vmovdqa64 %zmm10, %zmm19
	vpermt2b %zmm9, %zmm0, %zmm19
	vpermt2b %zmm9, %zmm1, %zmm10
	vmovdqa64 %zmm6, %zmm11
	vpermt2b %zmm16, %zmm0, %zmm11
	vpermt2b %zmm16, %zmm1, %zmm6
	vmovdqa64 %zmm17, %zmm16
	vpermt2b %zmm18, %zmm0, %zmm16
	vpermt2b %zmm18, %zmm1, %zmm17
	vmovdqa64 %zmm2, %zmm9
	vpermt2b %zmm12, %zmm0, %zmm9
	vpermt2b %zmm12, %zmm1, %zmm2
	vmovdqa64 %zmm7, %zmm18
	vpermt2b %zmm13, %zmm0, %zmm18
	vpermt2b %zmm13, %zmm1, %zmm7
	vmovdqa64 %zmm4, %zmm20
	vpermt2b %zmm14, %zmm0, %zmm20
	vpermt2b %zmm14, %zmm1, %zmm4
	vmovdqa64 %zmm8, %zmm14
	vpermt2b %zmm15, %zmm0, %zmm14
	vpermt2b %zmm15, %zmm1, %zmm8
	vmovdqa64 %zmm5, %zmm12
	vpermt2b %zmm19, %zmm0, %zmm12
	vpermt2b %zmm19, %zmm1, %zmm5
	vmovdqa64 %zmm11, %zmm15
	vpermt2b %zmm16, %zmm0, %zmm15
	vpermt2b %zmm16, %zmm1, %zmm11
	vmovdqa64 %zmm9, %zmm13
	vpermt2b %zmm18, %zmm0, %zmm13
	vpermt2b %zmm18, %zmm1, %zmm9
	vmovdqa64 %zmm20, %zmm16
	vpermt2b %zmm14, %zmm0, %zmm16
	vpermt2b %zmm14, %zmm1, %zmm20
	vmovdqa64 %zmm3, %zmm14
	vpermt2b %zmm10, %zmm0, %zmm14
	vpermt2b %zmm10, %zmm1, %zmm3
	vmovdqa64 %zmm6, %zmm10
	vpermt2b %zmm17, %zmm0, %zmm10
	vpermt2b %zmm17, %zmm1, %zmm6
	vmovdqa64 %zmm2, %zmm17
	vpermt2b %zmm7, %zmm0, %zmm17
	vpermt2b %zmm7, %zmm1, %zmm2
	vmovdqa64 %zmm4, %zmm7
	vpermt2b %zmm8, %zmm0, %zmm7
	vpermt2b %zmm8, %zmm1, %zmm4
	vmovdqa64 %zmm12, %zmm8
	vpermt2b %zmm15, %zmm0, %zmm8
	vpermt2b %zmm15, %zmm1, %zmm12
	vmovdqa64 %zmm13, %zmm15
	vpermt2b %zmm16, %zmm0, %zmm15
	vpermt2b %zmm16, %zmm1, %zmm13
	vmovdqa64 %zmm14, %zmm16
	vpermt2b %zmm10, %zmm0, %zmm16
	vpermt2b %zmm10, %zmm1, %zmm14
	vmovdqa64 %zmm17, %zmm10
	vpermt2b %zmm7, %zmm0, %zmm10
	vpermt2b %zmm7, %zmm1, %zmm17
	vmovdqa64 %zmm5, %zmm7
	vpermt2b %zmm11, %zmm0, %zmm7
	vpermt2b %zmm11, %zmm1, %zmm5
	vmovdqa64 %zmm9, %zmm11
	vpermt2b %zmm20, %zmm0, %zmm11
	vpermt2b %zmm20, %zmm1, %zmm9
	vmovdqa64 %zmm3, %zmm18
	vpermt2b %zmm6, %zmm0, %zmm18
	vpermt2b %zmm6, %zmm1, %zmm3
	vmovdqa64 %zmm2, %zmm6
	vpermt2b %zmm4, %zmm0, %zmm6
	vpermt2b %zmm4, %zmm1, %zmm2
	vmovdqu64 %zmm8, (%rdx,%r11)
	vmovdqu64 %zmm15, (%rsi,%r11)
	vmovdqu64 %zmm16, (%r10,%r11)
	vmovdqu64 %zmm10, (%rcx,%r11)
	vmovdqu64 %zmm7, (%rax,%r11)
	vmovdqu64 %zmm11, (%r8,%r11)
	vmovdqu64 %zmm18, (%r9,%r11)
	vmovdqu64 %zmm6, (%rbp,%r11)
	vmovdqu64 %zmm12, (%r13,%r11)
	vmovdqu64 %zmm13, (%r12,%r11)
	movq 16(%rsp), %r14
	vmovdqu64 %zmm14, (%r14,%r11)
	movq 48(%rsp), %r14
	vmovdqu64 %zmm17, (%r14,%r11)
	movq 24(%rsp), %r14
	vmovdqu64 %zmm5, (%r14,%r11)
	movq 40(%rsp), %r14
	vmovdqu64 %zmm9, (%r14,%r11)
	vmovdqu64 %zmm3, (%rbx,%r11)
	movq 32(%rsp), %r14
	vmovdqu64 %zmm2, (%r14,%r11)
	addq $64, %r11
	addq $1024, %rdi
	cmpq %r11, %r15
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
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
