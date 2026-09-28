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
	jb .LBB7_5
	movq %rsi, %rax
	shrq $9, %rax
	movq %rax, (%rsp)
	je .LBB7_4
	shrq $3, %rsi
	leaq (%rsi,%rsi), %rax
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rax,%rax,2), %r9
	leaq (,%rsi,8), %r10
	subq %rsi, %r10
	movq (%rsp), %r11
	shlq $6, %r11
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
	vpbroadcastq .LCPI7_0(%rip), %zmm0
	vpbroadcastq .LCPI7_1(%rip), %zmm1
	vpbroadcastq .LCPI7_2(%rip), %zmm2
.LBB7_3:
	vmovdqu64 (%rsi,%r15), %zmm3
	vmovdqu64 (%r14,%r15), %zmm4
	vmovdqu64 (%rcx,%r15), %zmm5
	vmovdqu64 (%r8,%r15), %zmm6
	vmovdqu64 (%r9,%r15), %zmm7
	vmovdqu64 (%r10,%r15), %zmm8
	vmovdqu64 (%rax,%r15), %zmm9
	vmovdqu64 (%rdi,%r15), %zmm10
	vpsrlq $4, %zmm10, %zmm11
	vpternlogq $40, %zmm0, %zmm9, %zmm11
	vpxorq %zmm9, %zmm11, %zmm9
	vpsllq $4, %zmm11, %zmm11
	vpxorq %zmm10, %zmm11, %zmm10
	vpsrlq $4, %zmm3, %zmm11
	vpternlogq $40, %zmm0, %zmm6, %zmm11
	vpxorq %zmm6, %zmm11, %zmm6
	vpsllq $4, %zmm11, %zmm11
	vpxorq %zmm3, %zmm11, %zmm3
	vpsrlq $4, %zmm4, %zmm11
	vpternlogq $40, %zmm0, %zmm7, %zmm11
	vpxorq %zmm7, %zmm11, %zmm7
	vpsllq $4, %zmm11, %zmm11
	vpxorq %zmm4, %zmm11, %zmm4
	vpsrlq $4, %zmm5, %zmm11
	vpternlogq $40, %zmm0, %zmm8, %zmm11
	vpxorq %zmm8, %zmm11, %zmm8
	vpsllq $4, %zmm11, %zmm11
	vpxorq %zmm5, %zmm11, %zmm5
	vpsrlq $2, %zmm10, %zmm11
	vpternlogq $40, %zmm1, %zmm4, %zmm11
	vpxorq %zmm4, %zmm11, %zmm4
	vpsllq $2, %zmm11, %zmm11
	vpxorq %zmm10, %zmm11, %zmm10
	vpsrlq $2, %zmm3, %zmm11
	vpternlogq $40, %zmm1, %zmm5, %zmm11
	vpxorq %zmm5, %zmm11, %zmm5
	vpsllq $2, %zmm11, %zmm11
	vpxorq %zmm3, %zmm11, %zmm3
	vpsrlq $2, %zmm9, %zmm11
	vpternlogq $40, %zmm1, %zmm7, %zmm11
	vpxorq %zmm7, %zmm11, %zmm7
	vpsllq $2, %zmm11, %zmm11
	vpxorq %zmm9, %zmm11, %zmm9
	vpsrlq $2, %zmm6, %zmm11
	vpternlogq $40, %zmm1, %zmm8, %zmm11
	vpxorq %zmm8, %zmm11, %zmm8
	vpsllq $2, %zmm11, %zmm11
	vpxorq %zmm6, %zmm11, %zmm6
	vpsrlq $1, %zmm10, %zmm11
	vpternlogq $40, %zmm2, %zmm3, %zmm11
	vpxorq %zmm3, %zmm11, %zmm3
	vpaddq %zmm11, %zmm11, %zmm11
	vpxorq %zmm10, %zmm11, %zmm10
	vpsrlq $1, %zmm4, %zmm11
	vpternlogq $40, %zmm2, %zmm5, %zmm11
	vpxorq %zmm5, %zmm11, %zmm5
	vpaddq %zmm11, %zmm11, %zmm11
	vpxorq %zmm4, %zmm11, %zmm4
	vpsrlq $1, %zmm9, %zmm11
	vpternlogq $40, %zmm2, %zmm6, %zmm11
	vpxorq %zmm6, %zmm11, %zmm6
	vpaddq %zmm11, %zmm11, %zmm11
	vpxorq %zmm9, %zmm11, %zmm9
	vpsrlq $1, %zmm7, %zmm11
	vpternlogq $40, %zmm2, %zmm8, %zmm11
	vpxorq %zmm8, %zmm11, %zmm8
	vpaddq %zmm11, %zmm11, %zmm11
	vpxorq %zmm7, %zmm11, %zmm7
	vmovdqu64 %zmm10, (%rdx,%r15)
	vmovdqu64 %zmm3, (%rbx,%r15)
	vmovdqu64 %zmm4, (%rbp,%r15)
	vmovdqu64 %zmm5, (%r13,%r15)
	movq 8(%rsp), %r12
	vmovdqu64 %zmm9, (%r12,%r15)
	movq 16(%rsp), %r12
	vmovdqu64 %zmm6, (%r12,%r15)
	movq 24(%rsp), %r12
	vmovdqu64 %zmm7, (%r12,%r15)
	movq 32(%rsp), %r12
	vmovdqu64 %zmm8, (%r12,%r15)
	addq $64, %r15
	cmpq %r15, %r11
	jne .LBB7_3
.LBB7_4:
	movq (%rsp), %rax
	shlq $6, %rax
	addq $40, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB7_5:
	leaq .Lanon.6931a3aa7c1713de1da1f697d6f7f66a.4(%rip), %rdi
	leaq .Lanon.6931a3aa7c1713de1da1f697d6f7f66a.5(%rip), %rdx
	movl $36, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
