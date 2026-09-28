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
	jb .LBB18_5
	movq %rsi, %rax
	shrq $8, %rax
	movq %rax, (%rsp)
	je .LBB18_4
	shrq $3, %rsi
	leaq (%rsi,%rsi), %rax
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rax,%rax,2), %r9
	leaq (,%rsi,8), %r10
	subq %rsi, %r10
	movq (%rsp), %r11
	shlq $5, %r11
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
	vpbroadcastq .LCPI18_0(%rip), %ymm0
	vpbroadcastq .LCPI18_1(%rip), %ymm1
	vpbroadcastq .LCPI18_2(%rip), %ymm2
.LBB18_3:
	vmovdqu (%rsi,%r15), %ymm5
	vmovdqu (%r14,%r15), %ymm6
	vmovdqu (%rcx,%r15), %ymm7
	vmovdqu (%r8,%r15), %ymm4
	vmovdqu (%r9,%r15), %ymm8
	vmovdqu (%r10,%r15), %ymm9
	vmovdqu (%rax,%r15), %ymm3
	vmovdqu (%rdi,%r15), %ymm10
	vpsrlq $4, %ymm10, %ymm11
	vpxor %ymm3, %ymm11, %ymm11
	vpand %ymm0, %ymm11, %ymm11
	vpxor %ymm3, %ymm11, %ymm3
	vpsllq $4, %ymm11, %ymm11
	vpxor %ymm10, %ymm11, %ymm10
	vpsrlq $4, %ymm5, %ymm11
	vpxor %ymm4, %ymm11, %ymm11
	vpand %ymm0, %ymm11, %ymm11
	vpxor %ymm4, %ymm11, %ymm4
	vpsllq $4, %ymm11, %ymm11
	vpxor %ymm5, %ymm11, %ymm5
	vpsrlq $4, %ymm6, %ymm11
	vpxor %ymm11, %ymm8, %ymm11
	vpand %ymm0, %ymm11, %ymm11
	vpxor %ymm8, %ymm11, %ymm8
	vpsllq $4, %ymm11, %ymm11
	vpxor %ymm6, %ymm11, %ymm6
	vpsrlq $4, %ymm7, %ymm11
	vpxor %ymm11, %ymm9, %ymm11
	vpand %ymm0, %ymm11, %ymm11
	vpxor %ymm9, %ymm11, %ymm9
	vpsllq $4, %ymm11, %ymm11
	vpxor %ymm7, %ymm11, %ymm7
	vpsrlq $2, %ymm10, %ymm11
	vpxor %ymm6, %ymm11, %ymm11
	vpand %ymm1, %ymm11, %ymm11
	vpxor %ymm6, %ymm11, %ymm6
	vpsllq $2, %ymm11, %ymm11
	vpxor %ymm10, %ymm11, %ymm10
	vpsrlq $2, %ymm5, %ymm11
	vpxor %ymm7, %ymm11, %ymm11
	vpand %ymm1, %ymm11, %ymm11
	vpxor %ymm7, %ymm11, %ymm7
	vpsllq $2, %ymm11, %ymm11
	vpxor %ymm5, %ymm11, %ymm5
	vpsrlq $2, %ymm3, %ymm11
	vpxor %ymm11, %ymm8, %ymm11
	vpand %ymm1, %ymm11, %ymm11
	vpxor %ymm8, %ymm11, %ymm8
	vpsllq $2, %ymm11, %ymm11
	vpxor %ymm3, %ymm11, %ymm3
	vpsrlq $2, %ymm4, %ymm11
	vpxor %ymm11, %ymm9, %ymm11
	vpand %ymm1, %ymm11, %ymm11
	vpxor %ymm9, %ymm11, %ymm9
	vpsllq $2, %ymm11, %ymm11
	vpxor %ymm4, %ymm11, %ymm4
	vpsrlq $1, %ymm10, %ymm11
	vpxor %ymm5, %ymm11, %ymm11
	vpand %ymm2, %ymm11, %ymm11
	vpxor %ymm5, %ymm11, %ymm5
	vpaddq %ymm11, %ymm11, %ymm11
	vpxor %ymm10, %ymm11, %ymm10
	vpsrlq $1, %ymm6, %ymm11
	vpxor %ymm7, %ymm11, %ymm11
	vpand %ymm2, %ymm11, %ymm11
	vpxor %ymm7, %ymm11, %ymm7
	vpaddq %ymm11, %ymm11, %ymm11
	vpxor %ymm6, %ymm11, %ymm6
	vpsrlq $1, %ymm3, %ymm11
	vpxor %ymm4, %ymm11, %ymm11
	vpand %ymm2, %ymm11, %ymm11
	vpxor %ymm4, %ymm11, %ymm4
	vpaddq %ymm11, %ymm11, %ymm11
	vpxor %ymm3, %ymm11, %ymm3
	vpsrlq $1, %ymm8, %ymm11
	vpxor %ymm11, %ymm9, %ymm11
	vpand %ymm2, %ymm11, %ymm11
	vpxor %ymm9, %ymm11, %ymm9
	vpaddq %ymm11, %ymm11, %ymm11
	vpxor %ymm8, %ymm11, %ymm8
	vmovdqu %ymm10, (%rdx,%r15)
	vmovdqu %ymm5, (%rbx,%r15)
	vmovdqu %ymm6, (%rbp,%r15)
	vmovdqu %ymm7, (%r13,%r15)
	movq 8(%rsp), %r12
	vmovdqu %ymm3, (%r12,%r15)
	movq 16(%rsp), %r12
	vmovdqu %ymm4, (%r12,%r15)
	movq 24(%rsp), %r12
	vmovdqu %ymm8, (%r12,%r15)
	movq 32(%rsp), %r12
	vmovdqu %ymm9, (%r12,%r15)
	addq $32, %r15
	cmpq %r15, %r11
	jne .LBB18_3
.LBB18_4:
	movq (%rsp), %rax
	shlq $5, %rax
	addq $40, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB18_5:
	leaq .Lanon.6931a3aa7c1713de1da1f697d6f7f66a.4(%rip), %rdi
	leaq .Lanon.6931a3aa7c1713de1da1f697d6f7f66a.5(%rip), %rdx
	movl $36, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
