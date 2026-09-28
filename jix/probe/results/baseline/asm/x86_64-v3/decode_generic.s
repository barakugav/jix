jix_probe::byte_shuffle::decode_impl_generic:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $24, %rsp
	testq %r8, %r8
	je .LBB8_6
	movq %rdx, %rcx
	movq %rdi, %r12
	movq %rsi, %rax
	orq %r8, %rax
	shrq $32, %rax
	je .LBB8_2
	movq %rsi, %rax
	xorl %edx, %edx
	divq %r8
	cmpq %rax, %r9
	jb .LBB8_5
	jmp .LBB8_20
.LBB8_2:
	movl %esi, %eax
	xorl %edx, %edx
	divl %r8d
	cmpq %rax, %r9
	jae .LBB8_20
.LBB8_5:
	movq %r8, %rdx
	andq $-32, %rdx
	movq %rdx, 8(%rsp)
	movq %r8, %rdi
	andq $-8, %rdi
	movq %r9, %r10
	imulq %r8, %r10
	addq %r10, %rcx
	jmp .LBB8_8
.LBB8_7:
	incq %r9
	addq %r8, %rcx
	cmpq %rax, %r9
	jae .LBB8_20
.LBB8_8:
	leaq (%r12,%r9), %r10
	cmpq $8, %r8
	jae .LBB8_10
	xorl %ebx, %ebx
	jmp .LBB8_19
.LBB8_10:
	cmpq $32, %r8
	jae .LBB8_12
	xorl %r11d, %r11d
	jmp .LBB8_16
.LBB8_12:
	movq %rcx, %r11
	movq 8(%rsp), %rbx
	xorl %r14d, %r14d
.LBB8_13:
	movq %r14, %r15
	imulq %rax, %r15
	movzbl (%r10,%r15), %ebp
	vmovd %ebp, %xmm0
	leaq 1(%r14), %r15
	imulq %rax, %r15
	vpinsrb $1, (%r10,%r15), %xmm0, %xmm0
	leaq 2(%r14), %r15
	imulq %rax, %r15
	vpinsrb $2, (%r10,%r15), %xmm0, %xmm0
	leaq 3(%r14), %r15
	imulq %rax, %r15
	vpinsrb $3, (%r10,%r15), %xmm0, %xmm0
	leaq 4(%r14), %r15
	imulq %rax, %r15
	vpinsrb $4, (%r10,%r15), %xmm0, %xmm0
	leaq 5(%r14), %r15
	imulq %rax, %r15
	vpinsrb $5, (%r10,%r15), %xmm0, %xmm0
	leaq 6(%r14), %r15
	imulq %rax, %r15
	vpinsrb $6, (%r10,%r15), %xmm0, %xmm0
	leaq 7(%r14), %r15
	imulq %rax, %r15
	vpinsrb $7, (%r10,%r15), %xmm0, %xmm0
	leaq 8(%r14), %r15
	imulq %rax, %r15
	vpinsrb $8, (%r10,%r15), %xmm0, %xmm0
	leaq 9(%r14), %r15
	imulq %rax, %r15
	vpinsrb $9, (%r10,%r15), %xmm0, %xmm0
	leaq 10(%r14), %r15
	imulq %rax, %r15
	vpinsrb $10, (%r10,%r15), %xmm0, %xmm0
	leaq 11(%r14), %r15
	imulq %rax, %r15
	vpinsrb $11, (%r10,%r15), %xmm0, %xmm0
	leaq 12(%r14), %r15
	imulq %rax, %r15
	vpinsrb $12, (%r10,%r15), %xmm0, %xmm0
	leaq 13(%r14), %r15
	imulq %rax, %r15
	vpinsrb $13, (%r10,%r15), %xmm0, %xmm0
	leaq 14(%r14), %r15
	imulq %rax, %r15
	vpinsrb $14, (%r10,%r15), %xmm0, %xmm0
	leaq 15(%r14), %r15
	imulq %rax, %r15
	vpinsrb $15, (%r10,%r15), %xmm0, %xmm0
	leaq 16(%r14), %r15
	imulq %rax, %r15
	movzbl (%r10,%r15), %ebp
	vmovd %ebp, %xmm1
	leaq 17(%r14), %r15
	imulq %rax, %r15
	vpinsrb $1, (%r10,%r15), %xmm1, %xmm1
	leaq 18(%r14), %r15
	imulq %rax, %r15
	vpinsrb $2, (%r10,%r15), %xmm1, %xmm1
	leaq 19(%r14), %r15
	imulq %rax, %r15
	vpinsrb $3, (%r10,%r15), %xmm1, %xmm1
	leaq 20(%r14), %r15
	imulq %rax, %r15
	vpinsrb $4, (%r10,%r15), %xmm1, %xmm1
	leaq 21(%r14), %r15
	imulq %rax, %r15
	vpinsrb $5, (%r10,%r15), %xmm1, %xmm1
	leaq 22(%r14), %r15
	imulq %rax, %r15
	vpinsrb $6, (%r10,%r15), %xmm1, %xmm1
	leaq 23(%r14), %r15
	imulq %rax, %r15
	vpinsrb $7, (%r10,%r15), %xmm1, %xmm1
	leaq 24(%r14), %r15
	imulq %rax, %r15
	vpinsrb $8, (%r10,%r15), %xmm1, %xmm1
	leaq 25(%r14), %r15
	imulq %rax, %r15
	vpinsrb $9, (%r10,%r15), %xmm1, %xmm1
	leaq 26(%r14), %r15
	imulq %rax, %r15
	vpinsrb $10, (%r10,%r15), %xmm1, %xmm1
	leaq 27(%r14), %r15
	imulq %rax, %r15
	vpinsrb $11, (%r10,%r15), %xmm1, %xmm1
	leaq 28(%r14), %r15
	imulq %rax, %r15
	vpinsrb $12, (%r10,%r15), %xmm1, %xmm1
	leaq 29(%r14), %r15
	imulq %rax, %r15
	vpinsrb $13, (%r10,%r15), %xmm1, %xmm1
	leaq 30(%r14), %r15
	imulq %rax, %r15
	vpinsrb $14, (%r10,%r15), %xmm1, %xmm1
	leaq 31(%r14), %r15
	imulq %rax, %r15
	vpinsrb $15, (%r10,%r15), %xmm1, %xmm1
	vmovdqu %xmm1, 16(%r11)
	vmovdqu %xmm0, (%r11)
	addq $32, %r14
	addq $32, %r11
	addq $-32, %rbx
	jne .LBB8_13
	cmpq 8(%rsp), %r8
	je .LBB8_7
	movq 8(%rsp), %rbx
	movq %rbx, %r11
	testb $24, %r8b
	je .LBB8_19
.LBB8_16:
	movq %r9, 16(%rsp)
	movq %r12, %r9
.LBB8_17:
	leaq 1(%r11), %rbx
	leaq 2(%r11), %r14
	leaq 3(%r11), %r15
	leaq 4(%r11), %r12
	leaq 5(%r11), %r13
	leaq 6(%r11), %rbp
	leaq 7(%r11), %rdx
	movq %r11, %rsi
	imulq %rax, %rsi
	imulq %rax, %rbx
	imulq %rax, %r14
	imulq %rax, %r15
	imulq %rax, %r12
	imulq %rax, %r13
	imulq %rax, %rbp
	imulq %rax, %rdx
	movzbl (%r10,%rsi), %esi
	vmovd %esi, %xmm0
	vpinsrb $1, (%r10,%rbx), %xmm0, %xmm0
	vpinsrb $2, (%r10,%r14), %xmm0, %xmm0
	vpinsrb $3, (%r10,%r15), %xmm0, %xmm0
	vpinsrb $4, (%r10,%r12), %xmm0, %xmm0
	vpinsrb $5, (%r10,%r13), %xmm0, %xmm0
	vpinsrb $6, (%r10,%rbp), %xmm0, %xmm0
	vpinsrb $7, (%r10,%rdx), %xmm0, %xmm0
	vmovq %xmm0, (%rcx,%r11)
	addq $8, %r11
	cmpq %r11, %rdi
	jne .LBB8_17
	movq %rdi, %rbx
	cmpq %rdi, %r8
	movq %r9, %r12
	movq 16(%rsp), %r9
	je .LBB8_7
.LBB8_19:
	movq %rbx, %rdx
	imulq %rax, %rdx
	movzbl (%r10,%rdx), %edx
	movb %dl, (%rcx,%rbx)
	incq %rbx
	cmpq %rbx, %r8
	jne .LBB8_19
	jmp .LBB8_7
.LBB8_20:
	addq $24, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	retq
.LBB8_6:
	leaq .Lanon.541dfd7828f123e2376bd9c83f3f80e1.1(%rip), %rdi
	callq *core::panicking::panic_const::panic_const_div_by_zero@GOTPCREL(%rip)
