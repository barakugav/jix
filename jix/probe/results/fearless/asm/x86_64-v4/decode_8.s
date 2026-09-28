probe_byte_shuffle_decode_8:
	pushq %r15
	pushq %r14
	pushq %rbx
	movabsq $9223372036854775800, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB13_5
	movq %rsi, %rax
	shrq $9, %rax
	je .LBB13_4
	shrq $3, %rsi
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rdi,%rsi,8), %r9
	subq %rsi, %r9
	leaq (%rdi,%rcx,2), %r10
	addq %rdi, %r8
	leaq (%rdi,%rsi,4), %r11
	addq %rdi, %rcx
	leaq (%rdi,%rsi,2), %rbx
	addq %rdi, %rsi
	movq %rax, %r14
	shlq $6, %r14
	xorl %r15d, %r15d
	vmovdqa64 .LCPI13_0(%rip), %zmm0
	vmovdqa64 .LCPI13_1(%rip), %zmm1
.LBB13_3:
	vmovdqu64 (%rdi,%r15), %zmm2
	vmovdqu64 (%rsi,%r15), %zmm3
	vmovdqu64 (%rbx,%r15), %zmm4
	vmovdqu64 (%rcx,%r15), %zmm5
	vmovdqu64 (%r11,%r15), %zmm6
	vmovdqu64 (%r8,%r15), %zmm7
	vmovdqu64 (%r10,%r15), %zmm8
	vmovdqu64 (%r9,%r15), %zmm9
	vmovdqa64 %zmm2, %zmm10
	vpermt2b %zmm6, %zmm0, %zmm10
	vpermt2b %zmm6, %zmm1, %zmm2
	vmovdqa64 %zmm3, %zmm6
	vpermt2b %zmm7, %zmm0, %zmm6
	vpermt2b %zmm7, %zmm1, %zmm3
	vmovdqa64 %zmm4, %zmm7
	vpermt2b %zmm8, %zmm0, %zmm7
	vpermt2b %zmm8, %zmm1, %zmm4
	vmovdqa64 %zmm5, %zmm8
	vpermt2b %zmm9, %zmm0, %zmm8
	vpermt2b %zmm9, %zmm1, %zmm5
	vmovdqa64 %zmm10, %zmm9
	vpermt2b %zmm7, %zmm0, %zmm9
	vpermt2b %zmm7, %zmm1, %zmm10
	vmovdqa64 %zmm2, %zmm7
	vpermt2b %zmm4, %zmm0, %zmm7
	vpermt2b %zmm4, %zmm1, %zmm2
	vmovdqa64 %zmm6, %zmm4
	vpermt2b %zmm8, %zmm0, %zmm4
	vpermt2b %zmm8, %zmm1, %zmm6
	vmovdqa64 %zmm3, %zmm8
	vpermt2b %zmm5, %zmm0, %zmm8
	vpermt2b %zmm5, %zmm1, %zmm3
	vmovdqa64 %zmm9, %zmm5
	vpermt2b %zmm4, %zmm0, %zmm5
	vpermt2b %zmm4, %zmm1, %zmm9
	vmovdqa64 %zmm10, %zmm4
	vpermt2b %zmm6, %zmm0, %zmm4
	vpermt2b %zmm6, %zmm1, %zmm10
	vmovdqa64 %zmm7, %zmm6
	vpermt2b %zmm8, %zmm0, %zmm6
	vpermt2b %zmm8, %zmm1, %zmm7
	vmovdqa64 %zmm2, %zmm8
	vpermt2b %zmm3, %zmm0, %zmm8
	vpermt2b %zmm3, %zmm1, %zmm2
	vmovdqu64 %zmm5, (%rdx,%r15,8)
	vmovdqu64 %zmm9, 64(%rdx,%r15,8)
	vmovdqu64 %zmm4, 128(%rdx,%r15,8)
	vmovdqu64 %zmm10, 192(%rdx,%r15,8)
	vmovdqu64 %zmm6, 256(%rdx,%r15,8)
	vmovdqu64 %zmm7, 320(%rdx,%r15,8)
	vmovdqu64 %zmm8, 384(%rdx,%r15,8)
	vmovdqu64 %zmm2, 448(%rdx,%r15,8)
	addq $64, %r15
	cmpq %r15, %r14
	jne .LBB13_3
.LBB13_4:
	shlq $6, %rax
	popq %rbx
	popq %r14
	popq %r15
	vzeroupper
	retq
.LBB13_5:
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
