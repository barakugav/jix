probe_byte_shuffle_decode_4:
	movabsq $9223372036854775804, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB12_5
	movq %rsi, %rax
	shrq $8, %rax
	je .LBB12_4
	shrq $2, %rsi
	leaq (%rsi,%rsi,2), %rcx
	addq %rdi, %rcx
	leaq (%rdi,%rsi,2), %r8
	addq %rdi, %rsi
	movq %rax, %r9
	shlq $6, %r9
	xorl %r10d, %r10d
	vmovdqa64 .LCPI12_0(%rip), %zmm0
	vmovdqa64 .LCPI12_1(%rip), %zmm1
.LBB12_3:
	vmovdqu64 (%rdi,%r10), %zmm2
	vmovdqu64 (%rsi,%r10), %zmm3
	vmovdqu64 (%r8,%r10), %zmm4
	vmovdqu64 (%rcx,%r10), %zmm5
	vmovdqa64 %zmm2, %zmm6
	vpermt2b %zmm4, %zmm0, %zmm6
	vpermt2b %zmm4, %zmm1, %zmm2
	vmovdqa64 %zmm3, %zmm4
	vpermt2b %zmm5, %zmm0, %zmm4
	vpermt2b %zmm5, %zmm1, %zmm3
	vmovdqa64 %zmm6, %zmm5
	vpermt2b %zmm4, %zmm0, %zmm5
	vpermt2b %zmm4, %zmm1, %zmm6
	vmovdqa64 %zmm2, %zmm4
	vpermt2b %zmm3, %zmm0, %zmm4
	vpermt2b %zmm3, %zmm1, %zmm2
	vmovdqu64 %zmm5, (%rdx,%r10,4)
	vmovdqu64 %zmm6, 64(%rdx,%r10,4)
	vmovdqu64 %zmm4, 128(%rdx,%r10,4)
	vmovdqu64 %zmm2, 192(%rdx,%r10,4)
	addq $64, %r10
	cmpq %r10, %r9
	jne .LBB12_3
.LBB12_4:
	shlq $6, %rax
	vzeroupper
	retq
.LBB12_5:
	pushq %rax
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
