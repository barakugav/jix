probe_byte_shuffle_decode_2:
	movabsq $9223372036854775806, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB11_5
	movq %rsi, %rax
	shrq %rax
	shrq $8, %rsi
	je .LBB11_4
	leaq (%rax,%rdi), %rcx
	addq $64, %rcx
	shlq $7, %rsi
	xorl %r8d, %r8d
	vmovdqa64 .LCPI11_0(%rip), %zmm0
	vmovdqa64 .LCPI11_1(%rip), %zmm1
.LBB11_3:
	vmovdqu64 (%rdi,%r8), %zmm2
	vmovdqu64 -64(%rcx,%r8), %zmm3
	vmovdqa64 %zmm2, %zmm4
	vpermt2b %zmm3, %zmm0, %zmm4
	vpermt2b %zmm3, %zmm1, %zmm2
	vmovdqu64 %zmm4, (%rdx,%r8,2)
	vmovdqu64 %zmm2, 64(%rdx,%r8,2)
	vmovdqu64 64(%rdi,%r8), %zmm2
	vmovdqu64 (%rcx,%r8), %zmm3
	vmovdqa64 %zmm2, %zmm4
	vpermt2b %zmm3, %zmm0, %zmm4
	vpermt2b %zmm3, %zmm1, %zmm2
	vmovdqu64 %zmm4, 128(%rdx,%r8,2)
	vmovdqu64 %zmm2, 192(%rdx,%r8,2)
	subq $-128, %r8
	cmpq %r8, %rsi
	jne .LBB11_3
.LBB11_4:
	movabsq $4611686018427387776, %rcx
	andq %rcx, %rax
	vzeroupper
	retq
.LBB11_5:
	pushq %rax
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
