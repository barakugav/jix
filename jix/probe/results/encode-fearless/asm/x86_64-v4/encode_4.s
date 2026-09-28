probe_byte_shuffle_encode_4:
	movabsq $9223372036854775804, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB12_5
	movq %rsi, %rax
	shrq $8, %rax
	je .LBB12_4
	shrq $2, %rsi
	leaq (%rsi,%rsi,2), %rcx
	addq %rdx, %rcx
	leaq (%rdx,%rsi,2), %r8
	addq %rdx, %rsi
	movq %rax, %r9
	shlq $6, %r9
	xorl %r10d, %r10d
	vmovdqa64 .LCPI12_0(%rip), %zmm0
	vmovdqa64 .LCPI12_1(%rip), %zmm1
.LBB12_3:
	vmovdqu64 (%rdi,%r10,4), %zmm2
	vmovdqu64 64(%rdi,%r10,4), %zmm3
	vmovdqu64 128(%rdi,%r10,4), %zmm4
	vmovdqu64 192(%rdi,%r10,4), %zmm5
	vmovdqa64 %zmm2, %zmm6
	vpermt2b %zmm3, %zmm0, %zmm6
	vpermt2b %zmm3, %zmm1, %zmm2
	vmovdqa64 %zmm4, %zmm3
	vpermt2b %zmm5, %zmm0, %zmm3
	vpermt2b %zmm5, %zmm1, %zmm4
	vmovdqa64 %zmm6, %zmm5
	vpermt2b %zmm3, %zmm0, %zmm5
	vpermt2b %zmm3, %zmm1, %zmm6
	vmovdqa64 %zmm2, %zmm3
	vpermt2b %zmm4, %zmm0, %zmm3
	vpermt2b %zmm4, %zmm1, %zmm2
	vmovdqu64 %zmm5, (%rdx,%r10)
	vmovdqu64 %zmm3, (%rsi,%r10)
	vmovdqu64 %zmm6, (%r8,%r10)
	vmovdqu64 %zmm2, (%rcx,%r10)
	addq $64, %r10
	cmpq %r10, %r9
	jne .LBB12_3
.LBB12_4:
	shlq $6, %rax
	vzeroupper
	retq
.LBB12_5:
	pushq %rax
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
