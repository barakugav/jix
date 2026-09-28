jix_probe::byte_shuffle::encode_impl::<2, 64>:
	movq %rsi, %rax
	shrq %rax
	movabsq $4611686018427387840, %rcx
	andq %rax, %rcx
	je .LBB1_1
	addq %rdx, %rax
	leaq 1(%rdi), %r8
	xorl %r9d, %r9d
	movabsq $6148914691236517205, %r10
	kmovq %r10, %k1
	vmovdqa64 .LCPI1_0(%rip), %zmm0
	vmovdqa64 .LCPI1_1(%rip), %zmm1
	vmovdqa64 .LCPI1_2(%rip), %zmm2
.LBB1_4:
	vmovdqu8 (%r8,%r9,2), %zmm3 {%k1} {z}
	vmovdqu8 64(%r8,%r9,2), %zmm4 {%k1} {z}
	vpshufb %zmm0, %zmm4, %zmm4
	vpshufb %zmm1, %zmm3, %zmm3
	vpermt2q %zmm4, %zmm2, %zmm3
	vmovdqu8 -1(%r8,%r9,2), %zmm4 {%k1} {z}
	vmovdqu8 63(%r8,%r9,2), %zmm5 {%k1} {z}
	vpshufb %zmm0, %zmm5, %zmm5
	vpshufb %zmm1, %zmm4, %zmm4
	vpermt2q %zmm5, %zmm2, %zmm4
	vmovdqu64 %zmm4, (%rdx,%r9)
	vmovdqu64 %zmm3, (%rax,%r9)
	addq $64, %r9
	cmpq %rcx, %r9
	jb .LBB1_4
	movl $2, %r8d
	vzeroupper
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
.LBB1_1:
	xorl %r9d, %r9d
	movl $2, %r8d
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
