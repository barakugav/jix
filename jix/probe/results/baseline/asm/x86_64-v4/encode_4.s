jix_probe::byte_shuffle::encode_impl::<4, 32>:
	pushq %rbx
	movq %rsi, %rax
	shrq $2, %rax
	movabsq $2305843009213693920, %rcx
	andq %rax, %rcx
	je .LBB6_1
	leaq (%rax,%rax,2), %r8
	addq %rdx, %r8
	leaq (%rdx,%rax,2), %r10
	addq %rdx, %rax
	leaq 3(%rdi), %r11
	xorl %r9d, %r9d
	movabsq $1229782938247303441, %rbx
	kmovq %rbx, %k1
	vmovdqa64 .LCPI6_0(%rip), %zmm0
	vmovdqa64 .LCPI6_1(%rip), %zmm1
	vmovdqa .LCPI6_2(%rip), %ymm2
.LBB6_4:
	vmovdqu8 -3(%r11,%r9,4), %zmm3 {%k1} {z}
	vmovdqu8 61(%r11,%r9,4), %zmm4 {%k1} {z}
	vpshufb %zmm0, %zmm4, %zmm4
	vpshufb %zmm1, %zmm3, %zmm3
	vpermt2d %zmm4, %zmm2, %zmm3
	vmovdqu %ymm3, (%rdx,%r9)
	vmovdqu8 (%r11,%r9,4), %zmm3 {%k1} {z}
	vmovdqu8 64(%r11,%r9,4), %zmm4 {%k1} {z}
	vpshufb %zmm0, %zmm4, %zmm4
	vpshufb %zmm1, %zmm3, %zmm3
	vpermt2d %zmm4, %zmm2, %zmm3
	vmovdqu8 -2(%r11,%r9,4), %zmm4 {%k1} {z}
	vmovdqu8 62(%r11,%r9,4), %zmm5 {%k1} {z}
	vpshufb %zmm0, %zmm5, %zmm5
	vpshufb %zmm1, %zmm4, %zmm4
	vpermt2d %zmm5, %zmm2, %zmm4
	vmovdqu %ymm4, (%rax,%r9)
	vmovdqu8 -1(%r11,%r9,4), %zmm4 {%k1} {z}
	vmovdqu8 63(%r11,%r9,4), %zmm5 {%k1} {z}
	vpshufb %zmm0, %zmm5, %zmm5
	vpshufb %zmm1, %zmm4, %zmm4
	vpermt2d %zmm5, %zmm2, %zmm4
	vmovdqu %ymm4, (%r10,%r9)
	vmovdqu %ymm3, (%r8,%r9)
	addq $32, %r9
	cmpq %rcx, %r9
	jb .LBB6_4
	movl $4, %r8d
	popq %rbx
	vzeroupper
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
.LBB6_1:
	xorl %r9d, %r9d
	movl $4, %r8d
	popq %rbx
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
