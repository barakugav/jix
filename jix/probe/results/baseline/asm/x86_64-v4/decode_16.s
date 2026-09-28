jix_probe::byte_shuffle::decode_impl::<16, 8>:
	movq %rsi, %rcx
	shrq $4, %rcx
	movabsq $576460752303423480, %rax
	andq %rcx, %rax
	je .LBB0_1
	vpbroadcastq %rcx, %zmm1
	vpmullq .LCPI0_0(%rip), %zmm1, %zmm0
	vpmullq .LCPI0_1(%rip), %zmm1, %zmm1
	leaq 120(%rdx), %rcx
	xorl %r9d, %r9d
.LBB0_4:
	leaq (%rdi,%r9), %r8
	kxnorb %k0, %k0, %k1
	vpxor %xmm2, %xmm2, %xmm2
	vpgatherqq (%r8,%zmm0), %zmm2 {%k1}
	vpsrlq $8, %zmm2, %zmm3
	vpsrlq $16, %zmm2, %zmm4
	vpsrlq $24, %zmm2, %zmm5
	vpsrlq $32, %zmm2, %zmm6
	vpsrlq $40, %zmm2, %zmm7
	vpsrlq $48, %zmm2, %zmm8
	vpsrlq $56, %zmm2, %zmm9
	kxnorb %k0, %k0, %k1
	vpxor %xmm10, %xmm10, %xmm10
	vpgatherqq (%r8,%zmm1), %zmm10 {%k1}
	vpsrlq $8, %zmm10, %zmm11
	vpsrlq $16, %zmm10, %zmm12
	vpsrlq $24, %zmm10, %zmm13
	vpsrlq $32, %zmm10, %zmm14
	vpsrlq $40, %zmm10, %zmm15
	vpsrlq $48, %zmm10, %zmm16
	vpsrlq $56, %zmm10, %zmm17
	vpmovqb %zmm2, -120(%rcx)
	vpmovqb %zmm10, -112(%rcx)
	vpmovqb %zmm3, -104(%rcx)
	vpmovqb %zmm11, -96(%rcx)
	vpmovqb %zmm4, -88(%rcx)
	vpmovqb %zmm12, -80(%rcx)
	vpmovqb %zmm5, -72(%rcx)
	vpmovqb %zmm13, -64(%rcx)
	vpmovqb %zmm6, -56(%rcx)
	vpmovqb %zmm14, -48(%rcx)
	vpmovqb %zmm7, -40(%rcx)
	vpmovqb %zmm15, -32(%rcx)
	vpmovqb %zmm8, -24(%rcx)
	vpmovqb %zmm16, -16(%rcx)
	vpmovqb %zmm9, -8(%rcx)
	vpmovqb %zmm17, (%rcx)
	addq $8, %r9
	subq $-128, %rcx
	cmpq %rax, %r9
	jb .LBB0_4
	movl $16, %r8d
	vzeroupper
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
.LBB0_1:
	xorl %r9d, %r9d
	movl $16, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
