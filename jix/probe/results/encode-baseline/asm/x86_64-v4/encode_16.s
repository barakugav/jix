jix_probe::byte_shuffle::encode_impl::<16, 8>:
	movq %rsi, %rcx
	shrq $4, %rcx
	movabsq $576460752303423480, %rax
	andq %rcx, %rax
	je .LBB0_1
	vpbroadcastq %rcx, %zmm1
	vpmullq .LCPI0_0(%rip), %zmm1, %zmm0
	vpmullq .LCPI0_1(%rip), %zmm1, %zmm1
	leaq 120(%rdi), %rcx
	xorl %r9d, %r9d
.LBB0_4:
	vpmovzxbq (%rcx), %zmm2
	vpmovzxbq -8(%rcx), %zmm3
	vpmovzxbq -16(%rcx), %zmm4
	vpmovzxbq -24(%rcx), %zmm5
	vpmovzxbq -32(%rcx), %zmm6
	vpmovzxbq -40(%rcx), %zmm7
	vpmovzxbq -48(%rcx), %zmm8
	vpmovzxbq -56(%rcx), %zmm9
	vpmovzxbq -64(%rcx), %zmm10
	vpmovzxbq -72(%rcx), %zmm11
	vpmovzxbq -80(%rcx), %zmm12
	vpmovzxbq -88(%rcx), %zmm13
	vpmovzxbq -96(%rcx), %zmm14
	vpmovzxbq -104(%rcx), %zmm15
	vpmovzxbq -112(%rcx), %zmm16
	leaq (%rdx,%r9), %r8
	vpmovzxbq -120(%rcx), %zmm17
	vpsllq $56, %zmm3, %zmm3
	vpsllq $48, %zmm5, %zmm5
	vpsllq $40, %zmm7, %zmm7
	vpsllq $32, %zmm9, %zmm9
	vpsllq $24, %zmm11, %zmm11
	vpsllq $16, %zmm13, %zmm13
	vpsllq $8, %zmm15, %zmm15
	vporq %zmm17, %zmm15, %zmm15
	vpternlogq $254, %zmm13, %zmm11, %zmm15
	vpternlogq $254, %zmm9, %zmm7, %zmm15
	vpternlogq $254, %zmm5, %zmm3, %zmm15
	kxnorb %k0, %k0, %k1
	vpscatterqq %zmm15, (%r8,%zmm0) {%k1}
	vpsllq $56, %zmm2, %zmm2
	vpsllq $48, %zmm4, %zmm3
	vpsllq $40, %zmm6, %zmm4
	vpsllq $32, %zmm8, %zmm5
	vpsllq $24, %zmm10, %zmm6
	vpsllq $16, %zmm12, %zmm7
	vpsllq $8, %zmm14, %zmm8
	vporq %zmm16, %zmm8, %zmm8
	vpternlogq $254, %zmm7, %zmm6, %zmm8
	vpternlogq $254, %zmm5, %zmm4, %zmm8
	vpternlogq $254, %zmm3, %zmm2, %zmm8
	kxnorb %k0, %k0, %k1
	vpscatterqq %zmm8, (%r8,%zmm1) {%k1}
	addq $8, %r9
	subq $-128, %rcx
	cmpq %rax, %r9
	jb .LBB0_4
	movl $16, %r8d
	vzeroupper
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
.LBB0_1:
	xorl %r9d, %r9d
	movl $16, %r8d
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
