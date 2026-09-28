jix_probe::byte_shuffle::decode_impl::<2, 64>:
	movq %rsi, %r8
	shrq %r8
	movabsq $4611686018427387840, %rax
	andq %r8, %rax
	je .LBB1_1
	leaq 48(%rdi), %rcx
	addq %rdi, %r8
	addq $48, %r8
	xorl %r9d, %r9d
.LBB1_4:
	vbroadcasti128 -48(%r8,%r9), %ymm0
	vpermq $244, %ymm0, %ymm0
	vbroadcasti128 -48(%rcx,%r9), %ymm1
	vpermq $244, %ymm1, %ymm1
	vpunpcklbw %ymm0, %ymm1, %ymm0
	vmovdqu %ymm0, (%rdx,%r9,2)
	vbroadcasti128 -32(%r8,%r9), %ymm0
	vpermq $244, %ymm0, %ymm0
	vbroadcasti128 -32(%rcx,%r9), %ymm1
	vpermq $244, %ymm1, %ymm1
	vpunpcklbw %ymm0, %ymm1, %ymm0
	vmovdqu %ymm0, 32(%rdx,%r9,2)
	vbroadcasti128 -16(%r8,%r9), %ymm0
	vpermq $244, %ymm0, %ymm0
	vbroadcasti128 -16(%rcx,%r9), %ymm1
	vpermq $244, %ymm1, %ymm1
	vpunpcklbw %ymm0, %ymm1, %ymm0
	vmovdqu %ymm0, 64(%rdx,%r9,2)
	vbroadcasti128 (%r8,%r9), %ymm0
	vpermq $244, %ymm0, %ymm0
	vbroadcasti128 (%rcx,%r9), %ymm1
	vpermq $244, %ymm1, %ymm1
	vpunpcklbw %ymm0, %ymm1, %ymm0
	vmovdqu %ymm0, 96(%rdx,%r9,2)
	addq $64, %r9
	cmpq %rax, %r9
	jb .LBB1_4
	movl $2, %r8d
	vzeroupper
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
.LBB1_1:
	xorl %r9d, %r9d
	movl $2, %r8d
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
