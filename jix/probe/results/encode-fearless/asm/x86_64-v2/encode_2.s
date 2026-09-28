probe_byte_shuffle_encode_2:
	movabsq $9223372036854775806, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB27_5
	movq %rsi, %rax
	shrq %rax
	shrq $8, %rsi
	je .LBB27_4
	leaq (%rax,%rdx), %rcx
	addq $112, %rcx
	shlq $7, %rsi
	xorl %r8d, %r8d
	movdqa .LCPI27_0(%rip), %xmm0
.LBB27_3:
	movdqu (%rdi,%r8,2), %xmm1
	movdqu 16(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, (%rdx,%r8)
	movdqu %xmm1, -112(%rcx,%r8)
	movdqu 32(%rdi,%r8,2), %xmm1
	movdqu 48(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, 16(%rdx,%r8)
	movdqu %xmm1, -96(%rcx,%r8)
	movdqu 64(%rdi,%r8,2), %xmm1
	movdqu 80(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, 32(%rdx,%r8)
	movdqu %xmm1, -80(%rcx,%r8)
	movdqu 96(%rdi,%r8,2), %xmm1
	movdqu 112(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, 48(%rdx,%r8)
	movdqu %xmm1, -64(%rcx,%r8)
	movdqu 128(%rdi,%r8,2), %xmm1
	movdqu 144(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, 64(%rdx,%r8)
	movdqu %xmm1, -48(%rcx,%r8)
	movdqu 160(%rdi,%r8,2), %xmm1
	movdqu 176(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, 80(%rdx,%r8)
	movdqu %xmm1, -32(%rcx,%r8)
	movdqu 192(%rdi,%r8,2), %xmm1
	movdqu 208(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, 96(%rdx,%r8)
	movdqu %xmm1, -16(%rcx,%r8)
	movdqu 224(%rdi,%r8,2), %xmm1
	movdqu 240(%rdi,%r8,2), %xmm2
	pshufb %xmm0, %xmm1
	pshufb %xmm0, %xmm2
	movdqa %xmm1, %xmm3
	punpcklqdq %xmm2, %xmm3
	punpckhqdq %xmm2, %xmm1
	movdqu %xmm3, 112(%rdx,%r8)
	movdqu %xmm1, (%rcx,%r8)
	subq $-128, %r8
	cmpq %r8, %rsi
	jne .LBB27_3
.LBB27_4:
	movabsq $4611686018427387776, %rcx
	andq %rcx, %rax
	retq
.LBB27_5:
	pushq %rax
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
