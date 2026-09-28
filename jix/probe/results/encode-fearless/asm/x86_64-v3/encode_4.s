probe_byte_shuffle_encode_4:
	movabsq $9223372036854775804, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB20_5
	movq %rsi, %rax
	shrq $2, %rax
	shrq $8, %rsi
	je .LBB20_4
	leaq (%rax,%rax,2), %rcx
	addq %rdx, %rcx
	addq $32, %rcx
	leaq (%rdx,%rax,2), %r8
	addq $32, %r8
	leaq (%rax,%rdx), %r9
	addq $32, %r9
	shlq $6, %rsi
	xorl %r10d, %r10d
	vmovdqa .LCPI20_0(%rip), %ymm0
	vmovdqa .LCPI20_1(%rip), %ymm1
	vmovdqa .LCPI20_2(%rip), %ymm2
	vmovdqa .LCPI20_3(%rip), %ymm3
	vmovdqa .LCPI20_4(%rip), %ymm4
.LBB20_3:
	vmovdqu (%rdi,%r10,4), %ymm5
	vmovdqu 32(%rdi,%r10,4), %ymm6
	vmovdqu 64(%rdi,%r10,4), %ymm7
	vmovdqu 96(%rdi,%r10,4), %ymm8
	vpshufb %ymm0, %ymm5, %ymm5
	vpshufb %ymm0, %ymm6, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm9
	vperm2i128 $49, %ymm6, %ymm5, %ymm5
	vpshufb %ymm0, %ymm7, %ymm6
	vpshufb %ymm0, %ymm8, %ymm7
	vinserti128 $1, %xmm7, %ymm6, %ymm8
	vperm2i128 $49, %ymm7, %ymm6, %ymm6
	vpshufb %ymm1, %ymm5, %ymm7
	vpshufb %ymm2, %ymm9, %ymm10
	vpblendd $170, %ymm7, %ymm10, %ymm7
	vpshufb %ymm1, %ymm6, %ymm10
	vpshufb %ymm2, %ymm8, %ymm11
	vpblendd $170, %ymm10, %ymm11, %ymm10
	vinserti128 $1, %xmm10, %ymm7, %ymm11
	vperm2i128 $49, %ymm10, %ymm7, %ymm7
	vpunpcklqdq %ymm7, %ymm11, %ymm10
	vpunpckhqdq %ymm7, %ymm11, %ymm7
	vpshufb %ymm3, %ymm5, %ymm5
	vpshufb %ymm4, %ymm9, %ymm9
	vpblendd $170, %ymm5, %ymm9, %ymm5
	vpshufb %ymm3, %ymm6, %ymm6
	vpshufb %ymm4, %ymm8, %ymm8
	vpblendd $170, %ymm6, %ymm8, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm8
	vperm2i128 $49, %ymm6, %ymm5, %ymm5
	vpunpcklqdq %ymm5, %ymm8, %ymm6
	vpunpckhqdq %ymm5, %ymm8, %ymm5
	vmovdqu %ymm10, (%rdx,%r10)
	vmovdqu %ymm6, -32(%r9,%r10)
	vmovdqu %ymm7, -32(%r8,%r10)
	vmovdqu %ymm5, -32(%rcx,%r10)
	vmovdqu 128(%rdi,%r10,4), %ymm5
	vmovdqu 160(%rdi,%r10,4), %ymm6
	vmovdqu 192(%rdi,%r10,4), %ymm7
	vmovdqu 224(%rdi,%r10,4), %ymm8
	vpshufb %ymm0, %ymm5, %ymm5
	vpshufb %ymm0, %ymm6, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm9
	vperm2i128 $49, %ymm6, %ymm5, %ymm5
	vpshufb %ymm0, %ymm7, %ymm6
	vpshufb %ymm0, %ymm8, %ymm7
	vinserti128 $1, %xmm7, %ymm6, %ymm8
	vperm2i128 $49, %ymm7, %ymm6, %ymm6
	vpshufb %ymm1, %ymm5, %ymm7
	vpshufb %ymm2, %ymm9, %ymm10
	vpblendd $170, %ymm7, %ymm10, %ymm7
	vpshufb %ymm1, %ymm6, %ymm10
	vpshufb %ymm2, %ymm8, %ymm11
	vpblendd $170, %ymm10, %ymm11, %ymm10
	vinserti128 $1, %xmm10, %ymm7, %ymm11
	vperm2i128 $49, %ymm10, %ymm7, %ymm7
	vpunpcklqdq %ymm7, %ymm11, %ymm10
	vpunpckhqdq %ymm7, %ymm11, %ymm7
	vpshufb %ymm3, %ymm5, %ymm5
	vpshufb %ymm4, %ymm9, %ymm9
	vpblendd $170, %ymm5, %ymm9, %ymm5
	vpshufb %ymm3, %ymm6, %ymm6
	vpshufb %ymm4, %ymm8, %ymm8
	vpblendd $170, %ymm6, %ymm8, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm8
	vperm2i128 $49, %ymm6, %ymm5, %ymm5
	vpunpcklqdq %ymm5, %ymm8, %ymm6
	vpunpckhqdq %ymm5, %ymm8, %ymm5
	vmovdqu %ymm10, 32(%rdx,%r10)
	vmovdqu %ymm6, (%r9,%r10)
	vmovdqu %ymm7, (%r8,%r10)
	vmovdqu %ymm5, (%rcx,%r10)
	addq $64, %r10
	cmpq %r10, %rsi
	jne .LBB20_3
.LBB20_4:
	movabsq $2305843009213693888, %rcx
	andq %rcx, %rax
	vzeroupper
	retq
.LBB20_5:
	pushq %rax
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
