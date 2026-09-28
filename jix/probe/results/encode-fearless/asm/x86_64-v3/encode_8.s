probe_byte_shuffle_encode_8:
	pushq %r15
	pushq %r14
	pushq %rbx
	subq $64, %rsp
	movabsq $9223372036854775800, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB21_5
	movq %rsi, %rax
	shrq $8, %rax
	je .LBB21_4
	shrq $3, %rsi
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rdx,%rsi,8), %r9
	subq %rsi, %r9
	leaq (%rdx,%rcx,2), %r10
	addq %rdx, %r8
	leaq (%rdx,%rsi,4), %r11
	addq %rdx, %rcx
	leaq (%rdx,%rsi,2), %rbx
	addq %rdx, %rsi
	movq %rax, %r14
	shlq $5, %r14
	xorl %r15d, %r15d
	vmovdqa .LCPI21_2(%rip), %ymm3
.LBB21_3:
	vmovdqu (%rdi,%r15,8), %ymm5
	vmovdqu 32(%rdi,%r15,8), %ymm6
	vmovdqu 64(%rdi,%r15,8), %ymm7
	vmovdqu 96(%rdi,%r15,8), %ymm8
	vmovdqu 128(%rdi,%r15,8), %ymm9
	vmovdqu 160(%rdi,%r15,8), %ymm10
	vmovdqu 192(%rdi,%r15,8), %ymm11
	vmovdqu 224(%rdi,%r15,8), %ymm12
	vmovdqa .LCPI21_0(%rip), %ymm2
	vpshufb %ymm2, %ymm5, %ymm5
	vpshufb %ymm2, %ymm6, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm13
	vperm2i128 $49, %ymm6, %ymm5, %ymm14
	vpshufb %ymm2, %ymm7, %ymm5
	vpshufb %ymm2, %ymm8, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm8
	vperm2i128 $49, %ymm6, %ymm5, %ymm0
	vpshufb %ymm2, %ymm9, %ymm5
	vpshufb %ymm2, %ymm10, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm4
	vperm2i128 $49, %ymm6, %ymm5, %ymm1
	vpshufb %ymm2, %ymm11, %ymm5
	vpshufb %ymm2, %ymm12, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm11
	vperm2i128 $49, %ymm6, %ymm5, %ymm12
	vmovdqa .LCPI21_1(%rip), %ymm15
	vpshufb %ymm15, %ymm14, %ymm5
	vpshufb %ymm3, %ymm13, %ymm6
	vpblendd $170, %ymm5, %ymm6, %ymm6
	vpshufb %ymm15, %ymm0, %ymm5
	vpshufb %ymm3, %ymm8, %ymm7
	vpblendd $170, %ymm5, %ymm7, %ymm7
	vinserti128 $1, %xmm7, %ymm6, %ymm2
	vmovdqu %ymm2, 32(%rsp)
	vperm2i128 $49, %ymm7, %ymm6, %ymm6
	vpshufb %ymm15, %ymm1, %ymm7
	vpshufb %ymm3, %ymm4, %ymm9
	vmovdqa %ymm4, %ymm2
	vpblendd $170, %ymm7, %ymm9, %ymm9
	vpshufb %ymm15, %ymm12, %ymm7
	vpshufb %ymm3, %ymm11, %ymm10
	vpblendd $170, %ymm7, %ymm10, %ymm10
	vinserti128 $1, %xmm10, %ymm9, %ymm5
	vmovdqu %ymm5, (%rsp)
	vperm2i128 $49, %ymm10, %ymm9, %ymm7
	vmovdqa .LCPI21_3(%rip), %ymm4
	vpshufb %ymm4, %ymm14, %ymm10
	vmovdqa .LCPI21_4(%rip), %ymm9
	vpshufb %ymm9, %ymm13, %ymm13
	vpblendd $170, %ymm10, %ymm13, %ymm10
	vpshufb %ymm4, %ymm0, %ymm13
	vpshufb %ymm9, %ymm8, %ymm8
	vpblendd $170, %ymm13, %ymm8, %ymm13
	vinserti128 $1, %xmm13, %ymm10, %ymm8
	vperm2i128 $49, %ymm13, %ymm10, %ymm10
	vpshufb %ymm4, %ymm1, %ymm1
	vpshufb %ymm9, %ymm2, %ymm0
	vpblendd $170, %ymm1, %ymm0, %ymm0
	vpshufb %ymm4, %ymm12, %ymm1
	vpshufb %ymm9, %ymm11, %ymm11
	vpblendd $170, %ymm1, %ymm11, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm11
	vperm2i128 $49, %ymm1, %ymm0, %ymm13
	vpshufb %ymm15, %ymm6, %ymm0
	vmovdqu 32(%rsp), %ymm2
	vpshufb %ymm3, %ymm2, %ymm1
	vpblendd $170, %ymm0, %ymm1, %ymm0
	vpshufb %ymm15, %ymm7, %ymm1
	vpshufb %ymm3, %ymm5, %ymm12
	vpblendd $170, %ymm1, %ymm12, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm12
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm12, %ymm1
	vpunpckhqdq %ymm0, %ymm12, %ymm12
	vpshufb %ymm15, %ymm10, %ymm0
	vpshufb %ymm3, %ymm8, %ymm14
	vpblendd $170, %ymm0, %ymm14, %ymm0
	vpshufb %ymm15, %ymm13, %ymm14
	vpshufb %ymm3, %ymm11, %ymm15
	vpblendd $170, %ymm14, %ymm15, %ymm14
	vinserti128 $1, %xmm14, %ymm0, %ymm15
	vperm2i128 $49, %ymm14, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm15, %ymm14
	vpunpckhqdq %ymm0, %ymm15, %ymm0
	vpshufb %ymm4, %ymm6, %ymm6
	vpshufb %ymm9, %ymm2, %ymm5
	vpblendd $170, %ymm6, %ymm5, %ymm5
	vpshufb %ymm4, %ymm7, %ymm6
	vmovdqu (%rsp), %ymm2
	vpshufb %ymm9, %ymm2, %ymm7
	vpblendd $170, %ymm6, %ymm7, %ymm6
	vinserti128 $1, %xmm6, %ymm5, %ymm7
	vperm2i128 $49, %ymm6, %ymm5, %ymm5
	vpunpcklqdq %ymm5, %ymm7, %ymm6
	vpunpckhqdq %ymm5, %ymm7, %ymm5
	vpshufb %ymm4, %ymm10, %ymm7
	vpshufb %ymm9, %ymm8, %ymm8
	vpblendd $170, %ymm7, %ymm8, %ymm7
	vpshufb %ymm4, %ymm13, %ymm8
	vpshufb %ymm9, %ymm11, %ymm9
	vpblendd $170, %ymm8, %ymm9, %ymm8
	vinserti128 $1, %xmm8, %ymm7, %ymm9
	vperm2i128 $49, %ymm8, %ymm7, %ymm7
	vpunpcklqdq %ymm7, %ymm9, %ymm8
	vpunpckhqdq %ymm7, %ymm9, %ymm7
	vmovdqu %ymm1, (%rdx,%r15)
	vmovdqu %ymm14, (%rsi,%r15)
	vmovdqu %ymm6, (%rbx,%r15)
	vmovdqu %ymm8, (%rcx,%r15)
	vmovdqu %ymm12, (%r11,%r15)
	vmovdqu %ymm0, (%r8,%r15)
	vmovdqu %ymm5, (%r10,%r15)
	vmovdqu %ymm7, (%r9,%r15)
	addq $32, %r15
	cmpq %r15, %r14
	jne .LBB21_3
.LBB21_4:
	shlq $5, %rax
	addq $64, %rsp
	popq %rbx
	popq %r14
	popq %r15
	vzeroupper
	retq
.LBB21_5:
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
