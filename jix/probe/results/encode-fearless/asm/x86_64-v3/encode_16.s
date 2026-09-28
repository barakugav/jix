probe_byte_shuffle_encode_16:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $936, %rsp
	movabsq $9223372036854775792, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB18_5
	movq %rsi, %rax
	shrq $9, %rax
	movq %rax, 8(%rsp)
	je .LBB18_4
	shrq $4, %rsi
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rsi,4), %r8
	leaq (%rsi,%rsi,8), %r9
	leaq (%rsi,%r8,2), %r10
	leaq (%rsi,%rcx,4), %r11
	movq %rsi, %rbx
	shlq $4, %rbx
	subq %rsi, %rbx
	subq %rsi, %rbx
	leaq (%r8,%r8,2), %r14
	addq %rdx, %r14
	addq %rdx, %rbx
	addq %rdx, %r11
	movq %r11, 136(%rsp)
	leaq (%rdx,%rcx,4), %rax
	movq %rax, 128(%rsp)
	addq %rdx, %r10
	movq %r10, 152(%rsp)
	leaq (%rdx,%r8,2), %rax
	movq %rax, 120(%rsp)
	addq %rdx, %r9
	movq %r9, 144(%rsp)
	leaq (%rdx,%rsi,8), %r13
	movq %r13, %rbp
	subq %rsi, %rbp
	leaq (%rdx,%rcx,2), %r11
	addq %rdx, %r8
	leaq (%rdx,%rsi,4), %rax
	addq %rdx, %rcx
	leaq (%rdx,%rsi,2), %r12
	addq %rdx, %rsi
	addq $480, %rdi
	movq 8(%rsp), %r9
	shlq $5, %r9
	xorl %r10d, %r10d
.LBB18_3:
	vmovups (%rdi), %ymm0
	vmovups %ymm0, 80(%rsp)
	vmovups -64(%rdi), %ymm0
	vmovups %ymm0, 48(%rsp)
	vmovups -96(%rdi), %ymm0
	vmovups %ymm0, 16(%rsp)
	vmovups -128(%rdi), %ymm0
	vmovups %ymm0, 288(%rsp)
	vmovdqu -160(%rdi), %ymm4
	vmovdqu -192(%rdi), %ymm3
	vmovdqu -224(%rdi), %ymm0
	vmovdqu -256(%rdi), %ymm5
	vmovdqu -288(%rdi), %ymm13
	vmovdqu -320(%rdi), %ymm8
	vmovdqu -352(%rdi), %ymm1
	vmovdqu -480(%rdi), %ymm7
	vmovdqu -448(%rdi), %ymm11
	vmovdqu -416(%rdi), %ymm15
	vmovdqu -384(%rdi), %ymm10
	vmovdqa .LCPI18_0(%rip), %ymm12
	vpshufb %ymm12, %ymm7, %ymm7
	vpshufb %ymm12, %ymm11, %ymm11
	vinserti128 $1, %xmm11, %ymm7, %ymm14
	vmovdqu %ymm14, 384(%rsp)
	vperm2i128 $49, %ymm11, %ymm7, %ymm11
	vpshufb %ymm12, %ymm15, %ymm7
	vpshufb %ymm12, %ymm10, %ymm10
	vinserti128 $1, %xmm10, %ymm7, %ymm9
	vmovdqu %ymm9, 256(%rsp)
	vperm2i128 $49, %ymm10, %ymm7, %ymm2
	vmovdqu %ymm2, 608(%rsp)
	vpshufb %ymm12, %ymm1, %ymm1
	vpshufb %ymm12, %ymm8, %ymm8
	vinserti128 $1, %xmm8, %ymm1, %ymm6
	vmovdqu %ymm6, 352(%rsp)
	vperm2i128 $49, %ymm8, %ymm1, %ymm8
	vpshufb %ymm12, %ymm13, %ymm1
	vpshufb %ymm12, %ymm5, %ymm5
	vinserti128 $1, %xmm5, %ymm1, %ymm10
	vmovdqu %ymm10, 224(%rsp)
	vperm2i128 $49, %ymm5, %ymm1, %ymm1
	vmovdqu %ymm1, 512(%rsp)
	vpshufb %ymm12, %ymm0, %ymm0
	vpshufb %ymm12, %ymm3, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm3
	vmovdqu %ymm3, 320(%rsp)
	vperm2i128 $49, %ymm1, %ymm0, %ymm6
	vmovdqu %ymm6, 192(%rsp)
	vpshufb %ymm12, %ymm4, %ymm0
	vmovdqu 288(%rsp), %ymm1
	vpshufb %ymm12, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm7
	vmovdqu %ymm7, 288(%rsp)
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vmovdqu %ymm0, 480(%rsp)
	vmovdqu 16(%rsp), %ymm0
	vpshufb %ymm12, %ymm0, %ymm0
	vmovdqu 48(%rsp), %ymm1
	vpshufb %ymm12, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm15
	vperm2i128 $49, %ymm1, %ymm0, %ymm4
	vmovdqu %ymm4, 160(%rsp)
	vmovdqu -32(%rdi), %ymm0
	vpshufb %ymm12, %ymm0, %ymm0
	vmovdqu 80(%rsp), %ymm1
	vpshufb %ymm12, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm5
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vmovdqu %ymm0, 576(%rsp)
	vmovdqa .LCPI18_1(%rip), %ymm12
	vpshufb %ymm12, %ymm11, %ymm1
	vmovdqa .LCPI18_2(%rip), %ymm3
	vpshufb %ymm3, %ymm14, %ymm13
	vpblendd $170, %ymm1, %ymm13, %ymm1
	vpshufb %ymm12, %ymm2, %ymm13
	vpshufb %ymm3, %ymm9, %ymm14
	vpblendd $170, %ymm13, %ymm14, %ymm13
	vinserti128 $1, %xmm13, %ymm1, %ymm2
	vmovdqu %ymm2, 640(%rsp)
	vperm2i128 $49, %ymm13, %ymm1, %ymm1
	vmovdqu %ymm1, 544(%rsp)
	vpshufb %ymm12, %ymm8, %ymm1
	vmovdqu 352(%rsp), %ymm2
	vpshufb %ymm3, %ymm2, %ymm13
	vpblendd $170, %ymm1, %ymm13, %ymm1
	vmovdqu 512(%rsp), %ymm9
	vpshufb %ymm12, %ymm9, %ymm13
	vpshufb %ymm3, %ymm10, %ymm14
	vpblendd $170, %ymm13, %ymm14, %ymm13
	vinserti128 $1, %xmm13, %ymm1, %ymm2
	vmovdqu %ymm2, 48(%rsp)
	vperm2i128 $49, %ymm13, %ymm1, %ymm1
	vmovdqu %ymm1, 448(%rsp)
	vpshufb %ymm12, %ymm6, %ymm1
	vmovdqu 320(%rsp), %ymm2
	vpshufb %ymm3, %ymm2, %ymm13
	vpblendd $170, %ymm1, %ymm13, %ymm1
	vmovdqu 480(%rsp), %ymm6
	vpshufb %ymm12, %ymm6, %ymm13
	vpshufb %ymm3, %ymm7, %ymm14
	vpblendd $170, %ymm13, %ymm14, %ymm13
	vinserti128 $1, %xmm13, %ymm1, %ymm2
	vmovdqu %ymm2, 416(%rsp)
	vperm2i128 $49, %ymm13, %ymm1, %ymm1
	vmovdqu %ymm1, 896(%rsp)
	vpshufb %ymm12, %ymm4, %ymm1
	vpshufb %ymm3, %ymm15, %ymm13
	vmovdqa %ymm15, %ymm10
	vpblendd $170, %ymm1, %ymm13, %ymm1
	vpshufb %ymm12, %ymm0, %ymm13
	vpshufb %ymm3, %ymm5, %ymm14
	vmovdqa %ymm5, %ymm15
	vpblendd $170, %ymm13, %ymm14, %ymm13
	vinserti128 $1, %xmm13, %ymm1, %ymm0
	vmovdqu %ymm0, 768(%rsp)
	vperm2i128 $49, %ymm13, %ymm1, %ymm0
	vmovdqu %ymm0, 736(%rsp)
	vmovdqa .LCPI18_3(%rip), %ymm4
	vpshufb %ymm4, %ymm11, %ymm1
	vmovdqa .LCPI18_4(%rip), %ymm2
	vmovdqu 384(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm11
	vpblendd $170, %ymm1, %ymm11, %ymm1
	vmovdqu 608(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm7
	vmovdqu 256(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm11
	vpblendd $170, %ymm7, %ymm11, %ymm7
	vinserti128 $1, %xmm7, %ymm1, %ymm11
	vmovdqu %ymm11, 80(%rsp)
	vperm2i128 $49, %ymm7, %ymm1, %ymm0
	vmovdqu %ymm0, 704(%rsp)
	vpshufb %ymm4, %ymm8, %ymm7
	vmovdqu 352(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm8
	vpblendd $170, %ymm7, %ymm8, %ymm7
	vpshufb %ymm4, %ymm9, %ymm5
	vmovdqu 224(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm8
	vpblendd $170, %ymm5, %ymm8, %ymm5
	vinserti128 $1, %xmm5, %ymm7, %ymm1
	vmovdqu %ymm1, 16(%rsp)
	vperm2i128 $49, %ymm5, %ymm7, %ymm1
	vmovdqu %ymm1, 256(%rsp)
	vmovdqu 192(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm5
	vmovdqu 320(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm7
	vpblendd $170, %ymm5, %ymm7, %ymm7
	vpshufb %ymm4, %ymm6, %ymm5
	vmovdqu 288(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm6
	vpblendd $170, %ymm5, %ymm6, %ymm6
	vinserti128 $1, %xmm6, %ymm7, %ymm1
	vmovdqu %ymm1, 224(%rsp)
	vperm2i128 $49, %ymm6, %ymm7, %ymm1
	vmovdqu %ymm1, 192(%rsp)
	vmovdqu 160(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm6
	vpshufb %ymm2, %ymm10, %ymm7
	vpblendd $170, %ymm6, %ymm7, %ymm6
	vmovdqu 576(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm0
	vpshufb %ymm2, %ymm15, %ymm7
	vpblendd $170, %ymm0, %ymm7, %ymm0
	vinserti128 $1, %xmm0, %ymm6, %ymm1
	vmovdqu %ymm1, 160(%rsp)
	vperm2i128 $49, %ymm0, %ymm6, %ymm0
	vmovdqu %ymm0, 672(%rsp)
	vmovdqu 544(%rsp), %ymm15
	vpshufb %ymm12, %ymm15, %ymm6
	vmovdqu 640(%rsp), %ymm14
	vpshufb %ymm3, %ymm14, %ymm10
	vpblendd $170, %ymm6, %ymm10, %ymm6
	vmovdqu 448(%rsp), %ymm11
	vpshufb %ymm12, %ymm11, %ymm10
	vmovdqu 48(%rsp), %ymm1
	vpshufb %ymm3, %ymm1, %ymm13
	vpblendd $170, %ymm10, %ymm13, %ymm10
	vinserti128 $1, %xmm10, %ymm6, %ymm1
	vmovdqu %ymm1, 608(%rsp)
	vperm2i128 $49, %ymm10, %ymm6, %ymm1
	vmovdqu %ymm1, 576(%rsp)
	vmovdqu 896(%rsp), %ymm7
	vpshufb %ymm12, %ymm7, %ymm6
	vmovdqu 416(%rsp), %ymm5
	vpshufb %ymm3, %ymm5, %ymm10
	vpblendd $170, %ymm6, %ymm10, %ymm6
	vmovdqu 736(%rsp), %ymm9
	vpshufb %ymm12, %ymm9, %ymm10
	vmovdqu 768(%rsp), %ymm8
	vpshufb %ymm3, %ymm8, %ymm13
	vpblendd $170, %ymm10, %ymm13, %ymm10
	vinserti128 $1, %xmm10, %ymm6, %ymm1
	vmovdqu %ymm1, 512(%rsp)
	vperm2i128 $49, %ymm10, %ymm6, %ymm6
	vmovdqu %ymm6, 384(%rsp)
	vmovdqu 704(%rsp), %ymm1
	vpshufb %ymm12, %ymm1, %ymm6
	vmovdqu 80(%rsp), %ymm10
	vpshufb %ymm3, %ymm10, %ymm10
	vpblendd $170, %ymm6, %ymm10, %ymm6
	vmovdqu 256(%rsp), %ymm10
	vpshufb %ymm12, %ymm10, %ymm10
	vmovdqu 16(%rsp), %ymm13
	vpshufb %ymm3, %ymm13, %ymm13
	vpblendd $170, %ymm10, %ymm13, %ymm10
	vinserti128 $1, %xmm10, %ymm6, %ymm13
	vmovdqu %ymm13, 480(%rsp)
	vperm2i128 $49, %ymm10, %ymm6, %ymm6
	vmovdqu %ymm6, 288(%rsp)
	vmovdqu 192(%rsp), %ymm6
	vpshufb %ymm12, %ymm6, %ymm6
	vmovdqu 224(%rsp), %ymm10
	vpshufb %ymm3, %ymm10, %ymm10
	vpblendd $170, %ymm6, %ymm10, %ymm6
	vpshufb %ymm12, %ymm0, %ymm10
	vmovdqu 160(%rsp), %ymm13
	vpshufb %ymm3, %ymm13, %ymm13
	vpblendd $170, %ymm10, %ymm13, %ymm10
	vinserti128 $1, %xmm10, %ymm6, %ymm0
	vmovdqu %ymm0, 864(%rsp)
	vperm2i128 $49, %ymm10, %ymm6, %ymm0
	vmovdqu %ymm0, 800(%rsp)
	vpshufb %ymm4, %ymm15, %ymm6
	vpshufb %ymm2, %ymm14, %ymm10
	vpblendd $170, %ymm6, %ymm10, %ymm6
	vpshufb %ymm4, %ymm11, %ymm10
	vmovdqu 48(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm13
	vpblendd $170, %ymm10, %ymm13, %ymm10
	vinserti128 $1, %xmm10, %ymm6, %ymm0
	vmovdqu %ymm0, 832(%rsp)
	vperm2i128 $49, %ymm10, %ymm6, %ymm0
	vmovdqu %ymm0, 352(%rsp)
	vpshufb %ymm4, %ymm7, %ymm6
	vpshufb %ymm2, %ymm5, %ymm13
	vpblendd $170, %ymm6, %ymm13, %ymm6
	vpshufb %ymm4, %ymm9, %ymm13
	vpshufb %ymm2, %ymm8, %ymm14
	vpblendd $170, %ymm13, %ymm14, %ymm14
	vinserti128 $1, %xmm14, %ymm6, %ymm0
	vmovdqu %ymm0, 320(%rsp)
	vperm2i128 $49, %ymm14, %ymm6, %ymm0
	vmovdqu %ymm0, 48(%rsp)
	vpshufb %ymm4, %ymm1, %ymm1
	vmovdqu 80(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm6
	vpblendd $170, %ymm1, %ymm6, %ymm1
	vmovdqu 256(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm6
	vmovdqu 16(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm8
	vpblendd $170, %ymm6, %ymm8, %ymm6
	vinserti128 $1, %xmm6, %ymm1, %ymm0
	vmovdqu %ymm0, 80(%rsp)
	vperm2i128 $49, %ymm6, %ymm1, %ymm0
	vmovdqu %ymm0, 16(%rsp)
	vmovdqu 192(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm1
	vmovdqu 224(%rsp), %ymm0
	vpshufb %ymm2, %ymm0, %ymm5
	vpblendd $170, %ymm1, %ymm5, %ymm1
	vmovdqu 672(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm0
	vmovdqu 160(%rsp), %ymm5
	vpshufb %ymm2, %ymm5, %ymm5
	vpblendd $170, %ymm0, %ymm5, %ymm0
	vinserti128 $1, %xmm0, %ymm1, %ymm7
	vperm2i128 $49, %ymm0, %ymm1, %ymm6
	vmovdqu 576(%rsp), %ymm10
	vpshufb %ymm12, %ymm10, %ymm0
	vmovdqu 608(%rsp), %ymm8
	vpshufb %ymm3, %ymm8, %ymm1
	vpblendd $170, %ymm0, %ymm1, %ymm0
	vmovdqu 384(%rsp), %ymm1
	vpshufb %ymm12, %ymm1, %ymm1
	vmovdqu 512(%rsp), %ymm13
	vpshufb %ymm3, %ymm13, %ymm5
	vpblendd $170, %ymm1, %ymm5, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm5
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm5, %ymm1
	vmovdqu %ymm1, 224(%rsp)
	vpunpckhqdq %ymm0, %ymm5, %ymm0
	vmovdqu %ymm0, 256(%rsp)
	vmovdqu 288(%rsp), %ymm0
	vpshufb %ymm12, %ymm0, %ymm0
	vmovdqu 480(%rsp), %ymm9
	vpshufb %ymm3, %ymm9, %ymm1
	vpblendd $170, %ymm0, %ymm1, %ymm0
	vmovdqu 800(%rsp), %ymm15
	vpshufb %ymm12, %ymm15, %ymm1
	vmovdqu 864(%rsp), %ymm11
	vpshufb %ymm3, %ymm11, %ymm5
	vpblendd $170, %ymm1, %ymm5, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm5
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm5, %ymm1
	vmovdqu %ymm1, 160(%rsp)
	vpunpckhqdq %ymm0, %ymm5, %ymm0
	vmovdqu %ymm0, 192(%rsp)
	vmovdqu 352(%rsp), %ymm0
	vpshufb %ymm12, %ymm0, %ymm0
	vmovdqu 832(%rsp), %ymm14
	vpshufb %ymm3, %ymm14, %ymm1
	vpblendd $170, %ymm0, %ymm1, %ymm0
	vmovdqu 48(%rsp), %ymm1
	vpshufb %ymm12, %ymm1, %ymm1
	vmovdqu 320(%rsp), %ymm5
	vpshufb %ymm3, %ymm5, %ymm5
	vpblendd $170, %ymm1, %ymm5, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm5
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm5, %ymm1
	vmovdqu %ymm1, 544(%rsp)
	vpunpckhqdq %ymm0, %ymm5, %ymm0
	vmovdqu %ymm0, 640(%rsp)
	vmovdqu 16(%rsp), %ymm0
	vpshufb %ymm12, %ymm0, %ymm0
	vmovdqu 80(%rsp), %ymm1
	vpshufb %ymm3, %ymm1, %ymm1
	vpblendd $170, %ymm0, %ymm1, %ymm0
	vpshufb %ymm12, %ymm6, %ymm1
	vpshufb %ymm3, %ymm7, %ymm5
	vpblendd $170, %ymm1, %ymm5, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm5
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm5, %ymm1
	vmovdqu %ymm1, 416(%rsp)
	vpunpckhqdq %ymm0, %ymm5, %ymm0
	vmovdqu %ymm0, 448(%rsp)
	vpshufb %ymm4, %ymm10, %ymm0
	vpshufb %ymm2, %ymm8, %ymm5
	vpblendd $170, %ymm0, %ymm5, %ymm0
	vmovdqu 384(%rsp), %ymm1
	vpshufb %ymm4, %ymm1, %ymm5
	vpshufb %ymm2, %ymm13, %ymm8
	vpblendd $170, %ymm5, %ymm8, %ymm5
	vinserti128 $1, %xmm5, %ymm0, %ymm8
	vperm2i128 $49, %ymm5, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm8, %ymm5
	vpunpckhqdq %ymm0, %ymm8, %ymm8
	vmovdqu 288(%rsp), %ymm0
	vpshufb %ymm4, %ymm0, %ymm0
	vpshufb %ymm2, %ymm9, %ymm1
	vpblendd $170, %ymm0, %ymm1, %ymm0
	vpshufb %ymm4, %ymm15, %ymm1
	vpshufb %ymm2, %ymm11, %ymm10
	vpblendd $170, %ymm1, %ymm10, %ymm1
	vinserti128 $1, %xmm1, %ymm0, %ymm10
	vperm2i128 $49, %ymm1, %ymm0, %ymm0
	vpunpcklqdq %ymm0, %ymm10, %ymm1
	vpunpckhqdq %ymm0, %ymm10, %ymm0
	vmovdqu 352(%rsp), %ymm3
	vpshufb %ymm4, %ymm3, %ymm10
	vpshufb %ymm2, %ymm14, %ymm13
	vpblendd $170, %ymm10, %ymm13, %ymm10
	vmovdqu 48(%rsp), %ymm3
	vpshufb %ymm4, %ymm3, %ymm13
	vmovdqu 320(%rsp), %ymm3
	vpshufb %ymm2, %ymm3, %ymm15
	vpblendd $170, %ymm13, %ymm15, %ymm13
	vinserti128 $1, %xmm13, %ymm10, %ymm15
	vperm2i128 $49, %ymm13, %ymm10, %ymm10
	vpunpcklqdq %ymm10, %ymm15, %ymm13
	vpunpckhqdq %ymm10, %ymm15, %ymm10
	vmovdqu 16(%rsp), %ymm3
	vpshufb %ymm4, %ymm3, %ymm14
	vmovdqu 80(%rsp), %ymm3
	vpshufb %ymm2, %ymm3, %ymm11
	vpblendd $170, %ymm14, %ymm11, %ymm11
	vpshufb %ymm4, %ymm6, %ymm6
	vpshufb %ymm2, %ymm7, %ymm7
	vpblendd $170, %ymm6, %ymm7, %ymm6
	vinserti128 $1, %xmm6, %ymm11, %ymm7
	vperm2i128 $49, %ymm6, %ymm11, %ymm6
	vpunpcklqdq %ymm6, %ymm7, %ymm11
	vpunpckhqdq %ymm6, %ymm7, %ymm6
	vmovups 224(%rsp), %ymm2
	vmovups %ymm2, (%rdx,%r10)
	vmovups 160(%rsp), %ymm2
	vmovups %ymm2, (%rsi,%r10)
	vmovups 544(%rsp), %ymm2
	vmovups %ymm2, (%r12,%r10)
	vmovups 416(%rsp), %ymm2
	vmovups %ymm2, (%rcx,%r10)
	vmovdqu %ymm5, (%rax,%r10)
	vmovdqu %ymm1, (%r8,%r10)
	vmovdqu %ymm13, (%r11,%r10)
	vmovdqu %ymm11, (%rbp,%r10)
	vmovups 256(%rsp), %ymm1
	vmovups %ymm1, (%r13,%r10)
	movq 144(%rsp), %r15
	vmovups 192(%rsp), %ymm1
	vmovups %ymm1, (%r15,%r10)
	movq 120(%rsp), %r15
	vmovups 640(%rsp), %ymm1
	vmovups %ymm1, (%r15,%r10)
	movq 152(%rsp), %r15
	vmovups 448(%rsp), %ymm1
	vmovups %ymm1, (%r15,%r10)
	movq 128(%rsp), %r15
	vmovdqu %ymm8, (%r15,%r10)
	movq 136(%rsp), %r15
	vmovdqu %ymm0, (%r15,%r10)
	vmovdqu %ymm10, (%rbx,%r10)
	vmovdqu %ymm6, (%r14,%r10)
	addq $32, %r10
	addq $512, %rdi
	cmpq %r10, %r9
	jne .LBB18_3
.LBB18_4:
	movq 8(%rsp), %rax
	shlq $5, %rax
	addq $936, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB18_5:
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
