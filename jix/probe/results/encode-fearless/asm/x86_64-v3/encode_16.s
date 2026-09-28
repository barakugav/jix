probe_byte_shuffle_encode_16:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $152, %rsp
	movabsq $9223372036854775792, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB18_7
	movq %rsi, %rax
	shrq $9, %rax
	movq %rax, (%rsp)
	je .LBB18_6
	shrq $4, %rsi
	leaq (%rsi,%rsi,2), %r11
	leaq (%rsi,%rsi,4), %r8
	leaq (%rsi,%rsi), %rax
	leaq (%rax,%rax,2), %r9
	leaq (,%rsi,8), %r10
	subq %rsi, %r10
	leaq (%rax,%rax,4), %rbx
	leaq (,%rsi,4), %rcx
	leaq (%rcx,%rcx,2), %r15
	movq %r11, %r14
	leaq (%rsi,%r11,4), %rcx
	movq %rcx, 8(%rsp)
	movq %rsi, %r13
	shlq $4, %r13
	subq %rax, %r13
	leaq (%r8,%r8,2), %rbp
	xorl %ecx, %ecx
	movq (%rsp), %rax
.LBB18_3:
	movq %rcx, %r11
	shlq $9, %r11
	vmovdqu (%rdi,%r11), %ymm1
	vmovdqu 32(%rdi,%r11), %ymm15
	vmovups 64(%rdi,%r11), %ymm0
	vmovups %ymm0, 80(%rsp)
	vmovdqu 96(%rdi,%r11), %ymm14
	vmovups 128(%rdi,%r11), %ymm0
	vmovups %ymm0, 16(%rsp)
	vmovdqu 160(%rdi,%r11), %ymm4
	vmovups 192(%rdi,%r11), %ymm0
	vmovups %ymm0, 48(%rsp)
	vmovdqu 224(%rdi,%r11), %ymm0
	vmovdqu 256(%rdi,%r11), %ymm13
	vmovdqu 288(%rdi,%r11), %ymm11
	vmovdqu 320(%rdi,%r11), %ymm10
	vmovdqu 352(%rdi,%r11), %ymm7
	vmovdqu 384(%rdi,%r11), %ymm8
	vmovdqu 416(%rdi,%r11), %ymm2
	vmovdqu 448(%rdi,%r11), %ymm6
	vmovups 480(%rdi,%r11), %ymm3
	vmovups %ymm3, 112(%rsp)
	movl $5, %r11d
.LBB18_4:
	vpunpcklbw %ymm13, %ymm1, %ymm3
	vpunpckhbw %ymm13, %ymm1, %ymm5
	vinserti128 $1, %xmm5, %ymm3, %ymm1
	vperm2i128 $49, %ymm5, %ymm3, %ymm5
	vpunpcklbw %ymm11, %ymm15, %ymm3
	vpunpckhbw %ymm11, %ymm15, %ymm9
	vmovdqu 80(%rsp), %ymm11
	vpunpcklbw %ymm10, %ymm11, %ymm12
	vpunpckhbw %ymm10, %ymm11, %ymm10
	vinserti128 $1, %xmm9, %ymm3, %ymm11
	vmovdqu %ymm11, 80(%rsp)
	vperm2i128 $49, %ymm9, %ymm3, %ymm3
	vmovdqu 16(%rsp), %ymm9
	vpunpcklbw %ymm8, %ymm9, %ymm11
	vpunpckhbw %ymm8, %ymm9, %ymm8
	vinserti128 $1, %xmm10, %ymm12, %ymm9
	vmovdqu %ymm9, 16(%rsp)
	vperm2i128 $49, %ymm10, %ymm12, %ymm12
	vpunpcklbw %ymm7, %ymm14, %ymm9
	vpunpckhbw %ymm7, %ymm14, %ymm7
	vmovdqu 48(%rsp), %ymm10
	vpunpcklbw %ymm6, %ymm10, %ymm14
	vpunpckhbw %ymm6, %ymm10, %ymm6
	vinserti128 $1, %xmm7, %ymm9, %ymm10
	vmovdqu %ymm10, 48(%rsp)
	vperm2i128 $49, %ymm7, %ymm9, %ymm9
	vinserti128 $1, %xmm8, %ymm11, %ymm13
	vperm2i128 $49, %ymm8, %ymm11, %ymm11
	vpunpcklbw %ymm2, %ymm4, %ymm7
	vpunpckhbw %ymm2, %ymm4, %ymm2
	vinserti128 $1, %xmm2, %ymm7, %ymm10
	vperm2i128 $49, %ymm2, %ymm7, %ymm7
	vinserti128 $1, %xmm6, %ymm14, %ymm8
	vperm2i128 $49, %ymm6, %ymm14, %ymm2
	vmovdqu 112(%rsp), %ymm6
	vpunpcklbw %ymm6, %ymm0, %ymm4
	vpunpckhbw %ymm6, %ymm0, %ymm0
	vinserti128 $1, %xmm0, %ymm4, %ymm6
	vperm2i128 $49, %ymm0, %ymm4, %ymm0
	vmovdqu %ymm0, 112(%rsp)
	vmovdqa %ymm9, %ymm0
	vmovdqa %ymm12, %ymm4
	vmovdqa %ymm3, %ymm14
	vmovdqa %ymm5, %ymm15
	decl %r11d
	jne .LBB18_4
	movq %rcx, %r11
	shlq $5, %r11
	vmovdqu %ymm1, (%rdx,%r11)
	addq %rdx, %r11
	vmovdqu %ymm5, (%rsi,%r11)
	vmovups 80(%rsp), %ymm0
	vmovups %ymm0, (%r11,%rsi,2)
	vmovdqu %ymm3, (%r14,%r11)
	vmovups 16(%rsp), %ymm0
	vmovups %ymm0, (%r11,%rsi,4)
	vmovdqu %ymm12, (%r8,%r11)
	vmovups 48(%rsp), %ymm0
	vmovups %ymm0, (%r9,%r11)
	vmovdqu %ymm9, (%r10,%r11)
	vmovdqu %ymm13, (%r11,%rsi,8)
	leaq (%rsi,%rsi,8), %r12
	vmovdqu %ymm11, (%r12,%r11)
	vmovdqu %ymm10, (%rbx,%r11)
	leaq (%rsi,%r8,2), %r12
	vmovdqu %ymm7, (%r12,%r11)
	vmovdqu %ymm8, (%r15,%r11)
	movq 8(%rsp), %r12
	vmovdqu %ymm2, (%r12,%r11)
	vmovdqu %ymm6, (%r13,%r11)
	vmovdqu 112(%rsp), %ymm0
	vmovdqu %ymm0, (%rbp,%r11)
	incq %rcx
	decq %rax
	jne .LBB18_3
.LBB18_6:
	movq (%rsp), %rax
	shlq $5, %rax
	addq $152, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB18_7:
	leaq .Lanon.b3414de80210f36676cad164b492ea25.3(%rip), %rdi
	leaq .Lanon.b3414de80210f36676cad164b492ea25.6(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
