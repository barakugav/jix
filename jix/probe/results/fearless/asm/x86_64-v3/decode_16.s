probe_byte_shuffle_decode_16:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $232, %rsp
	movabsq $9223372036854775792, %rax
	andq %rsi, %rax
	cmpq %rax, %rcx
	jb .LBB14_5
	movq %rsi, %rax
	shrq $9, %rax
	movq %rax, 8(%rsp)
	je .LBB14_4
	shrq $4, %rsi
	leaq (%rsi,%rsi,4), %rax
	leaq (%rax,%rax,2), %r12
	movq %rsi, %r8
	shlq $4, %r8
	subq %rsi, %r8
	subq %rsi, %r8
	leaq (%rsi,%rsi,2), %rcx
	leaq (%rsi,%rcx,4), %r9
	leaq (%rsi,%rax,2), %r10
	leaq (%rsi,%rsi,8), %r11
	addq $480, %rdx
	leaq (%rdi,%rsi), %rbx
	movq %rbx, 120(%rsp)
	leaq (%rdi,%rsi,2), %rbx
	movq %rbx, 112(%rsp)
	leaq (%rdi,%rcx), %rbx
	movq %rbx, 104(%rsp)
	leaq (%rdi,%rsi,4), %rbx
	movq %rbx, 96(%rsp)
	leaq (%rdi,%rax), %rbx
	movq %rbx, 88(%rsp)
	leaq (%rdi,%rsi,8), %rbx
	movq %rbx, %rbp
	subq %rsi, %rbp
	leaq (%rdi,%rcx,2), %rsi
	addq %rdi, %r11
	leaq (%rdi,%rax,2), %rax
	addq %rdi, %r10
	leaq (%rdi,%rcx,4), %r14
	addq %rdi, %r9
	addq %rdi, %r8
	addq %rdi, %r12
	movq 8(%rsp), %r15
	shlq $5, %r15
	xorl %ecx, %ecx
.LBB14_3:
	vmovups (%r12,%rcx), %ymm0
	vmovups %ymm0, 48(%rsp)
	vmovdqu (%r8,%rcx), %ymm1
	vmovdqu (%r9,%rcx), %ymm2
	vmovdqu (%r14,%rcx), %ymm3
	vmovdqu (%r10,%rcx), %ymm4
	vmovdqu (%rax,%rcx), %ymm5
	vmovdqu (%r11,%rcx), %ymm6
	vmovdqu (%rbx,%rcx), %ymm7
	vmovdqu (%rsi,%rcx), %ymm8
	movq 88(%rsp), %r13
	vmovdqu (%r13,%rcx), %ymm9
	movq 96(%rsp), %r13
	vmovdqu (%r13,%rcx), %ymm10
	movq 104(%rsp), %r13
	vmovdqu (%r13,%rcx), %ymm11
	movq 112(%rsp), %r13
	vmovdqu (%r13,%rcx), %ymm12
	movq 120(%rsp), %r13
	vmovdqu (%r13,%rcx), %ymm13
	vmovdqu (%rdi,%rcx), %ymm14
	vpunpcklbw %ymm7, %ymm14, %ymm15
	vpunpckhbw %ymm7, %ymm14, %ymm7
	vpunpcklbw %ymm6, %ymm13, %ymm14
	vpunpckhbw %ymm6, %ymm13, %ymm13
	vpunpcklbw %ymm5, %ymm12, %ymm0
	vpunpckhbw %ymm5, %ymm12, %ymm5
	vpunpcklbw %ymm4, %ymm11, %ymm6
	vmovdqu %ymm6, 16(%rsp)
	vpunpckhbw %ymm4, %ymm11, %ymm4
	vmovdqu %ymm4, 128(%rsp)
	vpunpcklbw %ymm3, %ymm10, %ymm11
	vpunpckhbw %ymm3, %ymm10, %ymm3
	vpunpcklbw %ymm2, %ymm9, %ymm10
	vpunpckhbw %ymm2, %ymm9, %ymm2
	vpunpcklbw %ymm1, %ymm8, %ymm9
	vpunpckhbw %ymm1, %ymm8, %ymm1
	vmovdqu (%rbp,%rcx), %ymm8
	vmovdqu 48(%rsp), %ymm4
	vpunpcklbw %ymm4, %ymm8, %ymm6
	vpunpckhbw %ymm4, %ymm8, %ymm8
	vinserti128 $1, %xmm7, %ymm15, %ymm4
	vperm2i128 $49, %ymm7, %ymm15, %ymm7
	vinserti128 $1, %xmm3, %ymm11, %ymm15
	vperm2i128 $49, %ymm3, %ymm11, %ymm3
	vinserti128 $1, %xmm13, %ymm14, %ymm11
	vperm2i128 $49, %ymm13, %ymm14, %ymm12
	vinserti128 $1, %xmm2, %ymm10, %ymm14
	vperm2i128 $49, %ymm2, %ymm10, %ymm2
	vinserti128 $1, %xmm5, %ymm0, %ymm10
	vperm2i128 $49, %ymm5, %ymm0, %ymm0
	vmovdqu %ymm0, 160(%rsp)
	vinserti128 $1, %xmm1, %ymm9, %ymm13
	vperm2i128 $49, %ymm1, %ymm9, %ymm1
	vmovdqu 16(%rsp), %ymm5
	vmovdqu 128(%rsp), %ymm0
	vinserti128 $1, %xmm0, %ymm5, %ymm9
	vperm2i128 $49, %ymm0, %ymm5, %ymm0
	vmovdqu %ymm0, 192(%rsp)
	vinserti128 $1, %xmm8, %ymm6, %ymm5
	vperm2i128 $49, %ymm8, %ymm6, %ymm0
	vpunpcklbw %ymm15, %ymm4, %ymm8
	vpunpckhbw %ymm15, %ymm4, %ymm4
	vpunpcklbw %ymm3, %ymm7, %ymm15
	vpunpckhbw %ymm3, %ymm7, %ymm7
	vpunpcklbw %ymm14, %ymm11, %ymm3
	vmovdqu %ymm3, 128(%rsp)
	vpunpckhbw %ymm14, %ymm11, %ymm11
	vpunpcklbw %ymm2, %ymm12, %ymm3
	vmovdqu %ymm3, 48(%rsp)
	vpunpckhbw %ymm2, %ymm12, %ymm2
	vmovdqu %ymm2, 16(%rsp)
	vpunpcklbw %ymm13, %ymm10, %ymm6
	vpunpckhbw %ymm13, %ymm10, %ymm10
	vmovdqu 160(%rsp), %ymm2
	vpunpcklbw %ymm1, %ymm2, %ymm13
	vpunpckhbw %ymm1, %ymm2, %ymm1
	vpunpcklbw %ymm5, %ymm9, %ymm12
	vpunpckhbw %ymm5, %ymm9, %ymm5
	vmovdqu 192(%rsp), %ymm2
	vpunpcklbw %ymm0, %ymm2, %ymm9
	vpunpckhbw %ymm0, %ymm2, %ymm2
	vinserti128 $1, %xmm4, %ymm8, %ymm3
	vperm2i128 $49, %ymm4, %ymm8, %ymm4
	vinserti128 $1, %xmm10, %ymm6, %ymm8
	vperm2i128 $49, %ymm10, %ymm6, %ymm6
	vinserti128 $1, %xmm7, %ymm15, %ymm10
	vperm2i128 $49, %ymm7, %ymm15, %ymm14
	vinserti128 $1, %xmm1, %ymm13, %ymm15
	vperm2i128 $49, %ymm1, %ymm13, %ymm1
	vmovdqu 128(%rsp), %ymm0
	vinserti128 $1, %xmm11, %ymm0, %ymm13
	vperm2i128 $49, %ymm11, %ymm0, %ymm0
	vmovdqu %ymm0, 128(%rsp)
	vinserti128 $1, %xmm5, %ymm12, %ymm11
	vperm2i128 $49, %ymm5, %ymm12, %ymm5
	vmovdqu 48(%rsp), %ymm7
	vmovdqu 16(%rsp), %ymm0
	vinserti128 $1, %xmm0, %ymm7, %ymm12
	vperm2i128 $49, %ymm0, %ymm7, %ymm0
	vmovdqu %ymm0, 16(%rsp)
	vinserti128 $1, %xmm2, %ymm9, %ymm7
	vperm2i128 $49, %ymm2, %ymm9, %ymm0
	vpunpcklbw %ymm8, %ymm3, %ymm9
	vpunpckhbw %ymm8, %ymm3, %ymm2
	vpunpcklbw %ymm6, %ymm4, %ymm8
	vpunpckhbw %ymm6, %ymm4, %ymm4
	vpunpcklbw %ymm15, %ymm10, %ymm6
	vpunpckhbw %ymm15, %ymm10, %ymm10
	vpunpcklbw %ymm1, %ymm14, %ymm3
	vmovdqu %ymm3, 48(%rsp)
	vpunpckhbw %ymm1, %ymm14, %ymm15
	vpunpcklbw %ymm11, %ymm13, %ymm3
	vpunpckhbw %ymm11, %ymm13, %ymm11
	vmovdqu 128(%rsp), %ymm1
	vpunpcklbw %ymm5, %ymm1, %ymm13
	vpunpckhbw %ymm5, %ymm1, %ymm5
	vpunpcklbw %ymm7, %ymm12, %ymm14
	vpunpckhbw %ymm7, %ymm12, %ymm7
	vmovdqu 16(%rsp), %ymm1
	vpunpcklbw %ymm0, %ymm1, %ymm12
	vpunpckhbw %ymm0, %ymm1, %ymm0
	vinserti128 $1, %xmm2, %ymm9, %ymm1
	vperm2i128 $49, %ymm2, %ymm9, %ymm2
	vinserti128 $1, %xmm11, %ymm3, %ymm9
	vperm2i128 $49, %ymm11, %ymm3, %ymm3
	vinserti128 $1, %xmm4, %ymm8, %ymm11
	vperm2i128 $49, %ymm4, %ymm8, %ymm4
	vinserti128 $1, %xmm5, %ymm13, %ymm8
	vperm2i128 $49, %ymm5, %ymm13, %ymm5
	vinserti128 $1, %xmm10, %ymm6, %ymm13
	vperm2i128 $49, %ymm10, %ymm6, %ymm6
	vmovdqu %ymm6, 16(%rsp)
	vinserti128 $1, %xmm7, %ymm14, %ymm10
	vperm2i128 $49, %ymm7, %ymm14, %ymm7
	vmovdqu 48(%rsp), %ymm6
	vinserti128 $1, %xmm15, %ymm6, %ymm14
	vperm2i128 $49, %ymm15, %ymm6, %ymm15
	vinserti128 $1, %xmm0, %ymm12, %ymm6
	vperm2i128 $49, %ymm0, %ymm12, %ymm0
	vpunpcklbw %ymm9, %ymm1, %ymm12
	vpunpckhbw %ymm9, %ymm1, %ymm1
	vinserti128 $1, %xmm1, %ymm12, %ymm9
	vperm2i128 $49, %ymm1, %ymm12, %ymm1
	vmovdqu %ymm1, 48(%rsp)
	vpunpcklbw %ymm3, %ymm2, %ymm12
	vpunpckhbw %ymm3, %ymm2, %ymm2
	vinserti128 $1, %xmm2, %ymm12, %ymm3
	vperm2i128 $49, %ymm2, %ymm12, %ymm2
	vpunpcklbw %ymm8, %ymm11, %ymm12
	vpunpckhbw %ymm8, %ymm11, %ymm8
	vinserti128 $1, %xmm8, %ymm12, %ymm11
	vperm2i128 $49, %ymm8, %ymm12, %ymm8
	vpunpcklbw %ymm5, %ymm4, %ymm12
	vpunpckhbw %ymm5, %ymm4, %ymm4
	vinserti128 $1, %xmm4, %ymm12, %ymm5
	vperm2i128 $49, %ymm4, %ymm12, %ymm4
	vpunpcklbw %ymm10, %ymm13, %ymm12
	vpunpckhbw %ymm10, %ymm13, %ymm10
	vinserti128 $1, %xmm10, %ymm12, %ymm13
	vperm2i128 $49, %ymm10, %ymm12, %ymm10
	vmovdqu 16(%rsp), %ymm1
	vpunpcklbw %ymm7, %ymm1, %ymm12
	vpunpckhbw %ymm7, %ymm1, %ymm7
	vinserti128 $1, %xmm7, %ymm12, %ymm1
	vperm2i128 $49, %ymm7, %ymm12, %ymm7
	vpunpcklbw %ymm6, %ymm14, %ymm12
	vpunpckhbw %ymm6, %ymm14, %ymm6
	vinserti128 $1, %xmm6, %ymm12, %ymm14
	vperm2i128 $49, %ymm6, %ymm12, %ymm6
	vpunpcklbw %ymm0, %ymm15, %ymm12
	vpunpckhbw %ymm0, %ymm15, %ymm0
	vinserti128 $1, %xmm0, %ymm12, %ymm15
	vperm2i128 $49, %ymm0, %ymm12, %ymm0
	vmovdqu %ymm9, -480(%rdx)
	vmovups 48(%rsp), %ymm9
	vmovups %ymm9, -448(%rdx)
	vmovdqu %ymm3, -416(%rdx)
	vmovdqu %ymm2, -384(%rdx)
	vmovdqu %ymm11, -352(%rdx)
	vmovdqu %ymm8, -320(%rdx)
	vmovdqu %ymm5, -288(%rdx)
	vmovdqu %ymm4, -256(%rdx)
	vmovdqu %ymm13, -224(%rdx)
	vmovdqu %ymm10, -192(%rdx)
	vmovdqu %ymm1, -160(%rdx)
	vmovdqu %ymm7, -128(%rdx)
	vmovdqu %ymm14, -96(%rdx)
	vmovdqu %ymm6, -64(%rdx)
	vmovdqu %ymm15, -32(%rdx)
	vmovdqu %ymm0, (%rdx)
	addq $512, %rdx
	addq $32, %rcx
	cmpq %rcx, %r15
	jne .LBB14_3
.LBB14_4:
	movq 8(%rsp), %rax
	shlq $5, %rax
	addq $232, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB14_5:
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.3(%rip), %rdi
	leaq .Lanon.f597599225a7e11d01edf736da1dddc0.5(%rip), %rdx
	movl $48, %esi
	callq *core::panicking::panic@GOTPCREL(%rip)
