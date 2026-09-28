jix_probe::byte_shuffle::decode_impl::<16, 8>:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $424, %rsp
	movq %rsi, -128(%rsp)
	movq %rsi, %rax
	shrq $4, %rax
	movabsq $576460752303423480, %r8
	andq %rax, %r8
	movq %rdx, -120(%rsp)
	je .LBB0_1
	leaq (%rax,%rax,2), %rsi
	leaq (%rax,%rax,4), %r10
	leaq (%rax,%rax,8), %r9
	leaq (%rax,%r10,2), %r11
	leaq (%rax,%rsi,4), %rbx
	movq %rax, %r15
	shlq $4, %r15
	subq %rax, %r15
	subq %rax, %r15
	leaq (%r10,%r10,2), %rcx
	leaq 124(%rdx), %r13
	addq %rdi, %rcx
	movq %rcx, -56(%rsp)
	addq %rdi, %r15
	addq %rdi, %rbx
	movq %rbx, -48(%rsp)
	leaq (%rdi,%rsi,4), %rcx
	movq %rcx, -64(%rsp)
	addq %rdi, %r11
	movq %r11, -40(%rsp)
	leaq (%rdi,%r10,2), %rcx
	movq %rcx, -72(%rsp)
	addq %rdi, %r9
	movq %r9, -32(%rsp)
	leaq (%rdi,%rax,8), %rcx
	movq %rcx, -80(%rsp)
	subq %rax, %rcx
	movq %rcx, -88(%rsp)
	leaq (%rdi,%rsi,2), %rcx
	movq %rcx, -96(%rsp)
	addq %rdi, %r10
	leaq (%rdi,%rax,4), %rcx
	movq %rcx, -104(%rsp)
	addq %rdi, %rsi
	movq %rsi, -24(%rsp)
	leaq (%rdi,%rax,2), %rcx
	movq %rcx, -112(%rsp)
	addq %rdi, %rax
	vmovd .LCPI0_3(%rip), %xmm0
	vpbroadcastq .LCPI0_2(%rip), %ymm2
	xorl %r9d, %r9d
.LBB0_4:
	movq (%rdi,%r9), %rcx
	movq (%rax,%r9), %rdx
	movq -112(%rsp), %rsi
	movq (%rsi,%r9), %rsi
	movq -24(%rsp), %r11
	movq (%r11,%r9), %r11
	vmovq %rdx, %xmm1
	vmovq %rcx, %xmm3
	vpunpcklqdq %xmm1, %xmm3, %xmm4
	vmovq %r11, %xmm1
	vmovq %rsi, %xmm3
	vpunpcklqdq %xmm1, %xmm3, %xmm3
	vmovd %ecx, %xmm1
	vpinsrb $1, %edx, %xmm1, %xmm1
	vpinsrb $2, %esi, %xmm1, %xmm1
	vinserti128 $1, %xmm3, %ymm4, %ymm5
	vmovdqa %ymm3, %ymm8
	vmovdqu %ymm3, 352(%rsp)
	vmovdqa %ymm4, %ymm6
	vmovdqu %ymm4, 384(%rsp)
	vpinsrb $3, %r11d, %xmm1, %xmm7
	vpsrlq $8, %ymm5, %ymm1
	vextracti128 $1, %ymm1, %xmm3
	vpshufb %xmm0, %xmm3, %xmm3
	vpshufb %xmm0, %xmm1, %xmm1
	vpunpcklwd %xmm3, %xmm1, %xmm1
	vmovdqu %ymm1, 224(%rsp)
	vpsrlq $16, %ymm5, %ymm1
	vextracti128 $1, %ymm1, %xmm3
	vpshufb %xmm0, %xmm3, %xmm3
	vpshufb %xmm0, %xmm1, %xmm1
	vpunpcklwd %xmm3, %xmm1, %xmm4
	vpsrlq $24, %ymm5, %ymm1
	vextracti128 $1, %ymm1, %xmm3
	vpshufb %xmm0, %xmm3, %xmm3
	vpshufb %xmm0, %xmm1, %xmm1
	vpunpcklwd %xmm3, %xmm1, %xmm1
	vmovdqu %ymm1, 288(%rsp)
	vmovd .LCPI0_4(%rip), %xmm12
	vpshufb %xmm12, %xmm8, %xmm1
	vpshufb %xmm12, %xmm6, %xmm3
	vpunpcklwd %xmm1, %xmm3, %xmm3
	vpsrlq $40, %ymm5, %ymm1
	vextracti128 $1, %ymm1, %xmm6
	vpshufb %xmm0, %xmm6, %xmm6
	vpshufb %xmm0, %xmm1, %xmm1
	vpunpcklwd %xmm6, %xmm1, %xmm1
	vmovdqu %ymm1, 320(%rsp)
	vpsrlq $48, %ymm5, %ymm1
	vextracti128 $1, %ymm1, %xmm5
	vpshufb %xmm0, %xmm5, %xmm5
	vpshufb %xmm0, %xmm1, %xmm1
	vpunpcklwd %xmm5, %xmm1, %xmm1
	movq -104(%rsp), %rcx
	movq (%rcx,%r9), %rcx
	movq (%r10,%r9), %rdx
	movq -96(%rsp), %rsi
	movq (%rsi,%r9), %rsi
	movq -88(%rsp), %r11
	movq (%r11,%r9), %r11
	vmovq %rdx, %xmm5
	vmovq %rcx, %xmm6
	vpunpcklqdq %xmm5, %xmm6, %xmm8
	vmovq %r11, %xmm5
	vmovq %rsi, %xmm6
	vpunpcklqdq %xmm5, %xmm6, %xmm11
	vmovd %ecx, %xmm5
	vpinsrb $1, %edx, %xmm5, %xmm5
	vpinsrb $2, %esi, %xmm5, %xmm5
	vpinsrb $3, %r11d, %xmm5, %xmm5
	vinserti128 $1, %xmm11, %ymm8, %ymm6
	vmovdqu %ymm8, 256(%rsp)
	vpunpckldq %xmm5, %xmm7, %xmm5
	vmovdqa %xmm5, 32(%rsp)
	vpsrlq $8, %ymm6, %ymm5
	vextracti128 $1, %ymm5, %xmm7
	vpshufb %xmm0, %xmm7, %xmm7
	vpshufb %xmm0, %xmm5, %xmm5
	vpunpcklwd %xmm7, %xmm5, %xmm5
	vmovdqa %xmm5, -16(%rsp)
	vpsrlq $16, %ymm6, %ymm5
	vextracti128 $1, %ymm5, %xmm7
	vpshufb %xmm0, %xmm7, %xmm7
	vpshufb %xmm0, %xmm5, %xmm5
	vpunpcklwd %xmm7, %xmm5, %xmm5
	vpunpckldq %xmm5, %xmm4, %xmm4
	vmovdqa %xmm4, 48(%rsp)
	vpsrlq $24, %ymm6, %ymm4
	vextracti128 $1, %ymm4, %xmm5
	vpshufb %xmm0, %xmm5, %xmm5
	vpshufb %xmm0, %xmm4, %xmm4
	vpunpcklwd %xmm5, %xmm4, %xmm4
	vmovdqa %xmm4, (%rsp)
	vpshufb %xmm12, %xmm11, %xmm4
	vpshufb %xmm12, %xmm8, %xmm5
	vpunpcklwd %xmm4, %xmm5, %xmm4
	vpunpckldq %xmm4, %xmm3, %xmm3
	vmovdqa %xmm3, 80(%rsp)
	vpsrlq $40, %ymm6, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm3
	vmovdqa %xmm3, 16(%rsp)
	vpsrlq $48, %ymm6, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm3
	vpunpckldq %xmm3, %xmm1, %xmm1
	vmovdqa %xmm1, 64(%rsp)
	movq -80(%rsp), %rcx
	movq (%rcx,%r9), %rbp
	movq -32(%rsp), %rcx
	movq (%rcx,%r9), %r11
	movq -72(%rsp), %rcx
	movq (%rcx,%r9), %r12
	movq -40(%rsp), %rcx
	movq (%rcx,%r9), %r14
	vmovq %r11, %xmm1
	vmovq %rbp, %xmm3
	vpunpcklqdq %xmm1, %xmm3, %xmm13
	vmovq %r14, %xmm1
	vmovq %r12, %xmm3
	vpunpcklqdq %xmm1, %xmm3, %xmm15
	vinserti128 $1, %xmm15, %ymm13, %ymm1
	vpsrlq $8, %ymm1, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm8
	vpsrlq $16, %ymm1, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm3
	vmovdqu %ymm3, 128(%rsp)
	vpsrlq $24, %ymm1, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm9
	vpshufb %xmm12, %xmm15, %xmm3
	vpshufb %xmm12, %xmm13, %xmm4
	vpunpcklwd %xmm3, %xmm4, %xmm3
	vmovdqu %ymm3, 192(%rsp)
	vpsrlq $40, %ymm1, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm3
	vmovdqu %ymm3, 96(%rsp)
	vpsrlq $48, %ymm1, %ymm1
	vextracti128 $1, %ymm1, %xmm3
	vpshufb %xmm0, %xmm3, %xmm3
	vpshufb %xmm0, %xmm1, %xmm1
	vpunpcklwd %xmm3, %xmm1, %xmm1
	vmovdqu %ymm1, 160(%rsp)
	movq -64(%rsp), %rcx
	movq (%rcx,%r9), %rbx
	movq -48(%rsp), %rcx
	movq (%rcx,%r9), %rcx
	vmovq %rcx, %xmm1
	vmovq %rbx, %xmm3
	vpunpcklqdq %xmm1, %xmm3, %xmm10
	movq (%r15,%r9), %rsi
	movq -56(%rsp), %rdx
	movq (%rdx,%r9), %rdx
	vmovq %rdx, %xmm1
	vmovq %rsi, %xmm3
	vpunpcklqdq %xmm1, %xmm3, %xmm14
	vinserti128 $1, %xmm14, %ymm10, %ymm5
	vpsrlq $8, %ymm5, %ymm1
	vextracti128 $1, %ymm1, %xmm3
	vpshufb %xmm0, %xmm3, %xmm3
	vpshufb %xmm0, %xmm1, %xmm1
	vpunpcklwd %xmm3, %xmm1, %xmm1
	vpsrlq $16, %ymm5, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm6
	vpsrlq $24, %ymm5, %ymm3
	vextracti128 $1, %ymm3, %xmm4
	vpshufb %xmm0, %xmm4, %xmm4
	vpshufb %xmm0, %xmm3, %xmm3
	vpunpcklwd %xmm4, %xmm3, %xmm3
	vpshufb %xmm12, %xmm14, %xmm4
	vpshufb %xmm12, %xmm10, %xmm7
	vpunpcklwd %xmm4, %xmm7, %xmm7
	vpsrlq $40, %ymm5, %ymm4
	vextracti128 $1, %ymm4, %xmm12
	vpshufb %xmm0, %xmm12, %xmm12
	vpshufb %xmm0, %xmm4, %xmm4
	vpunpcklwd %xmm12, %xmm4, %xmm4
	vpsrlq $48, %ymm5, %ymm5
	vextracti128 $1, %ymm5, %xmm12
	vpshufb %xmm0, %xmm12, %xmm12
	vpshufb %xmm0, %xmm5, %xmm5
	vpunpcklwd %xmm12, %xmm5, %xmm5
	vinserti128 $1, %xmm1, %ymm8, %ymm1
	vmovdqu 224(%rsp), %ymm8
	vinserti128 $1, -16(%rsp), %ymm8, %ymm8
	vmovd %ebp, %xmm12
	vpinsrb $1, %r11d, %xmm12, %xmm12
	vpinsrb $2, %r12d, %xmm12, %xmm12
	vpermd %ymm1, %ymm2, %ymm1
	vpermd %ymm8, %ymm2, %ymm8
	vpblendd $192, %ymm1, %ymm8, %ymm1
	vpinsrb $3, %r14d, %xmm12, %xmm8
	vmovd %ebx, %xmm12
	vpinsrb $1, %ecx, %xmm12, %xmm12
	vpinsrb $2, %esi, %xmm12, %xmm12
	vpinsrb $3, %edx, %xmm12, %xmm12
	vinserti128 $1, %xmm12, %ymm8, %ymm8
	vpermd %ymm8, %ymm2, %ymm8
	vpblendd $3, 32(%rsp), %xmm8, %xmm8
	vpblendd $240, %ymm1, %ymm8, %ymm1
	vmovdqu %ymm1, -124(%r13)
	vinserti128 $1, %xmm3, %ymm9, %ymm1
	vmovdqu 288(%rsp), %ymm3
	vinserti128 $1, (%rsp), %ymm3, %ymm3
	vpermd %ymm1, %ymm2, %ymm1
	vpermd %ymm3, %ymm2, %ymm3
	vpblendd $192, %ymm1, %ymm3, %ymm1
	vmovdqu 128(%rsp), %ymm3
	vinserti128 $1, %xmm6, %ymm3, %ymm3
	vpermd %ymm3, %ymm2, %ymm3
	vpblendd $3, 48(%rsp), %xmm3, %xmm3
	vpblendd $240, %ymm1, %ymm3, %ymm1
	vmovdqu %ymm1, -92(%r13)
	vmovdqu 96(%rsp), %ymm1
	vinserti128 $1, %xmm4, %ymm1, %ymm1
	vmovdqu 320(%rsp), %ymm3
	vinserti128 $1, 16(%rsp), %ymm3, %ymm3
	vpermd %ymm1, %ymm2, %ymm1
	vpermd %ymm3, %ymm2, %ymm3
	vpblendd $192, %ymm1, %ymm3, %ymm1
	vmovdqu 192(%rsp), %ymm3
	vinserti128 $1, %xmm7, %ymm3, %ymm3
	vpermd %ymm3, %ymm2, %ymm3
	vpblendd $3, 80(%rsp), %xmm3, %xmm3
	vpblendd $240, %ymm1, %ymm3, %ymm1
	vmovdqu %ymm1, -60(%r13)
	vmovdqu 160(%rsp), %ymm1
	vinserti128 $1, %xmm5, %ymm1, %ymm1
	vpermd %ymm1, %ymm2, %ymm1
	vpblendd $3, 64(%rsp), %xmm1, %xmm1
	vinserti128 $1, %xmm14, %ymm15, %ymm3
	vinserti128 $1, %xmm10, %ymm13, %ymm4
	vpsrlq $56, %ymm3, %ymm3
	vpsrlq $56, %ymm4, %ymm4
	vpackusdw %ymm3, %ymm4, %ymm3
	vmovdqu 352(%rsp), %ymm4
	vinserti128 $1, %xmm11, %ymm4, %ymm4
	vmovdqu 384(%rsp), %ymm5
	vinserti128 $1, 256(%rsp), %ymm5, %ymm5
	vpsrlq $56, %ymm4, %ymm4
	vpsrlq $56, %ymm5, %ymm5
	vpackusdw %ymm4, %ymm5, %ymm4
	vpackusdw %ymm0, %ymm3, %ymm3
	vpackuswb %ymm0, %ymm3, %ymm3
	vpermd %ymm3, %ymm2, %ymm3
	vpackusdw %ymm0, %ymm4, %ymm4
	vpackuswb %ymm0, %ymm4, %ymm4
	vpermd %ymm4, %ymm2, %ymm4
	vpblendd $192, %ymm3, %ymm4, %ymm3
	vpblendd $240, %ymm3, %ymm1, %ymm1
	vmovdqu %ymm1, -28(%r13)
	addq $8, %r9
	subq $-128, %r13
	cmpq %r8, %r9
	jb .LBB0_4
	jmp .LBB0_2
.LBB0_1:
	xorl %r9d, %r9d
.LBB0_2:
	movl $16, %r8d
	movq -128(%rsp), %rsi
	movq -120(%rsp), %rdx
	addq $424, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	jmpq *jix_probe::byte_shuffle::decode_impl_generic@GOTPCREL(%rip)
