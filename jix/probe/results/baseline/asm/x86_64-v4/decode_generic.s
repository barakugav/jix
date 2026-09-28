jix_probe::byte_shuffle::decode_impl_generic:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $24, %rsp
	testq %r8, %r8
	je .LBB8_6
	movq %rdx, %rcx
	movq %rsi, %rax
	orq %r8, %rax
	shrq $32, %rax
	je .LBB8_2
	movq %rsi, %rax
	xorl %edx, %edx
	divq %r8
	cmpq %rax, %r9
	jb .LBB8_5
	jmp .LBB8_19
.LBB8_2:
	movl %esi, %eax
	xorl %edx, %edx
	divl %r8d
	cmpq %rax, %r9
	jae .LBB8_19
.LBB8_5:
	movq %r8, %r14
	andq $-64, %r14
	vpbroadcastq %rax, %zmm0
	movq %r8, %rsi
	andq $-8, %rsi
	movq %r9, %r10
	imulq %r8, %r10
	addq %r10, %rcx
	vmovdqa64 .LCPI8_7(%rip), %zmm1
	vpbroadcastq .LCPI8_9(%rip), %zmm2
	vmovdqa64 .LCPI8_0(%rip), %zmm3
	vmovdqa64 .LCPI8_1(%rip), %zmm4
	vmovdqa64 .LCPI8_2(%rip), %zmm5
	vmovdqa64 .LCPI8_3(%rip), %zmm6
	vmovdqa64 .LCPI8_4(%rip), %zmm7
	vmovdqa64 .LCPI8_5(%rip), %zmm8
	vmovdqa64 .LCPI8_6(%rip), %zmm9
	vmovdqa64 .LCPI8_7(%rip), %zmm10
	vpbroadcastq .LCPI8_8(%rip), %zmm11
	movq %rdi, 16(%rsp)
	movq %r14, 8(%rsp)
	jmp .LBB8_8
.LBB8_7:
	incq %r9
	addq %r8, %rcx
	cmpq %rax, %r9
	jae .LBB8_19
.LBB8_8:
	leaq (%rdi,%r9), %r10
	cmpq $8, %r8
	jae .LBB8_10
	xorl %ebx, %ebx
	jmp .LBB8_18
.LBB8_10:
	xorl %r11d, %r11d
	cmpq $64, %r8
	jb .LBB8_15
	vmovdqa64 %zmm10, %zmm12
	vmovdqa64 %zmm9, %zmm13
	vmovdqa64 %zmm8, %zmm14
	vmovdqa64 %zmm7, %zmm15
	vmovdqa64 %zmm6, %zmm16
	vmovdqa64 %zmm5, %zmm17
	vmovdqa64 %zmm4, %zmm18
	vmovdqa64 %zmm3, %zmm19
.LBB8_12:
	vpmullq %zmm0, %zmm14, %zmm20
	vmovq %xmm20, %rbx
	movzbl (%r10,%rbx), %ebx
	vmovd %ebx, %xmm21
	vpextrq $1, %xmm20, %rbx
	vextracti32x4 $1, %ymm20, %xmm22
	vpinsrb $1, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	vpinsrb $2, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $2, %zmm20, %xmm22
	vpinsrb $3, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	vpinsrb $4, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $3, %zmm20, %xmm20
	vpinsrb $5, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm20, %rbx
	vpinsrb $6, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm20, %rbx
	vpmullq %zmm0, %zmm15, %zmm20
	vpinsrb $7, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm20, %rbx
	vpinsrb $8, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm20, %rbx
	vextracti32x4 $1, %ymm20, %xmm22
	vpinsrb $9, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	vpinsrb $10, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $2, %zmm20, %xmm22
	vpinsrb $11, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	vpinsrb $12, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $3, %zmm20, %xmm20
	vpinsrb $13, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm20, %rbx
	vpinsrb $14, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm20, %rbx
	vpmullq %zmm0, %zmm12, %zmm22
	vpinsrb $15, (%r10,%rbx), %xmm21, %xmm20
	vmovq %xmm22, %rbx
	movzbl (%r10,%rbx), %ebx
	vmovd %ebx, %xmm21
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $1, %ymm22, %xmm23
	vpinsrb $1, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm23, %rbx
	vpinsrb $2, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm23, %rbx
	vextracti32x4 $2, %zmm22, %xmm23
	vpinsrb $3, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm23, %rbx
	vpinsrb $4, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm23, %rbx
	vextracti32x4 $3, %zmm22, %xmm22
	vpinsrb $5, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	vpinsrb $6, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm22, %rbx
	vpmullq %zmm0, %zmm13, %zmm22
	vpinsrb $7, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	vpinsrb $8, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $1, %ymm22, %xmm23
	vpinsrb $9, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm23, %rbx
	vpinsrb $10, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm23, %rbx
	vextracti32x4 $2, %zmm22, %xmm23
	vpinsrb $11, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm23, %rbx
	vpinsrb $12, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm23, %rbx
	vextracti32x4 $3, %zmm22, %xmm22
	vpinsrb $13, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	vpinsrb $14, (%r10,%rbx), %xmm21, %xmm21
	vpextrq $1, %xmm22, %rbx
	vpmullq %zmm0, %zmm18, %zmm22
	vpinsrb $15, (%r10,%rbx), %xmm21, %xmm21
	vmovq %xmm22, %rbx
	movzbl (%r10,%rbx), %ebx
	vmovd %ebx, %xmm23
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $1, %ymm22, %xmm24
	vpinsrb $1, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm24, %rbx
	vpinsrb $2, (%r10,%rbx), %xmm23, %xmm23
	vpextrq $1, %xmm24, %rbx
	vextracti32x4 $2, %zmm22, %xmm24
	vpinsrb $3, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm24, %rbx
	vpinsrb $4, (%r10,%rbx), %xmm23, %xmm23
	vpextrq $1, %xmm24, %rbx
	vextracti32x4 $3, %zmm22, %xmm22
	vpinsrb $5, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm22, %rbx
	vpinsrb $6, (%r10,%rbx), %xmm23, %xmm23
	vpextrq $1, %xmm22, %rbx
	vpmullq %zmm0, %zmm19, %zmm22
	vpinsrb $7, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm22, %rbx
	vpinsrb $8, (%r10,%rbx), %xmm23, %xmm23
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $1, %ymm22, %xmm24
	vpinsrb $9, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm24, %rbx
	vpinsrb $10, (%r10,%rbx), %xmm23, %xmm23
	vpextrq $1, %xmm24, %rbx
	vextracti32x4 $2, %zmm22, %xmm24
	vpinsrb $11, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm24, %rbx
	vpinsrb $12, (%r10,%rbx), %xmm23, %xmm23
	vpextrq $1, %xmm24, %rbx
	vextracti32x4 $3, %zmm22, %xmm22
	vpinsrb $13, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm22, %rbx
	vpinsrb $14, (%r10,%rbx), %xmm23, %xmm23
	vpextrq $1, %xmm22, %rbx
	vpmullq %zmm0, %zmm16, %zmm22
	vpinsrb $15, (%r10,%rbx), %xmm23, %xmm23
	vmovq %xmm22, %rbx
	movzbl (%r10,%rbx), %ebx
	vmovd %ebx, %xmm24
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $1, %ymm22, %xmm25
	vpinsrb $1, (%r10,%rbx), %xmm24, %xmm24
	vmovq %xmm25, %rbx
	vpinsrb $2, (%r10,%rbx), %xmm24, %xmm24
	vpextrq $1, %xmm25, %rbx
	vextracti32x4 $2, %zmm22, %xmm25
	vpinsrb $3, (%r10,%rbx), %xmm24, %xmm24
	vmovq %xmm25, %rbx
	vpinsrb $4, (%r10,%rbx), %xmm24, %xmm24
	vpextrq $1, %xmm25, %rbx
	vextracti32x4 $3, %zmm22, %xmm22
	vpinsrb $5, (%r10,%rbx), %xmm24, %xmm24
	vmovq %xmm22, %rbx
	vpinsrb $6, (%r10,%rbx), %xmm24, %xmm24
	vpextrq $1, %xmm22, %rbx
	vpmullq %zmm0, %zmm17, %zmm22
	vpinsrb $7, (%r10,%rbx), %xmm24, %xmm24
	vmovq %xmm22, %rbx
	vpinsrb $8, (%r10,%rbx), %xmm24, %xmm24
	vpextrq $1, %xmm22, %rbx
	vextracti32x4 $1, %ymm22, %xmm25
	vpinsrb $9, (%r10,%rbx), %xmm24, %xmm24
	vmovq %xmm25, %rbx
	vpinsrb $10, (%r10,%rbx), %xmm24, %xmm24
	vpextrq $1, %xmm25, %rbx
	vextracti32x4 $2, %zmm22, %xmm25
	vpinsrb $11, (%r10,%rbx), %xmm24, %xmm24
	vmovq %xmm25, %rbx
	vpinsrb $12, (%r10,%rbx), %xmm24, %xmm24
	vpextrq $1, %xmm25, %rbx
	vextracti32x4 $3, %zmm22, %xmm22
	vpinsrb $13, (%r10,%rbx), %xmm24, %xmm24
	vmovq %xmm22, %rbx
	vpinsrb $14, (%r10,%rbx), %xmm24, %xmm24
	vpextrq $1, %xmm22, %rbx
	vpinsrb $15, (%r10,%rbx), %xmm24, %xmm22
	vinserti32x4 $1, %xmm20, %ymm21, %ymm20
	vinserti32x4 $1, %xmm23, %ymm22, %ymm21
	vinserti64x4 $1, %ymm21, %zmm20, %zmm20
	vpaddq %zmm11, %zmm12, %zmm12
	vpaddq %zmm11, %zmm13, %zmm13
	vmovdqu64 %zmm20, (%rcx,%r11)
	vpaddq %zmm11, %zmm14, %zmm14
	vpaddq %zmm11, %zmm15, %zmm15
	vpaddq %zmm11, %zmm16, %zmm16
	addq $64, %r11
	vpaddq %zmm11, %zmm17, %zmm17
	vpaddq %zmm11, %zmm18, %zmm18
	vpaddq %zmm11, %zmm19, %zmm19
	cmpq %r11, %r14
	jne .LBB8_12
	cmpq %r14, %r8
	je .LBB8_7
	movq %r14, %r11
	movq %r14, %rbx
	testb $56, %r8b
	je .LBB8_18
.LBB8_15:
	vpbroadcastq %r11, %zmm12
	vporq %zmm1, %zmm12, %zmm12
.LBB8_16:
	vpmullq %zmm0, %zmm12, %zmm13
	vmovq %xmm13, %rbx
	vpextrq $1, %xmm13, %r14
	vextracti128 $1, %ymm13, %xmm14
	vmovq %xmm14, %r15
	vpextrq $1, %xmm14, %r12
	vextracti32x4 $2, %zmm13, %xmm14
	vmovq %xmm14, %r13
	vpextrq $1, %xmm14, %rbp
	vextracti32x4 $3, %zmm13, %xmm13
	vmovq %xmm13, %rdx
	vpextrq $1, %xmm13, %rdi
	movzbl (%r10,%rbx), %ebx
	vmovd %ebx, %xmm13
	vpinsrb $1, (%r10,%r14), %xmm13, %xmm13
	vpinsrb $2, (%r10,%r15), %xmm13, %xmm13
	vpinsrb $3, (%r10,%r12), %xmm13, %xmm13
	vpinsrb $4, (%r10,%r13), %xmm13, %xmm13
	vpinsrb $5, (%r10,%rbp), %xmm13, %xmm13
	vpinsrb $6, (%r10,%rdx), %xmm13, %xmm13
	vpinsrb $7, (%r10,%rdi), %xmm13, %xmm13
	vmovq %xmm13, (%rcx,%r11)
	addq $8, %r11
	vpaddq %zmm2, %zmm12, %zmm12
	cmpq %r11, %rsi
	jne .LBB8_16
	movq %rsi, %rbx
	cmpq %rsi, %r8
	movq 16(%rsp), %rdi
	movq 8(%rsp), %r14
	je .LBB8_7
.LBB8_18:
	movq %rbx, %rdx
	imulq %rax, %rdx
	movzbl (%r10,%rdx), %edx
	movb %dl, (%rcx,%rbx)
	incq %rbx
	cmpq %rbx, %r8
	jne .LBB8_18
	jmp .LBB8_7
.LBB8_19:
	addq $24, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB8_6:
	leaq .Lanon.541dfd7828f123e2376bd9c83f3f80e1.1(%rip), %rdi
	callq *core::panicking::panic_const::panic_const_div_by_zero@GOTPCREL(%rip)
