jix_probe::byte_shuffle::encode_impl::<16, 8>:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	movq %rsi, -48(%rsp)
	movq %rsi, %rax
	shrq $4, %rax
	movabsq $576460752303423480, %rcx
	andq %rax, %rcx
	movq %rdi, -56(%rsp)
	je .LBB4_1
	leaq (%rax,%rax,2), %r8
	leaq (%rax,%rax,4), %r10
	movq %rcx, -8(%rsp)
	leaq (%rax,%rax,8), %rcx
	leaq (%rax,%r10,2), %r11
	leaq (%rax,%r8,4), %r9
	movq %rax, %r15
	shlq $4, %r15
	subq %rax, %r15
	subq %rax, %r15
	leaq (%r10,%r10,2), %rsi
	leaq 124(%rdi), %r13
	addq %rdx, %rsi
	movq %rsi, -24(%rsp)
	addq %rdx, %r15
	addq %rdx, %r9
	movq %r9, -16(%rsp)
	leaq (%rdx,%r8,4), %rsi
	movq %rsi, -32(%rsp)
	addq %rdx, %r11
	leaq (%rdx,%r10,2), %rsi
	movq %rsi, -40(%rsp)
	addq %rdx, %rcx
	leaq (%rdx,%rax,8), %rsi
	movq %rsi, %rdi
	subq %rax, %rdi
	leaq (%rdx,%r8,2), %rbx
	addq %rdx, %r10
	leaq (%rdx,%rax,4), %r14
	addq %rdx, %r8
	leaq (%rdx,%rax,2), %rbp
	addq %rdx, %rax
	xorl %r9d, %r9d
.LBB4_4:
	vpmovzxbq -4(%r13), %ymm0
	vpmovzxbq -8(%r13), %ymm1
	vpmovzxbq -12(%r13), %ymm2
	vpmovzxbq -28(%r13), %ymm3
	vpmovzxbq -44(%r13), %ymm4
	vpmovzxbq -60(%r13), %ymm5
	vpmovzxbq -76(%r13), %ymm6
	vpsllq $40, %ymm4, %ymm4
	vpsllq $32, %ymm5, %ymm5
	vpor %ymm4, %ymm5, %ymm4
	vpmovzxbq -92(%r13), %ymm5
	vpsllq $48, %ymm3, %ymm3
	vpor %ymm3, %ymm4, %ymm3
	vpmovzxbq -108(%r13), %ymm4
	vpsllq $24, %ymm6, %ymm6
	vpsllq $16, %ymm5, %ymm5
	vpor %ymm6, %ymm5, %ymm5
	vpmovzxbq -124(%r13), %ymm6
	vpsllq $8, %ymm4, %ymm4
	vpor %ymm6, %ymm4, %ymm4
	vpor %ymm5, %ymm4, %ymm4
	vpmovzxbq -24(%r13), %ymm5
	vpsllq $56, %ymm2, %ymm2
	vpor %ymm3, %ymm4, %ymm3
	vpor %ymm2, %ymm3, %ymm2
	vmovq %xmm2, (%rdx,%r9)
	vpextrq $1, %xmm2, (%rax,%r9)
	vpmovzxbq -40(%r13), %ymm3
	vextracti128 $1, %ymm2, %xmm2
	vmovq %xmm2, (%rbp,%r9)
	vpextrq $1, %xmm2, (%r8,%r9)
	vpmovzxbq -56(%r13), %ymm2
	vpmovzxbq -72(%r13), %ymm4
	vpsllq $40, %ymm3, %ymm3
	vpsllq $32, %ymm2, %ymm2
	vpor %ymm3, %ymm2, %ymm2
	vpmovzxbq -88(%r13), %ymm3
	vpsllq $48, %ymm5, %ymm5
	vpor %ymm5, %ymm2, %ymm2
	vpmovzxbq -104(%r13), %ymm5
	vpsllq $24, %ymm4, %ymm4
	vpsllq $16, %ymm3, %ymm3
	vpor %ymm4, %ymm3, %ymm3
	vpmovzxbq -120(%r13), %ymm4
	vpsllq $8, %ymm5, %ymm5
	vpor %ymm4, %ymm5, %ymm4
	vpor %ymm3, %ymm4, %ymm3
	vpmovzxbq -20(%r13), %ymm4
	vpsllq $56, %ymm1, %ymm1
	vpor %ymm2, %ymm3, %ymm2
	vpor %ymm1, %ymm2, %ymm1
	vmovq %xmm1, (%r14,%r9)
	vpextrq $1, %xmm1, (%r10,%r9)
	vpmovzxbq -36(%r13), %ymm2
	vextracti128 $1, %ymm1, %xmm1
	vmovq %xmm1, (%rbx,%r9)
	vpextrq $1, %xmm1, (%rdi,%r9)
	vpmovzxbq -52(%r13), %ymm1
	vpmovzxbq -68(%r13), %ymm3
	vpsllq $40, %ymm2, %ymm2
	vpsllq $32, %ymm1, %ymm1
	vpor %ymm2, %ymm1, %ymm1
	vpmovzxbq -84(%r13), %ymm2
	vpsllq $48, %ymm4, %ymm4
	vpor %ymm4, %ymm1, %ymm1
	vpmovzxbq -100(%r13), %ymm4
	vpsllq $24, %ymm3, %ymm3
	vpsllq $16, %ymm2, %ymm2
	vpor %ymm3, %ymm2, %ymm2
	vpmovzxbq -116(%r13), %ymm3
	vpsllq $8, %ymm4, %ymm4
	vpor %ymm3, %ymm4, %ymm3
	vpor %ymm2, %ymm3, %ymm2
	vpmovzxbq -16(%r13), %ymm3
	vpsllq $56, %ymm0, %ymm0
	vpor %ymm1, %ymm2, %ymm1
	vpor %ymm0, %ymm1, %ymm0
	vmovq %xmm0, (%rsi,%r9)
	vpextrq $1, %xmm0, (%rcx,%r9)
	vpmovzxbq -32(%r13), %ymm1
	vextracti128 $1, %ymm0, %xmm0
	movq -40(%rsp), %r12
	vmovq %xmm0, (%r12,%r9)
	vpextrq $1, %xmm0, (%r11,%r9)
	vpmovzxbq -48(%r13), %ymm0
	vpmovzxbq -64(%r13), %ymm2
	vpsllq $40, %ymm1, %ymm1
	vpsllq $32, %ymm0, %ymm0
	vpor %ymm1, %ymm0, %ymm0
	vpmovzxbq -80(%r13), %ymm1
	vpsllq $48, %ymm3, %ymm3
	vpor %ymm3, %ymm0, %ymm0
	vpmovzxbq -96(%r13), %ymm3
	vpsllq $24, %ymm2, %ymm2
	vpsllq $16, %ymm1, %ymm1
	vpor %ymm2, %ymm1, %ymm1
	vpmovzxbq -112(%r13), %ymm2
	vpsllq $8, %ymm3, %ymm3
	vpor %ymm2, %ymm3, %ymm2
	vpor %ymm1, %ymm2, %ymm1
	vpmovzxbq (%r13), %ymm2
	vpor %ymm0, %ymm1, %ymm0
	vpsllq $56, %ymm2, %ymm1
	vpor %ymm1, %ymm0, %ymm0
	movq -32(%rsp), %r12
	vmovq %xmm0, (%r12,%r9)
	movq -16(%rsp), %r12
	vpextrq $1, %xmm0, (%r12,%r9)
	vextracti128 $1, %ymm0, %xmm0
	vmovq %xmm0, (%r15,%r9)
	movq -24(%rsp), %r12
	vpextrq $1, %xmm0, (%r12,%r9)
	addq $8, %r9
	subq $-128, %r13
	cmpq -8(%rsp), %r9
	jb .LBB4_4
	jmp .LBB4_2
.LBB4_1:
	xorl %r9d, %r9d
.LBB4_2:
	movl $16, %r8d
	movq -56(%rsp), %rdi
	movq -48(%rsp), %rsi
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	jmpq *jix_probe::byte_shuffle::encode_impl_generic@GOTPCREL(%rip)
