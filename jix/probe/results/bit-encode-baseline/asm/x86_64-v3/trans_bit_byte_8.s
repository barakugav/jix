jix_probe::bit_shuffle::trans_bit_byte:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $376, %rsp
	movq %rdx, 8(%rsp)
	movq %r8, %rax
	shrq $3, %rax
	movq %r9, 32(%rsp)
	testq %r9, %r9
	movq %rax, %r9
	sete %al
	testq %r9, %r9
	sete %dl
	orb %al, %dl
	jne .LBB12_29
	movq %r8, %rbx
	movq %rcx, %r13
	movq 32(%rsp), %rax
	movq %rax, %r10
	imulq %r9, %r10
	leaq (%r10,%r10), %rdx
	leaq (%r10,%r10,4), %r11
	leaq (%rdx,%rdx,2), %r14
	leaq (,%r10,8), %r15
	subq %r10, %r15
	vmovq %rax, %xmm0
	vmovq %r9, %xmm1
	vpsrlq $32, %xmm0, %xmm2
	vpmuludq %xmm1, %xmm2, %xmm2
	vpsrlq $32, %xmm1, %xmm3
	vpmuludq %xmm3, %xmm0, %xmm3
	vpaddq %xmm2, %xmm3, %xmm2
	vpsllq $32, %xmm2, %xmm2
	vpmuludq %xmm1, %xmm0, %xmm0
	vpaddq %xmm2, %xmm0, %xmm0
	vpbroadcastq %xmm0, %ymm1
	vmovdqa .LCPI12_0(%rip), %ymm2
	vpmuludq %ymm2, %ymm1, %ymm3
	vpsrlq $32, %ymm1, %ymm4
	vpmuludq %ymm2, %ymm4, %ymm2
	vpsllq $32, %ymm2, %ymm2
	vpaddq %ymm2, %ymm3, %ymm9
	vpbroadcastq .LCPI12_1(%rip), %ymm2
	vpmuludq %ymm2, %ymm1, %ymm3
	vmovdqa .LCPI12_2(%rip), %ymm5
	vpmuludq %ymm5, %ymm4, %ymm4
	vpaddq %ymm4, %ymm3, %ymm3
	vpsllq $32, %ymm3, %ymm3
	vpmuludq %ymm5, %ymm1, %ymm1
	vpinsrq $1, %r10, %xmm0, %xmm4
	vpaddq %ymm3, %ymm1, %ymm10
	vmovdqa .LCPI12_3(%rip), %xmm1
	vpmuludq %xmm1, %xmm4, %xmm3
	vpsrlq $32, %xmm4, %xmm5
	vpmuludq %xmm1, %xmm5, %xmm1
	vpsllq $32, %xmm1, %xmm1
	vpaddq %xmm1, %xmm3, %xmm11
	vpmuludq %xmm2, %xmm4, %xmm1
	vmovdqa .LCPI12_4(%rip), %xmm2
	vpmuludq %xmm2, %xmm5, %xmm3
	vpaddq %xmm3, %xmm1, %xmm1
	vpsllq $32, %xmm1, %xmm1
	vpmuludq %xmm2, %xmm4, %xmm2
	vpaddq %xmm1, %xmm2, %xmm12
	leaq -1(%r9), %rax
	movq %rax, 144(%rsp)
	vmovq %rdx, %xmm1
	vmovq %r10, %xmm2
	vpbroadcastq %xmm2, %xmm2
	vpunpcklqdq %xmm1, %xmm2, %xmm1
	vmovdqa .LCPI12_5(%rip), %xmm3
	vpmuludq %xmm3, %xmm2, %xmm4
	vpsrlq $32, %xmm2, %xmm2
	vpmuludq %xmm3, %xmm2, %xmm2
	vpsllq $32, %xmm2, %xmm2
	vpaddq %xmm2, %xmm4, %xmm2
	vinserti128 $1, %xmm2, %ymm1, %ymm1
	vmovq %rsi, %xmm2
	vpbroadcastq %xmm2, %ymm13
	vmovq %rcx, %xmm2
	vpbroadcastq %xmm2, %ymm14
	vmovq %xmm0, 136(%rsp)
	vpcmpeqd %ymm0, %ymm0, %ymm0
	vpaddq %ymm0, %ymm1, %ymm1
	leaq (%r10,%r10,4), %rax
	decq %rax
	cmpq $15, %rax
	setb %al
	movq %rdx, 96(%rsp)
	leaq (%rdx,%rdx,2), %rdx
	decq %rdx
	cmpq $15, %rdx
	setb %dl
	leaq -1(%r15), %r8
	cmpq $15, %r8
	setb %bpl
	vpbroadcastq .LCPI12_6(%rip), %ymm15
	vpxor %ymm1, %ymm15, %ymm1
	vpbroadcastq .LCPI12_7(%rip), %ymm2
	vpcmpgtq %ymm2, %ymm1, %ymm1
	vtestpd %ymm0, %ymm1
	setae %r8b
	orb %dl, %bpl
	orb %al, %bpl
	orb %r8b, %bpl
	movb %bpl, 7(%rsp)
	movq %r15, 56(%rsp)
	movq 8(%rsp), %rcx
	leaq (%rcx,%r15), %r8
	movq %r14, 64(%rsp)
	leaq (%rcx,%r14), %r12
	movq %r11, 72(%rsp)
	addq %rcx, %r11
	leaq (,%r10,4), %rax
	movq %rax, 80(%rsp)
	leaq (%rcx,%r10,4), %r14
	leaq (%r10,%r10,2), %rax
	leaq 127(%rdi), %r15
	addq $3, %rdi
	movq %rdi, 104(%rsp)
	movq %rax, 88(%rsp)
	leaq (%rcx,%rax), %rbp
	leaq (%rcx,%r10,2), %rax
	movq %rax, 16(%rsp)
	movq %r10, 112(%rsp)
	leaq (%rcx,%r10), %rax
	movq $0, 40(%rsp)
	movq $0, 120(%rsp)
	xorl %ecx, %ecx
	movq %rbx, 128(%rsp)
	vmovdqu %ymm9, 336(%rsp)
	vmovdqu %ymm10, 304(%rsp)
	vmovdqa %xmm11, 160(%rsp)
	vmovdqu %ymm12, 272(%rsp)
	vmovdqu %ymm13, 240(%rsp)
	vmovdqu %ymm14, 208(%rsp)
	vmovdqu %ymm15, 176(%rsp)
	movq %r13, 152(%rsp)
	vmovdqa .LCPI12_20(%rip), %ymm7
.LBB12_3:
	movq %rax, 24(%rsp)
	movq %r9, %r10
	imulq %rcx, %r10
	movq 136(%rsp), %rax
	addq %r10, %rax
	cmpq %rax, %r13
	movq %rax, %rdx
	cmovaq %r13, %rdx
	cmpq %r10, %r13
	vmovq %r10, %xmm1
	cmovaq %r13, %r10
	movq %rcx, 48(%rsp)
	imulq %rcx, %rbx
	vmovq %rbx, %xmm0
	vpbroadcastq %xmm0, %ymm0
	vpaddq .LCPI12_8(%rip), %ymm0, %ymm2
	vpaddq .LCPI12_9(%rip), %ymm0, %ymm3
	vpxor %ymm15, %ymm13, %ymm4
	vpxor %ymm3, %ymm15, %ymm5
	vpcmpgtq %ymm5, %ymm4, %ymm5
	vblendvpd %ymm5, %ymm13, %ymm3, %ymm3
	vpxor %ymm2, %ymm15, %ymm5
	vpcmpgtq %ymm5, %ymm4, %ymm4
	vblendvpd %ymm4, %ymm13, %ymm2, %ymm2
	vpbroadcastq %xmm1, %ymm1
	vpaddq %ymm1, %ymm9, %ymm4
	vpxor %ymm15, %ymm14, %ymm5
	vpxor %ymm4, %ymm15, %ymm6
	vpcmpgtq %ymm6, %ymm5, %ymm5
	vblendvpd %ymm5, %ymm14, %ymm4, %ymm4
	vpaddq %xmm1, %xmm11, %xmm5
	vpxor %xmm15, %xmm14, %xmm6
	vpxor %xmm5, %xmm15, %xmm8
	vpcmpgtq %xmm8, %xmm6, %xmm6
	vblendvpd %xmm6, %xmm14, %xmm5, %xmm5
	vmovq %r10, %xmm6
	vmovq %rdx, %xmm8
	vpunpcklqdq %xmm6, %xmm8, %xmm6
	vmovq %rax, %xmm8
	vinserti128 $1, %xmm6, %ymm12, %ymm6
	vpunpcklqdq %xmm1, %xmm8, %xmm8
	vinserti128 $1, %xmm8, %ymm1, %ymm8
	vpsubq %ymm8, %ymm6, %ymm6
	vpsubq %ymm0, %ymm2, %ymm2
	vpsubq %ymm0, %ymm3, %ymm0
	vpaddq .LCPI12_10(%rip), %ymm0, %ymm0
	vpsubq %ymm1, %ymm10, %ymm1
	vpaddq %ymm4, %ymm1, %ymm1
	vpaddq %ymm5, %ymm6, %ymm3
	vpsrlq $3, %ymm0, %ymm0
	vpxor %ymm3, %ymm15, %ymm4
	vpor %ymm0, %ymm15, %ymm5
	vpcmpgtq %ymm5, %ymm4, %ymm4
	vpaddq .LCPI12_11(%rip), %ymm2, %ymm2
	vpsrlq $3, %ymm2, %ymm2
	vblendvpd %ymm4, %ymm0, %ymm3, %ymm0
	vpxor %ymm1, %ymm15, %ymm3
	vpor %ymm2, %ymm15, %ymm4
	vpcmpgtq %ymm4, %ymm3, %ymm3
	vblendvpd %ymm3, %ymm2, %ymm1, %ymm1
	vxorpd %ymm0, %ymm15, %ymm2
	vxorpd %ymm1, %ymm15, %ymm3
	vpcmpgtq %ymm3, %ymm2, %ymm2
	vblendvpd %ymm2, %ymm1, %ymm0, %ymm0
	vxorpd %xmm0, %xmm15, %xmm1
	vextractf128 $1, %ymm0, %xmm2
	vxorpd %xmm2, %xmm15, %xmm3
	vpcmpgtq %xmm1, %xmm3, %xmm1
	vblendvpd %xmm1, %xmm0, %xmm2, %xmm0
	vpextrq $1, %xmm0, %rcx
	vmovq %xmm0, %rax
	cmpq %rcx, %rax
	cmovbq %rax, %rcx
	movq 144(%rsp), %rax
	cmpq %rax, %rcx
	cmovaeq %rax, %rcx
	cmpq $16, %rcx
	setb %al
	orb 7(%rsp), %al
	testb $1, %al
	je .LBB12_5
	xorl %ecx, %ecx
	jmp .LBB12_8
.LBB12_5:
	incq %rcx
	movl %ecx, %eax
	andl $15, %eax
	movl $16, %edx
	cmoveq %rdx, %rax
	subq %rax, %rcx
	xorl %eax, %eax
	movq 16(%rsp), %rdi
	movq 24(%rsp), %rbx
.LBB12_6:
	movzbl -127(%r15,%rax,8), %edx
	vmovd %edx, %xmm0
	vpinsrb $1, -119(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $2, -111(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $3, -103(%r15,%rax,8), %xmm0, %xmm8
	vpinsrb $4, -95(%r15,%rax,8), %xmm8, %xmm0
	vpinsrb $5, -87(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $6, -79(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $7, -71(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $8, -63(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $9, -55(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $10, -47(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $11, -39(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $12, -31(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $13, -23(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $14, -15(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $15, -7(%r15,%rax,8), %xmm0, %xmm9
	movzbl -125(%r15,%rax,8), %edx
	vmovd %edx, %xmm0
	vpinsrb $1, -117(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $2, -109(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $3, -101(%r15,%rax,8), %xmm0, %xmm10
	vpinsrb $4, -93(%r15,%rax,8), %xmm10, %xmm0
	vpinsrb $5, -85(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $6, -77(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $7, -69(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $8, -61(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $9, -53(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $10, -45(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $11, -37(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $12, -29(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $13, -21(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $14, -13(%r15,%rax,8), %xmm0, %xmm0
	vpinsrb $15, -5(%r15,%rax,8), %xmm0, %xmm0
	movzbl -124(%r15,%rax,8), %edx
	vmovd %edx, %xmm1
	vpinsrb $1, -116(%r15,%rax,8), %xmm1, %xmm1
	vpinsrb $2, -108(%r15,%rax,8), %xmm1, %xmm1
	vpinsrb $3, -100(%r15,%rax,8), %xmm1, %xmm1
	vpinsrb $4, -92(%r15,%rax,8), %xmm1, %xmm2
	vpinsrb $5, -84(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $6, -76(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $7, -68(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $8, -60(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $9, -52(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $10, -44(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $11, -36(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $12, -28(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $13, -20(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $14, -12(%r15,%rax,8), %xmm2, %xmm2
	vpinsrb $15, -4(%r15,%rax,8), %xmm2, %xmm2
	movzbl -122(%r15,%rax,8), %edx
	vmovd %edx, %xmm3
	vpinsrb $1, -114(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $2, -106(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $3, -98(%r15,%rax,8), %xmm3, %xmm4
	vpinsrb $4, -90(%r15,%rax,8), %xmm4, %xmm3
	vpinsrb $5, -82(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $6, -74(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $7, -66(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $8, -58(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $9, -50(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $10, -42(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $11, -34(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $12, -26(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $13, -18(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $14, -10(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $15, -2(%r15,%rax,8), %xmm3, %xmm5
	movzbl -121(%r15,%rax,8), %edx
	vmovd %edx, %xmm3
	vpinsrb $1, -113(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $2, -105(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $3, -97(%r15,%rax,8), %xmm3, %xmm11
	vpinsrb $4, -89(%r15,%rax,8), %xmm11, %xmm3
	vpinsrb $5, -81(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $6, -73(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $7, -65(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $8, -57(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $9, -49(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $10, -41(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $11, -33(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $12, -25(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $13, -17(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $14, -9(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $15, -1(%r15,%rax,8), %xmm3, %xmm12
	movzbl -120(%r15,%rax,8), %edx
	vmovd %edx, %xmm3
	vpinsrb $1, -112(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $2, -104(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $3, -96(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $4, -88(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $5, -80(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $6, -72(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $7, -64(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $8, -56(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $9, -48(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $10, -40(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $11, -32(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $12, -24(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $13, -16(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $14, -8(%r15,%rax,8), %xmm3, %xmm3
	vpinsrb $15, (%r15,%rax,8), %xmm3, %xmm3
	movzbl -123(%r15,%rax,8), %edx
	vpmovzxbw %xmm3, %ymm6
	vpmovzxwd %xmm6, %ymm3
	vextracti128 $1, %ymm3, %xmm13
	vpmovzxdq %xmm13, %ymm13
	vextracti128 $1, %ymm6, %xmm6
	vpmovzxwd %xmm6, %ymm6
	vpsllq $56, %ymm13, %ymm13
	vpmovzxbq %xmm11, %ymm11
	vpmovzxbw %xmm12, %ymm12
	vpshufd $238, %xmm12, %xmm14
	vpmovzxwq %xmm14, %ymm14
	vextracti128 $1, %ymm12, %xmm12
	vpsllq $48, %ymm14, %ymm14
	vpsllq $48, %ymm11, %ymm11
	vpmovzxbw %xmm5, %ymm5
	vpmovzxbq %xmm4, %ymm4
	vpsllq $40, %ymm4, %ymm4
	vpor %ymm4, %ymm11, %ymm11
	vpshufd $238, %xmm5, %xmm4
	vpmovzxwq %xmm4, %ymm4
	vpsllq $40, %ymm4, %ymm4
	vpor %ymm4, %ymm14, %ymm4
	vpmovzxwq %xmm12, %ymm14
	vpsllq $48, %ymm14, %ymm14
	vextracti128 $1, %ymm5, %xmm15
	vpor %ymm4, %ymm13, %ymm4
	vpmovzxwq %xmm15, %ymm5
	vpsllq $40, %ymm5, %ymm5
	vpor %ymm5, %ymm14, %ymm5
	vpmovzxdq %xmm6, %ymm13
	vpsllq $56, %ymm13, %ymm13
	vpor %ymm5, %ymm13, %ymm5
	vmovd %edx, %xmm13
	vpinsrb $1, -115(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $2, -107(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $3, -99(%r15,%rax,8), %xmm13, %xmm13
	movzbl -126(%r15,%rax,8), %r10d
	vpshufd $238, %xmm12, %xmm12
	vpmovzxwq %xmm12, %ymm12
	vpsllq $48, %ymm12, %ymm12
	vpshufd $238, %xmm15, %xmm14
	vpmovzxwq %xmm14, %ymm14
	vpsllq $40, %ymm14, %ymm14
	vpor %ymm14, %ymm12, %ymm12
	vpinsrb $4, -91(%r15,%rax,8), %xmm13, %xmm14
	vpinsrb $5, -83(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $6, -75(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $7, -67(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $8, -59(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $9, -51(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $10, -43(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $11, -35(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $12, -27(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $13, -19(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $14, -11(%r15,%rax,8), %xmm14, %xmm14
	vpinsrb $15, -3(%r15,%rax,8), %xmm14, %xmm14
	vextracti128 $1, %ymm6, %xmm6
	vpmovzxdq %xmm6, %ymm6
	vpsllq $56, %ymm6, %ymm6
	vpor %ymm6, %ymm12, %ymm6
	vpmovzxbq %xmm13, %ymm12
	vpmovzxbw %xmm14, %ymm13
	vpshufd $238, %xmm13, %xmm14
	vpmovzxwq %xmm14, %ymm14
	vextracti128 $1, %ymm13, %xmm13
	vpsllq $32, %ymm12, %ymm12
	vpor %ymm12, %ymm11, %ymm11
	vpmovzxwq %xmm13, %ymm12
	vpshufd $238, %xmm13, %xmm13
	vpmovzxwq %xmm13, %ymm13
	vpsllq $32, %ymm13, %ymm13
	vpsllq $32, %ymm12, %ymm12
	vpsllq $32, %ymm14, %ymm14
	vpmovzxbw %xmm2, %ymm2
	vpmovzxbq %xmm1, %ymm1
	vpsllq $24, %ymm1, %ymm1
	vpor %ymm1, %ymm11, %ymm1
	vextracti128 $1, %ymm2, %xmm11
	vpshufd $238, %xmm2, %xmm2
	vpmovzxwq %xmm2, %ymm2
	vpsllq $24, %ymm2, %ymm2
	vpor %ymm2, %ymm14, %ymm2
	vpshufd $238, %xmm11, %xmm14
	vpmovzxwq %xmm14, %ymm14
	vpmovzxwq %xmm11, %ymm11
	vpsllq $24, %ymm11, %ymm11
	vpor %ymm11, %ymm12, %ymm12
	vpsllq $24, %ymm14, %ymm11
	vpor %ymm11, %ymm13, %ymm13
	vpmovzxbw %xmm0, %ymm11
	vextracti128 $1, %ymm11, %xmm14
	vpshufd $238, %xmm14, %xmm0
	vpmovzxwq %xmm0, %ymm0
	vpsllq $16, %ymm0, %ymm0
	vpor %ymm0, %ymm13, %ymm0
	vmovd %r10d, %xmm13
	vpinsrb $1, -118(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $2, -110(%r15,%rax,8), %xmm13, %xmm13
	vpor %ymm0, %ymm6, %ymm0
	vpinsrb $3, -102(%r15,%rax,8), %xmm13, %xmm6
	vpmovzxwq %xmm14, %ymm13
	vpsllq $16, %ymm13, %ymm13
	vpor %ymm13, %ymm12, %ymm12
	vpinsrb $4, -94(%r15,%rax,8), %xmm6, %xmm13
	vpinsrb $5, -86(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $6, -78(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $7, -70(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $8, -62(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $9, -54(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $10, -46(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $11, -38(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $12, -30(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $13, -22(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $14, -14(%r15,%rax,8), %xmm13, %xmm13
	vpinsrb $15, -6(%r15,%rax,8), %xmm13, %xmm13
	vpmovzxdq %xmm3, %ymm3
	vpsllq $56, %ymm3, %ymm3
	vpmovzxbq %xmm10, %ymm10
	vpshufd $238, %xmm11, %xmm11
	vpmovzxwq %xmm11, %ymm11
	vpor %ymm5, %ymm12, %ymm5
	vpsllq $16, %ymm11, %ymm11
	vpor %ymm2, %ymm11, %ymm2
	vpor %ymm2, %ymm4, %ymm2
	vpsllq $16, %ymm10, %ymm4
	vpor %ymm4, %ymm1, %ymm1
	vpmovzxbw %xmm13, %ymm4
	vextracti128 $1, %ymm4, %xmm10
	vpshufd $238, %xmm10, %xmm11
	vpmovzxwq %xmm11, %ymm11
	vpmovzxwq %xmm10, %ymm10
	vpshufd $238, %xmm4, %xmm4
	vpmovzxwq %xmm4, %ymm4
	vpmovzxbq %xmm6, %ymm6
	vpsllq $8, %ymm6, %ymm6
	vpor %ymm6, %ymm1, %ymm1
	vpor %ymm1, %ymm3, %ymm1
	vpsllq $8, %ymm4, %ymm3
	vpor %ymm3, %ymm2, %ymm2
	vpsllq $8, %ymm10, %ymm3
	vpor %ymm3, %ymm5, %ymm3
	vpsllq $8, %ymm11, %ymm4
	vpor %ymm4, %ymm0, %ymm0
	vpmovzxbw %xmm9, %ymm4
	vextracti128 $1, %ymm4, %xmm5
	vpshufd $238, %xmm5, %xmm6
	vpmovzxwq %xmm6, %ymm6
	vpmovzxwq %xmm5, %ymm5
	vpshufd $238, %xmm4, %xmm4
	vpmovzxwq %xmm4, %ymm4
	vpmovzxbq %xmm8, %ymm8
	vpor %ymm1, %ymm8, %ymm8
	vpor %ymm4, %ymm2, %ymm4
	vpor %ymm5, %ymm3, %ymm5
	vpor %ymm6, %ymm0, %ymm6
	vpsrlq $7, %ymm1, %ymm1
	vpsrlq $7, %ymm2, %ymm2
	vpsrlq $7, %ymm3, %ymm3
	vpsrlq $7, %ymm0, %ymm0
	vpxor %ymm6, %ymm0, %ymm0
	vpxor %ymm5, %ymm3, %ymm3
	vpxor %ymm4, %ymm2, %ymm2
	vpxor %ymm1, %ymm8, %ymm1
	vpbroadcastq .LCPI12_12(%rip), %ymm9
	vpand %ymm1, %ymm9, %ymm1
	vpand %ymm2, %ymm9, %ymm2
	vpand %ymm3, %ymm9, %ymm3
	vpand %ymm0, %ymm9, %ymm0
	vpsllq $7, %ymm0, %ymm9
	vpor %ymm0, %ymm9, %ymm0
	vpsllq $7, %ymm3, %ymm9
	vpor %ymm3, %ymm9, %ymm3
	vpsllq $7, %ymm2, %ymm9
	vpor %ymm2, %ymm9, %ymm2
	vpsllq $7, %ymm1, %ymm9
	vpor %ymm1, %ymm9, %ymm1
	vpxor %ymm1, %ymm8, %ymm1
	vpxor %ymm4, %ymm2, %ymm2
	vpxor %ymm5, %ymm3, %ymm3
	vpxor %ymm6, %ymm0, %ymm0
	vpsrlq $14, %ymm0, %ymm4
	vpsrlq $14, %ymm3, %ymm5
	vpsrlq $14, %ymm2, %ymm6
	vpsrlq $14, %ymm1, %ymm8
	vpxor %ymm1, %ymm8, %ymm8
	vpxor %ymm2, %ymm6, %ymm6
	vpxor %ymm3, %ymm5, %ymm5
	vpxor %ymm0, %ymm4, %ymm4
	vpbroadcastq .LCPI12_13(%rip), %ymm9
	vpand %ymm4, %ymm9, %ymm4
	vpand %ymm5, %ymm9, %ymm5
	vpand %ymm6, %ymm9, %ymm6
	vpand %ymm9, %ymm8, %ymm8
	vpsllq $14, %ymm8, %ymm9
	vpor %ymm8, %ymm9, %ymm8
	vpsllq $14, %ymm6, %ymm9
	vpor %ymm6, %ymm9, %ymm6
	vpsllq $14, %ymm5, %ymm9
	vpor %ymm5, %ymm9, %ymm5
	vpsllq $14, %ymm4, %ymm9
	vpor %ymm4, %ymm9, %ymm4
	vpxor %ymm0, %ymm4, %ymm4
	vpxor %ymm3, %ymm5, %ymm3
	vpxor %ymm2, %ymm6, %ymm2
	vpxor %ymm1, %ymm8, %ymm0
	vpsrlq $28, %ymm0, %ymm1
	vpsrlq $28, %ymm2, %ymm5
	vpsrlq $28, %ymm3, %ymm6
	vpsrlq $28, %ymm4, %ymm8
	vpxor %ymm4, %ymm8, %ymm8
	vpxor %ymm3, %ymm6, %ymm6
	vpxor %ymm2, %ymm5, %ymm5
	vpxor %ymm0, %ymm1, %ymm1
	vpbroadcastq .LCPI12_14(%rip), %ymm9
	vpand %ymm1, %ymm9, %ymm1
	vpand %ymm5, %ymm9, %ymm5
	vpand %ymm6, %ymm9, %ymm6
	vpand %ymm9, %ymm8, %ymm8
	vpsllq $28, %ymm8, %ymm9
	vpor %ymm8, %ymm9, %ymm8
	vpsllq $28, %ymm6, %ymm9
	vpor %ymm6, %ymm9, %ymm6
	vpsllq $28, %ymm5, %ymm9
	vpor %ymm5, %ymm9, %ymm5
	vpsllq $28, %ymm1, %ymm9
	vpor %ymm1, %ymm9, %ymm1
	vpxor %ymm0, %ymm1, %ymm0
	vpxor %ymm2, %ymm5, %ymm1
	vpxor %ymm3, %ymm6, %ymm3
	vpxor %ymm4, %ymm8, %ymm4
	vmovdqa .LCPI12_15(%rip), %ymm6
	vpshufb %ymm6, %ymm4, %ymm2
	vpshufb %ymm6, %ymm3, %ymm5
	vpackusdw %ymm2, %ymm5, %ymm2
	vpermq $216, %ymm2, %ymm2
	vpshufb %ymm6, %ymm1, %ymm5
	vpshufb %ymm6, %ymm0, %ymm6
	vpackusdw %ymm5, %ymm6, %ymm5
	vpermq $216, %ymm5, %ymm5
	vpackusdw %ymm2, %ymm5, %ymm2
	vextracti128 $1, %ymm2, %xmm5
	vpackuswb %xmm5, %xmm2, %xmm2
	vmovdqa .LCPI12_16(%rip), %ymm8
	vpshufb %ymm8, %ymm4, %ymm5
	vpshufb %ymm8, %ymm3, %ymm6
	vpackusdw %ymm5, %ymm6, %ymm5
	vpermq $216, %ymm5, %ymm5
	vpshufb %ymm8, %ymm1, %ymm6
	vpshufb %ymm8, %ymm0, %ymm8
	vpackusdw %ymm6, %ymm8, %ymm6
	vpermq $216, %ymm6, %ymm6
	vpackusdw %ymm5, %ymm6, %ymm5
	vextracti128 $1, %ymm5, %xmm6
	vpackuswb %xmm6, %xmm5, %xmm5
	vmovdqa .LCPI12_17(%rip), %ymm9
	vpshufb %ymm9, %ymm4, %ymm6
	vpshufb %ymm9, %ymm3, %ymm8
	vpackusdw %ymm6, %ymm8, %ymm6
	vpermq $216, %ymm6, %ymm6
	vpshufb %ymm9, %ymm1, %ymm8
	vpshufb %ymm9, %ymm0, %ymm9
	vpackusdw %ymm8, %ymm9, %ymm8
	vpermq $216, %ymm8, %ymm8
	vpackusdw %ymm6, %ymm8, %ymm6
	vextracti128 $1, %ymm6, %xmm8
	vpackuswb %xmm8, %xmm6, %xmm8
	vmovdqa .LCPI12_18(%rip), %ymm10
	vpshufb %ymm10, %ymm4, %ymm6
	vpshufb %ymm10, %ymm3, %ymm9
	vpackusdw %ymm6, %ymm9, %ymm6
	vpermq $216, %ymm6, %ymm6
	vpshufb %ymm10, %ymm1, %ymm9
	vpshufb %ymm10, %ymm0, %ymm10
	vpackusdw %ymm9, %ymm10, %ymm9
	vpermq $216, %ymm9, %ymm9
	vpackusdw %ymm6, %ymm9, %ymm6
	vextracti128 $1, %ymm6, %xmm9
	vpackuswb %xmm9, %xmm6, %xmm6
	vmovdqa .LCPI12_19(%rip), %ymm11
	vpshufb %ymm11, %ymm4, %ymm9
	vpshufb %ymm11, %ymm3, %ymm10
	vpackusdw %ymm9, %ymm10, %ymm9
	vpermq $216, %ymm9, %ymm9
	vpshufb %ymm11, %ymm1, %ymm10
	vpshufb %ymm11, %ymm0, %ymm11
	vpackusdw %ymm10, %ymm11, %ymm10
	vpermq $216, %ymm10, %ymm10
	vpackusdw %ymm9, %ymm10, %ymm9
	vextracti128 $1, %ymm9, %xmm10
	vpackuswb %xmm10, %xmm9, %xmm9
	vpshufb %ymm7, %ymm4, %ymm10
	vpshufb %ymm7, %ymm3, %ymm11
	vpackusdw %ymm10, %ymm11, %ymm10
	vpermq $216, %ymm10, %ymm10
	vpshufb %ymm7, %ymm1, %ymm11
	vpshufb %ymm7, %ymm0, %ymm12
	vpackusdw %ymm11, %ymm12, %ymm11
	vpermq $216, %ymm11, %ymm11
	vpackusdw %ymm10, %ymm11, %ymm10
	vextracti128 $1, %ymm10, %xmm11
	vpackuswb %xmm11, %xmm10, %xmm10
	vpsrlq $56, %ymm1, %ymm11
	vpsrlq $56, %ymm0, %ymm12
	vpackusdw %ymm11, %ymm12, %ymm11
	vpsrlq $56, %ymm4, %ymm12
	vpsrlq $56, %ymm3, %ymm13
	vpackusdw %ymm12, %ymm13, %ymm12
	vpermq $216, %ymm12, %ymm12
	vpermq $216, %ymm11, %ymm11
	vpackusdw %ymm12, %ymm11, %ymm11
	vextracti128 $1, %ymm11, %xmm12
	vpackuswb %xmm12, %xmm11, %xmm11
	vpbroadcastq .LCPI12_21(%rip), %ymm12
	vpand %ymm4, %ymm12, %ymm4
	vpand %ymm3, %ymm12, %ymm3
	vpackusdw %ymm4, %ymm3, %ymm3
	vpand %ymm1, %ymm12, %ymm1
	vpand %ymm0, %ymm12, %ymm0
	vpackusdw %ymm1, %ymm0, %ymm0
	vpermq $216, %ymm3, %ymm1
	vpermq $216, %ymm0, %ymm0
	vpackusdw %ymm1, %ymm0, %ymm0
	vextracti128 $1, %ymm0, %xmm1
	vpackuswb %xmm1, %xmm0, %xmm0
	vpshufd $216, %xmm0, %xmm0
	movq 8(%rsp), %rdx
	vmovdqu %xmm0, (%rdx,%rax)
	vpshufd $216, %xmm2, %xmm0
	vmovdqu %xmm0, (%rbx,%rax)
	vpshufd $216, %xmm5, %xmm0
	vmovdqu %xmm0, (%rdi,%rax)
	vpshufd $216, %xmm8, %xmm0
	vmovdqu %xmm0, (%rbp,%rax)
	vpshufd $216, %xmm6, %xmm0
	vmovdqu %xmm0, (%r14,%rax)
	vpshufd $216, %xmm9, %xmm0
	vmovdqu %xmm0, (%r11,%rax)
	vpshufd $216, %xmm10, %xmm0
	vmovdqu %xmm0, (%r12,%rax)
	vpshufd $216, %xmm11, %xmm0
	vmovdqu %xmm0, (%r8,%rax)
	addq $16, %rax
	cmpq %rax, %rcx
	jne .LBB12_6
	vmovdqu 336(%rsp), %ymm9
	vmovdqu 304(%rsp), %ymm10
	vmovdqa 160(%rsp), %xmm11
	vmovdqu 272(%rsp), %ymm12
	vmovupd 240(%rsp), %ymm13
	vmovupd 208(%rsp), %ymm14
	vmovupd 176(%rsp), %ymm15
.LBB12_8:
	incq 48(%rsp)
	movq 40(%rsp), %rax
	leaq (%rax,%rcx,8), %r10
.LBB12_9:
	cmpq %rsi, %r10
	jae .LBB12_10
	leaq 1(%r10), %rax
	cmpq %rsi, %rax
	jae .LBB12_11
	leaq 2(%r10), %rax
	cmpq %rsi, %rax
	jae .LBB12_11
	leaq 3(%r10), %rax
	cmpq %rsi, %rax
	jae .LBB12_11
	leaq 4(%r10), %rax
	cmpq %rsi, %rax
	jae .LBB12_11
	leaq 5(%r10), %rax
	cmpq %rsi, %rax
	jae .LBB12_11
	leaq 6(%r10), %rax
	cmpq %rsi, %rax
	jae .LBB12_11
	leaq 7(%r10), %rax
	cmpq %rsi, %rax
	jae .LBB12_11
	movq 120(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movq %rbp, %rdi
	leaq 1(%rcx), %rbx
	movq 104(%rsp), %rbp
	movzbl -11(%rbp,%rbx,8), %eax
	movzbl -5(%rbp,%rbx,8), %edx
	movzbl -4(%rbp,%rbx,8), %r13d
	shlq $56, %r13
	shlq $48, %rdx
	orq %r13, %rdx
	movzbl -6(%rbp,%rbx,8), %r13d
	shlq $40, %r13
	orq %rdx, %r13
	movzbl -7(%rbp,%rbx,8), %edx
	shlq $32, %rdx
	orq %r13, %rdx
	movzbl (%rbp,%rcx,8), %r13d
	shll $24, %r13d
	orq %rdx, %r13
	movzbl -9(%rbp,%rbx,8), %edx
	shll $16, %edx
	orq %r13, %rdx
	movzbl -10(%rbp,%rbx,8), %r13d
	shll $8, %r13d
	orq %rdx, %r13
	orq %r13, %rax
	shrq $7, %r13
	xorq %rax, %r13
	movabsq $47851476196393130, %rdx
	andq %rdx, %r13
	movq %r13, %rdx
	shlq $7, %rdx
	orq %r13, %rdx
	xorq %rax, %rdx
	movq %rdx, %rax
	shrq $14, %rax
	xorq %rdx, %rax
	movabsq $225176545447116, %r13
	andq %r13, %rax
	movq %rax, %r13
	shlq $14, %r13
	orq %rax, %r13
	xorq %rdx, %r13
	movq %r13, %rax
	shrq $28, %rax
	xorl %r13d, %eax
	andl $-252645136, %eax
	movq %rax, %rdx
	shlq $28, %rdx
	orq %rax, %rdx
	xorq %r13, %rdx
	movq 8(%rsp), %rax
	movq 152(%rsp), %r13
	movb %dl, (%rax,%rcx)
	movq 112(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movl %edx, %eax
	shrl $8, %eax
	movq 24(%rsp), %rbp
	movb %al, (%rbp,%rcx)
	movq 96(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movl %edx, %eax
	shrl $16, %eax
	movq 16(%rsp), %rbp
	movb %al, (%rbp,%rcx)
	movq 88(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movl %edx, %eax
	shrl $24, %eax
	movb %al, (%rdi,%rcx)
	movq 80(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movq %rdx, %rax
	shrq $32, %rax
	movb %al, (%r14,%rcx)
	movq 72(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movq %rdx, %rax
	shrq $40, %rax
	movb %al, (%r11,%rcx)
	movq 64(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movq %rdx, %rax
	shrq $48, %rax
	movb %al, (%r12,%rcx)
	movq 56(%rsp), %rax
	addq %rcx, %rax
	cmpq %r13, %rax
	jae .LBB12_28
	movq %rdi, %rbp
	shrq $56, %rdx
	movb %dl, (%r8,%rcx)
	addq $8, %r10
	movq %rbx, %rcx
	cmpq %rbx, %r9
	jne .LBB12_9
	addq %r9, %r8
	addq %r9, %r12
	addq %r9, %r11
	addq %r9, %r14
	addq %r9, %rbp
	addq %r9, 16(%rsp)
	movq 24(%rsp), %rax
	addq %r9, %rax
	addq %r9, 8(%rsp)
	movq 128(%rsp), %rbx
	addq %rbx, %r15
	addq %r9, 120(%rsp)
	addq %r9, 56(%rsp)
	addq %r9, 64(%rsp)
	addq %r9, 72(%rsp)
	addq %r9, 80(%rsp)
	addq %r9, 88(%rsp)
	addq %r9, 96(%rsp)
	addq %r9, 112(%rsp)
	addq %rbx, 40(%rsp)
	addq %rbx, 104(%rsp)
	movq 48(%rsp), %rcx
	cmpq 32(%rsp), %rcx
	jne .LBB12_3
.LBB12_29:
	addq $376, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	vzeroupper
	retq
.LBB12_28:
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.9(%rip), %rdx
	movq %rax, %rdi
	movq %r13, %rsi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
.LBB12_10:
	movq %r10, %rax
.LBB12_11:
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.10(%rip), %rdx
	movq %rax, %rdi
	vzeroupper
	callq *core::panicking::panic_bounds_check@GOTPCREL(%rip)
