probe_bit_shuffle_untrans_bit_byte:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $252, %esp
	movl 288(%esp), %ebx
	calll .L47$pb
.L47$pb:
	popl %eax
.Ltmp5566:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp5566-.L47$pb), %eax
	movl %eax, 12(%esp)
	shrl $3, %ebx
	cmpl $0, 292(%esp)
	sete %al
	testl %ebx, %ebx
	sete %cl
	orb %al, %cl
	jne .LBB47_19
	movl 12(%esp), %eax
	movl 292(%esp), %ecx
	movl 276(%esp), %edi
	pxor %xmm5, %xmm5
	movl $8, 124(%esp)
	movl $-1, 112(%esp)
	movl $7, 108(%esp)
	movl $0, 104(%esp)
	movl $0, 60(%esp)
	movl $0, 8(%esp)
	movl %ebx, 172(%esp)
	movdqa .LCPI47_0@GOTOFF(%eax), %xmm1
	imull %ebx, %ecx
	movdqa .LCPI47_1@GOTOFF(%eax), %xmm3
	movdqa .LCPI47_3@GOTOFF(%eax), %xmm6
	movdqa .LCPI47_4@GOTOFF(%eax), %xmm7
	movd %ecx, %xmm0
	leal (%ecx,%ecx), %esi
	leal (,%ecx,8), %edx
	movl %ecx, %ebp
	pshufd $0, %xmm0, %xmm0
	subl %edx, %ebp
	movl %esi, 140(%esp)
	leal (%esi,%esi,2), %esi
	subl %ecx, %edx
	movl %esi, 20(%esp)
	negl %esi
	movl %edx, 16(%esp)
	movl 272(%esp), %edx
	movl %ebp, 88(%esp)
	movl %ecx, %ebp
	pshufd $245, %xmm1, %xmm2
	pmuludq %xmm0, %xmm1
	movl %esi, 92(%esp)
	negl %ebp
	pmuludq %xmm0, %xmm2
	pshufd $232, %xmm1, %xmm1
	movl %ebp, 116(%esp)
	movl 280(%esp), %ebp
	pshufd $232, %xmm2, %xmm2
	punpckldq %xmm2, %xmm1
	pshufd $245, %xmm3, %xmm2
	movl %edx, 84(%esp)
	pmuludq %xmm0, %xmm2
	pmuludq %xmm3, %xmm0
	movdqa .LCPI47_2@GOTOFF(%eax), %xmm3
	movl 284(%esp), %eax
	movdqa %xmm1, 224(%esp)
	pshufd $232, %xmm2, %xmm2
	pshufd $232, %xmm0, %xmm0
	punpckldq %xmm2, %xmm0
	movdqa %xmm0, 208(%esp)
	movd %edi, %xmm0
	leal 1(%eax), %esi
	movl 16(%esp), %eax
	pshufd $0, %xmm0, %xmm0
	movl %esi, 152(%esp)
	leal -1(%ebx), %esi
	movdqa %xmm0, 192(%esp)
	pxor %xmm3, %xmm0
	movl %esi, 120(%esp)
	movdqa %xmm0, 176(%esp)
	leal (%edx,%eax), %esi
	movl 20(%esp), %eax
	movl %esi, 80(%esp)
	movl %ecx, %esi
	movl %esi, 156(%esp)
	leal (%edx,%eax), %eax
	movl %eax, 76(%esp)
	leal (%ecx,%ecx,4), %eax
	movl %eax, 128(%esp)
	leal (%edx,%eax), %eax
	movl %eax, 72(%esp)
	leal (,%ecx,4), %eax
	movl %eax, 132(%esp)
	leal (%edx,%ecx,4), %eax
	leal (%ecx,%ecx,2), %ecx
	movl %ecx, 136(%esp)
	leal (%edx,%ecx), %ecx
	movl %ecx, 48(%esp)
	leal (%edx,%esi,2), %ecx
	leal (%edx,%esi), %edx
	movl %edx, 44(%esp)
	movl 88(%esp), %edx
	movl %edx, 100(%esp)
	movl 16(%esp), %edx
	movl %edx, 52(%esp)
	movl 92(%esp), %edx
	movl %edx, 96(%esp)
	movl 20(%esp), %edx
	movl %edx, 56(%esp)
	xorl %edx, %edx
.LBB47_3:
	movl %ecx, 64(%esp)
	movl %ebp, 148(%esp)
	movl 152(%esp), %ecx
	movl 124(%esp), %ebp
	movl %eax, 68(%esp)
	movl %esi, 144(%esp)
	movl %edx, 40(%esp)
	movdqa 176(%esp), %xmm4
	cmpl %ecx, %ebp
	movl %ecx, %eax
	cmoval %ebp, %eax
	cmpl $-7, %edx
	movl $0, %ebp
	movl %eax, 4(%esp)
	movl $-8, %eax
	cmovael %edx, %eax
	cmpl %esi, %edi
	movl %eax, 160(%esp)
	movl 56(%esp), %eax
	cmoval %edi, %esi
	movl %esi, 164(%esp)
	movl $-8, %esi
	cmpl %eax, %edi
	cmoval %edi, %eax
	movl %eax, 32(%esp)
	movl 52(%esp), %eax
	cmpl %eax, %edi
	cmoval %edi, %eax
	movl %eax, 28(%esp)
	movl 60(%esp), %eax
	cmpl %eax, %edi
	cmoval %edi, %eax
	movl %eax, 24(%esp)
	movl 288(%esp), %eax
	imull 8(%esp), %eax
	subl %eax, %esi
	notl %eax
	movl %esi, %edx
	cmovbl %ebp, %edx
	negl %esi
	movl 88(%esp), %ebp
	cmpl %ecx, %esi
	cmovbel %ecx, %esi
	imull 8(%esp), %ebx
	addl $7, %edx
	movl $0, %ecx
	addl %eax, %esi
	movl 156(%esp), %eax
	shrl $3, %edx
	movl %edx, 36(%esp)
	movl %edi, %edx
	shrl $3, %esi
	movd %ebx, %xmm0
	pshufd $0, %xmm0, %xmm1
	movdqa 224(%esp), %xmm0
	addl %ebx, %eax
	subl %eax, %edx
	movl 20(%esp), %eax
	cmovbl %ecx, %edx
	movl %edx, 168(%esp)
	movl 92(%esp), %edx
	paddd %xmm1, %xmm0
	movdqa %xmm0, %xmm2
	addl %ebx, %eax
	pxor %xmm3, %xmm2
	cmpl %eax, %edi
	pcmpgtd %xmm2, %xmm4
	movdqa 192(%esp), %xmm2
	cmoval %edi, %eax
	subl %ebx, %edx
	addl %eax, %edx
	movl 16(%esp), %eax
	pand %xmm4, %xmm2
	pandn %xmm0, %xmm4
	movdqa 208(%esp), %xmm0
	por %xmm2, %xmm4
	addl %ebx, %eax
	cmpl %eax, %edi
	cmoval %edi, %eax
	subl %ebx, %ebp
	psubd %xmm1, %xmm0
	addl %eax, %ebp
	movl %edi, %eax
	paddd %xmm4, %xmm0
	subl %ebx, %eax
	pshufd $238, %xmm0, %xmm2
	movdqa %xmm0, %xmm1
	cmovbl %ecx, %eax
	movdqa %xmm2, %xmm4
	pxor %xmm3, %xmm1
	pxor %xmm3, %xmm4
	pcmpgtd %xmm1, %xmm4
	pand %xmm4, %xmm0
	pandn %xmm2, %xmm4
	por %xmm0, %xmm4
	pshufd $85, %xmm4, %xmm0
	movd %xmm4, %ecx
	movd %xmm0, %ebx
	cmpl %ebx, %ecx
	cmovbl %ecx, %ebx
	movl 168(%esp), %ecx
	cmpl %edx, %ebx
	cmovbl %ebx, %edx
	cmpl %ecx, %ebp
	cmovael %ecx, %ebp
	cmpl %esi, %eax
	movl 120(%esp), %ecx
	cmovael %esi, %eax
	movl 36(%esp), %esi
	cmpl %ecx, %esi
	cmovael %ecx, %esi
	cmpl %ebp, %edx
	cmovbl %edx, %ebp
	cmpl %esi, %eax
	cmovbl %eax, %esi
	movl %esi, %eax
	cmpl %esi, %ebp
	movl $0, %esi
	cmovbl %ebp, %eax
	cmpl $2, %eax
	jb .LBB47_7
	movl 32(%esp), %ebp
	movl 4(%esp), %ecx
	movl 160(%esp), %edx
	movl 164(%esp), %esi
	andl $536870910, %eax
	movl %eax, 36(%esp)
	addl 96(%esp), %ebp
	addl 112(%esp), %ecx
	addl 108(%esp), %edx
	addl 116(%esp), %esi
	movl %ebp, 32(%esp)
	movl 28(%esp), %ebp
	shrl $3, %ecx
	shrl $3, %edx
	movl 32(%esp), %eax
	addl 100(%esp), %ebp
	movl %ebp, 28(%esp)
	movl 24(%esp), %ebp
	addl 104(%esp), %ebp
	cmpl %ecx, %ebx
	cmovbl %ebx, %ecx
	cmpl %edx, %ecx
	cmovael %edx, %ecx
	movl 28(%esp), %edx
	cmpl %esi, %ecx
	movl %ebp, 24(%esp)
	movl 48(%esp), %ebp
	cmovael %esi, %ecx
	movl 24(%esp), %ebx
	cmpl %eax, %ecx
	cmovael %eax, %ecx
	cmpl %edx, %ecx
	cmovael %edx, %ecx
	movl 120(%esp), %edx
	cmpl %ebx, %ecx
	cmovael %ebx, %ecx
	movl 68(%esp), %ebx
	cmpl %edx, %ecx
	cmovael %edx, %ecx
	movl 64(%esp), %edx
	xorl %esi, %esi
	andl $-2, %ecx
	movl %ecx, 4(%esp)
	movl 44(%esp), %ecx
.LBB47_5:
	movzwl (%ebx,%esi), %eax
	movd %eax, %xmm1
	movl 76(%esp), %eax
	punpcklbw %xmm5, %xmm1
	punpcklwd %xmm5, %xmm1
	movzwl (%eax,%esi), %eax
	movd %eax, %xmm0
	movl 80(%esp), %eax
	movzwl (%eax,%esi), %eax
	movd %eax, %xmm2
	movl 72(%esp), %eax
	punpcklbw %xmm2, %xmm2
	punpcklwd %xmm2, %xmm2
	punpcklbw %xmm5, %xmm0
	punpcklwd %xmm5, %xmm0
	pshufd $212, %xmm2, %xmm2
	movzwl (%eax,%esi), %eax
	punpckldq %xmm5, %xmm0
	psllq $56, %xmm2
	psllq $48, %xmm0
	por %xmm2, %xmm0
	pxor %xmm2, %xmm2
	punpckldq %xmm1, %xmm2
	movd %eax, %xmm1
	movzwl (%edx,%esi), %eax
	punpcklbw %xmm5, %xmm1
	punpcklwd %xmm5, %xmm1
	punpckldq %xmm5, %xmm1
	psllq $40, %xmm1
	por %xmm1, %xmm2
	movd %eax, %xmm1
	movzwl (%ebp,%esi), %eax
	por %xmm0, %xmm2
	punpcklbw %xmm5, %xmm1
	punpcklwd %xmm5, %xmm1
	punpckldq %xmm5, %xmm1
	movd %eax, %xmm0
	movzwl (%ecx,%esi), %eax
	psllq $16, %xmm1
	punpcklbw %xmm5, %xmm0
	punpcklwd %xmm5, %xmm0
	punpckldq %xmm5, %xmm0
	psllq $24, %xmm0
	por %xmm0, %xmm1
	movd %eax, %xmm0
	movl 84(%esp), %eax
	punpcklbw %xmm5, %xmm0
	punpcklwd %xmm5, %xmm0
	punpckldq %xmm5, %xmm0
	movzwl (%eax,%esi), %eax
	psllq $8, %xmm0
	por %xmm1, %xmm0
	por %xmm2, %xmm0
	movd %eax, %xmm1
	movl 12(%esp), %eax
	punpcklbw %xmm5, %xmm1
	punpcklwd %xmm5, %xmm1
	punpckldq %xmm5, %xmm1
	por %xmm0, %xmm1
	psrlq $7, %xmm0
	pxor %xmm1, %xmm0
	pand %xmm6, %xmm0
	movdqa %xmm0, %xmm2
	psllq $7, %xmm2
	por %xmm0, %xmm2
	pxor %xmm1, %xmm2
	movdqa %xmm2, %xmm1
	psrlq $14, %xmm1
	pxor %xmm2, %xmm1
	pand %xmm7, %xmm1
	movdqa %xmm1, %xmm0
	psllq $14, %xmm0
	por %xmm1, %xmm0
	pxor %xmm2, %xmm0
	movdqa %xmm0, %xmm1
	psrlq $28, %xmm1
	pxor %xmm0, %xmm1
	pand .LCPI47_5@GOTOFF(%eax), %xmm1
	movl 148(%esp), %eax
	movdqa %xmm1, %xmm2
	psllq $28, %xmm2
	por %xmm1, %xmm2
	pxor %xmm0, %xmm2
	movdqu %xmm2, (%eax,%esi,8)
	addl $2, %esi
	cmpl %esi, 4(%esp)
	jne .LBB47_5
	movl 36(%esp), %esi
.LBB47_7:
	movl 40(%esp), %eax
	incl 8(%esp)
	leal (%eax,%esi,8), %eax
.LBB47_8:
	movl 60(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	movl 144(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	movl 140(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	movl 136(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	movl 132(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	movl 128(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	movl 56(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	movl 52(%esp), %ecx
	leal (%ecx,%esi), %edx
	cmpl %edi, %edx
	jae .LBB47_21
	cmpl $-9, %eax
	ja .LBB47_20
	leal 8(%eax), %ecx
	cmpl 284(%esp), %ecx
	ja .LBB47_20
	movl 80(%esp), %eax
	movl %ecx, 4(%esp)
	movl 76(%esp), %ecx
	movl 64(%esp), %edi
	movzbl (%eax,%esi), %eax
	movzbl (%ecx,%esi), %ecx
	movzbl (%edi,%esi), %ebx
	shll $24, %eax
	shll $16, %ecx
	shll $16, %ebx
	orl %eax, %ecx
	movl 72(%esp), %eax
	movzbl (%eax,%esi), %eax
	shll $8, %eax
	orl %ecx, %eax
	movl 68(%esp), %ecx
	movzbl (%ecx,%esi), %edx
	movl 48(%esp), %ecx
	movzbl (%ecx,%esi), %ecx
	orl %eax, %edx
	shrl $7, %eax
	xorl %edx, %eax
	andl $11141290, %eax
	shll $24, %ecx
	orl %ecx, %ebx
	movl 44(%esp), %ecx
	movzbl (%ecx,%esi), %edi
	movl 84(%esp), %ecx
	movzbl (%ecx,%esi), %ecx
	shll $8, %edi
	orl %ebx, %edi
	orl %edi, %ecx
	shrl $7, %edi
	xorl %ecx, %edi
	andl $11141290, %edi
	movl %edi, %ebp
	shll $7, %ebp
	orl %edi, %ebp
	movl %eax, %edi
	shll $7, %edi
	xorl %ecx, %ebp
	orl %eax, %edi
	movl %ebp, %eax
	xorl %edx, %edi
	shrl $14, %eax
	movl %edi, %edx
	xorl %ebp, %eax
	shrl $14, %edx
	andl $52428, %eax
	xorl %edi, %edx
	movl %eax, %ebx
	andl $52428, %edx
	shll $14, %ebx
	movl %edx, %ecx
	orl %eax, %ebx
	shll $14, %ecx
	xorl %ebp, %ebx
	movl 148(%esp), %ebp
	orl %edx, %ecx
	movl $268435457, %edx
	xorl %edi, %ecx
	movl 276(%esp), %edi
	movl %ecx, %eax
	shll $4, %eax
	xorl %ebx, %eax
	andl $-252645136, %eax
	mull %edx
	xorl %ebx, %eax
	movl 172(%esp), %ebx
	xorl %ecx, %edx
	movl %eax, (%ebp,%esi,8)
	movl 4(%esp), %eax
	movl %edx, 4(%ebp,%esi,8)
	incl %esi
	cmpl %esi, %ebx
	jne .LBB47_8
	movl 288(%esp), %eax
	movl 144(%esp), %esi
	movl 64(%esp), %ecx
	movl 8(%esp), %edx
	subl %ebx, 116(%esp)
	addl %ebx, 56(%esp)
	subl %ebx, 96(%esp)
	addl %ebx, 52(%esp)
	subl %ebx, 100(%esp)
	addl %ebx, 60(%esp)
	subl %ebx, 104(%esp)
	addl %ebx, 80(%esp)
	addl %ebx, 76(%esp)
	addl %ebx, 72(%esp)
	addl %ebx, 48(%esp)
	addl %ebx, 44(%esp)
	addl %ebx, 84(%esp)
	addl %ebx, 128(%esp)
	addl %ebx, 132(%esp)
	addl %ebx, 136(%esp)
	addl %ebx, 140(%esp)
	addl %eax, 124(%esp)
	subl %eax, 112(%esp)
	addl %eax, 40(%esp)
	subl %eax, 108(%esp)
	addl %eax, %ebp
	movl 68(%esp), %eax
	addl %ebx, %esi
	addl %ebx, %ecx
	addl %ebx, %eax
	cmpl 292(%esp), %edx
	movl 40(%esp), %edx
	jne .LBB47_3
.LBB47_19:
	addl $252, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB47_20:
	movl 12(%esp), %ebx
	leal 8(%eax), %ecx
	leal .Lanon.52bf38722a5bfecdfb3cf7aba91d383f.15@GOTOFF(%ebx), %edx
	pushl %edx
	pushl 288(%esp)
	pushl %ecx
	pushl %eax
	calll core::slice::index::slice_index_fail@PLT
.LBB47_21:
	subl $4, %esp
	movl 16(%esp), %ebx
	leal .Lanon.52bf38722a5bfecdfb3cf7aba91d383f.16@GOTOFF(%ebx), %eax
	pushl %eax
	pushl %edi
	pushl %edx
	calll core::panicking::panic_bounds_check@PLT
