probe_bit_shuffle_transpose_bit_rows:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $156, %esp
	movl 180(%esp), %ecx
	calll .L40$pb
.L40$pb:
	popl %ebx
.Ltmp5375:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp5375-.L40$pb), %ebx
	movl %ecx, %eax
	andl $2147483640, %eax
	cmpl %eax, 188(%esp)
	jb .LBB40_5
	movl %ecx, %eax
	shrl $7, %eax
	movl %eax, 4(%esp)
	je .LBB40_4
	movl 4(%esp), %edi
	movl 184(%esp), %edx
	movl 176(%esp), %ebp
	shrl $3, %ecx
	movdqa .LCPI40_0@GOTOFF(%ebx), %xmm0
	movdqa .LCPI40_1@GOTOFF(%ebx), %xmm2
	movaps .LCPI40_2@GOTOFF(%ebx), %xmm1
	leal (,%ecx,8), %esi
	leal (%ecx,%ecx), %eax
	subl %ecx, %esi
	leal (%eax,%eax,2), %eax
	shll $4, %edi
	leal (%ebp,%ecx,2), %ebx
	movdqa %xmm0, 128(%esp)
	movdqa %xmm2, 112(%esp)
	movaps %xmm1, 96(%esp)
	movl %edi, 36(%esp)
	leal (%edx,%esi), %edi
	addl %ebp, %esi
	movl %edi, 32(%esp)
	leal (%edx,%eax), %edi
	movl %esi, 40(%esp)
	addl %ebp, %eax
	leal (%ecx,%ecx,4), %esi
	movl %eax, 44(%esp)
	leal (%edx,%esi), %eax
	movl %edi, 28(%esp)
	addl %ebp, %esi
	movl %eax, 24(%esp)
	leal (%ecx,%ecx,2), %eax
	leal (%edx,%eax), %edi
	addl %ebp, %eax
	movl %edi, 20(%esp)
	leal (%edx,%ecx,4), %edi
	movl %edi, 16(%esp)
	leal (%edx,%ecx,2), %edi
	addl %ecx, %edx
	movl %edi, 12(%esp)
	leal (%ebp,%ecx,4), %edi
	addl %ebp, %ecx
	xorl %ebp, %ebp
	movl %edx, 8(%esp)
.LBB40_3:
	movl 44(%esp), %edx
	movdqa 128(%esp), %xmm3
	movdqu (%ecx,%ebp), %xmm5
	movdqu (%esi,%ebp), %xmm7
	movdqu (%ebx,%ebp), %xmm6
	movdqu (%eax,%ebp), %xmm1
	movups (%edx,%ebp), %xmm0
	movl 176(%esp), %edx
	movdqu (%edx,%ebp), %xmm2
	movaps %xmm0, 48(%esp)
	movdqu (%edi,%ebp), %xmm0
	movl 40(%esp), %edx
	movdqa %xmm2, %xmm4
	psrlq $4, %xmm4
	pxor %xmm0, %xmm4
	pand %xmm3, %xmm4
	pxor %xmm4, %xmm0
	psllq $4, %xmm4
	pxor %xmm2, %xmm4
	movdqa %xmm5, %xmm2
	psrlq $4, %xmm2
	pxor %xmm7, %xmm2
	pand %xmm3, %xmm2
	pxor %xmm2, %xmm7
	psllq $4, %xmm2
	pxor %xmm5, %xmm2
	movdqa 48(%esp), %xmm5
	movdqa %xmm7, 64(%esp)
	movdqa %xmm6, %xmm7
	psrlq $4, %xmm7
	pxor %xmm5, %xmm7
	pand %xmm3, %xmm7
	pxor %xmm7, %xmm5
	psllq $4, %xmm7
	pxor %xmm6, %xmm7
	movdqu (%edx,%ebp), %xmm6
	movdqa %xmm5, 48(%esp)
	movdqa %xmm1, %xmm5
	movl 184(%esp), %edx
	psrlq $4, %xmm5
	pxor %xmm6, %xmm5
	pand %xmm3, %xmm5
	movdqa %xmm4, %xmm3
	pxor %xmm5, %xmm6
	psllq $4, %xmm5
	psrlq $2, %xmm3
	pxor %xmm1, %xmm5
	movdqa 112(%esp), %xmm1
	movdqa %xmm6, 80(%esp)
	pxor %xmm7, %xmm3
	movdqa %xmm5, %xmm6
	movdqa %xmm2, %xmm5
	psrlq $2, %xmm5
	pxor %xmm6, %xmm5
	pand %xmm1, %xmm3
	pand %xmm1, %xmm5
	pxor %xmm3, %xmm7
	psllq $2, %xmm3
	pxor %xmm5, %xmm6
	psllq $2, %xmm5
	pxor %xmm4, %xmm3
	movdqa %xmm6, %xmm4
	movdqa 48(%esp), %xmm6
	pxor %xmm2, %xmm5
	movdqa %xmm0, %xmm2
	psrlq $2, %xmm2
	pxor %xmm6, %xmm2
	pand %xmm1, %xmm2
	pxor %xmm2, %xmm6
	psllq $2, %xmm2
	pxor %xmm0, %xmm2
	movdqa 64(%esp), %xmm0
	movdqa %xmm6, 48(%esp)
	movdqa 80(%esp), %xmm6
	psrlq $2, %xmm0
	pxor %xmm6, %xmm0
	pand %xmm1, %xmm0
	movdqa %xmm3, %xmm1
	pxor %xmm0, %xmm6
	psrlq $1, %xmm1
	psllq $2, %xmm0
	movdqa %xmm6, 80(%esp)
	movdqa 96(%esp), %xmm6
	pxor %xmm5, %xmm1
	pxor 64(%esp), %xmm0
	pand %xmm6, %xmm1
	pxor %xmm1, %xmm5
	paddq %xmm1, %xmm1
	pxor %xmm3, %xmm1
	movdqa %xmm7, %xmm3
	movdqu %xmm1, (%edx,%ebp)
	movl 8(%esp), %edx
	psrlq $1, %xmm3
	pxor %xmm4, %xmm3
	pand %xmm6, %xmm3
	pxor %xmm3, %xmm4
	paddq %xmm3, %xmm3
	movdqu %xmm5, (%edx,%ebp)
	movl 12(%esp), %edx
	movdqa %xmm4, 64(%esp)
	pxor %xmm7, %xmm3
	movdqa %xmm2, %xmm7
	movdqa 80(%esp), %xmm4
	psrlq $1, %xmm7
	movaps 64(%esp), %xmm1
	pxor %xmm0, %xmm7
	pand %xmm6, %xmm7
	movdqu %xmm3, (%edx,%ebp)
	movl 20(%esp), %edx
	pxor %xmm7, %xmm0
	paddq %xmm7, %xmm7
	pxor %xmm2, %xmm7
	movdqa 48(%esp), %xmm2
	movups %xmm1, (%edx,%ebp)
	movl 16(%esp), %edx
	psrlq $1, %xmm2
	pxor %xmm4, %xmm2
	pand %xmm6, %xmm2
	movdqu %xmm7, (%edx,%ebp)
	movl 24(%esp), %edx
	pxor %xmm2, %xmm4
	paddq %xmm2, %xmm2
	pxor 48(%esp), %xmm2
	movdqu %xmm0, (%edx,%ebp)
	movl 28(%esp), %edx
	movdqu %xmm2, (%edx,%ebp)
	movl 32(%esp), %edx
	movdqu %xmm4, (%edx,%ebp)
	addl $16, %ebp
	cmpl %ebp, 36(%esp)
	jne .LBB40_3
.LBB40_4:
	movl 4(%esp), %eax
	shll $4, %eax
	addl $156, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB40_5:
	subl $4, %esp
	leal .Lanon.c20a185645f18c1c99375284bd48df7e.5@GOTOFF(%ebx), %eax
	leal .Lanon.c20a185645f18c1c99375284bd48df7e.4@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $36
	pushl %ecx
	calll core::panicking::panic@PLT
