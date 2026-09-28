probe_byte_shuffle_encode_8:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $76, %esp
	movl 100(%esp), %ecx
	calll .L37$pb
.L37$pb:
	popl %ebx
.Ltmp4363:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp4363-.L37$pb), %ebx
	movl %ecx, %eax
	andl $2147483640, %eax
	cmpl %eax, 108(%esp)
	jb .LBB37_9
	movl %ecx, %eax
	shrl $3, %eax
	shrl $8, %ecx
	je .LBB37_8
	movl 96(%esp), %esi
	leal (%eax,%eax), %edx
	leal (,%eax,8), %edi
	xorl %ebx, %ebx
	leal (%edx,%edx,2), %edx
	subl %eax, %edi
	movl %edx, 12(%esp)
	movl %edi, 8(%esp)
.LBB37_3:
	movl %ebx, %edx
	shll $7, %edx
	movdqu (%esi,%edx), %xmm0
	movdqu 16(%esi,%edx), %xmm5
	movups 32(%esi,%edx), %xmm6
	movdqu 48(%esi,%edx), %xmm4
	movdqu 64(%esi,%edx), %xmm3
	movdqu 80(%esi,%edx), %xmm2
	movdqu 96(%esi,%edx), %xmm1
	movups 112(%esi,%edx), %xmm7
	movl $4, %edx
.LBB37_4:
	movaps %xmm6, 32(%esp)
	movdqa %xmm5, %xmm6
	movaps %xmm7, 16(%esp)
	movdqa %xmm4, %xmm7
	movdqa %xmm0, %xmm5
	punpcklbw %xmm3, %xmm0
	decl %edx
	movdqa %xmm6, %xmm4
	punpcklbw %xmm2, %xmm6
	movdqa %xmm0, 48(%esp)
	movdqa 16(%esp), %xmm0
	punpckhbw %xmm3, %xmm5
	punpckhbw %xmm2, %xmm4
	movdqa 32(%esp), %xmm2
	movdqa %xmm2, %xmm3
	punpckhbw %xmm1, %xmm2
	punpcklbw %xmm1, %xmm3
	movdqa %xmm7, %xmm1
	punpckhbw %xmm0, %xmm7
	punpcklbw %xmm0, %xmm1
	movdqa 48(%esp), %xmm0
	jne .LBB37_4
	movl 104(%esp), %edi
	movl %ebx, %edx
	shll $4, %edx
	leal (%edi,%edx), %ebp
	movdqu %xmm0, (%edi,%edx)
	leal (%eax,%eax,2), %edx
	movdqu %xmm5, (%eax,%ebp)
	movdqu %xmm6, (%ebp,%eax,2)
	movdqu %xmm4, (%edx,%ebp)
	leal (%eax,%eax,4), %edx
	movdqu %xmm3, (%ebp,%eax,4)
	movdqu %xmm2, (%edx,%ebp)
	movl 12(%esp), %edx
	movdqu %xmm1, (%edx,%ebp)
	movl 8(%esp), %edx
	movdqu %xmm7, (%edx,%ebp)
	leal 1(%ebx), %edx
	movl %edx, %ebp
	shll $7, %ebp
	movdqu (%esi,%ebp), %xmm1
	movdqu 16(%esi,%ebp), %xmm5
	movups 32(%esi,%ebp), %xmm6
	movdqu 48(%esi,%ebp), %xmm4
	movdqu 64(%esi,%ebp), %xmm3
	movdqu 80(%esi,%ebp), %xmm2
	movdqu 96(%esi,%ebp), %xmm0
	movups 112(%esi,%ebp), %xmm7
	movl $4, %ebp
.LBB37_6:
	movaps %xmm6, 32(%esp)
	movdqa %xmm5, %xmm6
	movaps %xmm7, 16(%esp)
	movdqa %xmm4, %xmm7
	movdqa %xmm1, %xmm5
	punpcklbw %xmm3, %xmm1
	decl %ebp
	movdqa %xmm6, %xmm4
	punpcklbw %xmm2, %xmm6
	movdqa %xmm1, 48(%esp)
	movdqa 16(%esp), %xmm1
	punpckhbw %xmm3, %xmm5
	punpckhbw %xmm2, %xmm4
	movdqa 32(%esp), %xmm2
	movdqa %xmm2, %xmm3
	punpckhbw %xmm0, %xmm2
	punpcklbw %xmm0, %xmm3
	movdqa %xmm7, %xmm0
	punpckhbw %xmm1, %xmm7
	punpcklbw %xmm1, %xmm0
	movdqa 48(%esp), %xmm1
	jne .LBB37_6
	movl 104(%esp), %edi
	shll $4, %edx
	movl 8(%esp), %ebp
	addl $2, %ebx
	decl %ecx
	movdqu %xmm1, (%edi,%edx)
	leal (%edi,%edx), %edx
	leal (%eax,%eax,2), %edi
	movdqu %xmm5, (%eax,%edx)
	movdqu %xmm6, (%edx,%eax,2)
	movdqu %xmm4, (%edi,%edx)
	leal (%eax,%eax,4), %edi
	movdqu %xmm3, (%edx,%eax,4)
	movdqu %xmm2, (%edi,%edx)
	movl 12(%esp), %edi
	movdqu %xmm0, (%edi,%edx)
	movdqu %xmm7, (%ebp,%edx)
	jne .LBB37_3
.LBB37_8:
	andl $268435424, %eax
	addl $76, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB37_9:
	subl $4, %esp
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.6@GOTOFF(%ebx), %eax
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
