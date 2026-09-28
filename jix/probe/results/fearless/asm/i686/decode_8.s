probe_byte_shuffle_decode_8:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $76, %esp
	movl 100(%esp), %ecx
	calll .L25$pb
.L25$pb:
	popl %ebx
.Ltmp3183:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp3183-.L25$pb), %ebx
	movl %ecx, %eax
	andl $2147483640, %eax
	cmpl %eax, 108(%esp)
	jb .LBB25_9
	movl %ecx, %eax
	shrl $3, %eax
	shrl $8, %ecx
	je .LBB25_8
	movl 104(%esp), %edx
	leal (%eax,%eax), %esi
	leal (,%eax,8), %edi
	xorl %ebx, %ebx
	leal (%esi,%esi,2), %esi
	subl %eax, %edi
	movl %esi, 12(%esp)
	movl %edi, 8(%esp)
.LBB25_3:
	movl 96(%esp), %edi
	movl %ebx, %esi
	movl 8(%esp), %ebp
	shll $4, %esi
	movdqu (%edi,%esi), %xmm0
	addl %edi, %esi
	leal (%eax,%eax,2), %edi
	movdqu (%edi,%esi), %xmm4
	leal (%eax,%eax,4), %edi
	movdqu (%eax,%esi), %xmm5
	movups (%esi,%eax,2), %xmm6
	movdqu (%esi,%eax,4), %xmm3
	movups (%ebp,%esi), %xmm7
	movdqu (%edi,%esi), %xmm2
	movl 12(%esp), %edi
	movdqu (%edi,%esi), %xmm1
	movl $3, %esi
.LBB25_4:
	movaps %xmm6, 32(%esp)
	movdqa %xmm5, %xmm6
	movaps %xmm7, 16(%esp)
	movdqa %xmm4, %xmm7
	movdqa %xmm0, %xmm5
	punpcklbw %xmm3, %xmm0
	decl %esi
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
	movaps 48(%esp), %xmm0
	jne .LBB25_4
	movl %ebx, %esi
	movl 96(%esp), %edi
	shll $7, %esi
	movups %xmm0, (%edx,%esi)
	movdqu %xmm5, 16(%edx,%esi)
	movdqu %xmm6, 32(%edx,%esi)
	movdqu %xmm4, 48(%edx,%esi)
	movdqu %xmm3, 64(%edx,%esi)
	movdqu %xmm2, 80(%edx,%esi)
	movdqu %xmm1, 96(%edx,%esi)
	movdqu %xmm7, 112(%edx,%esi)
	leal 1(%ebx), %esi
	movl %esi, %ebp
	shll $4, %ebp
	movdqu (%edi,%ebp), %xmm0
	addl %edi, %ebp
	leal (%eax,%eax,2), %edi
	movdqu (%edi,%ebp), %xmm4
	leal (%eax,%eax,4), %edi
	movdqu (%eax,%ebp), %xmm5
	movups (%ebp,%eax,2), %xmm6
	movdqu (%ebp,%eax,4), %xmm3
	movdqu (%edi,%ebp), %xmm2
	movl 12(%esp), %edi
	movdqu (%edi,%ebp), %xmm1
	movl 8(%esp), %edi
	movups (%edi,%ebp), %xmm7
	movl $3, %ebp
.LBB25_6:
	movaps %xmm6, 32(%esp)
	movdqa %xmm5, %xmm6
	movaps %xmm7, 16(%esp)
	movdqa %xmm4, %xmm7
	movdqa %xmm0, %xmm5
	punpcklbw %xmm3, %xmm0
	decl %ebp
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
	movaps 48(%esp), %xmm0
	jne .LBB25_6
	shll $7, %esi
	addl $2, %ebx
	decl %ecx
	movups %xmm0, (%edx,%esi)
	movdqu %xmm5, 16(%edx,%esi)
	movdqu %xmm6, 32(%edx,%esi)
	movdqu %xmm4, 48(%edx,%esi)
	movdqu %xmm3, 64(%edx,%esi)
	movdqu %xmm2, 80(%edx,%esi)
	movdqu %xmm1, 96(%edx,%esi)
	movdqu %xmm7, 112(%edx,%esi)
	jne .LBB25_3
.LBB25_8:
	andl $268435424, %eax
	addl $76, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB25_9:
	subl $4, %esp
	leal .Lanon.47755086f7148c9197b4fae71012f345.5@GOTOFF(%ebx), %eax
	leal .Lanon.47755086f7148c9197b4fae71012f345.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
