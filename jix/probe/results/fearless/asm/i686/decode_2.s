probe_byte_shuffle_decode_2:
	pushl %ebx
	pushl %edi
	pushl %esi
	movl 20(%esp), %ecx
	calll .L23$pb
.L23$pb:
	popl %ebx
.Ltmp3081:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp3081-.L23$pb), %ebx
	movl %ecx, %eax
	andl $2147483646, %eax
	cmpl %eax, 28(%esp)
	jb .LBB23_5
	movl %ecx, %eax
	shrl %eax
	shrl $8, %ecx
	je .LBB23_4
	movl 16(%esp), %esi
	movl 24(%esp), %edx
	shll $7, %ecx
	xorl %ebx, %ebx
	leal 112(%eax,%esi), %edi
.LBB23_3:
	movdqu (%esi,%ebx), %xmm0
	movdqu -112(%edi,%ebx), %xmm1
	movdqu 16(%esi,%ebx), %xmm3
	movdqu 32(%esi,%ebx), %xmm4
	movdqa %xmm0, %xmm2
	punpckhbw %xmm1, %xmm0
	punpcklbw %xmm1, %xmm2
	movdqu -96(%edi,%ebx), %xmm1
	movdqu %xmm2, (%edx,%ebx,2)
	movdqa %xmm3, %xmm2
	movdqu %xmm0, 16(%edx,%ebx,2)
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm3
	movdqu -80(%edi,%ebx), %xmm1
	movdqu %xmm2, 32(%edx,%ebx,2)
	movdqa %xmm4, %xmm2
	movdqu %xmm3, 48(%edx,%ebx,2)
	movdqu 48(%esi,%ebx), %xmm3
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm4
	movdqu -64(%edi,%ebx), %xmm1
	movdqu %xmm2, 64(%edx,%ebx,2)
	movdqa %xmm3, %xmm2
	movdqu %xmm4, 80(%edx,%ebx,2)
	movdqu 64(%esi,%ebx), %xmm4
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm3
	movdqu -48(%edi,%ebx), %xmm1
	movdqu %xmm2, 96(%edx,%ebx,2)
	movdqa %xmm4, %xmm2
	movdqu %xmm3, 112(%edx,%ebx,2)
	movdqu 80(%esi,%ebx), %xmm3
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm4
	movdqu -32(%edi,%ebx), %xmm1
	movdqu %xmm2, 128(%edx,%ebx,2)
	movdqa %xmm3, %xmm2
	movdqu %xmm4, 144(%edx,%ebx,2)
	movdqu 96(%esi,%ebx), %xmm4
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm3
	movdqu -16(%edi,%ebx), %xmm1
	movdqu %xmm2, 160(%edx,%ebx,2)
	movdqa %xmm4, %xmm2
	movdqu %xmm3, 176(%edx,%ebx,2)
	movdqu 112(%esi,%ebx), %xmm3
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm4
	movdqu (%edi,%ebx), %xmm1
	movdqu %xmm2, 192(%edx,%ebx,2)
	movdqa %xmm3, %xmm2
	movdqu %xmm4, 208(%edx,%ebx,2)
	punpcklbw %xmm1, %xmm2
	punpckhbw %xmm1, %xmm3
	movdqu %xmm2, 224(%edx,%ebx,2)
	movdqu %xmm3, 240(%edx,%ebx,2)
	subl $-128, %ebx
	cmpl %ebx, %ecx
	jne .LBB23_3
.LBB23_4:
	andl $1073741696, %eax
	popl %esi
	popl %edi
	popl %ebx
	retl
.LBB23_5:
	subl $4, %esp
	leal .Lanon.47755086f7148c9197b4fae71012f345.5@GOTOFF(%ebx), %eax
	leal .Lanon.47755086f7148c9197b4fae71012f345.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
