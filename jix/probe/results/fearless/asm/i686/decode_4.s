probe_byte_shuffle_decode_4:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $12, %esp
	movl 36(%esp), %eax
	calll .L24$pb
.L24$pb:
	popl %ebx
.Ltmp3144:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp3144-.L24$pb), %ebx
	movl %eax, %ecx
	andl $2147483644, %ecx
	cmpl %ecx, 44(%esp)
	jb .LBB24_7
	movl %eax, %ecx
	shrl $2, %ecx
	shrl $8, %eax
	movl %ecx, 8(%esp)
	je .LBB24_6
	movl 8(%esp), %ecx
	movl 40(%esp), %edx
	movl 32(%esp), %esi
	leal (%ecx,%ecx,2), %edi
	addl $48, %edx
	leal (%esi,%ecx,2), %ebx
	leal (%esi,%ecx), %ebp
	addl %esi, %edi
.LBB24_4:
	xorl %ecx, %ecx
.LBB24_5:
	movdqu (%esi,%ecx), %xmm0
	movdqu (%ebp,%ecx), %xmm1
	movdqu (%ebx,%ecx), %xmm3
	movdqu (%edi,%ecx), %xmm4
	movdqa %xmm0, %xmm2
	movdqa %xmm1, %xmm7
	punpckhbw %xmm3, %xmm0
	punpckhbw %xmm4, %xmm1
	punpcklbw %xmm3, %xmm2
	movdqa %xmm0, %xmm5
	punpcklbw %xmm4, %xmm7
	punpckhbw %xmm1, %xmm0
	movdqa %xmm2, %xmm6
	punpckhbw %xmm7, %xmm2
	punpcklbw %xmm1, %xmm5
	punpcklbw %xmm7, %xmm6
	movdqu %xmm6, -48(%edx,%ecx,4)
	movdqu %xmm2, -32(%edx,%ecx,4)
	movdqu %xmm5, -16(%edx,%ecx,4)
	movdqu %xmm0, (%edx,%ecx,4)
	addl $16, %ecx
	cmpl $64, %ecx
	jne .LBB24_5
	addl $256, %edx
	addl $64, %edi
	addl $64, %ebx
	addl $64, %ebp
	addl $64, %esi
	decl %eax
	jne .LBB24_4
.LBB24_6:
	movl 8(%esp), %eax
	andl $536870848, %eax
	addl $12, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB24_7:
	subl $4, %esp
	leal .Lanon.47755086f7148c9197b4fae71012f345.5@GOTOFF(%ebx), %eax
	leal .Lanon.47755086f7148c9197b4fae71012f345.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
