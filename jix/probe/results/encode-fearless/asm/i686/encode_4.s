probe_byte_shuffle_encode_4:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $12, %esp
	movl 36(%esp), %eax
	calll .L36$pb
.L36$pb:
	popl %ebx
.Ltmp4172:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp4172-.L36$pb), %ebx
	movl %eax, %ecx
	andl $2147483644, %ecx
	cmpl %ecx, 44(%esp)
	jb .LBB36_7
	movl %eax, %ecx
	shrl $2, %ecx
	shrl $8, %eax
	movl %ecx, 8(%esp)
	je .LBB36_6
	movl 8(%esp), %ecx
	movl 40(%esp), %edx
	movl 32(%esp), %esi
	leal (%ecx,%ecx,2), %edi
	leal (%edx,%ecx,2), %ebx
	leal (%edx,%ecx), %ebp
	addl $48, %esi
	addl %edx, %edi
.LBB36_4:
	xorl %ecx, %ecx
.LBB36_5:
	movdqu -48(%esi,%ecx,4), %xmm1
	movdqu -32(%esi,%ecx,4), %xmm3
	movdqu -16(%esi,%ecx,4), %xmm2
	movdqu (%esi,%ecx,4), %xmm4
	movdqa %xmm1, %xmm0
	movdqa %xmm3, %xmm5
	punpckhbw %xmm2, %xmm1
	punpckhbw %xmm4, %xmm3
	punpcklbw %xmm2, %xmm0
	punpcklbw %xmm4, %xmm5
	movdqa %xmm1, %xmm4
	punpckhbw %xmm3, %xmm1
	movdqa %xmm0, %xmm2
	punpckhbw %xmm5, %xmm0
	punpcklbw %xmm3, %xmm4
	punpcklbw %xmm5, %xmm2
	movdqa %xmm0, %xmm7
	punpckhbw %xmm1, %xmm0
	movdqa %xmm2, %xmm3
	punpckhbw %xmm4, %xmm2
	punpcklbw %xmm1, %xmm7
	punpcklbw %xmm4, %xmm3
	movdqa %xmm2, %xmm5
	punpckhbw %xmm0, %xmm2
	movdqa %xmm3, %xmm6
	punpckhbw %xmm7, %xmm3
	punpcklbw %xmm0, %xmm5
	punpcklbw %xmm7, %xmm6
	movdqu %xmm6, (%edx,%ecx)
	movdqu %xmm3, (%ebp,%ecx)
	movdqu %xmm5, (%ebx,%ecx)
	movdqu %xmm2, (%edi,%ecx)
	addl $16, %ecx
	cmpl $64, %ecx
	jne .LBB36_5
	addl $64, %edi
	addl $64, %ebx
	addl $64, %ebp
	addl $64, %edx
	addl $256, %esi
	decl %eax
	jne .LBB36_4
.LBB36_6:
	movl 8(%esp), %eax
	andl $536870848, %eax
	addl $12, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB36_7:
	subl $4, %esp
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.6@GOTOFF(%ebx), %eax
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
