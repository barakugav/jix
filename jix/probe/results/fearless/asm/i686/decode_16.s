probe_byte_shuffle_decode_16:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $268, %esp
	movl 292(%esp), %ecx
	calll .L22$pb
.L22$pb:
	popl %ebx
.Ltmp2967:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp2967-.L22$pb), %ebx
	movl %ecx, %eax
	andl $2147483632, %eax
	cmpl %eax, 300(%esp)
	jb .LBB22_7
	movl %ecx, %eax
	shrl $8, %eax
	movl %eax, 12(%esp)
	je .LBB22_6
	shrl $4, %ecx
	movl 296(%esp), %edx
	movl 12(%esp), %ebp
	leal (%ecx,%ecx), %eax
	leal (%eax,%eax,2), %esi
	movl %esi, 44(%esp)
	leal (,%ecx,8), %esi
	subl %ecx, %esi
	movl %esi, 40(%esp)
	leal (%eax,%eax,4), %esi
	movl %esi, 36(%esp)
	movl %ecx, %esi
	shll $4, %esi
	subl %eax, %esi
	leal (,%ecx,4), %eax
	movl %esi, 32(%esp)
	leal (%ecx,%ecx,4), %esi
	leal (%eax,%eax,2), %eax
	movl %eax, 28(%esp)
	leal (%ecx,%ecx,2), %eax
	leal (%ecx,%esi,2), %edi
	leal (%esi,%esi,2), %ebx
	movl %edi, 24(%esp)
	leal (%ecx,%eax,4), %eax
	xorl %edi, %edi
	movl %ebx, 16(%esp)
	movl %eax, 20(%esp)
.LBB22_3:
	movl 288(%esp), %esi
	movl %edi, %ebx
	shll $4, %ebx
	movups (%esi,%ebx), %xmm0
	leal (%esi,%ebx), %eax
	leal (%ecx,%ecx,2), %esi
	movl $4, %ebx
	movdqu (%ecx,%eax), %xmm3
	movaps %xmm0, 224(%esp)
	movups (%eax,%ecx,2), %xmm0
	movaps %xmm0, 160(%esp)
	movups (%esi,%eax), %xmm0
	leal (%ecx,%ecx,4), %esi
	movaps %xmm0, 176(%esp)
	movups (%eax,%ecx,4), %xmm0
	movaps %xmm0, 192(%esp)
	movups (%esi,%eax), %xmm0
	movl 44(%esp), %esi
	movaps %xmm0, 208(%esp)
	movups (%esi,%eax), %xmm0
	movl 40(%esp), %esi
	movaps %xmm0, 128(%esp)
	movups (%esi,%eax), %xmm0
	leal (%ecx,%ecx,8), %esi
	movdqu (%esi,%eax), %xmm6
	movl 36(%esp), %esi
	movaps %xmm0, 144(%esp)
	movdqu (%eax,%ecx,8), %xmm0
	movdqu (%esi,%eax), %xmm5
	movl 24(%esp), %esi
	movups (%esi,%eax), %xmm1
	movl 28(%esp), %esi
	movups (%esi,%eax), %xmm2
	movl 20(%esp), %esi
	movaps %xmm1, 96(%esp)
	movaps %xmm2, 48(%esp)
	movups (%esi,%eax), %xmm2
	movl 32(%esp), %esi
	movups (%esi,%eax), %xmm4
	movl 16(%esp), %esi
	movaps %xmm2, 64(%esp)
	movups (%esi,%eax), %xmm7
	movaps %xmm4, 80(%esp)
	movaps %xmm7, 112(%esp)
.LBB22_4:
	movdqa 224(%esp), %xmm7
	movdqa %xmm3, %xmm2
	movdqa 144(%esp), %xmm4
	decl %ebx
	movdqa %xmm7, %xmm1
	punpcklbw %xmm0, %xmm7
	movdqa %xmm1, %xmm3
	movdqa %xmm7, 224(%esp)
	movdqa %xmm2, %xmm7
	movdqa 160(%esp), %xmm1
	punpckhbw %xmm0, %xmm3
	movdqa %xmm2, %xmm0
	punpckhbw %xmm6, %xmm7
	movdqa 128(%esp), %xmm2
	movdqa %xmm3, 240(%esp)
	movdqa 176(%esp), %xmm3
	punpcklbw %xmm6, %xmm0
	movdqa %xmm7, 176(%esp)
	movdqa 112(%esp), %xmm7
	movdqa %xmm0, 160(%esp)
	movdqa %xmm3, %xmm0
	movdqa %xmm3, %xmm6
	movdqa 96(%esp), %xmm3
	punpcklbw %xmm3, %xmm6
	punpckhbw %xmm3, %xmm0
	movdqa 208(%esp), %xmm3
	movdqa %xmm6, 128(%esp)
	movdqa %xmm0, 144(%esp)
	movdqa %xmm1, %xmm0
	movdqa 192(%esp), %xmm6
	punpcklbw %xmm5, %xmm1
	movdqa %xmm1, 192(%esp)
	movdqa 48(%esp), %xmm1
	punpckhbw %xmm5, %xmm0
	movdqa %xmm0, 208(%esp)
	movdqa %xmm3, %xmm5
	movdqa %xmm6, %xmm0
	punpcklbw %xmm1, %xmm0
	punpckhbw %xmm1, %xmm6
	movdqa 64(%esp), %xmm1
	punpckhbw %xmm1, %xmm3
	punpcklbw %xmm1, %xmm5
	movdqa 80(%esp), %xmm1
	movdqa %xmm3, 96(%esp)
	movdqa %xmm2, %xmm3
	punpcklbw %xmm1, %xmm3
	punpckhbw %xmm1, %xmm2
	movdqa %xmm4, %xmm1
	punpckhbw %xmm7, %xmm4
	movdqa %xmm3, 48(%esp)
	movaps 240(%esp), %xmm3
	punpcklbw %xmm7, %xmm1
	movdqa %xmm2, 64(%esp)
	movdqa %xmm4, 112(%esp)
	movdqa %xmm1, 80(%esp)
	jne .LBB22_4
	movdqa 224(%esp), %xmm7
	movaps 160(%esp), %xmm2
	movaps 176(%esp), %xmm1
	movl %edi, %eax
	incl %edi
	shll $8, %eax
	decl %ebp
	movdqu %xmm7, (%edx,%eax)
	movups %xmm3, 16(%edx,%eax)
	movups %xmm2, 32(%edx,%eax)
	movups %xmm1, 48(%edx,%eax)
	movaps 192(%esp), %xmm2
	movaps 208(%esp), %xmm1
	movups %xmm2, 64(%edx,%eax)
	movups %xmm1, 80(%edx,%eax)
	movaps 128(%esp), %xmm2
	movaps 144(%esp), %xmm1
	movups %xmm2, 96(%edx,%eax)
	movups %xmm1, 112(%edx,%eax)
	movaps 96(%esp), %xmm1
	movdqu %xmm0, 128(%edx,%eax)
	movdqu %xmm6, 144(%edx,%eax)
	movdqu %xmm5, 160(%edx,%eax)
	movaps 48(%esp), %xmm2
	movups %xmm1, 176(%edx,%eax)
	movaps 64(%esp), %xmm1
	movups %xmm2, 192(%edx,%eax)
	movdqa 80(%esp), %xmm2
	movups %xmm1, 208(%edx,%eax)
	movdqa 112(%esp), %xmm1
	movdqu %xmm2, 224(%edx,%eax)
	movdqu %xmm1, 240(%edx,%eax)
	jne .LBB22_3
.LBB22_6:
	movl 12(%esp), %eax
	shll $4, %eax
	addl $268, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB22_7:
	subl $4, %esp
	leal .Lanon.47755086f7148c9197b4fae71012f345.5@GOTOFF(%ebx), %eax
	leal .Lanon.47755086f7148c9197b4fae71012f345.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
