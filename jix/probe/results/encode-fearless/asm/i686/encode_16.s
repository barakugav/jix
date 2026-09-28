probe_byte_shuffle_encode_16:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $252, %esp
	movl 276(%esp), %ecx
	calll .L34$pb
.L34$pb:
	popl %ebx
.Ltmp3989:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp3989-.L34$pb), %ebx
	movl %ecx, %eax
	andl $2147483632, %eax
	cmpl %eax, 284(%esp)
	jb .LBB34_7
	movl %ecx, %eax
	shrl $8, %eax
	movl %eax, (%esp)
	je .LBB34_6
	shrl $4, %ecx
	movl 272(%esp), %esi
	movl (%esp), %ebp
	leal (%ecx,%ecx), %eax
	leal (%ecx,%ecx,4), %edi
	leal (%eax,%eax,2), %edx
	leal (%ecx,%edi,2), %ebx
	movl %edx, 28(%esp)
	leal (,%ecx,8), %edx
	movl %ebx, 12(%esp)
	subl %ecx, %edx
	movl %edx, 24(%esp)
	leal (%eax,%eax,4), %edx
	movl %edx, 20(%esp)
	movl %ecx, %edx
	shll $4, %edx
	subl %eax, %edx
	leal (,%ecx,4), %eax
	leal (%eax,%eax,2), %eax
	movl %eax, 16(%esp)
	leal (%ecx,%ecx,2), %eax
	leal (%ecx,%eax,4), %eax
	movl %eax, 8(%esp)
	leal (%edi,%edi,2), %eax
	movl %eax, 4(%esp)
	xorl %eax, %eax
.LBB34_3:
	movl %eax, %edi
	movl $4, %ebx
	shll $8, %edi
	movups 32(%esi,%edi), %xmm2
	movups 48(%esi,%edi), %xmm1
	movups 176(%esi,%edi), %xmm4
	movups (%esi,%edi), %xmm0
	movups 240(%esi,%edi), %xmm7
	movdqu 16(%esi,%edi), %xmm3
	movdqu 144(%esi,%edi), %xmm6
	movdqu 160(%esi,%edi), %xmm5
	movaps %xmm2, 144(%esp)
	movups 64(%esi,%edi), %xmm2
	movaps %xmm1, 160(%esp)
	movups 80(%esi,%edi), %xmm1
	movaps %xmm4, 80(%esp)
	movups 224(%esi,%edi), %xmm4
	movaps %xmm0, 192(%esp)
	movaps %xmm7, 96(%esp)
	movaps %xmm2, 176(%esp)
	movups 96(%esi,%edi), %xmm2
	movaps %xmm1, 208(%esp)
	movups 112(%esi,%edi), %xmm1
	movaps %xmm4, 64(%esp)
	movaps %xmm2, 112(%esp)
	movups 192(%esi,%edi), %xmm2
	movaps %xmm1, 128(%esp)
	movdqu 128(%esi,%edi), %xmm1
	movaps %xmm2, 32(%esp)
	movups 208(%esi,%edi), %xmm2
	movaps %xmm2, 48(%esp)
.LBB34_4:
	movdqa 192(%esp), %xmm7
	movdqa %xmm3, %xmm2
	movdqa 128(%esp), %xmm4
	decl %ebx
	movdqa %xmm7, %xmm0
	punpcklbw %xmm1, %xmm7
	movdqa %xmm0, %xmm3
	movdqa %xmm7, 192(%esp)
	movdqa %xmm2, %xmm7
	movdqa 144(%esp), %xmm0
	punpckhbw %xmm1, %xmm3
	movdqa %xmm2, %xmm1
	punpckhbw %xmm6, %xmm7
	movdqa 112(%esp), %xmm2
	movdqa %xmm3, 224(%esp)
	movdqa 160(%esp), %xmm3
	punpcklbw %xmm6, %xmm1
	movdqa %xmm7, 160(%esp)
	movdqa 96(%esp), %xmm7
	movdqa %xmm1, 144(%esp)
	movdqa %xmm3, %xmm1
	movdqa %xmm3, %xmm6
	movdqa 80(%esp), %xmm3
	punpcklbw %xmm3, %xmm6
	punpckhbw %xmm3, %xmm1
	movdqa 208(%esp), %xmm3
	movdqa %xmm6, 112(%esp)
	movdqa %xmm1, 128(%esp)
	movdqa %xmm0, %xmm1
	movdqa 176(%esp), %xmm6
	punpcklbw %xmm5, %xmm0
	movdqa %xmm0, 176(%esp)
	movdqa 32(%esp), %xmm0
	punpckhbw %xmm5, %xmm1
	movdqa %xmm1, 208(%esp)
	movdqa %xmm3, %xmm5
	movdqa %xmm6, %xmm1
	punpcklbw %xmm0, %xmm1
	punpckhbw %xmm0, %xmm6
	movdqa 48(%esp), %xmm0
	punpckhbw %xmm0, %xmm3
	punpcklbw %xmm0, %xmm5
	movdqa 64(%esp), %xmm0
	movdqa %xmm3, 80(%esp)
	movdqa %xmm2, %xmm3
	punpcklbw %xmm0, %xmm3
	punpckhbw %xmm0, %xmm2
	movdqa %xmm4, %xmm0
	punpckhbw %xmm7, %xmm4
	movdqa %xmm3, 32(%esp)
	movaps 224(%esp), %xmm3
	punpcklbw %xmm7, %xmm0
	movdqa %xmm2, 48(%esp)
	movdqa %xmm4, 96(%esp)
	movdqa %xmm0, 64(%esp)
	jne .LBB34_4
	movl 280(%esp), %ebx
	movdqa 192(%esp), %xmm7
	movaps 144(%esp), %xmm2
	movaps 160(%esp), %xmm0
	movl %eax, %edi
	incl %eax
	shll $4, %edi
	decl %ebp
	movdqu %xmm7, (%ebx,%edi)
	leal (%ebx,%edi), %edi
	leal (%ecx,%ecx,2), %ebx
	movups %xmm3, (%ecx,%edi)
	movups %xmm2, (%edi,%ecx,2)
	movups %xmm0, (%ebx,%edi)
	movaps 176(%esp), %xmm2
	movaps 208(%esp), %xmm0
	leal (%ecx,%ecx,4), %ebx
	movups %xmm2, (%edi,%ecx,4)
	movups %xmm0, (%ebx,%edi)
	movl 28(%esp), %ebx
	movaps 112(%esp), %xmm2
	movaps 128(%esp), %xmm0
	movups %xmm2, (%ebx,%edi)
	movl 24(%esp), %ebx
	movdqa 64(%esp), %xmm2
	movups %xmm0, (%ebx,%edi)
	leal (%ecx,%ecx,8), %ebx
	movdqu %xmm1, (%edi,%ecx,8)
	movaps 80(%esp), %xmm1
	movdqa 32(%esp), %xmm0
	movdqu %xmm6, (%ebx,%edi)
	movl 20(%esp), %ebx
	movdqu %xmm5, (%ebx,%edi)
	movl 12(%esp), %ebx
	movups %xmm1, (%ebx,%edi)
	movl 16(%esp), %ebx
	movaps 48(%esp), %xmm1
	movdqu %xmm0, (%ebx,%edi)
	movl 8(%esp), %ebx
	movups %xmm1, (%ebx,%edi)
	movl 4(%esp), %ebx
	movdqa 96(%esp), %xmm1
	movdqu %xmm2, (%edx,%edi)
	movdqu %xmm1, (%ebx,%edi)
	jne .LBB34_3
.LBB34_6:
	movl (%esp), %eax
	shll $4, %eax
	addl $252, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB34_7:
	subl $4, %esp
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.6@GOTOFF(%ebx), %eax
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
