jix_probe::byte_shuffle::decode_impl::<2, 64>:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $28, %esp
	movl 52(%esp), %ecx
	movl 56(%esp), %edx
	movl 48(%esp), %eax
	calll .L1$pb
.L1$pb:
	popl %ebx
.Ltmp47:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp47-.L1$pb), %ebx
	shrl %ecx
	movl %ecx, %edi
	andl $1073741760, %edi
	je .LBB1_1
	leal 56(%eax), %ebp
	leal 56(%eax,%ecx), %ecx
	xorl %esi, %esi
.LBB1_4:
	movq -56(%ebp,%esi), %xmm0
	movq -56(%ecx,%esi), %xmm1
	movq -48(%ebp,%esi), %xmm2
	movq -40(%ebp,%esi), %xmm3
	punpcklbw %xmm1, %xmm0
	movq -48(%ecx,%esi), %xmm1
	movdqu %xmm0, (%edx,%esi,2)
	punpcklbw %xmm1, %xmm2
	movq -40(%ecx,%esi), %xmm1
	movdqu %xmm2, 16(%edx,%esi,2)
	movq -32(%ebp,%esi), %xmm2
	punpcklbw %xmm1, %xmm3
	movq -32(%ecx,%esi), %xmm1
	movdqu %xmm3, 32(%edx,%esi,2)
	movq -24(%ebp,%esi), %xmm3
	punpcklbw %xmm1, %xmm2
	movq -24(%ecx,%esi), %xmm1
	movdqu %xmm2, 48(%edx,%esi,2)
	movq -16(%ebp,%esi), %xmm2
	punpcklbw %xmm1, %xmm3
	movq -16(%ecx,%esi), %xmm1
	movdqu %xmm3, 64(%edx,%esi,2)
	movq -8(%ebp,%esi), %xmm3
	punpcklbw %xmm1, %xmm2
	movq -8(%ecx,%esi), %xmm1
	movdqu %xmm2, 80(%edx,%esi,2)
	movq (%ebp,%esi), %xmm2
	punpcklbw %xmm1, %xmm3
	movq (%ecx,%esi), %xmm1
	movdqu %xmm3, 96(%edx,%esi,2)
	punpcklbw %xmm1, %xmm2
	movdqu %xmm2, 112(%edx,%esi,2)
	addl $64, %esi
	cmpl %edi, %esi
	jb .LBB1_4
	jmp .LBB1_2
.LBB1_1:
	xorl %esi, %esi
.LBB1_2:
	movl 52(%esp), %ecx
	movl %esi, 20(%esp)
	movl %edx, 8(%esp)
	movl %eax, (%esp)
	movl $2, 16(%esp)
	movl %ecx, 4(%esp)
	calll jix_probe::byte_shuffle::decode_impl_generic@PLT
	addl $28, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
