jix_probe::byte_shuffle::encode_impl_generic:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $60, %esp
	movl 96(%esp), %esi
	calll .L9$pb
.L9$pb:
	popl %ebx
.Ltmp1721:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp1721-.L9$pb), %ebx
	testl %esi, %esi
	je .LBB9_3
	movl 84(%esp), %eax
	movl 100(%esp), %ebp
	xorl %edx, %edx
	xorl %ebx, %ebx
	movl %eax, %ecx
	divl %esi
	cmpl %eax, %ebp
	jae .LBB9_16
	subl %ebp, %ecx
	movl 80(%esp), %edi
	cmpl $8, %ecx
	movl %ecx, 20(%esp)
	setae %dl
	cmpl $1, %esi
	sete %dh
	andb %dl, %dh
	movl %edi, 36(%esp)
	movb %dh, 19(%esp)
	movl %ecx, %edx
	andl $-8, %ecx
	andl $-32, %edx
	movl %ecx, 52(%esp)
	movl %edx, 24(%esp)
	addl %ebp, %edx
	movl %edx, 40(%esp)
	movl %ecx, %edx
	addl %ebp, %ecx
	negl %edx
	movl %ecx, 44(%esp)
	leal 16(%edi,%ebp), %ecx
	movl %edx, 48(%esp)
	leal (%edi,%ebp), %edx
	movl %ecx, 28(%esp)
	movl %edx, 32(%esp)
	jmp .LBB9_5
.LBB9_4:
	movl 56(%esp), %ebx
	incl 28(%esp)
	incl 32(%esp)
	incl 36(%esp)
	incl %ebx
	cmpl %esi, %ebx
	je .LBB9_16
.LBB9_5:
	movl %ebx, 56(%esp)
	imull %eax, %ebx
	movl %ebp, %edi
	addl 88(%esp), %ebx
	cmpb $0, 19(%esp)
	je .LBB9_14
	xorl %edx, %edx
	cmpl $32, 20(%esp)
	jb .LBB9_11
	movl %ebp, %esi
	movl 28(%esp), %edi
	movl 24(%esp), %ebp
	xorl %edx, %edx
.LBB9_8:
	movups -16(%edi), %xmm0
	movups (%edi), %xmm1
	leal (%esi,%edx), %ecx
	addl $32, %edx
	addl $32, %edi
	addl $-32, %ebp
	movups %xmm0, (%ebx,%ecx)
	movups %xmm1, 16(%ebx,%ecx)
	jne .LBB9_8
	movl 24(%esp), %ecx
	movl %esi, %ebp
	movl 96(%esp), %esi
	cmpl %ecx, 20(%esp)
	je .LBB9_4
	testb $24, 20(%esp)
	movl 24(%esp), %edx
	movl 40(%esp), %edi
	je .LBB9_14
.LBB9_11:
	movl 48(%esp), %ecx
	movl %ebp, %esi
	movl 32(%esp), %ebp
	leal (%ecx,%edx), %edi
	addl %edx, %ebp
.LBB9_12:
	movsd (%ebp), %xmm0
	leal (%esi,%edx), %ecx
	addl $8, %edx
	addl $8, %ebp
	addl $8, %edi
	movsd %xmm0, (%ebx,%ecx)
	jne .LBB9_12
	movl 52(%esp), %ecx
	movl %esi, %ebp
	movl 44(%esp), %edi
	movl 96(%esp), %esi
	cmpl %ecx, 20(%esp)
	je .LBB9_4
.LBB9_14:
	movl %esi, %edx
	imull %edi, %edx
	addl 36(%esp), %edx
.LBB9_15:
	movzbl (%edx), %ecx
	addl %esi, %edx
	movb %cl, (%ebx,%edi)
	incl %edi
	cmpl %eax, %edi
	jb .LBB9_15
	jmp .LBB9_4
.LBB9_16:
	addl $60, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB9_3:
	leal .Lanon.8aa56d0f43c569882aff702370a80287.2@GOTOFF(%ebx), %eax
	movl %eax, (%esp)
	calll core::panicking::panic_const::panic_const_div_by_zero@PLT
