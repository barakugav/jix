jix_probe::byte_shuffle::decode_impl_generic:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $12, %esp
	movl 48(%esp), %ebp
	calll .L8$pb
.L8$pb:
	popl %ebx
.Ltmp1697:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp1697-.L8$pb), %ebx
	testl %ebp, %ebp
	je .LBB8_3
	movl 36(%esp), %eax
	movl 52(%esp), %ecx
	xorl %edx, %edx
	divl %ebp
	cmpl %eax, %ecx
	jae .LBB8_7
	movl %ecx, %esi
	imull %ebp, %esi
	addl 40(%esp), %esi
.LBB8_5:
	movl 32(%esp), %edx
	movl %esi, %edi
	movl %ecx, 8(%esp)
	leal (%edx,%ecx), %ebx
	xorl %edx, %edx
.LBB8_6:
	leal 1(%edx), %ecx
	imull %eax, %edx
	movzbl (%ebx,%edx), %edx
	movb %dl, (%edi)
	incl %edi
	decl %ebp
	movl %ecx, %edx
	jne .LBB8_6
	movl 8(%esp), %ecx
	movl 48(%esp), %ebp
	incl %ecx
	addl %ebp, %esi
	cmpl %eax, %ecx
	jb .LBB8_5
.LBB8_7:
	addl $12, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB8_3:
	leal .Lanon.8aa56d0f43c569882aff702370a80287.1@GOTOFF(%ebx), %eax
	movl %eax, (%esp)
	calll core::panicking::panic_const::panic_const_div_by_zero@PLT
