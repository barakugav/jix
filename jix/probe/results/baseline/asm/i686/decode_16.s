jix_probe::byte_shuffle::decode_impl::<16, 8>:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $172, %esp
	movl 196(%esp), %edi
	calll .L0$pb
.L0$pb:
	popl %eax
	movl 192(%esp), %ebp
.Ltmp1:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp1-.L0$pb), %eax
	movl %eax, 36(%esp)
	shrl $4, %edi
	movl %edi, %eax
	andl $134217720, %eax
	movl %eax, 40(%esp)
	je .LBB0_1
	xorl %esi, %esi
.LBB0_4:
	movl $-16, %eax
	movl %ebp, %ecx
.LBB0_5:
	movzbl (%ecx), %edx
	movb %dl, 34(%esp)
	movzbl 4(%ecx), %edx
	movb 1(%ecx), %dh
	movzbl 2(%ecx), %ebx
	movb 3(%ecx), %bh
	movb %dl, 35(%esp)
	movb 5(%ecx), %dl
	movb %dl, 33(%esp)
	movb 6(%ecx), %dl
	movb %dl, 32(%esp)
	movb 7(%ecx), %dl
	addl %edi, %ecx
	movb %dl, 31(%esp)
	movb 34(%esp), %dl
	movb %dl, 60(%esp,%eax)
	movb 35(%esp), %dl
	movb %dh, 76(%esp,%eax)
	movb %bl, 92(%esp,%eax)
	movb %bh, 108(%esp,%eax)
	movb 31(%esp), %dh
	movb %dl, 124(%esp,%eax)
	movb 33(%esp), %dl
	movb %dl, 140(%esp,%eax)
	movb 32(%esp), %dl
	movb %dl, 156(%esp,%eax)
	movb %dh, 172(%esp,%eax)
	incl %eax
	jne .LBB0_5
	movl %esi, %eax
	movl 36(%esp), %ebx
	leal 44(%esp), %ecx
	movl $128, 8(%esp)
	shll $4, %eax
	movl %ecx, 4(%esp)
	addl 200(%esp), %eax
	movl %eax, (%esp)
	calll memcpy@PLT
	addl $8, %esi
	addl $8, %ebp
	cmpl 40(%esp), %esi
	jb .LBB0_4
	jmp .LBB0_2
.LBB0_1:
	xorl %esi, %esi
.LBB0_2:
	movl 200(%esp), %eax
	movl 196(%esp), %edx
	movl 192(%esp), %ecx
	movl 36(%esp), %ebx
	movl %esi, 20(%esp)
	movl $16, 16(%esp)
	movl %eax, 8(%esp)
	movl %edx, 4(%esp)
	movl %ecx, (%esp)
	calll jix_probe::byte_shuffle::decode_impl_generic@PLT
	addl $172, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
