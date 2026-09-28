probe_byte_shuffle_encode_2:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $12, %esp
	movl 36(%esp), %ecx
	calll .L35$pb
.L35$pb:
	popl %ebx
.Ltmp4102:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp4102-.L35$pb), %ebx
	movl %ecx, %eax
	andl $2147483646, %eax
	cmpl %eax, 44(%esp)
	jb .LBB35_5
	movl %ecx, %eax
	shrl %eax
	shrl $8, %ecx
	je .LBB35_4
	movl 40(%esp), %edx
	movl 32(%esp), %esi
	movdqa .LCPI35_0@GOTOFF(%ebx), %xmm0
	shll $7, %ecx
	xorl %ebp, %ebp
	leal 112(%eax,%edx), %edi
.LBB35_3:
	movdqu (%esi,%ebp,2), %xmm1
	movdqu 16(%esi,%ebp,2), %xmm2
	movdqu 32(%esi,%ebp,2), %xmm5
	movdqu 64(%esi,%ebp,2), %xmm6
	movdqa %xmm2, %xmm3
	movdqa %xmm1, %xmm4
	psrlw $8, %xmm2
	psrlw $8, %xmm1
	packuswb %xmm2, %xmm1
	movdqu 48(%esi,%ebp,2), %xmm2
	pand %xmm0, %xmm3
	pand %xmm0, %xmm4
	packuswb %xmm3, %xmm4
	movdqu %xmm4, (%edx,%ebp)
	movdqa %xmm5, %xmm4
	psrlw $8, %xmm5
	movdqu %xmm1, -112(%edi,%ebp)
	pand %xmm0, %xmm4
	movdqa %xmm2, %xmm3
	psrlw $8, %xmm2
	packuswb %xmm2, %xmm5
	movdqu 80(%esi,%ebp,2), %xmm2
	pand %xmm0, %xmm3
	packuswb %xmm3, %xmm4
	movdqu %xmm4, 16(%edx,%ebp)
	movdqa %xmm6, %xmm4
	psrlw $8, %xmm6
	movdqu %xmm5, -96(%edi,%ebp)
	movdqu 96(%esi,%ebp,2), %xmm5
	pand %xmm0, %xmm4
	movdqa %xmm2, %xmm3
	psrlw $8, %xmm2
	packuswb %xmm2, %xmm6
	movdqu 112(%esi,%ebp,2), %xmm2
	pand %xmm0, %xmm3
	packuswb %xmm3, %xmm4
	movdqu %xmm4, 32(%edx,%ebp)
	movdqa %xmm5, %xmm4
	psrlw $8, %xmm5
	movdqu %xmm6, -80(%edi,%ebp)
	movdqu 128(%esi,%ebp,2), %xmm6
	pand %xmm0, %xmm4
	movdqa %xmm2, %xmm3
	psrlw $8, %xmm2
	packuswb %xmm2, %xmm5
	movdqu 144(%esi,%ebp,2), %xmm2
	pand %xmm0, %xmm3
	packuswb %xmm3, %xmm4
	movdqu %xmm4, 48(%edx,%ebp)
	movdqa %xmm6, %xmm4
	psrlw $8, %xmm6
	movdqu %xmm5, -64(%edi,%ebp)
	movdqu 160(%esi,%ebp,2), %xmm5
	pand %xmm0, %xmm4
	movdqa %xmm2, %xmm3
	psrlw $8, %xmm2
	packuswb %xmm2, %xmm6
	movdqu 176(%esi,%ebp,2), %xmm2
	pand %xmm0, %xmm3
	packuswb %xmm3, %xmm4
	movdqu %xmm4, 64(%edx,%ebp)
	movdqa %xmm5, %xmm4
	psrlw $8, %xmm5
	movdqu %xmm6, -48(%edi,%ebp)
	movdqu 192(%esi,%ebp,2), %xmm6
	pand %xmm0, %xmm4
	movdqa %xmm2, %xmm3
	psrlw $8, %xmm2
	packuswb %xmm2, %xmm5
	movdqu 208(%esi,%ebp,2), %xmm2
	pand %xmm0, %xmm3
	packuswb %xmm3, %xmm4
	movdqu %xmm4, 80(%edx,%ebp)
	movdqa %xmm6, %xmm4
	psrlw $8, %xmm6
	movdqu %xmm5, -32(%edi,%ebp)
	movdqu 224(%esi,%ebp,2), %xmm5
	pand %xmm0, %xmm4
	movdqa %xmm2, %xmm3
	psrlw $8, %xmm2
	packuswb %xmm2, %xmm6
	movdqu 240(%esi,%ebp,2), %xmm2
	pand %xmm0, %xmm3
	packuswb %xmm3, %xmm4
	movdqu %xmm4, 96(%edx,%ebp)
	movdqa %xmm5, %xmm4
	psrlw $8, %xmm5
	movdqu %xmm6, -16(%edi,%ebp)
	pand %xmm0, %xmm4
	movdqa %xmm2, %xmm3
	psrlw $8, %xmm2
	pand %xmm0, %xmm3
	packuswb %xmm2, %xmm5
	packuswb %xmm3, %xmm4
	movdqu %xmm4, 112(%edx,%ebp)
	movdqu %xmm5, (%edi,%ebp)
	subl $-128, %ebp
	cmpl %ebp, %ecx
	jne .LBB35_3
.LBB35_4:
	andl $1073741696, %eax
	addl $12, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB35_5:
	subl $4, %esp
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.6@GOTOFF(%ebx), %eax
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
