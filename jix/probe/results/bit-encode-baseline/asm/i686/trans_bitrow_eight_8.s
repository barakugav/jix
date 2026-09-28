jix_probe::bit_shuffle::trans_bitrow_eight:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $28, %esp
	movl 68(%esp), %ecx
	calll .L34$pb
.L34$pb:
	popl %eax
.Ltmp3360:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp3360-.L34$pb), %eax
	movl %eax, 8(%esp)
	testl %ecx, %ecx
	movl %ecx, 24(%esp)
	je .LBB34_49
	movl 64(%esp), %edx
	xorl %ebp, %ebp
	xorl %esi, %esi
	shrl $3, %edx
	leal (,%edx,8), %eax
	movl %edx, 16(%esp)
	movl %eax, 12(%esp)
	movl 68(%esp), %eax
.LBB34_2:
	movl %edx, %edi
	addl %esi, %edi
	jb .LBB34_51
	cmpl 60(%esp), %edi
	ja .LBB34_51
	movl %edx, %edi
	addl %ebp, %edi
	jb .LBB34_52
	cmpl 52(%esp), %edi
	ja .LBB34_52
	movl %eax, (%esp)
	movl 56(%esp), %eax
	addl 48(%esp), %ebp
	addl %esi, %eax
	subl $4, %esp
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ebp
	pushl %eax
	calll memcpy@PLT
	movl 16(%esp), %eax
	movl 32(%esp), %edx
	addl $16, %esp
	addl 12(%esp), %esi
	decl %eax
	movl %edi, %ebp
	jne .LBB34_2
	movl 68(%esp), %esi
	movl 56(%esp), %eax
	leal (%edx,%edx), %ecx
	xorl %edi, %edi
	movl %ecx, 4(%esp)
	movl %esi, %ebp
	leal 1(%esi), %ecx
	addl %edx, %eax
	imull %edx, %ebp
	imull %edx, %ecx
	movl %eax, 20(%esp)
.LBB34_8:
	movl 4(%esp), %eax
	movl %esi, %ebx
	leal (%edx,%edi), %esi
	addl %edi, %eax
	cmpl %esi, %eax
	jb .LBB34_50
	cmpl 60(%esp), %eax
	ja .LBB34_50
	cmpl %ebp, %ecx
	jb .LBB34_53
	cmpl 52(%esp), %ecx
	ja .LBB34_53
	movl %ecx, (%esp)
	movl 48(%esp), %ecx
	movl 20(%esp), %eax
	movl %ebx, %esi
	addl %ebp, %ecx
	addl %edi, %eax
	subl $4, %esp
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	calll memcpy@PLT
	movl 16(%esp), %ecx
	movl 32(%esp), %edx
	addl $16, %esp
	addl 12(%esp), %edi
	addl %edx, %ebp
	addl %edx, %ecx
	decl %esi
	jne .LBB34_8
	movl 68(%esp), %esi
	leal (%edx,%edx,2), %edi
	movl %esi, %eax
	leal 1(%esi,%esi), %ecx
	imull %edx, %eax
	imull %edx, %ecx
	movl %eax, 20(%esp)
	leal (%eax,%eax), %ebp
	movl %esi, %eax
	movl 4(%esp), %esi
.LBB34_14:
	cmpl %esi, %edi
	jb .LBB34_51
	cmpl 60(%esp), %edi
	ja .LBB34_51
	cmpl %ebp, %ecx
	jb .LBB34_53
	cmpl 52(%esp), %ecx
	ja .LBB34_53
	movl %eax, 4(%esp)
	movl 56(%esp), %eax
	movl %ecx, (%esp)
	movl 48(%esp), %ecx
	addl %esi, %eax
	addl %ebp, %ecx
	subl $4, %esp
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	calll memcpy@PLT
	movl 16(%esp), %ecx
	movl 32(%esp), %edx
	addl $16, %esp
	movl 12(%esp), %eax
	addl %edx, %ebp
	addl %edx, %ecx
	addl %eax, %esi
	addl %eax, %edi
	movl 4(%esp), %eax
	decl %eax
	jne .LBB34_14
	movl 68(%esp), %eax
	leal (%edx,%edx,2), %esi
	leal (,%edx,4), %edi
	leal (%eax,%eax,2), %ebp
	leal 1(%eax,%eax,2), %ecx
	imull %edx, %ebp
	imull %edx, %ecx
.LBB34_20:
	cmpl %esi, %edi
	jb .LBB34_51
	cmpl 60(%esp), %edi
	ja .LBB34_51
	cmpl %ebp, %ecx
	jb .LBB34_53
	cmpl 52(%esp), %ecx
	ja .LBB34_53
	movl %eax, 4(%esp)
	movl 56(%esp), %eax
	movl %ecx, (%esp)
	movl 48(%esp), %ecx
	addl %esi, %eax
	addl %ebp, %ecx
	subl $4, %esp
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	calll memcpy@PLT
	movl 16(%esp), %ecx
	movl 32(%esp), %edx
	addl $16, %esp
	movl 12(%esp), %eax
	addl %edx, %ebp
	addl %edx, %ecx
	addl %eax, %esi
	addl %eax, %edi
	movl 4(%esp), %eax
	decl %eax
	jne .LBB34_20
	movl 68(%esp), %eax
	movl 20(%esp), %ebp
	leal (,%edx,4), %esi
	leal (%edx,%edx,4), %edi
	leal 1(,%eax,4), %ecx
	shll $2, %ebp
	imull %edx, %ecx
.LBB34_26:
	cmpl %esi, %edi
	jb .LBB34_51
	cmpl 60(%esp), %edi
	ja .LBB34_51
	cmpl %ebp, %ecx
	jb .LBB34_53
	cmpl 52(%esp), %ecx
	ja .LBB34_53
	movl %eax, 4(%esp)
	movl 56(%esp), %eax
	movl %ecx, (%esp)
	movl 48(%esp), %ecx
	addl %esi, %eax
	addl %ebp, %ecx
	subl $4, %esp
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	calll memcpy@PLT
	movl 16(%esp), %ecx
	movl 32(%esp), %edx
	addl $16, %esp
	movl 12(%esp), %eax
	addl %edx, %ebp
	addl %edx, %ecx
	addl %eax, %esi
	addl %eax, %edi
	movl 4(%esp), %eax
	decl %eax
	jne .LBB34_26
	movl 68(%esp), %eax
	leal (%edx,%edx), %ecx
	leal (%edx,%edx,4), %esi
	movl %ecx, 20(%esp)
	leal (%ecx,%ecx,2), %edi
	leal (%eax,%eax,4), %ebp
	leal 1(%eax,%eax,4), %ecx
	imull %edx, %ebp
	imull %edx, %ecx
.LBB34_32:
	cmpl %esi, %edi
	jb .LBB34_51
	cmpl 60(%esp), %edi
	ja .LBB34_51
	cmpl %ebp, %ecx
	jb .LBB34_53
	cmpl 52(%esp), %ecx
	ja .LBB34_53
	movl %eax, 4(%esp)
	movl 56(%esp), %eax
	movl %ecx, (%esp)
	movl 48(%esp), %ecx
	addl %esi, %eax
	addl %ebp, %ecx
	subl $4, %esp
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	calll memcpy@PLT
	movl 16(%esp), %ecx
	movl 32(%esp), %edx
	addl $16, %esp
	movl 12(%esp), %eax
	addl %edx, %ebp
	addl %edx, %ecx
	addl %eax, %esi
	addl %eax, %edi
	movl 4(%esp), %eax
	decl %eax
	jne .LBB34_32
	movl 68(%esp), %ebx
	movl 20(%esp), %esi
	movl 12(%esp), %edi
	leal (%ebx,%ebx), %eax
	leal (%esi,%esi,2), %esi
	subl %edx, %edi
	leal (%eax,%eax,2), %ebp
	leal 1(%eax,%eax,2), %eax
	imull %edx, %eax
	imull %edx, %ebp
	movl %eax, %ecx
.LBB34_38:
	cmpl %esi, %edi
	jb .LBB34_51
	cmpl 60(%esp), %edi
	ja .LBB34_51
	cmpl %ebp, %ecx
	jb .LBB34_53
	cmpl 52(%esp), %ecx
	ja .LBB34_53
	movl %ecx, (%esp)
	movl 48(%esp), %ecx
	movl 56(%esp), %eax
	addl %ebp, %ecx
	addl %esi, %eax
	subl $4, %esp
	movl %ebx, 8(%esp)
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	calll memcpy@PLT
	movl 20(%esp), %ebx
	movl 16(%esp), %ecx
	movl 32(%esp), %edx
	addl $16, %esp
	movl 12(%esp), %eax
	addl %edx, %ebp
	addl %edx, %ecx
	addl %eax, %esi
	addl %eax, %edi
	decl %ebx
	jne .LBB34_38
	movl 68(%esp), %eax
	movl 12(%esp), %edi
	leal (,%eax,8), %ecx
	movl %edi, %esi
	subl %eax, %ecx
	subl %edx, %esi
	movl %ecx, %ebp
	incl %ecx
	imull %edx, %ebp
	imull %edx, %ecx
.LBB34_44:
	cmpl %esi, %edi
	jb .LBB34_51
	cmpl 60(%esp), %edi
	ja .LBB34_51
	cmpl %ebp, %ecx
	jb .LBB34_53
	cmpl 52(%esp), %ecx
	ja .LBB34_53
	movl %ecx, (%esp)
	movl 48(%esp), %ecx
	movl 56(%esp), %eax
	addl %ebp, %ecx
	addl %esi, %eax
	subl $4, %esp
	movl 12(%esp), %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	calll memcpy@PLT
	movl 16(%esp), %ecx
	movl 32(%esp), %edx
	addl $16, %esp
	movl 12(%esp), %eax
	addl %edx, %ebp
	addl %edx, %ecx
	addl %eax, %esi
	addl %eax, %edi
	decl 24(%esp)
	jne .LBB34_44
.LBB34_49:
	addl $28, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB34_52:
	addl %ebp, %edx
	movl %edx, %ecx
.LBB34_53:
	movl 8(%esp), %ebx
	leal .Lanon.a00c3abf785d8e01d4f036d98a3600d1.13@GOTOFF(%ebx), %eax
	pushl %eax
	pushl 56(%esp)
	pushl %ecx
	pushl %ebp
	calll core::slice::index::slice_index_fail@PLT
.LBB34_50:
	leal (%edi,%edx,2), %edi
.LBB34_51:
	movl 8(%esp), %ebx
	leal .Lanon.a00c3abf785d8e01d4f036d98a3600d1.14@GOTOFF(%ebx), %eax
	pushl %eax
	pushl 64(%esp)
	pushl %edi
	pushl %esi
	calll core::slice::index::slice_index_fail@PLT
