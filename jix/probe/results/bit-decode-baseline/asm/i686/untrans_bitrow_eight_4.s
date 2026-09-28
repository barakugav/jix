jix_probe::bit_shuffle::untrans_bitrow_eight:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $172, %esp
	calll .L35$pb
.L35$pb:
	popl %eax
.Ltmp3522:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp3522-.L35$pb), %eax
	cmpl $0, 212(%esp)
	movl %eax, 16(%esp)
	je .LBB35_35
	movl 212(%esp), %ebx
	movl 208(%esp), %edx
	movl 192(%esp), %eax
	movl $0, 8(%esp)
	leal (,%ebx,8), %edi
	shrl $3, %edx
	movl %ebx, %ecx
	leal (%ebx,%ebx), %esi
	subl %ebx, %edi
	imull %edx, %ecx
	movl %esi, 64(%esp)
	movl %edx, 4(%esp)
	incl %edi
	leal (,%ecx,8), %ebp
	movl %ecx, 12(%esp)
	imull %edx, %edi
	subl %ecx, %ebp
	movl %edi, 80(%esp)
	leal 1(%esi,%esi,2), %edi
	leal (%esi,%esi,2), %esi
	movl %ebp, 76(%esp)
	leal (,%edx,8), %ebp
	imull %edx, %edi
	movl %esi, 48(%esp)
	leal (%ecx,%ecx), %esi
	movl %ebp, 36(%esp)
	subl %edx, %ebp
	movl %edi, 96(%esp)
	leal 1(%ebx,%ebx,4), %edi
	movl %esi, 148(%esp)
	leal (%esi,%esi,2), %esi
	movl %ebp, 84(%esp)
	imull %edx, %edi
	movl %esi, 92(%esp)
	leal (%ecx,%ecx,4), %esi
	movl %edi, 112(%esp)
	leal 1(,%ebx,4), %edi
	movl %esi, 108(%esp)
	leal (,%ecx,4), %esi
	leal (%ecx,%ecx,2), %ecx
	imull %edx, %edi
	movl %ecx, 40(%esp)
	leal (%eax,%ebp), %ecx
	movl %esi, 124(%esp)
	leal (%eax,%edx,2), %esi
	xorl %ebp, %ebp
	movl %edi, 128(%esp)
	leal 1(%ebx,%ebx,2), %edi
	movl %ecx, 68(%esp)
	leal (%edx,%edx), %ecx
	movl %esi, 136(%esp)
	imull %edx, %edi
	movl %ecx, 156(%esp)
	leal (%ecx,%ecx,2), %ecx
	movl %edi, 140(%esp)
	leal 1(%ebx,%ebx), %edi
	movl %ecx, 100(%esp)
	addl %eax, %ecx
	imull %edx, %edi
	movl %ecx, 72(%esp)
	leal (%edx,%edx,4), %ecx
	movl %edi, 152(%esp)
	leal 1(%ebx), %edi
	movl %ecx, 116(%esp)
	addl %eax, %ecx
	imull %edx, %edi
	movl %ecx, 88(%esp)
	leal (,%edx,4), %ecx
	movl %edi, 160(%esp)
	leal (%ebx,%ebx,2), %edi
	movl %ecx, 132(%esp)
	leal (%eax,%edx,4), %ecx
	movl %edi, 60(%esp)
	leal (,%ebx,4), %edi
	movl %ecx, 104(%esp)
	leal (%edx,%edx,2), %ecx
	movl %edi, 56(%esp)
	leal (%ebx,%ebx,4), %edi
	movl %ecx, 44(%esp)
	addl %eax, %ecx
	addl %edx, %eax
	movl %edi, 52(%esp)
	movl %eax, 144(%esp)
	movl %ebx, %edi
	xorl %eax, %eax
	movl %ecx, 120(%esp)
.LBB35_2:
	movl %ebp, %ecx
	movl %edx, %ebx
	movl %eax, 32(%esp)
	imull %edx, %ecx
	addl %edx, %ecx
	addl %eax, %edx
	jb .LBB35_53
	cmpl 204(%esp), %edx
	ja .LBB35_53
	leal (,%ebp,8), %ecx
	movl %ebx, %esi
	movl %edx, 164(%esp)
	movl %edi, 168(%esp)
	movl %ecx, %eax
	imull %ebx, %eax
	addl %ebx, %eax
	addl 8(%esp), %esi
	jb .LBB35_56
	cmpl 196(%esp), %esi
	ja .LBB35_56
	movl 200(%esp), %eax
	movl 32(%esp), %edi
	movl %ebp, 20(%esp)
	movl %ebx, %edx
	movl %ecx, 24(%esp)
	leal (%eax,%edi), %ebp
	movl 192(%esp), %eax
	leal (%eax,%edi,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl %edx
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	addl $16, %esp
	movl 12(%esp), %eax
	movl 160(%esp), %ecx
	addl %edi, %eax
	addl %edi, %ecx
	cmpl %eax, %ecx
	jb .LBB35_36
	cmpl 204(%esp), %ecx
	ja .LBB35_36
	movl 8(%esp), %eax
	movl 156(%esp), %ecx
	addl %ecx, %eax
	cmpl %esi, %eax
	jb .LBB35_37
	cmpl 196(%esp), %eax
	ja .LBB35_37
	movl %eax, 28(%esp)
	movl 144(%esp), %eax
	addl 12(%esp), %ebp
	leal (%eax,%edi,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl 8(%esp)
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	addl $16, %esp
	movl 148(%esp), %eax
	leal (%eax,%edi), %ecx
	movl 152(%esp), %eax
	addl %edi, %eax
	cmpl %ecx, %eax
	jb .LBB35_38
	cmpl 204(%esp), %eax
	ja .LBB35_38
	movl 8(%esp), %eax
	movl 44(%esp), %ecx
	leal (%ecx,%eax), %esi
	cmpl 28(%esp), %esi
	jb .LBB35_39
	cmpl 196(%esp), %esi
	ja .LBB35_39
	movl 136(%esp), %eax
	addl 12(%esp), %ebp
	leal (%eax,%edi,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl 8(%esp)
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	addl $16, %esp
	movl 40(%esp), %eax
	movl 140(%esp), %ecx
	addl %edi, %eax
	addl %edi, %ecx
	cmpl %eax, %ecx
	jb .LBB35_40
	cmpl 204(%esp), %ecx
	ja .LBB35_40
	movl 8(%esp), %eax
	movl 132(%esp), %ecx
	addl %ecx, %eax
	cmpl %esi, %eax
	jb .LBB35_41
	cmpl 196(%esp), %eax
	ja .LBB35_41
	movl %eax, 28(%esp)
	movl 120(%esp), %eax
	addl 12(%esp), %ebp
	leal (%eax,%edi,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl 8(%esp)
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	addl $16, %esp
	movl 124(%esp), %eax
	leal (%eax,%edi), %ecx
	movl 128(%esp), %eax
	addl %edi, %eax
	cmpl %ecx, %eax
	jb .LBB35_42
	cmpl 204(%esp), %eax
	ja .LBB35_42
	movl 8(%esp), %eax
	movl 116(%esp), %ecx
	leal (%ecx,%eax), %esi
	cmpl 28(%esp), %esi
	jb .LBB35_43
	cmpl 196(%esp), %esi
	ja .LBB35_43
	movl 104(%esp), %eax
	addl 12(%esp), %ebp
	leal (%eax,%edi,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl 8(%esp)
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	addl $16, %esp
	movl 108(%esp), %eax
	movl 112(%esp), %ecx
	addl %edi, %eax
	addl %edi, %ecx
	cmpl %eax, %ecx
	jb .LBB35_44
	cmpl 204(%esp), %ecx
	ja .LBB35_44
	movl 8(%esp), %eax
	movl 100(%esp), %ecx
	addl %ecx, %eax
	cmpl %esi, %eax
	jb .LBB35_46
	cmpl 196(%esp), %eax
	ja .LBB35_46
	movl %eax, %esi
	movl 88(%esp), %eax
	addl 12(%esp), %ebp
	leal (%eax,%edi,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl 8(%esp)
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	addl $16, %esp
	movl 92(%esp), %eax
	leal (%eax,%edi), %ecx
	movl 96(%esp), %eax
	addl %edi, %eax
	cmpl %ecx, %eax
	jb .LBB35_48
	cmpl 204(%esp), %eax
	ja .LBB35_48
	movl 84(%esp), %eax
	movl 8(%esp), %ecx
	addl %ecx, %eax
	cmpl %esi, %eax
	jb .LBB35_49
	cmpl 196(%esp), %eax
	ja .LBB35_49
	movl %eax, 28(%esp)
	movl 72(%esp), %eax
	addl 12(%esp), %ebp
	leal (%eax,%edi,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl 8(%esp)
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	addl $16, %esp
	movl 76(%esp), %eax
	movl 80(%esp), %ecx
	addl %edi, %eax
	addl %edi, %ecx
	cmpl %eax, %ecx
	jb .LBB35_50
	cmpl 204(%esp), %ecx
	ja .LBB35_50
	movl 36(%esp), %eax
	movl 8(%esp), %ecx
	movl 168(%esp), %edi
	leal (%eax,%ecx), %esi
	cmpl 28(%esp), %esi
	jb .LBB35_51
	cmpl 196(%esp), %esi
	ja .LBB35_51
	movl 68(%esp), %eax
	movl 32(%esp), %ecx
	addl 12(%esp), %ebp
	leal (%eax,%ecx,8), %eax
	subl $4, %esp
	movl 20(%esp), %ebx
	pushl 8(%esp)
	pushl %eax
	pushl %ebp
	calll memcpy@PLT
	movl 20(%esp), %edx
	addl $16, %esp
	movl 20(%esp), %ebp
	movl 164(%esp), %eax
	movl %esi, 8(%esp)
	incl %ebp
	decl %edi
	jne .LBB35_2
.LBB35_35:
	addl $172, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB35_53:
	movl 32(%esp), %eax
	jmp .LBB35_54
.LBB35_56:
	movl 8(%esp), %esi
	jmp .LBB35_57
.LBB35_36:
	movl 212(%esp), %edx
	jmp .LBB35_45
.LBB35_37:
	movl 24(%esp), %ecx
	incl %ecx
	jmp .LBB35_47
.LBB35_38:
	movl 64(%esp), %ecx
	movl 4(%esp), %eax
	addl 20(%esp), %ecx
	imull %eax, %ecx
	addl %eax, %ecx
	movl 12(%esp), %eax
	leal (%edi,%eax,2), %eax
	jmp .LBB35_54
.LBB35_39:
	movl 24(%esp), %edx
	movl 4(%esp), %eax
	movl 8(%esp), %ecx
	orl $2, %edx
	leal (%ecx,%eax,2), %esi
	imull %eax, %edx
	addl %eax, %edx
	movl %edx, %eax
	jmp .LBB35_57
.LBB35_40:
	movl 60(%esp), %edx
	jmp .LBB35_45
.LBB35_41:
	movl 24(%esp), %ecx
	orl $3, %ecx
	jmp .LBB35_47
.LBB35_42:
	movl 56(%esp), %ecx
	movl 4(%esp), %eax
	addl 20(%esp), %ecx
	imull %eax, %ecx
	addl %eax, %ecx
	movl 12(%esp), %eax
	leal (%edi,%eax,4), %eax
	jmp .LBB35_54
.LBB35_43:
	movl 24(%esp), %edx
	movl 4(%esp), %eax
	movl 8(%esp), %ecx
	orl $4, %edx
	leal (%ecx,%eax,4), %esi
	imull %eax, %edx
	addl %eax, %edx
	movl %edx, %eax
	jmp .LBB35_57
.LBB35_44:
	movl 52(%esp), %edx
.LBB35_45:
	addl 20(%esp), %edx
	movl 4(%esp), %ecx
	imull %ecx, %edx
	addl %ecx, %edx
	movl %edx, %ecx
	jmp .LBB35_54
.LBB35_46:
	movl 24(%esp), %ecx
	orl $5, %ecx
.LBB35_47:
	movl 4(%esp), %eax
	imull %eax, %ecx
	addl %eax, %ecx
	movl %ecx, %eax
	jmp .LBB35_57
.LBB35_48:
	movl 48(%esp), %ecx
	movl 4(%esp), %eax
	addl 20(%esp), %ecx
	imull %eax, %ecx
	addl %eax, %ecx
	movl 40(%esp), %eax
	leal (%edi,%eax,2), %eax
	jmp .LBB35_54
.LBB35_49:
	movl 24(%esp), %edx
	movl 4(%esp), %eax
	movl 44(%esp), %ecx
	orl $6, %edx
	imull %eax, %edx
	addl %eax, %edx
	movl 8(%esp), %eax
	leal (%eax,%ecx,2), %esi
	movl %edx, %eax
	jmp .LBB35_57
.LBB35_50:
	movl 12(%esp), %edx
	leal (%edi,%edx,8), %eax
	subl %edx, %eax
.LBB35_54:
	movl 16(%esp), %ebx
	leal .Lanon.a00c3abf785d8e01d4f036d98a3600d1.16@GOTOFF(%ebx), %edx
	pushl %edx
	pushl 208(%esp)
	pushl %ecx
	pushl %eax
	calll core::slice::index::slice_index_fail@PLT
.LBB35_51:
	movl 4(%esp), %eax
	movl 8(%esp), %ecx
	leal (%ecx,%eax,8), %esi
	subl %eax, %esi
	movl 36(%esp), %eax
	addl %ecx, %eax
.LBB35_57:
	movl 16(%esp), %ebx
	leal .Lanon.a00c3abf785d8e01d4f036d98a3600d1.15@GOTOFF(%ebx), %ecx
	pushl %ecx
	pushl 200(%esp)
	pushl %eax
	pushl %esi
	calll core::slice::index::slice_index_fail@PLT
