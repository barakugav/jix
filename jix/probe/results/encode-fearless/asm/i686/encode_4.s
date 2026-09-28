probe_byte_shuffle_encode_4:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $12, %esp
	movl 36(%esp), %ecx
	calll .L36$pb
.L36$pb:
	popl %ebx
.Ltmp4607:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp4607-.L36$pb), %ebx
	movl %ecx, %eax
	andl $2147483644, %eax
	cmpl %eax, 44(%esp)
	jb .LBB36_7
	movl %ecx, %eax
	shrl $2, %eax
	shrl $8, %ecx
	movl %eax, 4(%esp)
	je .LBB36_6
	movl 4(%esp), %ebp
	movl 32(%esp), %esi
	movl 40(%esp), %edx
	leal (%ebp,%ebp,2), %eax
	leal 15(%edx,%ebp,2), %ebx
	leal 15(%edx,%ebp), %ebp
	addl $63, %esi
	leal 15(%edx,%eax), %edi
	addl $15, %edx
.LBB36_4:
	movl %ecx, 8(%esp)
	xorl %ecx, %ecx
.LBB36_5:
	movzbl -63(%esi,%ecx,4), %eax
	movb (%esi,%ecx,4), %ah
	movb %al, -15(%edx,%ecx)
	movb -59(%esi,%ecx,4), %al
	movb %al, -14(%edx,%ecx)
	movb -55(%esi,%ecx,4), %al
	movb %al, -13(%edx,%ecx)
	movb -51(%esi,%ecx,4), %al
	movb %al, -12(%edx,%ecx)
	movb -47(%esi,%ecx,4), %al
	movb %al, -11(%edx,%ecx)
	movb -43(%esi,%ecx,4), %al
	movb %al, -10(%edx,%ecx)
	movb -39(%esi,%ecx,4), %al
	movb %al, -9(%edx,%ecx)
	movb -35(%esi,%ecx,4), %al
	movb %al, -8(%edx,%ecx)
	movb -31(%esi,%ecx,4), %al
	movb %al, -7(%edx,%ecx)
	movb -27(%esi,%ecx,4), %al
	movb %al, -6(%edx,%ecx)
	movb -23(%esi,%ecx,4), %al
	movb %al, -5(%edx,%ecx)
	movb -19(%esi,%ecx,4), %al
	movb %al, -4(%edx,%ecx)
	movb -15(%esi,%ecx,4), %al
	movb %al, -3(%edx,%ecx)
	movb -11(%esi,%ecx,4), %al
	movb %al, -2(%edx,%ecx)
	movb -7(%esi,%ecx,4), %al
	movb %al, -1(%edx,%ecx)
	movb -3(%esi,%ecx,4), %al
	movb %al, (%edx,%ecx)
	movb -62(%esi,%ecx,4), %al
	movb %al, -15(%ebp,%ecx)
	movb -58(%esi,%ecx,4), %al
	movb %al, -14(%ebp,%ecx)
	movb -54(%esi,%ecx,4), %al
	movb %al, -13(%ebp,%ecx)
	movb -50(%esi,%ecx,4), %al
	movb %al, -12(%ebp,%ecx)
	movb -46(%esi,%ecx,4), %al
	movb %al, -11(%ebp,%ecx)
	movb -42(%esi,%ecx,4), %al
	movb %al, -10(%ebp,%ecx)
	movb -38(%esi,%ecx,4), %al
	movb %al, -9(%ebp,%ecx)
	movb -34(%esi,%ecx,4), %al
	movb %al, -8(%ebp,%ecx)
	movb -30(%esi,%ecx,4), %al
	movb %al, -7(%ebp,%ecx)
	movb -26(%esi,%ecx,4), %al
	movb %al, -6(%ebp,%ecx)
	movb -22(%esi,%ecx,4), %al
	movb %al, -5(%ebp,%ecx)
	movb -18(%esi,%ecx,4), %al
	movb %al, -4(%ebp,%ecx)
	movb -14(%esi,%ecx,4), %al
	movb %al, -3(%ebp,%ecx)
	movb -10(%esi,%ecx,4), %al
	movb %al, -2(%ebp,%ecx)
	movb -6(%esi,%ecx,4), %al
	movb %al, -1(%ebp,%ecx)
	movb -2(%esi,%ecx,4), %al
	movb %al, (%ebp,%ecx)
	movb -61(%esi,%ecx,4), %al
	movb %al, -15(%ebx,%ecx)
	movb -57(%esi,%ecx,4), %al
	movb %al, -14(%ebx,%ecx)
	movb -53(%esi,%ecx,4), %al
	movb %al, -13(%ebx,%ecx)
	movb -49(%esi,%ecx,4), %al
	movb %al, -12(%ebx,%ecx)
	movb -45(%esi,%ecx,4), %al
	movb %al, -11(%ebx,%ecx)
	movb -41(%esi,%ecx,4), %al
	movb %al, -10(%ebx,%ecx)
	movb -37(%esi,%ecx,4), %al
	movb %al, -9(%ebx,%ecx)
	movb -33(%esi,%ecx,4), %al
	movb %al, -8(%ebx,%ecx)
	movb -29(%esi,%ecx,4), %al
	movb %al, -7(%ebx,%ecx)
	movb -25(%esi,%ecx,4), %al
	movb %al, -6(%ebx,%ecx)
	movb -21(%esi,%ecx,4), %al
	movb %al, -5(%ebx,%ecx)
	movb -17(%esi,%ecx,4), %al
	movb %al, -4(%ebx,%ecx)
	movb -13(%esi,%ecx,4), %al
	movb %al, -3(%ebx,%ecx)
	movb -9(%esi,%ecx,4), %al
	movb %al, -2(%ebx,%ecx)
	movb -5(%esi,%ecx,4), %al
	movb %al, -1(%ebx,%ecx)
	movb -1(%esi,%ecx,4), %al
	movb %al, (%ebx,%ecx)
	movb -60(%esi,%ecx,4), %al
	movb %al, -15(%edi,%ecx)
	movb -56(%esi,%ecx,4), %al
	movb %al, -14(%edi,%ecx)
	movb -52(%esi,%ecx,4), %al
	movb %al, -13(%edi,%ecx)
	movb -48(%esi,%ecx,4), %al
	movb %al, -12(%edi,%ecx)
	movb -44(%esi,%ecx,4), %al
	movb %al, -11(%edi,%ecx)
	movb -40(%esi,%ecx,4), %al
	movb %al, -10(%edi,%ecx)
	movb -36(%esi,%ecx,4), %al
	movb %al, -9(%edi,%ecx)
	movb -32(%esi,%ecx,4), %al
	movb %al, -8(%edi,%ecx)
	movb -28(%esi,%ecx,4), %al
	movb %al, -7(%edi,%ecx)
	movb -24(%esi,%ecx,4), %al
	movb %al, -6(%edi,%ecx)
	movb -20(%esi,%ecx,4), %al
	movb %al, -5(%edi,%ecx)
	movb -16(%esi,%ecx,4), %al
	movb %al, -4(%edi,%ecx)
	movb -12(%esi,%ecx,4), %al
	movb %al, -3(%edi,%ecx)
	movb -8(%esi,%ecx,4), %al
	movb %al, -2(%edi,%ecx)
	movb -4(%esi,%ecx,4), %al
	movb %al, -1(%edi,%ecx)
	movb %ah, (%edi,%ecx)
	addl $16, %ecx
	cmpl $64, %ecx
	jne .LBB36_5
	movl 8(%esp), %ecx
	addl $64, %edi
	addl $64, %ebx
	addl $64, %ebp
	addl $64, %edx
	addl $256, %esi
	decl %ecx
	jne .LBB36_4
.LBB36_6:
	movl 4(%esp), %eax
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
