probe_byte_shuffle_encode_2:
	pushl %ebp
	pushl %ebx
	pushl %edi
	pushl %esi
	subl $12, %esp
	movl 36(%esp), %ebp
	calll .L35$pb
.L35$pb:
	popl %ebx
.Ltmp4525:
	addl $_GLOBAL_OFFSET_TABLE_+(.Ltmp4525-.L35$pb), %ebx
	movl %ebp, %eax
	andl $2147483646, %eax
	cmpl %eax, 44(%esp)
	jb .LBB35_7
	movl %ebp, %eax
	shrl %eax
	shrl $8, %ebp
	je .LBB35_6
	movl 32(%esp), %esi
	movl 40(%esp), %edx
	leal 15(%edx,%eax), %edi
	addl $15, %edx
	addl $31, %esi
.LBB35_4:
	movl $-128, %ebx
.LBB35_5:
	movzbl 225(%esi,%ebx,2), %ecx
	movb 256(%esi,%ebx,2), %ch
	movb %cl, 113(%edx,%ebx)
	movb 227(%esi,%ebx,2), %cl
	movb %cl, 114(%edx,%ebx)
	movb 229(%esi,%ebx,2), %cl
	movb %cl, 115(%edx,%ebx)
	movb 231(%esi,%ebx,2), %cl
	movb %cl, 116(%edx,%ebx)
	movb 233(%esi,%ebx,2), %cl
	movb %cl, 117(%edx,%ebx)
	movb 235(%esi,%ebx,2), %cl
	movb %cl, 118(%edx,%ebx)
	movb 237(%esi,%ebx,2), %cl
	movb %cl, 119(%edx,%ebx)
	movb 239(%esi,%ebx,2), %cl
	movb %cl, 120(%edx,%ebx)
	movb 241(%esi,%ebx,2), %cl
	movb %cl, 121(%edx,%ebx)
	movb 243(%esi,%ebx,2), %cl
	movb %cl, 122(%edx,%ebx)
	movb 245(%esi,%ebx,2), %cl
	movb %cl, 123(%edx,%ebx)
	movb 247(%esi,%ebx,2), %cl
	movb %cl, 124(%edx,%ebx)
	movb 249(%esi,%ebx,2), %cl
	movb %cl, 125(%edx,%ebx)
	movb 251(%esi,%ebx,2), %cl
	movb %cl, 126(%edx,%ebx)
	movb 253(%esi,%ebx,2), %cl
	movb %cl, 127(%edx,%ebx)
	movb 255(%esi,%ebx,2), %cl
	movb %cl, 128(%edx,%ebx)
	movb 226(%esi,%ebx,2), %cl
	movb %cl, 113(%edi,%ebx)
	movb 228(%esi,%ebx,2), %cl
	movb %cl, 114(%edi,%ebx)
	movb 230(%esi,%ebx,2), %cl
	movb %cl, 115(%edi,%ebx)
	movb 232(%esi,%ebx,2), %cl
	movb %cl, 116(%edi,%ebx)
	movb 234(%esi,%ebx,2), %cl
	movb %cl, 117(%edi,%ebx)
	movb 236(%esi,%ebx,2), %cl
	movb %cl, 118(%edi,%ebx)
	movb 238(%esi,%ebx,2), %cl
	movb %cl, 119(%edi,%ebx)
	movb 240(%esi,%ebx,2), %cl
	movb %cl, 120(%edi,%ebx)
	movb 242(%esi,%ebx,2), %cl
	movb %cl, 121(%edi,%ebx)
	movb 244(%esi,%ebx,2), %cl
	movb %cl, 122(%edi,%ebx)
	movb 246(%esi,%ebx,2), %cl
	movb %cl, 123(%edi,%ebx)
	movb 248(%esi,%ebx,2), %cl
	movb %cl, 124(%edi,%ebx)
	movb 250(%esi,%ebx,2), %cl
	movb %cl, 125(%edi,%ebx)
	movb 252(%esi,%ebx,2), %cl
	movb %cl, 126(%edi,%ebx)
	movb 254(%esi,%ebx,2), %cl
	movb %cl, 127(%edi,%ebx)
	movb %ch, 128(%edi,%ebx)
	addl $16, %ebx
	jne .LBB35_5
	subl $-128, %edi
	subl $-128, %edx
	addl $256, %esi
	decl %ebp
	jne .LBB35_4
.LBB35_6:
	andl $1073741696, %eax
	addl $12, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
.LBB35_7:
	subl $4, %esp
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.6@GOTOFF(%ebx), %eax
	leal .Lanon.60b08d49a6e92428fc97acd24b5c1e0b.3@GOTOFF(%ebx), %ecx
	pushl %eax
	pushl $48
	pushl %ecx
	calll core::panicking::panic@PLT
