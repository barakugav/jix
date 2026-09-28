jix_probe::bit_shuffle::trans_bitrow_eight:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $56, %rsp
	movq %rcx, 16(%rsp)
	movq %rdx, 32(%rsp)
	movq %rsi, 8(%rsp)
	movq %rdi, 24(%rsp)
	testq %r9, %r9
	je .LBB16_49
	movq %r8, %rbp
	shrq $3, %rbp
	xorl %r12d, %r12d
	movq %r9, 40(%rsp)
	movq %r9, %rbx
	xorl %r15d, %r15d
.LBB16_2:
	movq %rbp, %r13
	addq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	movq %rbp, %r14
	addq %r12, %r14
	jb .LBB16_51
	cmpq 8(%rsp), %r14
	ja .LBB16_51
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	addq 24(%rsp), %r12
	movq %r12, %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	movq %r14, %r12
	decq %rbx
	jne .LBB16_2
	leaq (,%rbp,2), %r13
	movq 40(%rsp), %rbx
	movq %rbx, %r12
	imulq %rbp, %r12
	leaq 1(%rbx), %r14
	imulq %rbp, %r14
	movq %rbp, %r15
.LBB16_8:
	cmpq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	cmpq %r12, %r14
	jb .LBB16_52
	cmpq 8(%rsp), %r14
	ja .LBB16_52
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	movq 24(%rsp), %rax
	leaq (%rax,%r12), %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	addq %rax, %r13
	addq %rbp, %r12
	addq %rbp, %r14
	decq %rbx
	jne .LBB16_8
	leaq (,%rbp,2), %r15
	leaq (,%rbp,2), %r13
	addq %rbp, %r13
	movq 40(%rsp), %rbx
	movq %rbx, %rax
	imulq %rbp, %rax
	movq %rax, 48(%rsp)
	leaq (%rax,%rax), %r12
	leaq 1(,%rbx,2), %r14
	imulq %rbp, %r14
.LBB16_14:
	cmpq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	cmpq %r12, %r14
	jb .LBB16_52
	cmpq 8(%rsp), %r14
	ja .LBB16_52
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	movq 24(%rsp), %rax
	leaq (%rax,%r12), %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	addq %rax, %r13
	addq %rbp, %r12
	addq %rbp, %r14
	decq %rbx
	jne .LBB16_14
	movq 40(%rsp), %rbx
	leaq (%rbx,%rbx,2), %r12
	leaq (,%rbp,2), %r15
	addq %rbp, %r15
	leaq (,%rbp,4), %r13
	imulq %rbp, %r12
	leaq (%rbx,%rbx,2), %r14
	incq %r14
	imulq %rbp, %r14
.LBB16_20:
	cmpq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	cmpq %r12, %r14
	jb .LBB16_52
	cmpq 8(%rsp), %r14
	ja .LBB16_52
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	movq 24(%rsp), %rax
	leaq (%rax,%r12), %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	addq %rax, %r13
	addq %rbp, %r12
	addq %rbp, %r14
	decq %rbx
	jne .LBB16_20
	leaq (,%rbp,4), %r15
	leaq (,%rbp,4), %r13
	addq %rbp, %r13
	movq 48(%rsp), %r12
	shlq $2, %r12
	movq 40(%rsp), %rbx
	leaq 1(,%rbx,4), %r14
	imulq %rbp, %r14
.LBB16_26:
	cmpq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	cmpq %r12, %r14
	jb .LBB16_52
	cmpq 8(%rsp), %r14
	ja .LBB16_52
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	movq 24(%rsp), %rax
	leaq (%rax,%r12), %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	addq %rax, %r13
	addq %rbp, %r12
	addq %rbp, %r14
	decq %rbx
	jne .LBB16_26
	movq 40(%rsp), %rbx
	leaq (%rbx,%rbx,4), %r12
	leaq (,%rbp,4), %r15
	addq %rbp, %r15
	leaq (,%rbp,2), %rax
	leaq (%rax,%rax,2), %r13
	imulq %rbp, %r12
	leaq (%rbx,%rbx,4), %r14
	incq %r14
	imulq %rbp, %r14
.LBB16_32:
	cmpq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	cmpq %r12, %r14
	jb .LBB16_52
	cmpq 8(%rsp), %r14
	ja .LBB16_52
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	movq 24(%rsp), %rax
	leaq (%rax,%r12), %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	addq %rax, %r13
	addq %rbp, %r12
	addq %rbp, %r14
	decq %rbx
	jne .LBB16_32
	movq 40(%rsp), %rbx
	leaq (%rbx,%rbx), %rax
	leaq (%rax,%rax,2), %r12
	leaq (,%rbp,2), %rcx
	leaq (%rcx,%rcx,2), %r15
	leaq (,%rbp,8), %r13
	subq %rbp, %r13
	imulq %rbp, %r12
	leaq (%rax,%rax,2), %r14
	incq %r14
	imulq %rbp, %r14
.LBB16_38:
	cmpq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	cmpq %r12, %r14
	jb .LBB16_52
	cmpq 8(%rsp), %r14
	ja .LBB16_52
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	movq 24(%rsp), %rax
	leaq (%rax,%r12), %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	addq %rax, %r13
	addq %rbp, %r12
	addq %rbp, %r14
	decq %rbx
	jne .LBB16_38
	movq 40(%rsp), %rbx
	leaq (,%rbx,8), %r14
	subq %rbx, %r14
	leaq (,%rbp,8), %r13
	movq %r13, %r15
	subq %rbp, %r15
	movq %r14, %r12
	imulq %rbp, %r12
	incq %r14
	imulq %rbp, %r14
.LBB16_44:
	cmpq %r15, %r13
	jb .LBB16_50
	cmpq 16(%rsp), %r13
	ja .LBB16_50
	cmpq %r12, %r14
	jb .LBB16_52
	cmpq 8(%rsp), %r14
	ja .LBB16_52
	movq 32(%rsp), %rax
	leaq (%rax,%r15), %rdi
	movq 24(%rsp), %rax
	leaq (%rax,%r12), %rsi
	movq %rbp, %rdx
	callq *memcpy@GOTPCREL(%rip)
	leaq (,%rbp,8), %rax
	addq %rax, %r15
	addq %rax, %r13
	addq %rbp, %r12
	addq %rbp, %r14
	decq %rbx
	jne .LBB16_44
.LBB16_49:
	addq $56, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	retq
.LBB16_50:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.10(%rip), %rcx
	movq %r15, %rdi
	movq %r13, %rsi
	movq 16(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB16_51:
	addq %r12, %rbp
	movq %rbp, %r14
.LBB16_52:
	leaq .Lanon.a4d27c521192673f4882b6c46f85c6f4.9(%rip), %rcx
	movq %r12, %rdi
	movq %r14, %rsi
	movq 8(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
