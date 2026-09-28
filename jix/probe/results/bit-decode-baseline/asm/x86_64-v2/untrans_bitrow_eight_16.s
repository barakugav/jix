jix_probe::bit_shuffle::untrans_bitrow_eight:
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12
	pushq %rbx
	subq $344, %rsp
	movq %rcx, 32(%rsp)
	movq %rdx, 328(%rsp)
	movq %rsi, 24(%rsp)
	movq %rdi, 88(%rsp)
	movq %r9, 56(%rsp)
	testq %r9, %r9
	je .LBB25_35
	movq %r8, %rbx
	shrq $3, %rbx
	movq 56(%rsp), %rax
	leaq (%rax,%rax), %rdx
	leaq (%rax,%rax,2), %rcx
	movq %rcx, 120(%rsp)
	leaq (,%rax,4), %rcx
	movq %rcx, 112(%rsp)
	leaq (%rax,%rax,4), %rcx
	movq %rcx, 104(%rsp)
	leaq (,%rax,8), %rsi
	subq %rax, %rsi
	movq %rax, %rcx
	imulq %rbx, %rcx
	leaq (,%rcx,8), %rdi
	subq %rcx, %rdi
	movq %rdi, 152(%rsp)
	incq %rsi
	imulq %rbx, %rsi
	movq %rsi, 160(%rsp)
	leaq (%rdx,%rdx,2), %rsi
	incq %rsi
	imulq %rbx, %rsi
	movq %rsi, 192(%rsp)
	leaq (%rax,%rax,4), %rsi
	incq %rsi
	imulq %rbx, %rsi
	movq %rsi, 224(%rsp)
	leaq 1(,%rax,4), %rsi
	imulq %rbx, %rsi
	movq %rsi, 256(%rsp)
	leaq (%rax,%rax,2), %rsi
	incq %rsi
	imulq %rbx, %rsi
	movq %rsi, 280(%rsp)
	leaq 1(,%rax,2), %rsi
	imulq %rbx, %rsi
	movq %rsi, 304(%rsp)
	leaq 1(%rax), %rsi
	imulq %rbx, %rsi
	movq %rsi, 320(%rsp)
	leaq (,%rbx,8), %rsi
	movq %rsi, 64(%rsp)
	subq %rbx, %rsi
	movq %rdx, 128(%rsp)
	leaq (%rdx,%rdx,2), %rdx
	movq %rdx, 96(%rsp)
	leaq (%rcx,%rcx), %rdx
	movq %rdx, 296(%rsp)
	leaq (%rdx,%rdx,2), %rdx
	movq %rdx, 184(%rsp)
	leaq (%rcx,%rcx,4), %rdx
	movq %rdx, 216(%rsp)
	leaq (,%rcx,4), %rdx
	movq %rdx, 248(%rsp)
	movq %rcx, 16(%rsp)
	leaq (%rcx,%rcx,2), %rcx
	movq %rcx, 72(%rsp)
	movq 88(%rsp), %rcx
	movq %rsi, 168(%rsp)
	leaq (%rcx,%rsi), %rdx
	movq %rdx, 136(%rsp)
	leaq (%rbx,%rbx), %rdx
	movq %rdx, 312(%rsp)
	leaq (%rdx,%rdx,2), %rdx
	movq %rdx, 200(%rsp)
	leaq (%rcx,%rdx), %rdx
	movq %rdx, 144(%rsp)
	leaq (%rbx,%rbx,4), %rdx
	movq %rdx, 232(%rsp)
	leaq (%rcx,%rdx), %rdx
	movq %rdx, 176(%rsp)
	leaq (,%rbx,4), %rdx
	movq %rdx, 264(%rsp)
	leaq (%rcx,%rbx,4), %rdx
	movq %rdx, 208(%rsp)
	leaq (%rbx,%rbx,2), %rdx
	movq %rdx, 80(%rsp)
	addq %rcx, %rdx
	movq %rdx, 240(%rsp)
	leaq (%rcx,%rbx,2), %rdx
	movq %rdx, 272(%rsp)
	addq %rbx, %rcx
	movq %rcx, 288(%rsp)
	movq $0, 8(%rsp)
	movq %rax, %r14
	xorl %ebp, %ebp
	xorl %eax, %eax
.LBB25_2:
	movq %rax, %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq %rbx, %r13
	addq %rbp, %r13
	jb .LBB25_53
	cmpq 32(%rsp), %r13
	ja .LBB25_53
	leaq (,%rax,8), %rcx
	movq %rcx, %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq %rbx, %r15
	addq 8(%rsp), %r15
	jb .LBB25_55
	cmpq 24(%rsp), %r15
	ja .LBB25_55
	movq %rcx, 48(%rsp)
	movq %rax, 40(%rsp)
	movq 328(%rsp), %rax
	leaq (%rax,%rbp), %r12
	movq 88(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	movq %r12, %rdi
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 16(%rsp), %rax
	leaq (%rax,%rbp), %rdi
	movq 320(%rsp), %rax
	addq %rbp, %rax
	cmpq %rdi, %rax
	jb .LBB25_36
	cmpq 32(%rsp), %rax
	ja .LBB25_36
	movq %r14, 336(%rsp)
	movq 8(%rsp), %rax
	movq 312(%rsp), %rcx
	leaq (%rcx,%rax), %r14
	cmpq %r15, %r14
	jb .LBB25_37
	cmpq 24(%rsp), %r14
	ja .LBB25_37
	movq 288(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	addq 16(%rsp), %r12
	movq %r12, %rdi
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 296(%rsp), %rax
	leaq (%rax,%rbp), %rcx
	movq 304(%rsp), %rax
	addq %rbp, %rax
	cmpq %rcx, %rax
	jb .LBB25_38
	cmpq 32(%rsp), %rax
	ja .LBB25_38
	movq 8(%rsp), %rax
	movq 80(%rsp), %rcx
	leaq (%rcx,%rax), %r15
	cmpq %r14, %r15
	jb .LBB25_39
	cmpq 24(%rsp), %r15
	ja .LBB25_39
	movq 272(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	addq 16(%rsp), %r12
	movq %r12, %rdi
	movq %rbx, %r14
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 72(%rsp), %rax
	leaq (%rax,%rbp), %rdi
	movq 280(%rsp), %rax
	addq %rbp, %rax
	cmpq %rdi, %rax
	jb .LBB25_40
	cmpq 32(%rsp), %rax
	ja .LBB25_40
	movq 8(%rsp), %rax
	movq 264(%rsp), %rcx
	leaq (%rcx,%rax), %r14
	cmpq %r15, %r14
	jb .LBB25_41
	cmpq 24(%rsp), %r14
	ja .LBB25_41
	movq 240(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	addq 16(%rsp), %r12
	movq %r12, %rdi
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 248(%rsp), %rax
	leaq (%rax,%rbp), %rcx
	movq 256(%rsp), %rax
	addq %rbp, %rax
	cmpq %rcx, %rax
	jb .LBB25_42
	cmpq 32(%rsp), %rax
	ja .LBB25_42
	movq 8(%rsp), %rax
	movq 232(%rsp), %rcx
	leaq (%rcx,%rax), %r15
	cmpq %r14, %r15
	jb .LBB25_43
	cmpq 24(%rsp), %r15
	ja .LBB25_43
	movq 208(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	addq 16(%rsp), %r12
	movq %r12, %rdi
	movq %rbx, %r14
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 216(%rsp), %rax
	leaq (%rax,%rbp), %rdi
	movq 224(%rsp), %rax
	addq %rbp, %rax
	cmpq %rdi, %rax
	jb .LBB25_44
	cmpq 32(%rsp), %rax
	ja .LBB25_44
	movq 8(%rsp), %rax
	movq 200(%rsp), %rcx
	leaq (%rcx,%rax), %r14
	cmpq %r15, %r14
	jb .LBB25_46
	cmpq 24(%rsp), %r14
	ja .LBB25_46
	movq 176(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	addq 16(%rsp), %r12
	movq %r12, %rdi
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 184(%rsp), %rax
	leaq (%rax,%rbp), %rcx
	movq 192(%rsp), %rax
	addq %rbp, %rax
	cmpq %rcx, %rax
	jb .LBB25_48
	cmpq 32(%rsp), %rax
	ja .LBB25_48
	movq 168(%rsp), %rax
	movq 8(%rsp), %rcx
	leaq (%rax,%rcx), %r15
	cmpq %r14, %r15
	jb .LBB25_49
	cmpq 24(%rsp), %r15
	ja .LBB25_49
	movq 144(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	addq 16(%rsp), %r12
	movq %r12, %rdi
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 152(%rsp), %rax
	addq %rbp, %rax
	movq 160(%rsp), %rcx
	leaq (%rcx,%rbp), %rsi
	cmpq %rax, %rsi
	jb .LBB25_50
	cmpq 32(%rsp), %rsi
	ja .LBB25_50
	movq 64(%rsp), %rax
	movq 8(%rsp), %rcx
	leaq (%rax,%rcx), %r14
	cmpq %r15, %r14
	movq 336(%rsp), %r15
	jb .LBB25_51
	cmpq 24(%rsp), %r14
	ja .LBB25_51
	movq 136(%rsp), %rax
	leaq (%rax,%rbp,8), %rsi
	addq 16(%rsp), %r12
	movq %r12, %rdi
	movq %rbx, %rdx
	callq *memcpy@GOTPCREL(%rip)
	movq 40(%rsp), %rax
	incq %rax
	movq %r14, 8(%rsp)
	movq %r13, %rbp
	decq %r15
	movq %r15, %r14
	jne .LBB25_2
.LBB25_35:
	addq $344, %rsp
	popq %rbx
	popq %r12
	popq %r13
	popq %r14
	popq %r15
	popq %rbp
	retq
.LBB25_53:
	movq %rbp, %rdi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.16(%rip), %rcx
	movq 32(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_55:
	movq 8(%rsp), %r15
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.15(%rip), %rcx
	movq %r15, %rdi
	movq 24(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_36:
	movq 56(%rsp), %rsi
	addq 40(%rsp), %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.16(%rip), %rcx
	movq 32(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_37:
	movq 48(%rsp), %rsi
	incq %rsi
	jmp .LBB25_47
.LBB25_38:
	movq 128(%rsp), %rsi
	addq 40(%rsp), %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq 16(%rsp), %rax
	leaq (,%rax,2), %rdi
	addq %rbp, %rdi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.16(%rip), %rcx
	movq 32(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_39:
	movq 48(%rsp), %rsi
	orq $2, %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq 8(%rsp), %rax
	leaq (%rax,%rbx,2), %r15
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.15(%rip), %rcx
	movq %r15, %rdi
	movq 24(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_40:
	movq 120(%rsp), %rsi
	jmp .LBB25_45
.LBB25_41:
	movq 48(%rsp), %rsi
	orq $3, %rsi
	jmp .LBB25_47
.LBB25_42:
	movq 112(%rsp), %rsi
	addq 40(%rsp), %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq 16(%rsp), %rax
	leaq (,%rax,4), %rdi
	addq %rbp, %rdi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.16(%rip), %rcx
	movq 32(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_43:
	movq 48(%rsp), %rsi
	orq $4, %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq 8(%rsp), %rax
	leaq (%rax,%rbx,4), %r15
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.15(%rip), %rcx
	movq %r15, %rdi
	movq 24(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_44:
	movq 104(%rsp), %rsi
.LBB25_45:
	addq 40(%rsp), %rsi
	imulq %r14, %rsi
	addq %r14, %rsi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.16(%rip), %rcx
	movq 32(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_46:
	movq 48(%rsp), %rsi
	orq $5, %rsi
.LBB25_47:
	imulq %rbx, %rsi
	addq %rbx, %rsi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.15(%rip), %rcx
	movq %r15, %rdi
	movq 24(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_48:
	movq 96(%rsp), %rsi
	addq 40(%rsp), %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq 72(%rsp), %rax
	leaq (,%rax,2), %rdi
	addq %rbp, %rdi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.16(%rip), %rcx
	movq 32(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_49:
	movq 48(%rsp), %rsi
	orq $6, %rsi
	imulq %rbx, %rsi
	addq %rbx, %rsi
	movq 8(%rsp), %rax
	movq 80(%rsp), %rcx
	leaq (%rax,%rcx,2), %r15
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.15(%rip), %rcx
	movq %r15, %rdi
	movq 24(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_50:
	movq 16(%rsp), %rax
	leaq (,%rax,8), %rdi
	addq %rbp, %rdi
	subq %rax, %rdi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.16(%rip), %rcx
	movq 32(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
.LBB25_51:
	movq 8(%rsp), %rax
	leaq (%rax,%rbx,8), %r15
	subq %rbx, %r15
	movq 64(%rsp), %rsi
	addq %rax, %rsi
	leaq .Lanon.927cf18b8d3ae4b4a6a609b888a1827a.15(%rip), %rcx
	movq %r15, %rdi
	movq 24(%rsp), %rdx
	callq *core::slice::index::slice_index_fail@GOTPCREL(%rip)
