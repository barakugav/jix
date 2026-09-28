jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	mov rcx, r8
	and rcx, -32
	je .LBB160_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor eax, eax
.LBB160_17:
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax]
	vpmulld ymm0, ymm0, ymmword ptr [r10 + 4*rax]
	vpsubd ymm0, ymm0, ymmword ptr [r11 + 4*rax]
	vmovdqu ymmword ptr [rsi + 4*rax], ymm0
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax + 32]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax + 32]
	vpmulld ymm0, ymm0, ymmword ptr [r10 + 4*rax + 32]
	vpsubd ymm0, ymm0, ymmword ptr [r11 + 4*rax + 32]
	vmovdqu ymmword ptr [rsi + 4*rax + 32], ymm0
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax + 64]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax + 64]
	vpmulld ymm0, ymm0, ymmword ptr [r10 + 4*rax + 64]
	vpsubd ymm0, ymm0, ymmword ptr [r11 + 4*rax + 64]
	vmovdqu ymmword ptr [rsi + 4*rax + 64], ymm0
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax + 96]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax + 96]
	vpmulld ymm0, ymm0, ymmword ptr [r10 + 4*rax + 96]
	vpsubd ymm0, ymm0, ymmword ptr [r11 + 4*rax + 96]
	vmovdqu ymmword ptr [rsi + 4*rax + 96], ymm0
	add rax, 32
	cmp rax, rcx
	jb .LBB160_17
	mov rbp, r8
	sub rbp, rax
	ja .LBB160_3
	jmp .LBB160_15
.LBB160_1:
	xor eax, eax
	mov rbp, r8
	sub rbp, rax
	jbe .LBB160_15
.LBB160_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp rbp, 8
	jae .LBB160_6
	mov r11, rax
	jmp .LBB160_5
.LBB160_6:
	cmp rbp, 32
	jae .LBB160_8
	xor ebx, ebx
	jmp .LBB160_12
.LBB160_8:
	mov qword ptr [rsp - 8], rbp
	mov rbx, rbp
	and rbx, -32
	lea r11, [rax + rbx]
	lea r14, [rsi + 4*rax]
	add r14, 96
	lea r15, [rdi + 4*rax + 96]
	lea r12, [r9 + 4*rax]
	add r12, 96
	lea r13, [rdx + 4*rax]
	add r13, 96
	lea rbp, [rcx + 4*rax]
	add rbp, 96
	xor r10d, r10d
.LBB160_9:
	vmovdqu ymm0, ymmword ptr [r13 + 4*r10 - 96]
	vmovdqu ymm1, ymmword ptr [r13 + 4*r10 - 64]
	vmovdqu ymm2, ymmword ptr [r13 + 4*r10 - 32]
	vmovdqu ymm3, ymmword ptr [r13 + 4*r10]
	vpaddd ymm0, ymm0, ymmword ptr [rbp + 4*r10 - 96]
	vpaddd ymm1, ymm1, ymmword ptr [rbp + 4*r10 - 64]
	vpaddd ymm2, ymm2, ymmword ptr [rbp + 4*r10 - 32]
	vpaddd ymm3, ymm3, ymmword ptr [rbp + 4*r10]
	vpmulld ymm0, ymm0, ymmword ptr [r12 + 4*r10 - 96]
	vpmulld ymm1, ymm1, ymmword ptr [r12 + 4*r10 - 64]
	vpmulld ymm2, ymm2, ymmword ptr [r12 + 4*r10 - 32]
	vpmulld ymm3, ymm3, ymmword ptr [r12 + 4*r10]
	vpsubd ymm0, ymm0, ymmword ptr [r15 + 4*r10 - 96]
	vpsubd ymm1, ymm1, ymmword ptr [r15 + 4*r10 - 64]
	vpsubd ymm2, ymm2, ymmword ptr [r15 + 4*r10 - 32]
	vpsubd ymm3, ymm3, ymmword ptr [r15 + 4*r10]
	vmovdqu ymmword ptr [r14 + 4*r10 - 96], ymm0
	vmovdqu ymmword ptr [r14 + 4*r10 - 64], ymm1
	vmovdqu ymmword ptr [r14 + 4*r10 - 32], ymm2
	vmovdqu ymmword ptr [r14 + 4*r10], ymm3
	add r10, 32
	cmp rbx, r10
	jne .LBB160_9
	mov rbp, qword ptr [rsp - 8]
	cmp rbp, rbx
	je .LBB160_15
	test bpl, 24
	je .LBB160_5
.LBB160_12:
	mov r10, rbp
	and r10, -8
	lea r11, [rax + r10]
	lea r14, [rsi + 4*rax]
	lea r15, [rdi + 4*rax]
	lea r12, [r9 + 4*rax]
	lea r13, [rdx + 4*rax]
	lea rax, [rcx + 4*rax]
.LBB160_13:
	vmovdqu ymm0, ymmword ptr [r13 + 4*rbx]
	vpaddd ymm0, ymm0, ymmword ptr [rax + 4*rbx]
	vpmulld ymm0, ymm0, ymmword ptr [r12 + 4*rbx]
	vpsubd ymm0, ymm0, ymmword ptr [r15 + 4*rbx]
	vmovdqu ymmword ptr [r14 + 4*rbx], ymm0
	add rbx, 8
	cmp r10, rbx
	jne .LBB160_13
	cmp rbp, r10
	je .LBB160_15
.LBB160_5:
	mov eax, dword ptr [rdx + 4*r11]
	add eax, dword ptr [rcx + 4*r11]
	imul eax, dword ptr [r9 + 4*r11]
	sub eax, dword ptr [rdi + 4*r11]
	mov dword ptr [rsi + 4*r11], eax
	inc r11
	cmp r8, r11
	jne .LBB160_5
.LBB160_15:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
