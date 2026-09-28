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
.LBB160_11:
	movdqu xmm0, xmmword ptr [rdx + 4*rax]
	movdqu xmm1, xmmword ptr [r9 + 4*rax]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax], xmm0
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 16]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 16]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax + 16]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax + 16]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax + 16], xmm0
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 32]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 32]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax + 32]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax + 32]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax + 32], xmm0
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 48]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 48]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax + 48]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax + 48]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax + 48], xmm0
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 64]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 64]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax + 64]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax + 64]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax + 64], xmm0
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 80]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 80]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax + 80]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax + 80]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax + 80], xmm0
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 96]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 96]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax + 96]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax + 96]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax + 96], xmm0
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 112]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 112]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r10 + 4*rax + 112]
	pmulld xmm0, xmm1
	movdqu xmm1, xmmword ptr [r11 + 4*rax + 112]
	psubd xmm0, xmm1
	movdqu xmmword ptr [rsi + 4*rax + 112], xmm0
	add rax, 32
	cmp rax, rcx
	jb .LBB160_11
	mov r11, r8
	sub r11, rax
	ja .LBB160_3
	jmp .LBB160_9
.LBB160_1:
	xor eax, eax
	mov r11, r8
	sub r11, rax
	jbe .LBB160_9
.LBB160_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r11, 8
	jae .LBB160_6
	mov r10, rax
	jmp .LBB160_5
.LBB160_6:
	mov rbx, r11
	and rbx, -8
	lea r10, [rax + rbx]
	lea r14, [rsi + 4*rax]
	add r14, 16
	lea r15, [rdi + 4*rax + 16]
	lea r12, [r9 + 4*rax]
	add r12, 16
	lea r13, [rdx + 4*rax]
	add r13, 16
	lea rax, [rcx + 4*rax]
	add rax, 16
	xor ebp, ebp
.LBB160_7:
	movdqu xmm0, xmmword ptr [rax + 4*rbp - 16]
	movdqu xmm1, xmmword ptr [rax + 4*rbp]
	movdqu xmm2, xmmword ptr [r13 + 4*rbp - 16]
	paddd xmm2, xmm0
	movdqu xmm0, xmmword ptr [r13 + 4*rbp]
	paddd xmm0, xmm1
	movdqu xmm1, xmmword ptr [r12 + 4*rbp - 16]
	pmulld xmm1, xmm2
	movdqu xmm2, xmmword ptr [r12 + 4*rbp]
	pmulld xmm2, xmm0
	movdqu xmm0, xmmword ptr [r15 + 4*rbp - 16]
	psubd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r15 + 4*rbp]
	psubd xmm2, xmm0
	movdqu xmmword ptr [r14 + 4*rbp - 16], xmm1
	movdqu xmmword ptr [r14 + 4*rbp], xmm2
	add rbp, 8
	cmp rbx, rbp
	jne .LBB160_7
	cmp r11, rbx
	je .LBB160_9
.LBB160_5:
	mov eax, dword ptr [rdx + 4*r10]
	add eax, dword ptr [rcx + 4*r10]
	imul eax, dword ptr [r9 + 4*r10]
	sub eax, dword ptr [rdi + 4*r10]
	mov dword ptr [rsi + 4*r10], eax
	inc r10
	cmp r8, r10
	jne .LBB160_5
.LBB160_9:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	ret
