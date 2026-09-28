jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
	push r14
	push rbx
	mov rcx, r8
	and rcx, -32
	je .LBB162_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor eax, eax
.LBB162_11:
	movdqu xmm0, xmmword ptr [rdx + 4*rax]
	movdqu xmm1, xmmword ptr [r9 + 4*rax]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 16]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 16]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 16], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 32]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 32]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 32], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 48]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 48]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 48], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 64]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 64]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 64], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 80]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 80]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 80], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 96]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 96]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 96], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 112]
	movdqu xmm1, xmmword ptr [r9 + 4*rax + 112]
	paddd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 112], xmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB162_11
	mov r9, r8
	sub r9, rax
	ja .LBB162_3
	jmp .LBB162_9
.LBB162_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB162_9
.LBB162_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jae .LBB162_6
	mov rdi, rax
	jmp .LBB162_5
.LBB162_6:
	mov r10, r9
	and r10, -8
	lea rdi, [rax + r10]
	lea r11, [rsi + 4*rax]
	add r11, 16
	lea rbx, [rdx + 4*rax]
	add rbx, 16
	lea rax, [rcx + 4*rax]
	add rax, 16
	xor r14d, r14d
.LBB162_7:
	movdqu xmm0, xmmword ptr [rax + 4*r14 - 16]
	movdqu xmm1, xmmword ptr [rax + 4*r14]
	movdqu xmm2, xmmword ptr [rbx + 4*r14 - 16]
	paddd xmm2, xmm0
	movdqu xmm0, xmmword ptr [rbx + 4*r14]
	paddd xmm0, xmm1
	movdqu xmmword ptr [r11 + 4*r14 - 16], xmm2
	movdqu xmmword ptr [r11 + 4*r14], xmm0
	add r14, 8
	cmp r10, r14
	jne .LBB162_7
	cmp r9, r10
	je .LBB162_9
.LBB162_5:
	mov eax, dword ptr [rdx + 4*rdi]
	add eax, dword ptr [rcx + 4*rdi]
	mov dword ptr [rsi + 4*rdi], eax
	inc rdi
	cmp r8, rdi
	jne .LBB162_5
.LBB162_9:
	pop rbx
	pop r14
	ret
