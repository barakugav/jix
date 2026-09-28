jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>:
	push r14
	push rbx
	mov rcx, r8
	and rcx, -32
	je .LBB146_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor eax, eax
.LBB146_11:
	movups xmm0, xmmword ptr [rdx + 4*rax]
	movups xmm1, xmmword ptr [r9 + 4*rax]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax], xmm1
	movups xmm0, xmmword ptr [rdx + 4*rax + 16]
	movups xmm1, xmmword ptr [r9 + 4*rax + 16]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 16], xmm1
	movups xmm0, xmmword ptr [rdx + 4*rax + 32]
	movups xmm1, xmmword ptr [r9 + 4*rax + 32]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 32], xmm1
	movups xmm0, xmmword ptr [rdx + 4*rax + 48]
	movups xmm1, xmmword ptr [r9 + 4*rax + 48]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 48], xmm1
	movups xmm0, xmmword ptr [rdx + 4*rax + 64]
	movups xmm1, xmmword ptr [r9 + 4*rax + 64]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 64], xmm1
	movups xmm0, xmmword ptr [rdx + 4*rax + 80]
	movups xmm1, xmmword ptr [r9 + 4*rax + 80]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 80], xmm1
	movups xmm0, xmmword ptr [rdx + 4*rax + 96]
	movups xmm1, xmmword ptr [r9 + 4*rax + 96]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 96], xmm1
	movups xmm0, xmmword ptr [rdx + 4*rax + 112]
	movups xmm1, xmmword ptr [r9 + 4*rax + 112]
	addps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 112], xmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB146_11
	mov r9, r8
	sub r9, rax
	ja .LBB146_3
	jmp .LBB146_9
.LBB146_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB146_9
.LBB146_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jae .LBB146_6
	mov rdi, rax
	jmp .LBB146_5
.LBB146_6:
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
.LBB146_7:
	movups xmm0, xmmword ptr [rax + 4*r14 - 16]
	movups xmm1, xmmword ptr [rax + 4*r14]
	movups xmm2, xmmword ptr [rbx + 4*r14 - 16]
	addps xmm2, xmm0
	movups xmm0, xmmword ptr [rbx + 4*r14]
	addps xmm0, xmm1
	movups xmmword ptr [r11 + 4*r14 - 16], xmm2
	movups xmmword ptr [r11 + 4*r14], xmm0
	add r14, 8
	cmp r10, r14
	jne .LBB146_7
	cmp r9, r10
	je .LBB146_9
.LBB146_5:
	movss xmm0, dword ptr [rcx + 4*rdi]
	addss xmm0, dword ptr [rdx + 4*rdi]
	movss dword ptr [rsi + 4*rdi], xmm0
	inc rdi
	cmp r8, rdi
	jne .LBB146_5
.LBB146_9:
	pop rbx
	pop r14
	ret
