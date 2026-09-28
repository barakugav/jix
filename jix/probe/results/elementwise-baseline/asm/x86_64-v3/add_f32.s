jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>:
	push r15
	push r14
	push rbx
	mov rcx, r8
	and rcx, -32
	je .LBB146_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor eax, eax
.LBB146_17:
	vmovups ymm0, ymmword ptr [rdx + 4*rax]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax]
	vmovups ymmword ptr [rsi + 4*rax], ymm0
	vmovups ymm0, ymmword ptr [rdx + 4*rax + 32]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax + 32]
	vmovups ymmword ptr [rsi + 4*rax + 32], ymm0
	vmovups ymm0, ymmword ptr [rdx + 4*rax + 64]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax + 64]
	vmovups ymmword ptr [rsi + 4*rax + 64], ymm0
	vmovups ymm0, ymmword ptr [rdx + 4*rax + 96]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax + 96]
	vmovups ymmword ptr [rsi + 4*rax + 96], ymm0
	add rax, 32
	cmp rax, rcx
	jb .LBB146_17
	mov r9, r8
	sub r9, rax
	ja .LBB146_3
	jmp .LBB146_15
.LBB146_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB146_15
.LBB146_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jae .LBB146_6
	mov rdi, rax
	jmp .LBB146_5
.LBB146_6:
	cmp r9, 32
	jae .LBB146_8
	xor r10d, r10d
	jmp .LBB146_12
.LBB146_8:
	mov r10, r9
	and r10, -32
	lea rdi, [rax + r10]
	lea r11, [rsi + 4*rax]
	add r11, 96
	lea rbx, [rdx + 4*rax]
	add rbx, 96
	lea r14, [rcx + 4*rax]
	add r14, 96
	xor r15d, r15d
.LBB146_9:
	vmovups ymm0, ymmword ptr [r14 + 4*r15 - 96]
	vmovups ymm1, ymmword ptr [r14 + 4*r15 - 64]
	vmovups ymm2, ymmword ptr [r14 + 4*r15 - 32]
	vmovups ymm3, ymmword ptr [r14 + 4*r15]
	vaddps ymm0, ymm0, ymmword ptr [rbx + 4*r15 - 96]
	vaddps ymm1, ymm1, ymmword ptr [rbx + 4*r15 - 64]
	vaddps ymm2, ymm2, ymmword ptr [rbx + 4*r15 - 32]
	vaddps ymm3, ymm3, ymmword ptr [rbx + 4*r15]
	vmovups ymmword ptr [r11 + 4*r15 - 96], ymm0
	vmovups ymmword ptr [r11 + 4*r15 - 64], ymm1
	vmovups ymmword ptr [r11 + 4*r15 - 32], ymm2
	vmovups ymmword ptr [r11 + 4*r15], ymm3
	add r15, 32
	cmp r10, r15
	jne .LBB146_9
	cmp r9, r10
	je .LBB146_15
	test r9b, 24
	je .LBB146_5
.LBB146_12:
	mov r11, r9
	and r11, -8
	lea rdi, [rax + r11]
	lea rbx, [rsi + 4*rax]
	lea r14, [rdx + 4*rax]
	lea rax, [rcx + 4*rax]
.LBB146_13:
	vmovups ymm0, ymmword ptr [rax + 4*r10]
	vaddps ymm0, ymm0, ymmword ptr [r14 + 4*r10]
	vmovups ymmword ptr [rbx + 4*r10], ymm0
	add r10, 8
	cmp r11, r10
	jne .LBB146_13
	cmp r9, r11
	je .LBB146_15
.LBB146_5:
	vmovss xmm0, dword ptr [rcx + 4*rdi]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*rdi]
	vmovss dword ptr [rsi + 4*rdi], xmm0
	inc rdi
	cmp r8, rdi
	jne .LBB146_5
.LBB146_15:
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
