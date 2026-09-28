jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	mov rcx, r8
	and rcx, -32
	je .LBB144_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor eax, eax
.LBB144_17:
	vmovups ymm0, ymmword ptr [rdx + 4*rax]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax]
	vmulps ymm0, ymm0, ymmword ptr [r10 + 4*rax]
	vsubps ymm0, ymm0, ymmword ptr [r11 + 4*rax]
	vmovups ymmword ptr [rsi + 4*rax], ymm0
	vmovups ymm0, ymmword ptr [rdx + 4*rax + 32]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax + 32]
	vmulps ymm0, ymm0, ymmword ptr [r10 + 4*rax + 32]
	vsubps ymm0, ymm0, ymmword ptr [r11 + 4*rax + 32]
	vmovups ymmword ptr [rsi + 4*rax + 32], ymm0
	vmovups ymm0, ymmword ptr [rdx + 4*rax + 64]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax + 64]
	vmulps ymm0, ymm0, ymmword ptr [r10 + 4*rax + 64]
	vsubps ymm0, ymm0, ymmword ptr [r11 + 4*rax + 64]
	vmovups ymmword ptr [rsi + 4*rax + 64], ymm0
	vmovups ymm0, ymmword ptr [rdx + 4*rax + 96]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*rax + 96]
	vmulps ymm0, ymm0, ymmword ptr [r10 + 4*rax + 96]
	vsubps ymm0, ymm0, ymmword ptr [r11 + 4*rax + 96]
	vmovups ymmword ptr [rsi + 4*rax + 96], ymm0
	add rax, 32
	cmp rax, rcx
	jb .LBB144_17
	mov rbp, r8
	sub rbp, rax
	ja .LBB144_3
	jmp .LBB144_15
.LBB144_1:
	xor eax, eax
	mov rbp, r8
	sub rbp, rax
	jbe .LBB144_15
.LBB144_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp rbp, 8
	jae .LBB144_6
	mov r11, rax
	jmp .LBB144_5
.LBB144_6:
	cmp rbp, 32
	jae .LBB144_8
	xor ebx, ebx
	jmp .LBB144_12
.LBB144_8:
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
.LBB144_9:
	vmovups ymm0, ymmword ptr [rbp + 4*r10 - 96]
	vmovups ymm1, ymmword ptr [rbp + 4*r10 - 64]
	vmovups ymm2, ymmword ptr [rbp + 4*r10 - 32]
	vmovups ymm3, ymmword ptr [rbp + 4*r10]
	vaddps ymm0, ymm0, ymmword ptr [r13 + 4*r10 - 96]
	vaddps ymm1, ymm1, ymmword ptr [r13 + 4*r10 - 64]
	vaddps ymm2, ymm2, ymmword ptr [r13 + 4*r10 - 32]
	vaddps ymm3, ymm3, ymmword ptr [r13 + 4*r10]
	vmulps ymm0, ymm0, ymmword ptr [r12 + 4*r10 - 96]
	vmulps ymm1, ymm1, ymmword ptr [r12 + 4*r10 - 64]
	vmulps ymm2, ymm2, ymmword ptr [r12 + 4*r10 - 32]
	vmulps ymm3, ymm3, ymmword ptr [r12 + 4*r10]
	vsubps ymm0, ymm0, ymmword ptr [r15 + 4*r10 - 96]
	vsubps ymm1, ymm1, ymmword ptr [r15 + 4*r10 - 64]
	vsubps ymm2, ymm2, ymmword ptr [r15 + 4*r10 - 32]
	vsubps ymm3, ymm3, ymmword ptr [r15 + 4*r10]
	vmovups ymmword ptr [r14 + 4*r10 - 96], ymm0
	vmovups ymmword ptr [r14 + 4*r10 - 64], ymm1
	vmovups ymmword ptr [r14 + 4*r10 - 32], ymm2
	vmovups ymmword ptr [r14 + 4*r10], ymm3
	add r10, 32
	cmp rbx, r10
	jne .LBB144_9
	mov rbp, qword ptr [rsp - 8]
	cmp rbp, rbx
	je .LBB144_15
	test bpl, 24
	je .LBB144_5
.LBB144_12:
	mov r10, rbp
	and r10, -8
	lea r11, [rax + r10]
	lea r14, [rsi + 4*rax]
	lea r15, [rdi + 4*rax]
	lea r12, [r9 + 4*rax]
	lea r13, [rdx + 4*rax]
	lea rax, [rcx + 4*rax]
.LBB144_13:
	vmovups ymm0, ymmword ptr [rax + 4*rbx]
	vaddps ymm0, ymm0, ymmword ptr [r13 + 4*rbx]
	vmulps ymm0, ymm0, ymmword ptr [r12 + 4*rbx]
	vsubps ymm0, ymm0, ymmword ptr [r15 + 4*rbx]
	vmovups ymmword ptr [r14 + 4*rbx], ymm0
	add rbx, 8
	cmp r10, rbx
	jne .LBB144_13
	cmp rbp, r10
	je .LBB144_15
.LBB144_5:
	vmovss xmm0, dword ptr [rcx + 4*r11]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r11]
	vmulss xmm0, xmm0, dword ptr [r9 + 4*r11]
	vsubss xmm0, xmm0, dword ptr [rdi + 4*r11]
	vmovss dword ptr [rsi + 4*r11], xmm0
	inc r11
	cmp r8, r11
	jne .LBB144_5
.LBB144_15:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
