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
.LBB144_11:
	movups xmm0, xmmword ptr [rdx + 4*rax]
	movups xmm1, xmmword ptr [r9 + 4*rax]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax], xmm0
	movups xmm0, xmmword ptr [rdx + 4*rax + 16]
	movups xmm1, xmmword ptr [r9 + 4*rax + 16]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax + 16]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax + 16]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax + 16], xmm0
	movups xmm0, xmmword ptr [rdx + 4*rax + 32]
	movups xmm1, xmmword ptr [r9 + 4*rax + 32]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax + 32]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax + 32]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax + 32], xmm0
	movups xmm0, xmmword ptr [rdx + 4*rax + 48]
	movups xmm1, xmmword ptr [r9 + 4*rax + 48]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax + 48]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax + 48]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax + 48], xmm0
	movups xmm0, xmmword ptr [rdx + 4*rax + 64]
	movups xmm1, xmmword ptr [r9 + 4*rax + 64]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax + 64]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax + 64]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax + 64], xmm0
	movups xmm0, xmmword ptr [rdx + 4*rax + 80]
	movups xmm1, xmmword ptr [r9 + 4*rax + 80]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax + 80]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax + 80]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax + 80], xmm0
	movups xmm0, xmmword ptr [rdx + 4*rax + 96]
	movups xmm1, xmmword ptr [r9 + 4*rax + 96]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax + 96]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax + 96]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax + 96], xmm0
	movups xmm0, xmmword ptr [rdx + 4*rax + 112]
	movups xmm1, xmmword ptr [r9 + 4*rax + 112]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [r10 + 4*rax + 112]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [r11 + 4*rax + 112]
	subps xmm0, xmm1
	movups xmmword ptr [rsi + 4*rax + 112], xmm0
	add rax, 32
	cmp rax, rcx
	jb .LBB144_11
	mov r11, r8
	sub r11, rax
	ja .LBB144_3
	jmp .LBB144_9
.LBB144_1:
	xor eax, eax
	mov r11, r8
	sub r11, rax
	jbe .LBB144_9
.LBB144_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r11, 8
	jae .LBB144_6
	mov r10, rax
	jmp .LBB144_5
.LBB144_6:
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
.LBB144_7:
	movups xmm0, xmmword ptr [rax + 4*rbp - 16]
	movups xmm1, xmmword ptr [rax + 4*rbp]
	movups xmm2, xmmword ptr [r13 + 4*rbp - 16]
	addps xmm2, xmm0
	movups xmm0, xmmword ptr [r13 + 4*rbp]
	addps xmm0, xmm1
	movups xmm1, xmmword ptr [r12 + 4*rbp - 16]
	mulps xmm1, xmm2
	movups xmm2, xmmword ptr [r12 + 4*rbp]
	mulps xmm2, xmm0
	movups xmm0, xmmword ptr [r15 + 4*rbp - 16]
	subps xmm1, xmm0
	movups xmm0, xmmword ptr [r15 + 4*rbp]
	subps xmm2, xmm0
	movups xmmword ptr [r14 + 4*rbp - 16], xmm1
	movups xmmword ptr [r14 + 4*rbp], xmm2
	add rbp, 8
	cmp rbx, rbp
	jne .LBB144_7
	cmp r11, rbx
	je .LBB144_9
.LBB144_5:
	movss xmm0, dword ptr [rcx + 4*r10]
	addss xmm0, dword ptr [rdx + 4*r10]
	mulss xmm0, dword ptr [r9 + 4*r10]
	subss xmm0, dword ptr [rdi + 4*r10]
	movss dword ptr [rsi + 4*r10], xmm0
	inc r10
	cmp r8, r10
	jne .LBB144_5
.LBB144_9:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	ret
