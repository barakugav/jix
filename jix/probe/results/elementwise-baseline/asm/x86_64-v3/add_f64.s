jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	push r15
	push r14
	push rbx
	mov rcx, r8
	and rcx, -16
	je .LBB130_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor eax, eax
.LBB130_17:
	vmovupd ymm0, ymmword ptr [rdx + 8*rax]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax]
	vmovupd ymmword ptr [rsi + 8*rax], ymm0
	vmovupd ymm0, ymmword ptr [rdx + 8*rax + 32]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax + 32]
	vmovupd ymmword ptr [rsi + 8*rax + 32], ymm0
	vmovupd ymm0, ymmword ptr [rdx + 8*rax + 64]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax + 64]
	vmovupd ymmword ptr [rsi + 8*rax + 64], ymm0
	vmovupd ymm0, ymmword ptr [rdx + 8*rax + 96]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax + 96]
	vmovupd ymmword ptr [rsi + 8*rax + 96], ymm0
	add rax, 16
	cmp rax, rcx
	jb .LBB130_17
	mov r9, r8
	sub r9, rax
	ja .LBB130_3
	jmp .LBB130_15
.LBB130_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB130_15
.LBB130_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 4
	jae .LBB130_6
	mov rdi, rax
	jmp .LBB130_5
.LBB130_6:
	cmp r9, 16
	jae .LBB130_8
	xor r10d, r10d
	jmp .LBB130_12
.LBB130_8:
	mov r10, r9
	and r10, -16
	lea rdi, [rax + r10]
	lea r11, [rsi + 8*rax]
	add r11, 96
	lea rbx, [rdx + 8*rax]
	add rbx, 96
	lea r14, [rcx + 8*rax]
	add r14, 96
	xor r15d, r15d
.LBB130_9:
	vmovupd ymm0, ymmword ptr [r14 + 8*r15 - 96]
	vmovupd ymm1, ymmword ptr [r14 + 8*r15 - 64]
	vmovupd ymm2, ymmword ptr [r14 + 8*r15 - 32]
	vmovupd ymm3, ymmword ptr [r14 + 8*r15]
	vaddpd ymm0, ymm0, ymmword ptr [rbx + 8*r15 - 96]
	vaddpd ymm1, ymm1, ymmword ptr [rbx + 8*r15 - 64]
	vaddpd ymm2, ymm2, ymmword ptr [rbx + 8*r15 - 32]
	vaddpd ymm3, ymm3, ymmword ptr [rbx + 8*r15]
	vmovupd ymmword ptr [r11 + 8*r15 - 96], ymm0
	vmovupd ymmword ptr [r11 + 8*r15 - 64], ymm1
	vmovupd ymmword ptr [r11 + 8*r15 - 32], ymm2
	vmovupd ymmword ptr [r11 + 8*r15], ymm3
	add r15, 16
	cmp r10, r15
	jne .LBB130_9
	cmp r9, r10
	je .LBB130_15
	test r9b, 12
	je .LBB130_5
.LBB130_12:
	mov r11, r9
	and r11, -4
	lea rdi, [rax + r11]
	lea rbx, [rsi + 8*rax]
	lea r14, [rdx + 8*rax]
	lea rax, [rcx + 8*rax]
.LBB130_13:
	vmovupd ymm0, ymmword ptr [rax + 8*r10]
	vaddpd ymm0, ymm0, ymmword ptr [r14 + 8*r10]
	vmovupd ymmword ptr [rbx + 8*r10], ymm0
	add r10, 4
	cmp r11, r10
	jne .LBB130_13
	cmp r9, r11
	je .LBB130_15
.LBB130_5:
	vmovsd xmm0, qword ptr [rcx + 8*rdi]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*rdi]
	vmovsd qword ptr [rsi + 8*rdi], xmm0
	inc rdi
	cmp r8, rdi
	jne .LBB130_5
.LBB130_15:
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
