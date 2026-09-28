jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	mov rcx, r8
	and rcx, -16
	je .LBB128_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor eax, eax
.LBB128_17:
	vmovupd ymm0, ymmword ptr [rdx + 8*rax]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax]
	vmulpd ymm0, ymm0, ymmword ptr [r10 + 8*rax]
	vsubpd ymm0, ymm0, ymmword ptr [r11 + 8*rax]
	vmovupd ymmword ptr [rsi + 8*rax], ymm0
	vmovupd ymm0, ymmword ptr [rdx + 8*rax + 32]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax + 32]
	vmulpd ymm0, ymm0, ymmword ptr [r10 + 8*rax + 32]
	vsubpd ymm0, ymm0, ymmword ptr [r11 + 8*rax + 32]
	vmovupd ymmword ptr [rsi + 8*rax + 32], ymm0
	vmovupd ymm0, ymmword ptr [rdx + 8*rax + 64]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax + 64]
	vmulpd ymm0, ymm0, ymmword ptr [r10 + 8*rax + 64]
	vsubpd ymm0, ymm0, ymmword ptr [r11 + 8*rax + 64]
	vmovupd ymmword ptr [rsi + 8*rax + 64], ymm0
	vmovupd ymm0, ymmword ptr [rdx + 8*rax + 96]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*rax + 96]
	vmulpd ymm0, ymm0, ymmword ptr [r10 + 8*rax + 96]
	vsubpd ymm0, ymm0, ymmword ptr [r11 + 8*rax + 96]
	vmovupd ymmword ptr [rsi + 8*rax + 96], ymm0
	add rax, 16
	cmp rax, rcx
	jb .LBB128_17
	mov rbp, r8
	sub rbp, rax
	ja .LBB128_3
	jmp .LBB128_15
.LBB128_1:
	xor eax, eax
	mov rbp, r8
	sub rbp, rax
	jbe .LBB128_15
.LBB128_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp rbp, 4
	jae .LBB128_6
	mov r11, rax
	jmp .LBB128_5
.LBB128_6:
	cmp rbp, 16
	jae .LBB128_8
	xor ebx, ebx
	jmp .LBB128_12
.LBB128_8:
	mov qword ptr [rsp - 8], rbp
	mov rbx, rbp
	and rbx, -16
	lea r11, [rax + rbx]
	lea r14, [rsi + 8*rax]
	add r14, 96
	lea r15, [rdi + 8*rax + 96]
	lea r12, [r9 + 8*rax]
	add r12, 96
	lea r13, [rdx + 8*rax]
	add r13, 96
	lea rbp, [rcx + 8*rax]
	add rbp, 96
	xor r10d, r10d
.LBB128_9:
	vmovupd ymm0, ymmword ptr [rbp + 8*r10 - 96]
	vmovupd ymm1, ymmword ptr [rbp + 8*r10 - 64]
	vmovupd ymm2, ymmword ptr [rbp + 8*r10 - 32]
	vmovupd ymm3, ymmword ptr [rbp + 8*r10]
	vaddpd ymm0, ymm0, ymmword ptr [r13 + 8*r10 - 96]
	vaddpd ymm1, ymm1, ymmword ptr [r13 + 8*r10 - 64]
	vaddpd ymm2, ymm2, ymmword ptr [r13 + 8*r10 - 32]
	vaddpd ymm3, ymm3, ymmword ptr [r13 + 8*r10]
	vmulpd ymm0, ymm0, ymmword ptr [r12 + 8*r10 - 96]
	vmulpd ymm1, ymm1, ymmword ptr [r12 + 8*r10 - 64]
	vmulpd ymm2, ymm2, ymmword ptr [r12 + 8*r10 - 32]
	vmulpd ymm3, ymm3, ymmword ptr [r12 + 8*r10]
	vsubpd ymm0, ymm0, ymmword ptr [r15 + 8*r10 - 96]
	vsubpd ymm1, ymm1, ymmword ptr [r15 + 8*r10 - 64]
	vsubpd ymm2, ymm2, ymmword ptr [r15 + 8*r10 - 32]
	vsubpd ymm3, ymm3, ymmword ptr [r15 + 8*r10]
	vmovupd ymmword ptr [r14 + 8*r10 - 96], ymm0
	vmovupd ymmword ptr [r14 + 8*r10 - 64], ymm1
	vmovupd ymmword ptr [r14 + 8*r10 - 32], ymm2
	vmovupd ymmword ptr [r14 + 8*r10], ymm3
	add r10, 16
	cmp rbx, r10
	jne .LBB128_9
	mov rbp, qword ptr [rsp - 8]
	cmp rbp, rbx
	je .LBB128_15
	test bpl, 12
	je .LBB128_5
.LBB128_12:
	mov r10, rbp
	and r10, -4
	lea r11, [rax + r10]
	lea r14, [rsi + 8*rax]
	lea r15, [rdi + 8*rax]
	lea r12, [r9 + 8*rax]
	lea r13, [rdx + 8*rax]
	lea rax, [rcx + 8*rax]
.LBB128_13:
	vmovupd ymm0, ymmword ptr [rax + 8*rbx]
	vaddpd ymm0, ymm0, ymmword ptr [r13 + 8*rbx]
	vmulpd ymm0, ymm0, ymmword ptr [r12 + 8*rbx]
	vsubpd ymm0, ymm0, ymmword ptr [r15 + 8*rbx]
	vmovupd ymmword ptr [r14 + 8*rbx], ymm0
	add rbx, 4
	cmp r10, rbx
	jne .LBB128_13
	cmp rbp, r10
	je .LBB128_15
.LBB128_5:
	vmovsd xmm0, qword ptr [rcx + 8*r11]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r11]
	vmulsd xmm0, xmm0, qword ptr [r9 + 8*r11]
	vsubsd xmm0, xmm0, qword ptr [rdi + 8*r11]
	vmovsd qword ptr [rsi + 8*r11], xmm0
	inc r11
	cmp r8, r11
	jne .LBB128_5
.LBB128_15:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
