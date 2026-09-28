jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	push r14
	push rbx
	mov rcx, r8
	and rcx, -16
	je .LBB130_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor eax, eax
.LBB130_11:
	movupd xmm0, xmmword ptr [rdx + 8*rax]
	movupd xmm1, xmmword ptr [r9 + 8*rax]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax], xmm1
	movupd xmm0, xmmword ptr [rdx + 8*rax + 16]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 16]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax + 16], xmm1
	movupd xmm0, xmmword ptr [rdx + 8*rax + 32]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 32]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax + 32], xmm1
	movupd xmm0, xmmword ptr [rdx + 8*rax + 48]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 48]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax + 48], xmm1
	movupd xmm0, xmmword ptr [rdx + 8*rax + 64]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 64]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax + 64], xmm1
	movupd xmm0, xmmword ptr [rdx + 8*rax + 80]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 80]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax + 80], xmm1
	movupd xmm0, xmmword ptr [rdx + 8*rax + 96]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 96]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax + 96], xmm1
	movupd xmm0, xmmword ptr [rdx + 8*rax + 112]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 112]
	addpd xmm1, xmm0
	movupd xmmword ptr [rsi + 8*rax + 112], xmm1
	add rax, 16
	cmp rax, rcx
	jb .LBB130_11
	mov r9, r8
	sub r9, rax
	ja .LBB130_3
	jmp .LBB130_9
.LBB130_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB130_9
.LBB130_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 4
	jae .LBB130_6
	mov rdi, rax
	jmp .LBB130_5
.LBB130_6:
	mov r10, r9
	and r10, -4
	lea rdi, [rax + r10]
	lea r11, [rsi + 8*rax]
	add r11, 16
	lea rbx, [rdx + 8*rax]
	add rbx, 16
	lea rax, [rcx + 8*rax]
	add rax, 16
	xor r14d, r14d
.LBB130_7:
	movupd xmm0, xmmword ptr [rax + 8*r14 - 16]
	movupd xmm1, xmmword ptr [rax + 8*r14]
	movupd xmm2, xmmword ptr [rbx + 8*r14 - 16]
	addpd xmm2, xmm0
	movupd xmm0, xmmword ptr [rbx + 8*r14]
	addpd xmm0, xmm1
	movupd xmmword ptr [r11 + 8*r14 - 16], xmm2
	movupd xmmword ptr [r11 + 8*r14], xmm0
	add r14, 4
	cmp r10, r14
	jne .LBB130_7
	cmp r9, r10
	je .LBB130_9
.LBB130_5:
	movsd xmm0, qword ptr [rcx + 8*rdi]
	addsd xmm0, qword ptr [rdx + 8*rdi]
	movsd qword ptr [rsi + 8*rdi], xmm0
	inc rdi
	cmp r8, rdi
	jne .LBB130_5
.LBB130_9:
	pop rbx
	pop r14
	ret
