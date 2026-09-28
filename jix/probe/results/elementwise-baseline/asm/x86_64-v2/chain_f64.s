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
.LBB128_11:
	movupd xmm0, xmmword ptr [rdx + 8*rax]
	movupd xmm1, xmmword ptr [r9 + 8*rax]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax], xmm0
	movupd xmm0, xmmword ptr [rdx + 8*rax + 16]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 16]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax + 16]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax + 16]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax + 16], xmm0
	movupd xmm0, xmmword ptr [rdx + 8*rax + 32]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 32]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax + 32]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax + 32]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax + 32], xmm0
	movupd xmm0, xmmword ptr [rdx + 8*rax + 48]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 48]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax + 48]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax + 48]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax + 48], xmm0
	movupd xmm0, xmmword ptr [rdx + 8*rax + 64]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 64]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax + 64]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax + 64]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax + 64], xmm0
	movupd xmm0, xmmword ptr [rdx + 8*rax + 80]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 80]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax + 80]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax + 80]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax + 80], xmm0
	movupd xmm0, xmmword ptr [rdx + 8*rax + 96]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 96]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax + 96]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax + 96]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax + 96], xmm0
	movupd xmm0, xmmword ptr [rdx + 8*rax + 112]
	movupd xmm1, xmmword ptr [r9 + 8*rax + 112]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r10 + 8*rax + 112]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r11 + 8*rax + 112]
	subpd xmm0, xmm1
	movupd xmmword ptr [rsi + 8*rax + 112], xmm0
	add rax, 16
	cmp rax, rcx
	jb .LBB128_11
	mov r11, r8
	sub r11, rax
	ja .LBB128_3
	jmp .LBB128_9
.LBB128_1:
	xor eax, eax
	mov r11, r8
	sub r11, rax
	jbe .LBB128_9
.LBB128_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r11, 4
	jae .LBB128_6
	mov r10, rax
	jmp .LBB128_5
.LBB128_6:
	mov rbx, r11
	and rbx, -4
	lea r10, [rax + rbx]
	lea r14, [rsi + 8*rax]
	add r14, 16
	lea r15, [rdi + 8*rax + 16]
	lea r12, [r9 + 8*rax]
	add r12, 16
	lea r13, [rdx + 8*rax]
	add r13, 16
	lea rax, [rcx + 8*rax]
	add rax, 16
	xor ebp, ebp
.LBB128_7:
	movupd xmm0, xmmword ptr [rax + 8*rbp - 16]
	movupd xmm1, xmmword ptr [rax + 8*rbp]
	movupd xmm2, xmmword ptr [r13 + 8*rbp - 16]
	addpd xmm2, xmm0
	movupd xmm0, xmmword ptr [r13 + 8*rbp]
	addpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r12 + 8*rbp - 16]
	mulpd xmm1, xmm2
	movupd xmm2, xmmword ptr [r12 + 8*rbp]
	mulpd xmm2, xmm0
	movupd xmm0, xmmword ptr [r15 + 8*rbp - 16]
	subpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r15 + 8*rbp]
	subpd xmm2, xmm0
	movupd xmmword ptr [r14 + 8*rbp - 16], xmm1
	movupd xmmword ptr [r14 + 8*rbp], xmm2
	add rbp, 4
	cmp rbx, rbp
	jne .LBB128_7
	cmp r11, rbx
	je .LBB128_9
.LBB128_5:
	movsd xmm0, qword ptr [rcx + 8*r10]
	addsd xmm0, qword ptr [rdx + 8*r10]
	mulsd xmm0, qword ptr [r9 + 8*r10]
	subsd xmm0, qword ptr [rdi + 8*r10]
	movsd qword ptr [rsi + 8*r10], xmm0
	inc r10
	cmp r8, r10
	jne .LBB128_5
.LBB128_9:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	ret
