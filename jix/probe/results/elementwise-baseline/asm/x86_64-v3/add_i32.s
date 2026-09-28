jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
	push r15
	push r14
	push rbx
	mov rcx, r8
	and rcx, -32
	je .LBB162_1
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor eax, eax
.LBB162_17:
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax]
	vmovdqu ymmword ptr [rsi + 4*rax], ymm0
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax + 32]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax + 32]
	vmovdqu ymmword ptr [rsi + 4*rax + 32], ymm0
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax + 64]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax + 64]
	vmovdqu ymmword ptr [rsi + 4*rax + 64], ymm0
	vmovdqu ymm0, ymmword ptr [r9 + 4*rax + 96]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*rax + 96]
	vmovdqu ymmword ptr [rsi + 4*rax + 96], ymm0
	add rax, 32
	cmp rax, rcx
	jb .LBB162_17
	mov r9, r8
	sub r9, rax
	ja .LBB162_3
	jmp .LBB162_15
.LBB162_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB162_15
.LBB162_3:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jae .LBB162_6
	mov rdi, rax
	jmp .LBB162_5
.LBB162_6:
	cmp r9, 32
	jae .LBB162_8
	xor r10d, r10d
	jmp .LBB162_12
.LBB162_8:
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
.LBB162_9:
	vmovdqu ymm0, ymmword ptr [rbx + 4*r15 - 96]
	vmovdqu ymm1, ymmword ptr [rbx + 4*r15 - 64]
	vmovdqu ymm2, ymmword ptr [rbx + 4*r15 - 32]
	vmovdqu ymm3, ymmword ptr [rbx + 4*r15]
	vpaddd ymm0, ymm0, ymmword ptr [r14 + 4*r15 - 96]
	vpaddd ymm1, ymm1, ymmword ptr [r14 + 4*r15 - 64]
	vpaddd ymm2, ymm2, ymmword ptr [r14 + 4*r15 - 32]
	vpaddd ymm3, ymm3, ymmword ptr [r14 + 4*r15]
	vmovdqu ymmword ptr [r11 + 4*r15 - 96], ymm0
	vmovdqu ymmword ptr [r11 + 4*r15 - 64], ymm1
	vmovdqu ymmword ptr [r11 + 4*r15 - 32], ymm2
	vmovdqu ymmword ptr [r11 + 4*r15], ymm3
	add r15, 32
	cmp r10, r15
	jne .LBB162_9
	cmp r9, r10
	je .LBB162_15
	test r9b, 24
	je .LBB162_5
.LBB162_12:
	mov r11, r9
	and r11, -8
	lea rdi, [rax + r11]
	lea rbx, [rsi + 4*rax]
	lea r14, [rdx + 4*rax]
	lea rax, [rcx + 4*rax]
.LBB162_13:
	vmovdqu ymm0, ymmword ptr [r14 + 4*r10]
	vpaddd ymm0, ymm0, ymmword ptr [rax + 4*r10]
	vmovdqu ymmword ptr [rbx + 4*r10], ymm0
	add r10, 8
	cmp r11, r10
	jne .LBB162_13
	cmp r9, r11
	je .LBB162_15
.LBB162_5:
	mov eax, dword ptr [rdx + 4*rdi]
	add eax, dword ptr [rcx + 4*rdi]
	mov dword ptr [rsi + 4*rdi], eax
	inc rdi
	cmp r8, rdi
	jne .LBB162_5
.LBB162_15:
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
