jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
	push rbx
	mov rax, r8
	and rax, -32
	je .LBB159_1
	mov rdx, qword ptr [rdi + 136]
	xor ecx, ecx
	vpxor xmm0, xmm0, xmm0
.LBB159_17:
	vpsubd ymm1, ymm0, ymmword ptr [rdx + 4*rcx]
	vmovdqu ymmword ptr [rsi + 4*rcx], ymm1
	vpsubd ymm1, ymm0, ymmword ptr [rdx + 4*rcx + 32]
	vmovdqu ymmword ptr [rsi + 4*rcx + 32], ymm1
	vpsubd ymm1, ymm0, ymmword ptr [rdx + 4*rcx + 64]
	vmovdqu ymmword ptr [rsi + 4*rcx + 64], ymm1
	vpsubd ymm1, ymm0, ymmword ptr [rdx + 4*rcx + 96]
	vmovdqu ymmword ptr [rsi + 4*rcx + 96], ymm1
	add rcx, 32
	cmp rcx, rax
	jb .LBB159_17
	mov rdx, r8
	sub rdx, rcx
	ja .LBB159_3
	jmp .LBB159_15
.LBB159_1:
	xor ecx, ecx
	mov rdx, r8
	sub rdx, rcx
	jbe .LBB159_15
.LBB159_3:
	mov rax, qword ptr [rdi + 136]
	cmp rdx, 8
	jae .LBB159_6
	mov rdi, rcx
	jmp .LBB159_5
.LBB159_6:
	cmp rdx, 32
	jae .LBB159_8
	xor r9d, r9d
	jmp .LBB159_12
.LBB159_8:
	mov r9, rdx
	and r9, -32
	lea rdi, [rcx + r9]
	lea r10, [rsi + 4*rcx]
	add r10, 96
	lea r11, [rax + 4*rcx]
	add r11, 96
	xor ebx, ebx
	vpxor xmm0, xmm0, xmm0
.LBB159_9:
	vpsubd ymm1, ymm0, ymmword ptr [r11 + 4*rbx - 96]
	vpsubd ymm2, ymm0, ymmword ptr [r11 + 4*rbx - 64]
	vpsubd ymm3, ymm0, ymmword ptr [r11 + 4*rbx - 32]
	vpsubd ymm4, ymm0, ymmword ptr [r11 + 4*rbx]
	vmovdqu ymmword ptr [r10 + 4*rbx - 96], ymm1
	vmovdqu ymmword ptr [r10 + 4*rbx - 64], ymm2
	vmovdqu ymmword ptr [r10 + 4*rbx - 32], ymm3
	vmovdqu ymmword ptr [r10 + 4*rbx], ymm4
	add rbx, 32
	cmp r9, rbx
	jne .LBB159_9
	cmp rdx, r9
	je .LBB159_15
	test dl, 24
	je .LBB159_5
.LBB159_12:
	mov r10, rdx
	and r10, -8
	lea rdi, [rcx + r10]
	lea r11, [rsi + 4*rcx]
	lea rcx, [rax + 4*rcx]
	vpxor xmm0, xmm0, xmm0
.LBB159_13:
	vpsubd ymm1, ymm0, ymmword ptr [rcx + 4*r9]
	vmovdqu ymmword ptr [r11 + 4*r9], ymm1
	add r9, 8
	cmp r10, r9
	jne .LBB159_13
	cmp rdx, r10
	je .LBB159_15
.LBB159_5:
	xor ecx, ecx
	sub ecx, dword ptr [rax + 4*rdi]
	mov dword ptr [rsi + 4*rdi], ecx
	inc rdi
	cmp r8, rdi
	jne .LBB159_5
.LBB159_15:
	pop rbx
	vzeroupper
	ret
