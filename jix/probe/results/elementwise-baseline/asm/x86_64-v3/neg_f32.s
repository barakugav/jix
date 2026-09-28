jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
	push rbx
	mov rax, r8
	and rax, -32
	je .LBB143_1
	mov rdx, qword ptr [rdi + 136]
	xor ecx, ecx
	vbroadcastss ymm0, dword ptr [rip + .LCPI143_0]
.LBB143_18:
	vxorps ymm1, ymm0, ymmword ptr [rdx + 4*rcx]
	vmovups ymmword ptr [rsi + 4*rcx], ymm1
	vxorps ymm1, ymm0, ymmword ptr [rdx + 4*rcx + 32]
	vmovups ymmword ptr [rsi + 4*rcx + 32], ymm1
	vxorps ymm1, ymm0, ymmword ptr [rdx + 4*rcx + 64]
	vmovups ymmword ptr [rsi + 4*rcx + 64], ymm1
	vxorps ymm1, ymm0, ymmword ptr [rdx + 4*rcx + 96]
	vmovups ymmword ptr [rsi + 4*rcx + 96], ymm1
	add rcx, 32
	cmp rcx, rax
	jb .LBB143_18
	mov rdx, r8
	sub rdx, rcx
	ja .LBB143_3
	jmp .LBB143_16
.LBB143_1:
	xor ecx, ecx
	mov rdx, r8
	sub rdx, rcx
	jbe .LBB143_16
.LBB143_3:
	mov rax, qword ptr [rdi + 136]
	cmp rdx, 8
	jae .LBB143_5
	mov rdi, rcx
	jmp .LBB143_14
.LBB143_5:
	cmp rdx, 32
	jae .LBB143_7
	xor r9d, r9d
	jmp .LBB143_11
.LBB143_7:
	mov r9, rdx
	and r9, -32
	lea rdi, [rcx + r9]
	lea r10, [rsi + 4*rcx]
	add r10, 96
	lea r11, [rax + 4*rcx]
	add r11, 96
	xor ebx, ebx
	vbroadcastss ymm0, dword ptr [rip + .LCPI143_0]
.LBB143_8:
	vxorps ymm1, ymm0, ymmword ptr [r11 + 4*rbx - 96]
	vxorps ymm2, ymm0, ymmword ptr [r11 + 4*rbx - 64]
	vxorps ymm3, ymm0, ymmword ptr [r11 + 4*rbx - 32]
	vxorps ymm4, ymm0, ymmword ptr [r11 + 4*rbx]
	vmovups ymmword ptr [r10 + 4*rbx - 96], ymm1
	vmovups ymmword ptr [r10 + 4*rbx - 64], ymm2
	vmovups ymmword ptr [r10 + 4*rbx - 32], ymm3
	vmovups ymmword ptr [r10 + 4*rbx], ymm4
	add rbx, 32
	cmp r9, rbx
	jne .LBB143_8
	cmp rdx, r9
	je .LBB143_16
	test dl, 24
	je .LBB143_14
.LBB143_11:
	mov r10, rdx
	and r10, -8
	lea rdi, [rcx + r10]
	lea r11, [rsi + 4*rcx]
	lea rcx, [rax + 4*rcx]
	vbroadcastss ymm0, dword ptr [rip + .LCPI143_0]
.LBB143_12:
	vxorps ymm1, ymm0, ymmword ptr [rcx + 4*r9]
	vmovups ymmword ptr [r11 + 4*r9], ymm1
	add r9, 8
	cmp r10, r9
	jne .LBB143_12
	cmp rdx, r10
	je .LBB143_16
.LBB143_14:
	mov ecx, -2147483648
.LBB143_15:
	mov edx, dword ptr [rax + 4*rdi]
	xor edx, ecx
	mov dword ptr [rsi + 4*rdi], edx
	inc rdi
	cmp r8, rdi
	jne .LBB143_15
.LBB143_16:
	pop rbx
	vzeroupper
	ret
