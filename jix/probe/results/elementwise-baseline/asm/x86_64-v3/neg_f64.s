jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
	push rbx
	mov rax, r8
	and rax, -16
	je .LBB127_1
	mov rdx, qword ptr [rdi + 136]
	xor ecx, ecx
	vbroadcastsd ymm0, qword ptr [rip + .LCPI127_0]
.LBB127_18:
	vxorps ymm1, ymm0, ymmword ptr [rdx + 8*rcx]
	vmovups ymmword ptr [rsi + 8*rcx], ymm1
	vxorps ymm1, ymm0, ymmword ptr [rdx + 8*rcx + 32]
	vmovups ymmword ptr [rsi + 8*rcx + 32], ymm1
	vxorps ymm1, ymm0, ymmword ptr [rdx + 8*rcx + 64]
	vmovups ymmword ptr [rsi + 8*rcx + 64], ymm1
	vxorps ymm1, ymm0, ymmword ptr [rdx + 8*rcx + 96]
	vmovups ymmword ptr [rsi + 8*rcx + 96], ymm1
	add rcx, 16
	cmp rcx, rax
	jb .LBB127_18
	mov rdx, r8
	sub rdx, rcx
	ja .LBB127_3
	jmp .LBB127_16
.LBB127_1:
	xor ecx, ecx
	mov rdx, r8
	sub rdx, rcx
	jbe .LBB127_16
.LBB127_3:
	mov rax, qword ptr [rdi + 136]
	cmp rdx, 4
	jae .LBB127_5
	mov rdi, rcx
	jmp .LBB127_14
.LBB127_5:
	cmp rdx, 16
	jae .LBB127_7
	xor r9d, r9d
	jmp .LBB127_11
.LBB127_7:
	mov r9, rdx
	and r9, -16
	lea rdi, [rcx + r9]
	lea r10, [rsi + 8*rcx]
	add r10, 96
	lea r11, [rax + 8*rcx]
	add r11, 96
	xor ebx, ebx
	vbroadcastsd ymm0, qword ptr [rip + .LCPI127_0]
.LBB127_8:
	vxorps ymm1, ymm0, ymmword ptr [r11 + 8*rbx - 96]
	vxorps ymm2, ymm0, ymmword ptr [r11 + 8*rbx - 64]
	vxorps ymm3, ymm0, ymmword ptr [r11 + 8*rbx - 32]
	vxorps ymm4, ymm0, ymmword ptr [r11 + 8*rbx]
	vmovups ymmword ptr [r10 + 8*rbx - 96], ymm1
	vmovups ymmword ptr [r10 + 8*rbx - 64], ymm2
	vmovups ymmword ptr [r10 + 8*rbx - 32], ymm3
	vmovups ymmword ptr [r10 + 8*rbx], ymm4
	add rbx, 16
	cmp r9, rbx
	jne .LBB127_8
	cmp rdx, r9
	je .LBB127_16
	test dl, 12
	je .LBB127_14
.LBB127_11:
	mov r10, rdx
	and r10, -4
	lea rdi, [rcx + r10]
	lea r11, [rsi + 8*rcx]
	lea rcx, [rax + 8*rcx]
	vbroadcastsd ymm0, qword ptr [rip + .LCPI127_0]
.LBB127_12:
	vxorps ymm1, ymm0, ymmword ptr [rcx + 8*r9]
	vmovups ymmword ptr [r11 + 8*r9], ymm1
	add r9, 4
	cmp r10, r9
	jne .LBB127_12
	cmp rdx, r10
	je .LBB127_16
.LBB127_14:
	movabs rcx, -9223372036854775808
.LBB127_15:
	mov rdx, qword ptr [rax + 8*rdi]
	xor rdx, rcx
	mov qword ptr [rsi + 8*rdi], rdx
	inc rdi
	cmp r8, rdi
	jne .LBB127_15
.LBB127_16:
	pop rbx
	vzeroupper
	ret
