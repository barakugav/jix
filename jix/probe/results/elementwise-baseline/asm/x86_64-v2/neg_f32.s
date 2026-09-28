jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
	mov rcx, r8
	and rcx, -32
	je .LBB143_1
	mov rdx, qword ptr [rdi + 136]
	xor eax, eax
	movaps xmm0, xmmword ptr [rip + .LCPI143_0]
.LBB143_12:
	movups xmm1, xmmword ptr [rdx + 4*rax]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax], xmm1
	movups xmm1, xmmword ptr [rdx + 4*rax + 16]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 16], xmm1
	movups xmm1, xmmword ptr [rdx + 4*rax + 32]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 32], xmm1
	movups xmm1, xmmword ptr [rdx + 4*rax + 48]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 48], xmm1
	movups xmm1, xmmword ptr [rdx + 4*rax + 64]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 64], xmm1
	movups xmm1, xmmword ptr [rdx + 4*rax + 80]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 80], xmm1
	movups xmm1, xmmword ptr [rdx + 4*rax + 96]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 96], xmm1
	movups xmm1, xmmword ptr [rdx + 4*rax + 112]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 4*rax + 112], xmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB143_12
	mov r9, r8
	sub r9, rax
	ja .LBB143_3
	jmp .LBB143_10
.LBB143_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB143_10
.LBB143_3:
	mov rcx, qword ptr [rdi + 136]
	cmp r9, 8
	jae .LBB143_5
	mov rdx, rax
	jmp .LBB143_8
.LBB143_5:
	mov rdi, r9
	and rdi, -8
	lea rdx, [rax + rdi]
	lea r10, [rsi + 4*rax]
	add r10, 16
	lea rax, [rcx + 4*rax]
	add rax, 16
	xor r11d, r11d
	movaps xmm0, xmmword ptr [rip + .LCPI143_0]
.LBB143_6:
	movups xmm1, xmmword ptr [rax + 4*r11 - 16]
	movups xmm2, xmmword ptr [rax + 4*r11]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [r10 + 4*r11 - 16], xmm1
	movups xmmword ptr [r10 + 4*r11], xmm2
	add r11, 8
	cmp rdi, r11
	jne .LBB143_6
	cmp r9, rdi
	je .LBB143_10
.LBB143_8:
	mov eax, -2147483648
.LBB143_9:
	mov edi, dword ptr [rcx + 4*rdx]
	xor edi, eax
	mov dword ptr [rsi + 4*rdx], edi
	inc rdx
	cmp r8, rdx
	jne .LBB143_9
.LBB143_10:
	ret
