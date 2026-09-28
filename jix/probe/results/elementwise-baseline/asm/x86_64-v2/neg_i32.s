jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
	mov rcx, r8
	and rcx, -32
	je .LBB159_1
	mov rdx, qword ptr [rdi + 136]
	xor eax, eax
.LBB159_11:
	movdqu xmm0, xmmword ptr [rdx + 4*rax]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 16]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 16], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 32]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 32], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 48]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 48], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 64]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 64], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 80]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 80], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 96]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 96], xmm1
	movdqu xmm0, xmmword ptr [rdx + 4*rax + 112]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [rsi + 4*rax + 112], xmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB159_11
	mov r9, r8
	sub r9, rax
	ja .LBB159_3
	jmp .LBB159_9
.LBB159_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB159_9
.LBB159_3:
	mov rcx, qword ptr [rdi + 136]
	cmp r9, 8
	jae .LBB159_6
	mov rdx, rax
	jmp .LBB159_5
.LBB159_6:
	mov rdi, r9
	and rdi, -8
	lea rdx, [rax + rdi]
	lea r10, [rsi + 4*rax]
	add r10, 16
	lea rax, [rcx + 4*rax]
	add rax, 16
	xor r11d, r11d
.LBB159_7:
	movdqu xmm0, xmmword ptr [rax + 4*r11 - 16]
	movdqu xmm1, xmmword ptr [rax + 4*r11]
	pxor xmm2, xmm2
	psubd xmm2, xmm0
	pxor xmm0, xmm0
	psubd xmm0, xmm1
	movdqu xmmword ptr [r10 + 4*r11 - 16], xmm2
	movdqu xmmword ptr [r10 + 4*r11], xmm0
	add r11, 8
	cmp rdi, r11
	jne .LBB159_7
	cmp r9, rdi
	je .LBB159_9
.LBB159_5:
	xor eax, eax
	sub eax, dword ptr [rcx + 4*rdx]
	mov dword ptr [rsi + 4*rdx], eax
	inc rdx
	cmp r8, rdx
	jne .LBB159_5
.LBB159_9:
	ret
