jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
	mov rcx, r8
	and rcx, -16
	je .LBB127_1
	mov rdx, qword ptr [rdi + 136]
	xor eax, eax
	movaps xmm0, xmmword ptr [rip + .LCPI127_0]
.LBB127_12:
	movups xmm1, xmmword ptr [rdx + 8*rax]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax], xmm1
	movups xmm1, xmmword ptr [rdx + 8*rax + 16]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax + 16], xmm1
	movups xmm1, xmmword ptr [rdx + 8*rax + 32]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax + 32], xmm1
	movups xmm1, xmmword ptr [rdx + 8*rax + 48]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax + 48], xmm1
	movups xmm1, xmmword ptr [rdx + 8*rax + 64]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax + 64], xmm1
	movups xmm1, xmmword ptr [rdx + 8*rax + 80]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax + 80], xmm1
	movups xmm1, xmmword ptr [rdx + 8*rax + 96]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax + 96], xmm1
	movups xmm1, xmmword ptr [rdx + 8*rax + 112]
	xorps xmm1, xmm0
	movups xmmword ptr [rsi + 8*rax + 112], xmm1
	add rax, 16
	cmp rax, rcx
	jb .LBB127_12
	mov r9, r8
	sub r9, rax
	ja .LBB127_3
	jmp .LBB127_10
.LBB127_1:
	xor eax, eax
	mov r9, r8
	sub r9, rax
	jbe .LBB127_10
.LBB127_3:
	mov rcx, qword ptr [rdi + 136]
	cmp r9, 4
	jae .LBB127_5
	mov rdx, rax
	jmp .LBB127_8
.LBB127_5:
	mov rdi, r9
	and rdi, -4
	lea rdx, [rax + rdi]
	lea r10, [rsi + 8*rax]
	add r10, 16
	lea rax, [rcx + 8*rax]
	add rax, 16
	xor r11d, r11d
	movaps xmm0, xmmword ptr [rip + .LCPI127_0]
.LBB127_6:
	movups xmm1, xmmword ptr [rax + 8*r11 - 16]
	movups xmm2, xmmword ptr [rax + 8*r11]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [r10 + 8*r11 - 16], xmm1
	movups xmmword ptr [r10 + 8*r11], xmm2
	add r11, 4
	cmp rdi, r11
	jne .LBB127_6
	cmp r9, rdi
	je .LBB127_10
.LBB127_8:
	movabs rax, -9223372036854775808
.LBB127_9:
	mov rdi, qword ptr [rcx + 8*rdx]
	xor rdi, rax
	mov qword ptr [rsi + 8*rdx], rdi
	inc rdx
	cmp r8, rdx
	jne .LBB127_9
.LBB127_10:
	ret
