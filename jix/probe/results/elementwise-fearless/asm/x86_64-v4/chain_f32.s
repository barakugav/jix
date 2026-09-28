jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	sub rsp, 24
	mov rax, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov ecx, dword ptr [rax + 8]
	test ecx, ecx
	jne .LBB172_1
	mov rax, r8
	and rax, -32
	je .LBB172_3
.LBB172_18:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor r9d, r9d
.LBB172_19:
	vmovups zmm0, zmmword ptr [rcx + 4*r9]
	vmovups zmm1, zmmword ptr [rcx + 4*r9 + 64]
	vaddps zmm0, zmm0, zmmword ptr [rdx + 4*r9]
	vaddps zmm1, zmm1, zmmword ptr [rdx + 4*r9 + 64]
	vmulps zmm0, zmm0, zmmword ptr [r10 + 4*r9]
	vmulps zmm1, zmm1, zmmword ptr [r10 + 4*r9 + 64]
	vsubps zmm0, zmm0, zmmword ptr [r11 + 4*r9]
	vsubps zmm1, zmm1, zmmword ptr [r11 + 4*r9 + 64]
	vmovups zmmword ptr [rsi + 4*r9], zmm0
	vmovups zmmword ptr [rsi + 4*r9 + 64], zmm1
	add r9, 32
	cmp r9, rax
	jb .LBB172_19
	mov rbp, r8
	sub rbp, r9
	ja .LBB172_5
	jmp .LBB172_17
.LBB172_1:
	mov qword ptr [rsp + 8], rax
	lea rcx, [rsp + 8]
	mov qword ptr [rsp + 16], rcx
	add rax, 8
	lea rcx, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.18]
	lea r9, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.20]
	lea rdx, [rsp + 16]
	mov rbx, rdi
	mov rdi, rax
	mov r14, rsi
	mov esi, 1
	mov r15, r8
	mov r8, r9
	call qword ptr [rip + <std::sys::sync::once::futex::Once>::call@GOTPCREL]
	mov rdi, rbx
	mov rsi, r14
	mov r8, r15
	mov rax, r8
	and rax, -32
	jne .LBB172_18
.LBB172_3:
	xor r9d, r9d
	mov rbp, r8
	sub rbp, r9
	jbe .LBB172_17
.LBB172_5:
	mov rax, qword ptr [rdi + 136]
	mov rcx, qword ptr [rdi + 288]
	mov rdx, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp rbp, 8
	jae .LBB172_7
	mov r11, r9
	jmp .LBB172_16
.LBB172_7:
	cmp rbp, 64
	jae .LBB172_9
	xor ebx, ebx
	jmp .LBB172_13
.LBB172_9:
	mov qword ptr [rsp], rbp
	mov rbx, rbp
	and rbx, -64
	lea r11, [r9 + rbx]
	lea r14, [rsi + 4*r9]
	add r14, 192
	lea r15, [rdi + 4*r9 + 192]
	lea r12, [rdx + 4*r9]
	add r12, 192
	lea r13, [rcx + 4*r9]
	add r13, 192
	lea rbp, [rax + 4*r9]
	add rbp, 192
	xor r10d, r10d
.LBB172_10:
	vmovups zmm0, zmmword ptr [rbp + 4*r10 - 192]
	vmovups zmm1, zmmword ptr [rbp + 4*r10 - 128]
	vmovups zmm2, zmmword ptr [rbp + 4*r10 - 64]
	vmovups zmm3, zmmword ptr [rbp + 4*r10]
	vaddps zmm0, zmm0, zmmword ptr [r13 + 4*r10 - 192]
	vaddps zmm1, zmm1, zmmword ptr [r13 + 4*r10 - 128]
	vaddps zmm2, zmm2, zmmword ptr [r13 + 4*r10 - 64]
	vaddps zmm3, zmm3, zmmword ptr [r13 + 4*r10]
	vmulps zmm0, zmm0, zmmword ptr [r12 + 4*r10 - 192]
	vmulps zmm1, zmm1, zmmword ptr [r12 + 4*r10 - 128]
	vmulps zmm2, zmm2, zmmword ptr [r12 + 4*r10 - 64]
	vmulps zmm3, zmm3, zmmword ptr [r12 + 4*r10]
	vsubps zmm0, zmm0, zmmword ptr [r15 + 4*r10 - 192]
	vsubps zmm1, zmm1, zmmword ptr [r15 + 4*r10 - 128]
	vsubps zmm2, zmm2, zmmword ptr [r15 + 4*r10 - 64]
	vsubps zmm3, zmm3, zmmword ptr [r15 + 4*r10]
	vmovups zmmword ptr [r14 + 4*r10 - 192], zmm0
	vmovups zmmword ptr [r14 + 4*r10 - 128], zmm1
	vmovups zmmword ptr [r14 + 4*r10 - 64], zmm2
	vmovups zmmword ptr [r14 + 4*r10], zmm3
	add r10, 64
	cmp rbx, r10
	jne .LBB172_10
	mov rbp, qword ptr [rsp]
	cmp rbp, rbx
	je .LBB172_17
	test bpl, 56
	je .LBB172_16
.LBB172_13:
	mov r10, rbp
	and r10, -8
	lea r11, [r9 + r10]
	lea r14, [rsi + 4*r9]
	lea r15, [rdi + 4*r9]
	lea r12, [rdx + 4*r9]
	lea r13, [rcx + 4*r9]
	lea r9, [rax + 4*r9]
.LBB172_14:
	vmovups ymm0, ymmword ptr [r9 + 4*rbx]
	vaddps ymm0, ymm0, ymmword ptr [r13 + 4*rbx]
	vmulps ymm0, ymm0, ymmword ptr [r12 + 4*rbx]
	vsubps ymm0, ymm0, ymmword ptr [r15 + 4*rbx]
	vmovups ymmword ptr [r14 + 4*rbx], ymm0
	add rbx, 8
	cmp r10, rbx
	jne .LBB172_14
	cmp rbp, r10
	je .LBB172_17
.LBB172_16:
	vmovss xmm0, dword ptr [rax + 4*r11]
	vaddss xmm0, xmm0, dword ptr [rcx + 4*r11]
	vmulss xmm0, xmm0, dword ptr [rdx + 4*r11]
	vsubss xmm0, xmm0, dword ptr [rdi + 4*r11]
	vmovss dword ptr [rsi + 4*r11], xmm0
	inc r11
	cmp r8, r11
	jne .LBB172_16
.LBB172_17:
	add rsp, 24
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
