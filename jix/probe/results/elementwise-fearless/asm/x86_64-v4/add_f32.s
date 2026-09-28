jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>:
	push r15
	push r14
	push rbx
	sub rsp, 16
	mov rax, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov ecx, dword ptr [rax + 8]
	test ecx, ecx
	jne .LBB174_1
	cmp r8, 32
	jae .LBB174_4
.LBB174_3:
	xor eax, eax
	jmp .LBB174_8
.LBB174_1:
	mov qword ptr [rsp], rax
	mov rcx, rsp
	mov qword ptr [rsp + 8], rcx
	add rax, 8
	lea rcx, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.18]
	lea r9, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.20]
	lea rdx, [rsp + 8]
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
	cmp r8, 32
	jb .LBB174_3
.LBB174_4:
	mov rdx, qword ptr [rdi + 136]
	mov rcx, qword ptr [rdi + 288]
	lea r9, [r8 - 32]
	mov r10, r9
	shr r10, 5
	je .LBB174_5
	inc r10
	and r10, -2
	xor eax, eax
.LBB174_23:
	vmovups zmm0, zmmword ptr [rdx + 4*rax]
	vmovups zmm1, zmmword ptr [rdx + 4*rax + 64]
	vaddps zmm0, zmm0, zmmword ptr [rcx + 4*rax]
	vaddps zmm1, zmm1, zmmword ptr [rcx + 4*rax + 64]
	vmovups zmmword ptr [rsi + 4*rax], zmm0
	vmovups zmmword ptr [rsi + 4*rax + 64], zmm1
	vmovups zmm0, zmmword ptr [rdx + 4*rax + 128]
	vmovups zmm1, zmmword ptr [rdx + 4*rax + 192]
	vaddps zmm0, zmm0, zmmword ptr [rcx + 4*rax + 128]
	vaddps zmm1, zmm1, zmmword ptr [rcx + 4*rax + 192]
	vmovups zmmword ptr [rsi + 4*rax + 128], zmm0
	vmovups zmmword ptr [rsi + 4*rax + 192], zmm1
	add rax, 64
	add r10, -2
	jne .LBB174_23
	test r9b, 32
	je .LBB174_7
	jmp .LBB174_8
.LBB174_5:
	xor eax, eax
.LBB174_7:
	vmovups zmm0, zmmword ptr [rdx + 4*rax]
	vmovups zmm1, zmmword ptr [rdx + 4*rax + 64]
	vaddps zmm0, zmm0, zmmword ptr [rcx + 4*rax]
	vaddps zmm1, zmm1, zmmword ptr [rcx + 4*rax + 64]
	vmovups zmmword ptr [rsi + 4*rax], zmm0
	vmovups zmmword ptr [rsi + 4*rax + 64], zmm1
	add rax, 32
.LBB174_8:
	mov r9, r8
	sub r9, rax
	jbe .LBB174_21
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jae .LBB174_11
	mov rdi, rax
	jmp .LBB174_20
.LBB174_11:
	cmp r9, 64
	jae .LBB174_13
	xor r10d, r10d
	jmp .LBB174_17
.LBB174_13:
	mov r10, r9
	and r10, -64
	lea rdi, [rax + r10]
	lea r11, [rsi + 4*rax]
	add r11, 192
	lea rbx, [rdx + 4*rax]
	add rbx, 192
	lea r14, [rcx + 4*rax]
	add r14, 192
	xor r15d, r15d
.LBB174_14:
	vmovups zmm0, zmmword ptr [r14 + 4*r15 - 192]
	vmovups zmm1, zmmword ptr [r14 + 4*r15 - 128]
	vmovups zmm2, zmmword ptr [r14 + 4*r15 - 64]
	vmovups zmm3, zmmword ptr [r14 + 4*r15]
	vaddps zmm0, zmm0, zmmword ptr [rbx + 4*r15 - 192]
	vaddps zmm1, zmm1, zmmword ptr [rbx + 4*r15 - 128]
	vaddps zmm2, zmm2, zmmword ptr [rbx + 4*r15 - 64]
	vaddps zmm3, zmm3, zmmword ptr [rbx + 4*r15]
	vmovups zmmword ptr [r11 + 4*r15 - 192], zmm0
	vmovups zmmword ptr [r11 + 4*r15 - 128], zmm1
	vmovups zmmword ptr [r11 + 4*r15 - 64], zmm2
	vmovups zmmword ptr [r11 + 4*r15], zmm3
	add r15, 64
	cmp r10, r15
	jne .LBB174_14
	cmp r9, r10
	je .LBB174_21
	test r9b, 56
	je .LBB174_20
.LBB174_17:
	mov r11, r9
	and r11, -8
	lea rdi, [rax + r11]
	lea rbx, [rsi + 4*rax]
	lea r14, [rdx + 4*rax]
	lea rax, [rcx + 4*rax]
.LBB174_18:
	vmovups ymm0, ymmword ptr [rax + 4*r10]
	vaddps ymm0, ymm0, ymmword ptr [r14 + 4*r10]
	vmovups ymmword ptr [rbx + 4*r10], ymm0
	add r10, 8
	cmp r11, r10
	jne .LBB174_18
	cmp r9, r11
	je .LBB174_21
.LBB174_20:
	vmovss xmm0, dword ptr [rcx + 4*rdi]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*rdi]
	vmovss dword ptr [rsi + 4*rdi], xmm0
	inc rdi
	cmp r8, rdi
	jne .LBB174_20
.LBB174_21:
	add rsp, 16
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
