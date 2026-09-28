jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	push r15
	push r14
	push rbx
	sub rsp, 16
	mov rax, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov ecx, dword ptr [rax + 8]
	test ecx, ecx
	jne .LBB170_1
	cmp r8, 16
	jae .LBB170_4
.LBB170_3:
	xor eax, eax
	jmp .LBB170_8
.LBB170_1:
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
	cmp r8, 16
	jb .LBB170_3
.LBB170_4:
	mov rdx, qword ptr [rdi + 136]
	mov rcx, qword ptr [rdi + 288]
	lea r9, [r8 - 16]
	mov r10, r9
	shr r10, 4
	je .LBB170_5
	inc r10
	and r10, -2
	xor eax, eax
.LBB170_23:
	vmovupd zmm0, zmmword ptr [rdx + 8*rax]
	vmovupd zmm1, zmmword ptr [rdx + 8*rax + 64]
	vaddpd zmm0, zmm0, zmmword ptr [rcx + 8*rax]
	vaddpd zmm1, zmm1, zmmword ptr [rcx + 8*rax + 64]
	vmovupd zmmword ptr [rsi + 8*rax], zmm0
	vmovupd zmmword ptr [rsi + 8*rax + 64], zmm1
	vmovupd zmm0, zmmword ptr [rdx + 8*rax + 128]
	vmovupd zmm1, zmmword ptr [rdx + 8*rax + 192]
	vaddpd zmm0, zmm0, zmmword ptr [rcx + 8*rax + 128]
	vaddpd zmm1, zmm1, zmmword ptr [rcx + 8*rax + 192]
	vmovupd zmmword ptr [rsi + 8*rax + 128], zmm0
	vmovupd zmmword ptr [rsi + 8*rax + 192], zmm1
	add rax, 32
	add r10, -2
	jne .LBB170_23
	test r9b, 16
	je .LBB170_7
	jmp .LBB170_8
.LBB170_5:
	xor eax, eax
.LBB170_7:
	vmovupd zmm0, zmmword ptr [rdx + 8*rax]
	vmovupd zmm1, zmmword ptr [rdx + 8*rax + 64]
	vaddpd zmm0, zmm0, zmmword ptr [rcx + 8*rax]
	vaddpd zmm1, zmm1, zmmword ptr [rcx + 8*rax + 64]
	vmovupd zmmword ptr [rsi + 8*rax], zmm0
	vmovupd zmmword ptr [rsi + 8*rax + 64], zmm1
	add rax, 16
.LBB170_8:
	mov r9, r8
	sub r9, rax
	jbe .LBB170_21
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jae .LBB170_11
	mov rdi, rax
	jmp .LBB170_20
.LBB170_11:
	cmp r9, 32
	jae .LBB170_13
	xor r10d, r10d
	jmp .LBB170_17
.LBB170_13:
	mov r10, r9
	and r10, -32
	lea rdi, [rax + r10]
	lea r11, [rsi + 8*rax]
	add r11, 192
	lea rbx, [rdx + 8*rax]
	add rbx, 192
	lea r14, [rcx + 8*rax]
	add r14, 192
	xor r15d, r15d
.LBB170_14:
	vmovupd zmm0, zmmword ptr [r14 + 8*r15 - 192]
	vmovupd zmm1, zmmword ptr [r14 + 8*r15 - 128]
	vmovupd zmm2, zmmword ptr [r14 + 8*r15 - 64]
	vmovupd zmm3, zmmword ptr [r14 + 8*r15]
	vaddpd zmm0, zmm0, zmmword ptr [rbx + 8*r15 - 192]
	vaddpd zmm1, zmm1, zmmword ptr [rbx + 8*r15 - 128]
	vaddpd zmm2, zmm2, zmmword ptr [rbx + 8*r15 - 64]
	vaddpd zmm3, zmm3, zmmword ptr [rbx + 8*r15]
	vmovupd zmmword ptr [r11 + 8*r15 - 192], zmm0
	vmovupd zmmword ptr [r11 + 8*r15 - 128], zmm1
	vmovupd zmmword ptr [r11 + 8*r15 - 64], zmm2
	vmovupd zmmword ptr [r11 + 8*r15], zmm3
	add r15, 32
	cmp r10, r15
	jne .LBB170_14
	cmp r9, r10
	je .LBB170_21
	test r9b, 24
	je .LBB170_20
.LBB170_17:
	mov r11, r9
	and r11, -8
	lea rdi, [rax + r11]
	lea rbx, [rsi + 8*rax]
	lea r14, [rdx + 8*rax]
	lea rax, [rcx + 8*rax]
.LBB170_18:
	vmovupd zmm0, zmmword ptr [rax + 8*r10]
	vaddpd zmm0, zmm0, zmmword ptr [r14 + 8*r10]
	vmovupd zmmword ptr [rbx + 8*r10], zmm0
	add r10, 8
	cmp r11, r10
	jne .LBB170_18
	cmp r9, r11
	je .LBB170_21
.LBB170_20:
	vmovsd xmm0, qword ptr [rcx + 8*rdi]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*rdi]
	vmovsd qword ptr [rsi + 8*rdi], xmm0
	inc rdi
	cmp r8, rdi
	jne .LBB170_20
.LBB170_21:
	add rsp, 16
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
