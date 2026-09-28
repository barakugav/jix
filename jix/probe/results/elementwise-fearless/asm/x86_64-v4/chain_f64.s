jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
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
	jne .LBB168_1
	mov rax, r8
	and rax, -16
	je .LBB168_3
.LBB168_18:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor r9d, r9d
.LBB168_19:
	vmovupd zmm0, zmmword ptr [rcx + 8*r9]
	vmovupd zmm1, zmmword ptr [rcx + 8*r9 + 64]
	vaddpd zmm0, zmm0, zmmword ptr [rdx + 8*r9]
	vaddpd zmm1, zmm1, zmmword ptr [rdx + 8*r9 + 64]
	vmulpd zmm0, zmm0, zmmword ptr [r10 + 8*r9]
	vmulpd zmm1, zmm1, zmmword ptr [r10 + 8*r9 + 64]
	vsubpd zmm0, zmm0, zmmword ptr [r11 + 8*r9]
	vsubpd zmm1, zmm1, zmmword ptr [r11 + 8*r9 + 64]
	vmovupd zmmword ptr [rsi + 8*r9], zmm0
	vmovupd zmmword ptr [rsi + 8*r9 + 64], zmm1
	add r9, 16
	cmp r9, rax
	jb .LBB168_19
	mov rbp, r8
	sub rbp, r9
	ja .LBB168_5
	jmp .LBB168_17
.LBB168_1:
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
	and rax, -16
	jne .LBB168_18
.LBB168_3:
	xor r9d, r9d
	mov rbp, r8
	sub rbp, r9
	jbe .LBB168_17
.LBB168_5:
	mov rax, qword ptr [rdi + 136]
	mov rcx, qword ptr [rdi + 288]
	mov rdx, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp rbp, 8
	jae .LBB168_7
	mov r11, r9
	jmp .LBB168_16
.LBB168_7:
	cmp rbp, 32
	jae .LBB168_9
	xor ebx, ebx
	jmp .LBB168_13
.LBB168_9:
	mov qword ptr [rsp], rbp
	mov rbx, rbp
	and rbx, -32
	lea r11, [r9 + rbx]
	lea r14, [rsi + 8*r9]
	add r14, 192
	lea r15, [rdi + 8*r9 + 192]
	lea r12, [rdx + 8*r9]
	add r12, 192
	lea r13, [rcx + 8*r9]
	add r13, 192
	lea rbp, [rax + 8*r9]
	add rbp, 192
	xor r10d, r10d
.LBB168_10:
	vmovupd zmm0, zmmword ptr [rbp + 8*r10 - 192]
	vmovupd zmm1, zmmword ptr [rbp + 8*r10 - 128]
	vmovupd zmm2, zmmword ptr [rbp + 8*r10 - 64]
	vmovupd zmm3, zmmword ptr [rbp + 8*r10]
	vaddpd zmm0, zmm0, zmmword ptr [r13 + 8*r10 - 192]
	vaddpd zmm1, zmm1, zmmword ptr [r13 + 8*r10 - 128]
	vaddpd zmm2, zmm2, zmmword ptr [r13 + 8*r10 - 64]
	vaddpd zmm3, zmm3, zmmword ptr [r13 + 8*r10]
	vmulpd zmm0, zmm0, zmmword ptr [r12 + 8*r10 - 192]
	vmulpd zmm1, zmm1, zmmword ptr [r12 + 8*r10 - 128]
	vmulpd zmm2, zmm2, zmmword ptr [r12 + 8*r10 - 64]
	vmulpd zmm3, zmm3, zmmword ptr [r12 + 8*r10]
	vsubpd zmm0, zmm0, zmmword ptr [r15 + 8*r10 - 192]
	vsubpd zmm1, zmm1, zmmword ptr [r15 + 8*r10 - 128]
	vsubpd zmm2, zmm2, zmmword ptr [r15 + 8*r10 - 64]
	vsubpd zmm3, zmm3, zmmword ptr [r15 + 8*r10]
	vmovupd zmmword ptr [r14 + 8*r10 - 192], zmm0
	vmovupd zmmword ptr [r14 + 8*r10 - 128], zmm1
	vmovupd zmmword ptr [r14 + 8*r10 - 64], zmm2
	vmovupd zmmword ptr [r14 + 8*r10], zmm3
	add r10, 32
	cmp rbx, r10
	jne .LBB168_10
	mov rbp, qword ptr [rsp]
	cmp rbp, rbx
	je .LBB168_17
	test bpl, 24
	je .LBB168_16
.LBB168_13:
	mov r10, rbp
	and r10, -8
	lea r11, [r9 + r10]
	lea r14, [rsi + 8*r9]
	lea r15, [rdi + 8*r9]
	lea r12, [rdx + 8*r9]
	lea r13, [rcx + 8*r9]
	lea r9, [rax + 8*r9]
.LBB168_14:
	vmovupd zmm0, zmmword ptr [r9 + 8*rbx]
	vaddpd zmm0, zmm0, zmmword ptr [r13 + 8*rbx]
	vmulpd zmm0, zmm0, zmmword ptr [r12 + 8*rbx]
	vsubpd zmm0, zmm0, zmmword ptr [r15 + 8*rbx]
	vmovupd zmmword ptr [r14 + 8*rbx], zmm0
	add rbx, 8
	cmp r10, rbx
	jne .LBB168_14
	cmp rbp, r10
	je .LBB168_17
.LBB168_16:
	vmovsd xmm0, qword ptr [rax + 8*r11]
	vaddsd xmm0, xmm0, qword ptr [rcx + 8*r11]
	vmulsd xmm0, xmm0, qword ptr [rdx + 8*r11]
	vsubsd xmm0, xmm0, qword ptr [rdi + 8*r11]
	vmovsd qword ptr [rsi + 8*r11], xmm0
	inc r11
	cmp r8, r11
	jne .LBB168_16
.LBB168_17:
	add rsp, 24
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
