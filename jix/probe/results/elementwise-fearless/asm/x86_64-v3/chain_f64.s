jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	sub rsp, 88
	mov qword ptr [rsp + 24], rcx
	mov qword ptr [rsp + 8], r8
	mov rbx, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov eax, dword ptr [rbx + 8]
	test eax, eax
	jne .LBB168_3
	cmp byte ptr [rbx], 2
	jne .LBB168_4
.LBB168_2:
	mov qword ptr [rsp + 40], rsi
	mov qword ptr [rsp + 48], rdx
	lea rax, [rsp + 24]
	mov qword ptr [rsp + 56], rax
	lea rax, [rsp + 8]
	mov qword ptr [rsp + 64], rax
	mov qword ptr [rsp + 72], rdi
	lea rax, [rsp + 7]
	mov qword ptr [rsp + 80], rax
	lea rdi, [rsp + 40]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>::{closure#1}, ()>
	jmp .LBB168_27
.LBB168_3:
	mov qword ptr [rsp + 32], rbx
	lea rax, [rsp + 32]
	mov qword ptr [rsp + 40], rax
	lea rax, [rbx + 8]
	lea rcx, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.18]
	lea r8, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.20]
	lea r9, [rsp + 40]
	mov r14, rdi
	mov rdi, rax
	mov r15, rsi
	mov esi, 1
	mov r12, rdx
	mov rdx, r9
	call qword ptr [rip + <std::sys::sync::once::futex::Once>::call@GOTPCREL]
	mov rdx, r12
	mov rdi, r14
	mov rsi, r15
	cmp byte ptr [rbx], 2
	je .LBB168_2
.LBB168_4:
	mov rax, qword ptr [rsp + 8]
	mov rcx, rax
	and rcx, -16
	je .LBB168_8
	mov rdx, qword ptr [rdi + 136]
	mov r8, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor r9d, r9d
.LBB168_6:
	vmovupd ymm0, ymmword ptr [rdx + 8*r9]
	vmovupd ymm1, ymmword ptr [rdx + 8*r9 + 32]
	vmovupd ymm2, ymmword ptr [rdx + 8*r9 + 64]
	vmovupd ymm3, ymmword ptr [rdx + 8*r9 + 96]
	vaddpd ymm0, ymm0, ymmword ptr [r8 + 8*r9]
	vaddpd ymm1, ymm1, ymmword ptr [r8 + 8*r9 + 32]
	vaddpd ymm2, ymm2, ymmword ptr [r8 + 8*r9 + 64]
	vaddpd ymm3, ymm3, ymmword ptr [r8 + 8*r9 + 96]
	vmulpd ymm0, ymm0, ymmword ptr [r10 + 8*r9]
	vmulpd ymm1, ymm1, ymmword ptr [r10 + 8*r9 + 32]
	vmulpd ymm2, ymm2, ymmword ptr [r10 + 8*r9 + 64]
	vmulpd ymm3, ymm3, ymmword ptr [r10 + 8*r9 + 96]
	vsubpd ymm0, ymm0, ymmword ptr [r11 + 8*r9]
	vsubpd ymm1, ymm1, ymmword ptr [r11 + 8*r9 + 32]
	vsubpd ymm2, ymm2, ymmword ptr [r11 + 8*r9 + 64]
	vsubpd ymm3, ymm3, ymmword ptr [r11 + 8*r9 + 96]
	vmovupd ymmword ptr [rsi + 8*r9], ymm0
	vmovupd ymmword ptr [rsi + 8*r9 + 32], ymm1
	vmovupd ymmword ptr [rsi + 8*r9 + 64], ymm2
	vmovupd ymmword ptr [rsi + 8*r9 + 96], ymm3
	add r9, 16
	cmp r9, rcx
	jb .LBB168_6
	mov r10, rax
	sub r10, r9
	ja .LBB168_9
	jmp .LBB168_27
.LBB168_8:
	xor r9d, r9d
	mov r10, rax
	sub r10, r9
	jbe .LBB168_27
.LBB168_9:
	mov rbp, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r8, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r10, 4
	jb .LBB168_10
	mov rcx, rbp
	sub rcx, rsi
	cmp rcx, -127
	setae cl
	mov r11, rdx
	sub r11, rsi
	cmp r11, -127
	setae r11b
	or r11b, cl
	mov rcx, r8
	sub rcx, rsi
	cmp rcx, -127
	setae cl
	mov rbx, rdi
	sub rbx, rsi
	cmp rbx, -127
	setae bl
	or bl, cl
	or bl, r11b
	je .LBB168_13
.LBB168_10:
	mov r11, r9
.LBB168_22:
	mov r9d, eax
	sub r9d, r11d
	lea rcx, [r11 + 1]
	test r9b, 1
	je .LBB168_24
	vmovsd xmm0, qword ptr [rbp + 8*r11]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r11]
	vmulsd xmm0, xmm0, qword ptr [r8 + 8*r11]
	vsubsd xmm0, xmm0, qword ptr [rdi + 8*r11]
	vmovsd qword ptr [rsi + 8*r11], xmm0
	mov r11, rcx
.LBB168_24:
	cmp rax, rcx
	je .LBB168_27
	sub rax, r11
	lea rcx, [rsi + 8*r11]
	add rcx, 8
	lea rsi, [rdi + 8*r11 + 8]
	lea rdi, [r8 + 8*r11]
	add rdi, 8
	lea rdx, [rdx + 8*r11]
	add rdx, 8
	lea r8, [8*r11 + 8]
	add r8, rbp
	xor r9d, r9d
.LBB168_26:
	vmovsd xmm0, qword ptr [r8 + 8*r9 - 8]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r9 - 8]
	vmulsd xmm0, xmm0, qword ptr [rdi + 8*r9 - 8]
	vsubsd xmm0, xmm0, qword ptr [rsi + 8*r9 - 8]
	vmovsd qword ptr [rcx + 8*r9 - 8], xmm0
	vmovsd xmm0, qword ptr [r8 + 8*r9]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r9]
	vmulsd xmm0, xmm0, qword ptr [rdi + 8*r9]
	vsubsd xmm0, xmm0, qword ptr [rsi + 8*r9]
	vmovsd qword ptr [rcx + 8*r9], xmm0
	add r9, 2
	cmp rax, r9
	jne .LBB168_26
.LBB168_27:
	add rsp, 88
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
.LBB168_13:
	cmp r10, 16
	jae .LBB168_15
	xor ebx, ebx
	jmp .LBB168_19
.LBB168_15:
	mov rbx, r10
	and rbx, -16
	lea r11, [r9 + rbx]
	lea r14, [rsi + 8*r9]
	add r14, 96
	lea r15, [rdi + 8*r9 + 96]
	lea r12, [r8 + 8*r9 + 96]
	lea r13, [rdx + 8*r9]
	add r13, 96
	mov qword ptr [rsp + 16], rbp
	lea rbp, [rbp + 8*r9 + 96]
	xor ecx, ecx
.LBB168_16:
	vmovupd ymm0, ymmword ptr [rbp + 8*rcx - 96]
	vmovupd ymm1, ymmword ptr [rbp + 8*rcx - 64]
	vmovupd ymm2, ymmword ptr [rbp + 8*rcx - 32]
	vmovupd ymm3, ymmword ptr [rbp + 8*rcx]
	vaddpd ymm0, ymm0, ymmword ptr [r13 + 8*rcx - 96]
	vaddpd ymm1, ymm1, ymmword ptr [r13 + 8*rcx - 64]
	vaddpd ymm2, ymm2, ymmword ptr [r13 + 8*rcx - 32]
	vaddpd ymm3, ymm3, ymmword ptr [r13 + 8*rcx]
	vmulpd ymm0, ymm0, ymmword ptr [r12 + 8*rcx - 96]
	vmulpd ymm1, ymm1, ymmword ptr [r12 + 8*rcx - 64]
	vmulpd ymm2, ymm2, ymmword ptr [r12 + 8*rcx - 32]
	vmulpd ymm3, ymm3, ymmword ptr [r12 + 8*rcx]
	vsubpd ymm0, ymm0, ymmword ptr [r15 + 8*rcx - 96]
	vsubpd ymm1, ymm1, ymmword ptr [r15 + 8*rcx - 64]
	vsubpd ymm2, ymm2, ymmword ptr [r15 + 8*rcx - 32]
	vsubpd ymm3, ymm3, ymmword ptr [r15 + 8*rcx]
	vmovupd ymmword ptr [r14 + 8*rcx - 96], ymm0
	vmovupd ymmword ptr [r14 + 8*rcx - 64], ymm1
	vmovupd ymmword ptr [r14 + 8*rcx - 32], ymm2
	vmovupd ymmword ptr [r14 + 8*rcx], ymm3
	add rcx, 16
	cmp rbx, rcx
	jne .LBB168_16
	cmp r10, rbx
	mov rbp, qword ptr [rsp + 16]
	je .LBB168_27
	test r10b, 12
	je .LBB168_22
.LBB168_19:
	mov rcx, r10
	and rcx, -4
	lea r11, [r9 + rcx]
	lea r14, [rsi + 8*r9]
	lea r15, [rdi + 8*r9]
	lea r12, [r8 + 8*r9]
	lea r13, [rdx + 8*r9]
	lea r9, [8*r9]
	add r9, rbp
.LBB168_20:
	vmovupd ymm0, ymmword ptr [r9 + 8*rbx]
	vaddpd ymm0, ymm0, ymmword ptr [r13 + 8*rbx]
	vmulpd ymm0, ymm0, ymmword ptr [r12 + 8*rbx]
	vsubpd ymm0, ymm0, ymmword ptr [r15 + 8*rbx]
	vmovupd ymmword ptr [r14 + 8*rbx], ymm0
	add rbx, 4
	cmp rcx, rbx
	jne .LBB168_20
	cmp r10, rcx
	je .LBB168_27
	jmp .LBB168_22
