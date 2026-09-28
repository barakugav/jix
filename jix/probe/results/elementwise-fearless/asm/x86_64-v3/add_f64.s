jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	push r15
	push r14
	push r12
	push rbx
	sub rsp, 88
	mov qword ptr [rsp + 24], rcx
	mov qword ptr [rsp + 16], r8
	mov rbx, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov eax, dword ptr [rbx + 8]
	test eax, eax
	jne .LBB170_3
	cmp byte ptr [rbx], 2
	jne .LBB170_4
.LBB170_2:
	mov qword ptr [rsp + 40], rsi
	mov qword ptr [rsp + 48], rdx
	lea rax, [rsp + 24]
	mov qword ptr [rsp + 56], rax
	lea rax, [rsp + 16]
	mov qword ptr [rsp + 64], rax
	mov qword ptr [rsp + 72], rdi
	lea rax, [rsp + 15]
	mov qword ptr [rsp + 80], rax
	lea rdi, [rsp + 40]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>::{closure#1}, ()>
	jmp .LBB170_27
.LBB170_3:
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
	je .LBB170_2
.LBB170_4:
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -16
	je .LBB170_8
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor r8d, r8d
.LBB170_6:
	vmovupd ymm0, ymmword ptr [rdx + 8*r8]
	vmovupd ymm1, ymmword ptr [rdx + 8*r8 + 32]
	vmovupd ymm2, ymmword ptr [rdx + 8*r8 + 64]
	vmovupd ymm3, ymmword ptr [rdx + 8*r8 + 96]
	vaddpd ymm0, ymm0, ymmword ptr [r9 + 8*r8]
	vaddpd ymm1, ymm1, ymmword ptr [r9 + 8*r8 + 32]
	vaddpd ymm2, ymm2, ymmword ptr [r9 + 8*r8 + 64]
	vaddpd ymm3, ymm3, ymmword ptr [r9 + 8*r8 + 96]
	vmovupd ymmword ptr [rsi + 8*r8], ymm0
	vmovupd ymmword ptr [rsi + 8*r8 + 32], ymm1
	vmovupd ymmword ptr [rsi + 8*r8 + 64], ymm2
	vmovupd ymmword ptr [rsi + 8*r8 + 96], ymm3
	add r8, 16
	cmp r8, rcx
	jb .LBB170_6
	mov r9, rax
	sub r9, r8
	ja .LBB170_9
	jmp .LBB170_27
.LBB170_8:
	xor r8d, r8d
	mov r9, rax
	sub r9, r8
	jbe .LBB170_27
.LBB170_9:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 4
	jb .LBB170_10
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -127
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -127
	setae r10b
	or r10b, dil
	je .LBB170_13
.LBB170_10:
	mov rdi, r8
.LBB170_22:
	mov r9d, eax
	sub r9d, edi
	mov r8, rdi
	and r9d, 3
	je .LBB170_25
	mov r8, rdi
.LBB170_24:
	vmovsd xmm0, qword ptr [rcx + 8*r8]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r8]
	vmovsd qword ptr [rsi + 8*r8], xmm0
	inc r8
	dec r9
	jne .LBB170_24
.LBB170_25:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB170_27
.LBB170_26:
	vmovsd xmm0, qword ptr [rcx + 8*r8]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r8]
	vmovsd qword ptr [rsi + 8*r8], xmm0
	vmovsd xmm0, qword ptr [rcx + 8*r8 + 8]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r8 + 8]
	vmovsd qword ptr [rsi + 8*r8 + 8], xmm0
	vmovsd xmm0, qword ptr [rcx + 8*r8 + 16]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r8 + 16]
	vmovsd qword ptr [rsi + 8*r8 + 16], xmm0
	vmovsd xmm0, qword ptr [rcx + 8*r8 + 24]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r8 + 24]
	vmovsd qword ptr [rsi + 8*r8 + 24], xmm0
	add r8, 4
	cmp rax, r8
	jne .LBB170_26
.LBB170_27:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	vzeroupper
	ret
.LBB170_13:
	cmp r9, 16
	jae .LBB170_15
	xor r10d, r10d
	jmp .LBB170_19
.LBB170_15:
	mov r10, r9
	and r10, -16
	lea rdi, [r8 + r10]
	lea r11, [rsi + 8*r8]
	add r11, 96
	lea rbx, [rdx + 8*r8]
	add rbx, 96
	lea r14, [rcx + 8*r8]
	add r14, 96
	xor r15d, r15d
.LBB170_16:
	vmovupd ymm0, ymmword ptr [r14 + 8*r15 - 96]
	vmovupd ymm1, ymmword ptr [r14 + 8*r15 - 64]
	vmovupd ymm2, ymmword ptr [r14 + 8*r15 - 32]
	vmovupd ymm3, ymmword ptr [r14 + 8*r15]
	vaddpd ymm0, ymm0, ymmword ptr [rbx + 8*r15 - 96]
	vaddpd ymm1, ymm1, ymmword ptr [rbx + 8*r15 - 64]
	vaddpd ymm2, ymm2, ymmword ptr [rbx + 8*r15 - 32]
	vaddpd ymm3, ymm3, ymmword ptr [rbx + 8*r15]
	vmovupd ymmword ptr [r11 + 8*r15 - 96], ymm0
	vmovupd ymmword ptr [r11 + 8*r15 - 64], ymm1
	vmovupd ymmword ptr [r11 + 8*r15 - 32], ymm2
	vmovupd ymmword ptr [r11 + 8*r15], ymm3
	add r15, 16
	cmp r10, r15
	jne .LBB170_16
	cmp r9, r10
	je .LBB170_27
	test r9b, 12
	je .LBB170_22
.LBB170_19:
	mov r11, r9
	and r11, -4
	lea rdi, [r8 + r11]
	lea rbx, [rsi + 8*r8]
	lea r14, [rdx + 8*r8]
	lea r8, [rcx + 8*r8]
.LBB170_20:
	vmovupd ymm0, ymmword ptr [r8 + 8*r10]
	vaddpd ymm0, ymm0, ymmword ptr [r14 + 8*r10]
	vmovupd ymmword ptr [rbx + 8*r10], ymm0
	add r10, 4
	cmp r11, r10
	jne .LBB170_20
	cmp r9, r11
	je .LBB170_27
	jmp .LBB170_22
