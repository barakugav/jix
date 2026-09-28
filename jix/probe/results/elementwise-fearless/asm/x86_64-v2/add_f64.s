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
	jne .LBB170_23
.LBB170_1:
	movzx eax, byte ptr [rbx]
	cmp al, 3
	sete cl
	add cl, cl
	inc cl
	cmp al, 2
	movzx ecx, cl
	mov eax, 2
	cmovne eax, ecx
	cmp al, 1
	je .LBB170_5
	movzx eax, al
	cmp eax, 2
	je .LBB170_4
	mov qword ptr [rsp + 32], rsi
	mov qword ptr [rsp + 40], rdx
	lea rax, [rsp + 24]
	mov qword ptr [rsp + 48], rax
	lea rax, [rsp + 16]
	mov qword ptr [rsp + 56], rax
	mov qword ptr [rsp + 64], rdi
	lea rax, [rsp + 15]
	mov qword ptr [rsp + 72], rax
	lea rdi, [rsp + 32]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>::{closure#1}, ()>
	jmp .LBB170_22
.LBB170_4:
	mov qword ptr [rsp + 32], rsi
	mov qword ptr [rsp + 40], rdx
	lea rax, [rsp + 24]
	mov qword ptr [rsp + 48], rax
	lea rax, [rsp + 16]
	mov qword ptr [rsp + 56], rax
	mov qword ptr [rsp + 64], rdi
	lea rax, [rsp + 15]
	mov qword ptr [rsp + 72], rax
	lea rdi, [rsp + 32]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>::{closure#2}, ()>
	jmp .LBB170_22
.LBB170_5:
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -16
	je .LBB170_9
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor r8d, r8d
.LBB170_7:
	movupd xmm0, xmmword ptr [rdx + 8*r8]
	movupd xmm1, xmmword ptr [rdx + 8*r8 + 16]
	movupd xmm2, xmmword ptr [rdx + 8*r8 + 32]
	movupd xmm3, xmmword ptr [rdx + 8*r8 + 48]
	movupd xmm4, xmmword ptr [rdx + 8*r8 + 64]
	movupd xmm5, xmmword ptr [rdx + 8*r8 + 80]
	movupd xmm6, xmmword ptr [rdx + 8*r8 + 96]
	movupd xmm7, xmmword ptr [rdx + 8*r8 + 112]
	movupd xmm8, xmmword ptr [r9 + 8*r8]
	addpd xmm8, xmm0
	movupd xmm0, xmmword ptr [r9 + 8*r8 + 16]
	addpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r9 + 8*r8 + 32]
	addpd xmm1, xmm2
	movupd xmm2, xmmword ptr [r9 + 8*r8 + 48]
	addpd xmm2, xmm3
	movupd xmm3, xmmword ptr [r9 + 8*r8 + 64]
	addpd xmm3, xmm4
	movupd xmm4, xmmword ptr [r9 + 8*r8 + 80]
	addpd xmm4, xmm5
	movupd xmm5, xmmword ptr [r9 + 8*r8 + 96]
	addpd xmm5, xmm6
	movupd xmm6, xmmword ptr [r9 + 8*r8 + 112]
	addpd xmm6, xmm7
	movupd xmmword ptr [rsi + 8*r8], xmm8
	movupd xmmword ptr [rsi + 8*r8 + 16], xmm0
	movupd xmmword ptr [rsi + 8*r8 + 32], xmm1
	movupd xmmword ptr [rsi + 8*r8 + 48], xmm2
	movupd xmmword ptr [rsi + 8*r8 + 64], xmm3
	movupd xmmword ptr [rsi + 8*r8 + 80], xmm4
	movupd xmmword ptr [rsi + 8*r8 + 96], xmm5
	movupd xmmword ptr [rsi + 8*r8 + 112], xmm6
	add r8, 16
	cmp r8, rcx
	jb .LBB170_7
	mov r9, rax
	sub r9, r8
	ja .LBB170_10
	jmp .LBB170_22
.LBB170_9:
	xor r8d, r8d
	mov r9, rax
	sub r9, r8
	jbe .LBB170_22
.LBB170_10:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 12
	jb .LBB170_11
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -31
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -31
	setae r10b
	or r10b, dil
	je .LBB170_14
.LBB170_11:
	mov rdi, r8
.LBB170_17:
	mov r9d, eax
	sub r9d, edi
	mov r8, rdi
	and r9d, 3
	je .LBB170_20
	mov r8, rdi
.LBB170_19:
	movsd xmm0, qword ptr [rcx + 8*r8]
	addsd xmm0, qword ptr [rdx + 8*r8]
	movsd qword ptr [rsi + 8*r8], xmm0
	inc r8
	dec r9
	jne .LBB170_19
.LBB170_20:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB170_22
.LBB170_21:
	movsd xmm0, qword ptr [rcx + 8*r8]
	addsd xmm0, qword ptr [rdx + 8*r8]
	movsd qword ptr [rsi + 8*r8], xmm0
	movsd xmm0, qword ptr [rcx + 8*r8 + 8]
	addsd xmm0, qword ptr [rdx + 8*r8 + 8]
	movsd qword ptr [rsi + 8*r8 + 8], xmm0
	movsd xmm0, qword ptr [rcx + 8*r8 + 16]
	addsd xmm0, qword ptr [rdx + 8*r8 + 16]
	movsd qword ptr [rsi + 8*r8 + 16], xmm0
	movsd xmm0, qword ptr [rcx + 8*r8 + 24]
	addsd xmm0, qword ptr [rdx + 8*r8 + 24]
	movsd qword ptr [rsi + 8*r8 + 24], xmm0
	add r8, 4
	cmp rax, r8
	jne .LBB170_21
	jmp .LBB170_22
.LBB170_14:
	mov r10, r9
	and r10, -4
	lea rdi, [r8 + r10]
	lea r11, [rsi + 8*r8]
	add r11, 16
	lea rbx, [rdx + 8*r8]
	add rbx, 16
	lea r8, [rcx + 8*r8]
	add r8, 16
	xor r14d, r14d
.LBB170_15:
	movupd xmm0, xmmword ptr [r8 + 8*r14 - 16]
	movupd xmm1, xmmword ptr [r8 + 8*r14]
	movupd xmm2, xmmword ptr [rbx + 8*r14 - 16]
	addpd xmm2, xmm0
	movupd xmm0, xmmword ptr [rbx + 8*r14]
	addpd xmm0, xmm1
	movupd xmmword ptr [r11 + 8*r14 - 16], xmm2
	movupd xmmword ptr [r11 + 8*r14], xmm0
	add r14, 4
	cmp r10, r14
	jne .LBB170_15
	cmp r9, r10
	jne .LBB170_17
.LBB170_22:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	ret
.LBB170_23:
	mov qword ptr [rsp + 80], rbx
	lea rax, [rsp + 80]
	mov qword ptr [rsp + 32], rax
	lea rax, [rbx + 8]
	lea rcx, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.18]
	lea r8, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.20]
	lea r9, [rsp + 32]
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
	jmp .LBB170_1
