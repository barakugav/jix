jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	sub rsp, 88
	mov qword ptr [rsp + 24], rcx
	mov qword ptr [rsp + 16], r8
	mov rbx, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov eax, dword ptr [rbx + 8]
	test eax, eax
	jne .LBB176_23
.LBB176_1:
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
	je .LBB176_5
	movzx eax, al
	cmp eax, 2
	je .LBB176_4
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#1}, ()>
	jmp .LBB176_22
.LBB176_4:
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#2}, ()>
	jmp .LBB176_22
.LBB176_5:
	mov rax, qword ptr [rsp + 16]
	mov rdx, rax
	and rdx, -32
	je .LBB176_9
	mov r8, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor ecx, ecx
.LBB176_7:
	movdqu xmm0, xmmword ptr [r8 + 4*rcx]
	movdqu xmm1, xmmword ptr [r8 + 4*rcx + 16]
	movdqu xmm2, xmmword ptr [r8 + 4*rcx + 32]
	movdqu xmm3, xmmword ptr [r8 + 4*rcx + 48]
	movdqu xmm4, xmmword ptr [r8 + 4*rcx + 64]
	movdqu xmm5, xmmword ptr [r8 + 4*rcx + 80]
	movdqu xmm6, xmmword ptr [r8 + 4*rcx + 96]
	movdqu xmm7, xmmword ptr [r8 + 4*rcx + 112]
	movdqu xmm8, xmmword ptr [r9 + 4*rcx]
	paddd xmm8, xmm0
	movdqu xmm9, xmmword ptr [r9 + 4*rcx + 16]
	paddd xmm9, xmm1
	movdqu xmm10, xmmword ptr [r9 + 4*rcx + 32]
	paddd xmm10, xmm2
	movdqu xmm11, xmmword ptr [r9 + 4*rcx + 48]
	paddd xmm11, xmm3
	movdqu xmm12, xmmword ptr [r9 + 4*rcx + 64]
	paddd xmm12, xmm4
	movdqu xmm13, xmmword ptr [r9 + 4*rcx + 80]
	paddd xmm13, xmm5
	movdqu xmm14, xmmword ptr [r9 + 4*rcx + 96]
	paddd xmm14, xmm6
	movdqu xmm15, xmmword ptr [r9 + 4*rcx + 112]
	paddd xmm15, xmm7
	movdqu xmm0, xmmword ptr [r10 + 4*rcx]
	pmulld xmm0, xmm8
	movdqu xmm1, xmmword ptr [r10 + 4*rcx + 16]
	pmulld xmm1, xmm9
	movdqu xmm2, xmmword ptr [r10 + 4*rcx + 32]
	pmulld xmm2, xmm10
	movdqu xmm3, xmmword ptr [r10 + 4*rcx + 48]
	pmulld xmm3, xmm11
	movdqu xmm4, xmmword ptr [r10 + 4*rcx + 64]
	pmulld xmm4, xmm12
	movdqu xmm5, xmmword ptr [r10 + 4*rcx + 80]
	pmulld xmm5, xmm13
	movdqu xmm6, xmmword ptr [r10 + 4*rcx + 96]
	pmulld xmm6, xmm14
	movdqu xmm7, xmmword ptr [r10 + 4*rcx + 112]
	pmulld xmm7, xmm15
	movdqu xmm8, xmmword ptr [r11 + 4*rcx]
	psubd xmm0, xmm8
	movdqu xmm8, xmmword ptr [r11 + 4*rcx + 16]
	psubd xmm1, xmm8
	movdqu xmm8, xmmword ptr [r11 + 4*rcx + 32]
	psubd xmm2, xmm8
	movdqu xmm8, xmmword ptr [r11 + 4*rcx + 48]
	psubd xmm3, xmm8
	movdqu xmm8, xmmword ptr [r11 + 4*rcx + 64]
	psubd xmm4, xmm8
	movdqu xmm8, xmmword ptr [r11 + 4*rcx + 80]
	psubd xmm5, xmm8
	movdqu xmm8, xmmword ptr [r11 + 4*rcx + 96]
	psubd xmm6, xmm8
	movdqu xmm8, xmmword ptr [r11 + 4*rcx + 112]
	psubd xmm7, xmm8
	movdqu xmmword ptr [rsi + 4*rcx], xmm0
	movdqu xmmword ptr [rsi + 4*rcx + 16], xmm1
	movdqu xmmword ptr [rsi + 4*rcx + 32], xmm2
	movdqu xmmword ptr [rsi + 4*rcx + 48], xmm3
	movdqu xmmword ptr [rsi + 4*rcx + 64], xmm4
	movdqu xmmword ptr [rsi + 4*rcx + 80], xmm5
	movdqu xmmword ptr [rsi + 4*rcx + 96], xmm6
	movdqu xmmword ptr [rsi + 4*rcx + 112], xmm7
	add rcx, 32
	cmp rcx, rdx
	jb .LBB176_7
	mov r11, rax
	sub r11, rcx
	ja .LBB176_10
	jmp .LBB176_22
.LBB176_9:
	xor ecx, ecx
	mov r11, rax
	sub r11, rcx
	jbe .LBB176_22
.LBB176_10:
	mov rdx, qword ptr [rdi + 136]
	mov r8, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r11, 16
	jb .LBB176_11
	mov r10, rdx
	sub r10, rsi
	cmp r10, -31
	setae r10b
	mov rbx, r8
	sub rbx, rsi
	cmp rbx, -31
	setae bl
	or bl, r10b
	mov r10, r9
	sub r10, rsi
	cmp r10, -31
	setae r10b
	mov r14, rdi
	sub r14, rsi
	cmp r14, -31
	setae bpl
	or bpl, r10b
	or bpl, bl
	je .LBB176_14
.LBB176_11:
	mov r10, rcx
.LBB176_17:
	mov r11d, eax
	sub r11d, r10d
	lea rcx, [r10 + 1]
	test r11b, 1
	je .LBB176_19
	mov r11d, dword ptr [r8 + 4*r10]
	add r11d, dword ptr [rdx + 4*r10]
	imul r11d, dword ptr [r9 + 4*r10]
	sub r11d, dword ptr [rdi + 4*r10]
	mov dword ptr [rsi + 4*r10], r11d
	mov r10, rcx
.LBB176_19:
	cmp rax, rcx
	je .LBB176_22
	sub rax, r10
	lea rcx, [rsi + 4*r10]
	add rcx, 4
	lea rsi, [rdi + 4*r10 + 4]
	lea rdi, [r9 + 4*r10]
	add rdi, 4
	lea r8, [r8 + 4*r10]
	add r8, 4
	lea rdx, [rdx + 4*r10]
	add rdx, 4
	xor r9d, r9d
.LBB176_21:
	mov r10d, dword ptr [r8 + 4*r9 - 4]
	add r10d, dword ptr [rdx + 4*r9 - 4]
	imul r10d, dword ptr [rdi + 4*r9 - 4]
	sub r10d, dword ptr [rsi + 4*r9 - 4]
	mov dword ptr [rcx + 4*r9 - 4], r10d
	mov r10d, dword ptr [r8 + 4*r9]
	add r10d, dword ptr [rdx + 4*r9]
	imul r10d, dword ptr [rdi + 4*r9]
	sub r10d, dword ptr [rsi + 4*r9]
	mov dword ptr [rcx + 4*r9], r10d
	add r9, 2
	cmp rax, r9
	jne .LBB176_21
	jmp .LBB176_22
.LBB176_14:
	mov rbx, r11
	and rbx, -8
	lea r10, [rcx + rbx]
	lea r14, [rsi + 4*rcx]
	add r14, 16
	lea r15, [rdi + 4*rcx + 16]
	lea r12, [r9 + 4*rcx]
	add r12, 16
	lea r13, [r8 + 4*rcx]
	add r13, 16
	lea rcx, [rdx + 4*rcx]
	add rcx, 16
	xor ebp, ebp
.LBB176_15:
	movdqu xmm0, xmmword ptr [rcx + 4*rbp - 16]
	movdqu xmm1, xmmword ptr [rcx + 4*rbp]
	movdqu xmm2, xmmword ptr [r13 + 4*rbp - 16]
	paddd xmm2, xmm0
	movdqu xmm0, xmmword ptr [r13 + 4*rbp]
	paddd xmm0, xmm1
	movdqu xmm1, xmmword ptr [r12 + 4*rbp - 16]
	pmulld xmm1, xmm2
	movdqu xmm2, xmmword ptr [r12 + 4*rbp]
	pmulld xmm2, xmm0
	movdqu xmm0, xmmword ptr [r15 + 4*rbp - 16]
	psubd xmm1, xmm0
	movdqu xmm0, xmmword ptr [r15 + 4*rbp]
	psubd xmm2, xmm0
	movdqu xmmword ptr [r14 + 4*rbp - 16], xmm1
	movdqu xmmword ptr [r14 + 4*rbp], xmm2
	add rbp, 8
	cmp rbx, rbp
	jne .LBB176_15
	cmp r11, rbx
	jne .LBB176_17
.LBB176_22:
	add rsp, 88
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	ret
.LBB176_23:
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
	jmp .LBB176_1
