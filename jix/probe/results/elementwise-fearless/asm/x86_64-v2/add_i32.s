jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
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
	jne .LBB178_23
.LBB178_1:
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
	je .LBB178_5
	movzx eax, al
	cmp eax, 2
	je .LBB178_4
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>::{closure#1}, ()>
	jmp .LBB178_22
.LBB178_4:
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>::{closure#2}, ()>
	jmp .LBB178_22
.LBB178_5:
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -32
	je .LBB178_9
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor r8d, r8d
.LBB178_7:
	movdqu xmm0, xmmword ptr [rdx + 4*r8]
	movdqu xmm1, xmmword ptr [rdx + 4*r8 + 16]
	movdqu xmm2, xmmword ptr [rdx + 4*r8 + 32]
	movdqu xmm3, xmmword ptr [rdx + 4*r8 + 48]
	movdqu xmm4, xmmword ptr [rdx + 4*r8 + 64]
	movdqu xmm5, xmmword ptr [rdx + 4*r8 + 80]
	movdqu xmm6, xmmword ptr [rdx + 4*r8 + 96]
	movdqu xmm7, xmmword ptr [rdx + 4*r8 + 112]
	movdqu xmm8, xmmword ptr [r9 + 4*r8]
	paddd xmm8, xmm0
	movdqu xmm0, xmmword ptr [r9 + 4*r8 + 16]
	paddd xmm0, xmm1
	movdqu xmm1, xmmword ptr [r9 + 4*r8 + 32]
	paddd xmm1, xmm2
	movdqu xmm2, xmmword ptr [r9 + 4*r8 + 48]
	paddd xmm2, xmm3
	movdqu xmm3, xmmword ptr [r9 + 4*r8 + 64]
	paddd xmm3, xmm4
	movdqu xmm4, xmmword ptr [r9 + 4*r8 + 80]
	paddd xmm4, xmm5
	movdqu xmm5, xmmword ptr [r9 + 4*r8 + 96]
	paddd xmm5, xmm6
	movdqu xmm6, xmmword ptr [r9 + 4*r8 + 112]
	paddd xmm6, xmm7
	movdqu xmmword ptr [rsi + 4*r8], xmm8
	movdqu xmmword ptr [rsi + 4*r8 + 16], xmm0
	movdqu xmmword ptr [rsi + 4*r8 + 32], xmm1
	movdqu xmmword ptr [rsi + 4*r8 + 48], xmm2
	movdqu xmmword ptr [rsi + 4*r8 + 64], xmm3
	movdqu xmmword ptr [rsi + 4*r8 + 80], xmm4
	movdqu xmmword ptr [rsi + 4*r8 + 96], xmm5
	movdqu xmmword ptr [rsi + 4*r8 + 112], xmm6
	add r8, 32
	cmp r8, rcx
	jb .LBB178_7
	mov r9, rax
	sub r9, r8
	ja .LBB178_10
	jmp .LBB178_22
.LBB178_9:
	xor r8d, r8d
	mov r9, rax
	sub r9, r8
	jbe .LBB178_22
.LBB178_10:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 12
	jb .LBB178_11
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -31
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -31
	setae r10b
	or r10b, dil
	je .LBB178_14
.LBB178_11:
	mov rdi, r8
.LBB178_17:
	mov r9d, eax
	sub r9d, edi
	mov r8, rdi
	and r9d, 3
	je .LBB178_20
	mov r8, rdi
.LBB178_19:
	mov r10d, dword ptr [rdx + 4*r8]
	add r10d, dword ptr [rcx + 4*r8]
	mov dword ptr [rsi + 4*r8], r10d
	inc r8
	dec r9
	jne .LBB178_19
.LBB178_20:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB178_22
.LBB178_21:
	mov edi, dword ptr [rdx + 4*r8]
	add edi, dword ptr [rcx + 4*r8]
	mov dword ptr [rsi + 4*r8], edi
	mov edi, dword ptr [rdx + 4*r8 + 4]
	add edi, dword ptr [rcx + 4*r8 + 4]
	mov dword ptr [rsi + 4*r8 + 4], edi
	mov edi, dword ptr [rdx + 4*r8 + 8]
	add edi, dword ptr [rcx + 4*r8 + 8]
	mov dword ptr [rsi + 4*r8 + 8], edi
	mov edi, dword ptr [rdx + 4*r8 + 12]
	add edi, dword ptr [rcx + 4*r8 + 12]
	mov dword ptr [rsi + 4*r8 + 12], edi
	add r8, 4
	cmp rax, r8
	jne .LBB178_21
	jmp .LBB178_22
.LBB178_14:
	mov r10, r9
	and r10, -8
	lea rdi, [r8 + r10]
	lea r11, [rsi + 4*r8]
	add r11, 16
	lea rbx, [rdx + 4*r8]
	add rbx, 16
	lea r8, [rcx + 4*r8]
	add r8, 16
	xor r14d, r14d
.LBB178_15:
	movdqu xmm0, xmmword ptr [r8 + 4*r14 - 16]
	movdqu xmm1, xmmword ptr [r8 + 4*r14]
	movdqu xmm2, xmmword ptr [rbx + 4*r14 - 16]
	paddd xmm2, xmm0
	movdqu xmm0, xmmword ptr [rbx + 4*r14]
	paddd xmm0, xmm1
	movdqu xmmword ptr [r11 + 4*r14 - 16], xmm2
	movdqu xmmword ptr [r11 + 4*r14], xmm0
	add r14, 8
	cmp r10, r14
	jne .LBB178_15
	cmp r9, r10
	jne .LBB178_17
.LBB178_22:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	ret
.LBB178_23:
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
	jmp .LBB178_1
