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
	lea rcx, [rip + .LJTI176_0]
	movsxd rax, dword ptr [rcx + 4*rax]
	add rax, rcx
	jmp rax
	mov rax, qword ptr [rsp + 16]
	mov rdx, rax
	and rdx, -32
	je .LBB176_10
	mov r8, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor ecx, ecx
.LBB176_4:
	movdqu xmm0, xmmword ptr [r8 + 4*rcx]
	movdqu xmm1, xmmword ptr [r8 + 4*rcx + 16]
	movdqu xmm3, xmmword ptr [r8 + 4*rcx + 32]
	movdqu xmm5, xmmword ptr [r8 + 4*rcx + 48]
	movdqu xmm7, xmmword ptr [r8 + 4*rcx + 64]
	movdqu xmm9, xmmword ptr [r8 + 4*rcx + 80]
	movdqu xmm10, xmmword ptr [r8 + 4*rcx + 96]
	movdqu xmm11, xmmword ptr [r8 + 4*rcx + 112]
	movdqu xmm12, xmmword ptr [r9 + 4*rcx]
	paddd xmm12, xmm0
	movdqu xmm2, xmmword ptr [r9 + 4*rcx + 16]
	paddd xmm2, xmm1
	movdqu xmm4, xmmword ptr [r9 + 4*rcx + 32]
	paddd xmm4, xmm3
	movdqu xmm6, xmmword ptr [r9 + 4*rcx + 48]
	paddd xmm6, xmm5
	movdqu xmm8, xmmword ptr [r9 + 4*rcx + 64]
	paddd xmm8, xmm7
	movdqu xmm7, xmmword ptr [r9 + 4*rcx + 80]
	paddd xmm7, xmm9
	movdqu xmm5, xmmword ptr [r9 + 4*rcx + 96]
	paddd xmm5, xmm10
	movdqu xmm3, xmmword ptr [r9 + 4*rcx + 112]
	paddd xmm3, xmm11
	movdqu xmm0, xmmword ptr [r10 + 4*rcx]
	movdqu xmm1, xmmword ptr [r10 + 4*rcx + 16]
	movdqu xmm10, xmmword ptr [r10 + 4*rcx + 32]
	movdqu xmm9, xmmword ptr [r10 + 4*rcx + 48]
	pshufd xmm11, xmm0, 245
	pmuludq xmm0, xmm12
	pshufd xmm0, xmm0, 232
	pshufd xmm12, xmm12, 245
	pmuludq xmm12, xmm11
	pshufd xmm11, xmm12, 232
	punpckldq xmm0, xmm11
	pshufd xmm11, xmm1, 245
	pmuludq xmm1, xmm2
	pshufd xmm1, xmm1, 232
	pshufd xmm2, xmm2, 245
	pmuludq xmm2, xmm11
	pshufd xmm2, xmm2, 232
	punpckldq xmm1, xmm2
	pshufd xmm11, xmm10, 245
	pmuludq xmm10, xmm4
	pshufd xmm2, xmm10, 232
	pshufd xmm4, xmm4, 245
	pmuludq xmm4, xmm11
	pshufd xmm4, xmm4, 232
	punpckldq xmm2, xmm4
	pshufd xmm10, xmm9, 245
	pmuludq xmm9, xmm6
	pshufd xmm4, xmm9, 232
	pshufd xmm6, xmm6, 245
	pmuludq xmm6, xmm10
	pshufd xmm6, xmm6, 232
	punpckldq xmm4, xmm6
	movdqu xmm6, xmmword ptr [r10 + 4*rcx + 64]
	pshufd xmm9, xmm6, 245
	pmuludq xmm6, xmm8
	pshufd xmm6, xmm6, 232
	pshufd xmm8, xmm8, 245
	pmuludq xmm9, xmm8
	pshufd xmm8, xmm9, 232
	punpckldq xmm6, xmm8
	movdqu xmm8, xmmword ptr [r10 + 4*rcx + 80]
	pshufd xmm9, xmm8, 245
	pmuludq xmm8, xmm7
	pshufd xmm8, xmm8, 232
	pshufd xmm7, xmm7, 245
	pmuludq xmm9, xmm7
	pshufd xmm7, xmm9, 232
	punpckldq xmm8, xmm7
	movdqu xmm7, xmmword ptr [r10 + 4*rcx + 96]
	pshufd xmm9, xmm7, 245
	pmuludq xmm7, xmm5
	pshufd xmm7, xmm7, 232
	pshufd xmm5, xmm5, 245
	pmuludq xmm9, xmm5
	pshufd xmm5, xmm9, 232
	punpckldq xmm7, xmm5
	movdqu xmm5, xmmword ptr [r10 + 4*rcx + 112]
	pshufd xmm9, xmm5, 245
	pmuludq xmm5, xmm3
	pshufd xmm5, xmm5, 232
	pshufd xmm3, xmm3, 245
	pmuludq xmm9, xmm3
	pshufd xmm3, xmm9, 232
	punpckldq xmm5, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx]
	psubd xmm0, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx + 16]
	psubd xmm1, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx + 32]
	psubd xmm2, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx + 48]
	psubd xmm4, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx + 64]
	psubd xmm6, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx + 80]
	psubd xmm8, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx + 96]
	psubd xmm7, xmm3
	movdqu xmm3, xmmword ptr [r11 + 4*rcx + 112]
	psubd xmm5, xmm3
	movdqu xmmword ptr [rsi + 4*rcx], xmm0
	movdqu xmmword ptr [rsi + 4*rcx + 16], xmm1
	movdqu xmmword ptr [rsi + 4*rcx + 32], xmm2
	movdqu xmmword ptr [rsi + 4*rcx + 48], xmm4
	movdqu xmmword ptr [rsi + 4*rcx + 64], xmm6
	movdqu xmmword ptr [rsi + 4*rcx + 80], xmm8
	movdqu xmmword ptr [rsi + 4*rcx + 96], xmm7
	movdqu xmmword ptr [rsi + 4*rcx + 112], xmm5
	add rcx, 32
	cmp rcx, rdx
	jb .LBB176_4
	mov r11, rax
	sub r11, rcx
	jbe .LBB176_9
	jmp .LBB176_11
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#3}, ()>
	jmp .LBB176_9
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#2}, ()>
	jmp .LBB176_9
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
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#1}, ()>
.LBB176_9:
	add rsp, 88
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	ret
.LBB176_10:
	xor ecx, ecx
	mov r11, rax
	sub r11, rcx
	jbe .LBB176_9
.LBB176_11:
	mov rdx, qword ptr [rdi + 136]
	mov r8, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r11, 16
	jb .LBB176_12
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
	je .LBB176_15
.LBB176_12:
	mov r10, rcx
.LBB176_18:
	mov r11d, eax
	sub r11d, r10d
	lea rcx, [r10 + 1]
	test r11b, 1
	je .LBB176_20
	mov r11d, dword ptr [r8 + 4*r10]
	add r11d, dword ptr [rdx + 4*r10]
	imul r11d, dword ptr [r9 + 4*r10]
	sub r11d, dword ptr [rdi + 4*r10]
	mov dword ptr [rsi + 4*r10], r11d
	mov r10, rcx
.LBB176_20:
	cmp rax, rcx
	je .LBB176_9
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
.LBB176_22:
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
	jne .LBB176_22
	jmp .LBB176_9
.LBB176_15:
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
.LBB176_16:
	movdqu xmm0, xmmword ptr [rcx + 4*rbp - 16]
	movdqu xmm1, xmmword ptr [rcx + 4*rbp]
	movdqu xmm2, xmmword ptr [r13 + 4*rbp - 16]
	paddd xmm2, xmm0
	movdqu xmm0, xmmword ptr [r13 + 4*rbp]
	paddd xmm0, xmm1
	movdqu xmm1, xmmword ptr [r12 + 4*rbp - 16]
	movdqu xmm3, xmmword ptr [r12 + 4*rbp]
	movdqu xmm4, xmmword ptr [r15 + 4*rbp - 16]
	movdqu xmm5, xmmword ptr [r15 + 4*rbp]
	pshufd xmm6, xmm1, 245
	pmuludq xmm1, xmm2
	pshufd xmm1, xmm1, 232
	pshufd xmm2, xmm2, 245
	pmuludq xmm2, xmm6
	pshufd xmm2, xmm2, 232
	punpckldq xmm1, xmm2
	psubd xmm1, xmm4
	pshufd xmm2, xmm3, 245
	pmuludq xmm3, xmm0
	pshufd xmm3, xmm3, 232
	pshufd xmm0, xmm0, 245
	pmuludq xmm0, xmm2
	pshufd xmm0, xmm0, 232
	punpckldq xmm3, xmm0
	psubd xmm3, xmm5
	movdqu xmmword ptr [r14 + 4*rbp - 16], xmm1
	movdqu xmmword ptr [r14 + 4*rbp], xmm3
	add rbp, 8
	cmp rbx, rbp
	jne .LBB176_16
	cmp r11, rbx
	je .LBB176_9
	jmp .LBB176_18
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
