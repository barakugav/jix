jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
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
	jne .LBB175_1
.LBB175_2:
	movzx eax, byte ptr [rbx]
	lea rcx, [rip + .LJTI175_0]
	movsxd rax, dword ptr [rcx + 4*rax]
	add rax, rcx
	jmp rax
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -32
	je .LBB175_4
	mov r8, qword ptr [rdi + 136]
	xor edx, edx
.LBB175_17:
	movdqu xmm0, xmmword ptr [r8 + 4*rdx]
	movdqu xmm1, xmmword ptr [r8 + 4*rdx + 16]
	movdqu xmm2, xmmword ptr [r8 + 4*rdx + 32]
	movdqu xmm3, xmmword ptr [r8 + 4*rdx + 48]
	movdqu xmm4, xmmword ptr [r8 + 4*rdx + 64]
	movdqu xmm5, xmmword ptr [r8 + 4*rdx + 80]
	movdqu xmm6, xmmword ptr [r8 + 4*rdx + 96]
	movdqu xmm7, xmmword ptr [r8 + 4*rdx + 112]
	pxor xmm8, xmm8
	psubd xmm8, xmm0
	pxor xmm0, xmm0
	psubd xmm0, xmm1
	pxor xmm1, xmm1
	psubd xmm1, xmm2
	pxor xmm2, xmm2
	psubd xmm2, xmm3
	pxor xmm3, xmm3
	psubd xmm3, xmm4
	pxor xmm4, xmm4
	psubd xmm4, xmm5
	pxor xmm5, xmm5
	psubd xmm5, xmm6
	pxor xmm6, xmm6
	psubd xmm6, xmm7
	movdqu xmmword ptr [rsi + 4*rdx], xmm8
	movdqu xmmword ptr [rsi + 4*rdx + 16], xmm0
	movdqu xmmword ptr [rsi + 4*rdx + 32], xmm1
	movdqu xmmword ptr [rsi + 4*rdx + 48], xmm2
	movdqu xmmword ptr [rsi + 4*rdx + 64], xmm3
	movdqu xmmword ptr [rsi + 4*rdx + 80], xmm4
	movdqu xmmword ptr [rsi + 4*rdx + 96], xmm5
	movdqu xmmword ptr [rsi + 4*rdx + 112], xmm6
	add rdx, 32
	cmp rdx, rcx
	jb .LBB175_17
	mov r8, rax
	sub r8, rdx
	jbe .LBB175_21
	jmp .LBB175_6
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>::{closure#3}, ()>
	jmp .LBB175_21
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>::{closure#2}, ()>
	jmp .LBB175_21
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
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>::{closure#1}, ()>
.LBB175_21:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	ret
.LBB175_4:
	xor edx, edx
	mov r8, rax
	sub r8, rdx
	jbe .LBB175_21
.LBB175_6:
	mov rcx, qword ptr [rdi + 136]
	cmp r8, 8
	setb dil
	mov r9, rcx
	sub r9, rsi
	cmp r9, -31
	setae r9b
	or r9b, dil
	je .LBB175_8
	mov rdi, rdx
	jmp .LBB175_11
.LBB175_8:
	mov r9, r8
	and r9, -8
	lea rdi, [rdx + r9]
	lea r10, [rsi + 4*rdx]
	add r10, 16
	lea rdx, [rcx + 4*rdx]
	add rdx, 16
	xor r11d, r11d
.LBB175_9:
	movdqu xmm0, xmmword ptr [rdx + 4*r11 - 16]
	movdqu xmm1, xmmword ptr [rdx + 4*r11]
	pxor xmm2, xmm2
	psubd xmm2, xmm0
	pxor xmm0, xmm0
	psubd xmm0, xmm1
	movdqu xmmword ptr [r10 + 4*r11 - 16], xmm2
	movdqu xmmword ptr [r10 + 4*r11], xmm0
	add r11, 8
	cmp r9, r11
	jne .LBB175_9
	cmp r8, r9
	je .LBB175_21
.LBB175_11:
	mov r8d, eax
	sub r8d, edi
	mov rdx, rdi
	and r8d, 3
	je .LBB175_14
	mov rdx, rdi
.LBB175_13:
	xor r9d, r9d
	sub r9d, dword ptr [rcx + 4*rdx]
	mov dword ptr [rsi + 4*rdx], r9d
	inc rdx
	dec r8
	jne .LBB175_13
.LBB175_14:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB175_21
.LBB175_15:
	xor edi, edi
	sub edi, dword ptr [rcx + 4*rdx]
	mov dword ptr [rsi + 4*rdx], edi
	xor edi, edi
	sub edi, dword ptr [rcx + 4*rdx + 4]
	mov dword ptr [rsi + 4*rdx + 4], edi
	xor edi, edi
	sub edi, dword ptr [rcx + 4*rdx + 8]
	mov dword ptr [rsi + 4*rdx + 8], edi
	xor edi, edi
	sub edi, dword ptr [rcx + 4*rdx + 12]
	mov dword ptr [rsi + 4*rdx + 12], edi
	add rdx, 4
	cmp rax, rdx
	jne .LBB175_15
	jmp .LBB175_21
.LBB175_1:
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
	jmp .LBB175_2
