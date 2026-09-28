jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
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
	jne .LBB171_1
.LBB171_2:
	movzx eax, byte ptr [rbx]
	lea rcx, [rip + .LJTI171_0]
	movsxd rax, dword ptr [rcx + 4*rax]
	add rax, rcx
	jmp rax
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -32
	je .LBB171_4
	mov r8, qword ptr [rdi + 136]
	xor edx, edx
	movaps xmm0, xmmword ptr [rip + .LCPI171_0]
.LBB171_18:
	movups xmm1, xmmword ptr [r8 + 4*rdx]
	movups xmm2, xmmword ptr [r8 + 4*rdx + 16]
	movups xmm3, xmmword ptr [r8 + 4*rdx + 32]
	movups xmm4, xmmword ptr [r8 + 4*rdx + 48]
	movups xmm5, xmmword ptr [r8 + 4*rdx + 64]
	movups xmm6, xmmword ptr [r8 + 4*rdx + 80]
	movups xmm7, xmmword ptr [r8 + 4*rdx + 96]
	movups xmm8, xmmword ptr [r8 + 4*rdx + 112]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	xorps xmm3, xmm0
	xorps xmm4, xmm0
	xorps xmm5, xmm0
	xorps xmm6, xmm0
	xorps xmm7, xmm0
	xorps xmm8, xmm0
	movups xmmword ptr [rsi + 4*rdx], xmm1
	movups xmmword ptr [rsi + 4*rdx + 16], xmm2
	movups xmmword ptr [rsi + 4*rdx + 32], xmm3
	movups xmmword ptr [rsi + 4*rdx + 48], xmm4
	movups xmmword ptr [rsi + 4*rdx + 64], xmm5
	movups xmmword ptr [rsi + 4*rdx + 80], xmm6
	movups xmmword ptr [rsi + 4*rdx + 96], xmm7
	movups xmmword ptr [rsi + 4*rdx + 112], xmm8
	add rdx, 32
	cmp rdx, rcx
	jb .LBB171_18
	mov r8, rax
	sub r8, rdx
	jbe .LBB171_22
	jmp .LBB171_6
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>::{closure#3}, ()>
	jmp .LBB171_22
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>::{closure#2}, ()>
	jmp .LBB171_22
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
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>::{closure#1}, ()>
.LBB171_22:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	ret
.LBB171_4:
	xor edx, edx
	mov r8, rax
	sub r8, rdx
	jbe .LBB171_22
.LBB171_6:
	mov rcx, qword ptr [rdi + 136]
	cmp r8, 8
	setb dil
	mov r9, rcx
	sub r9, rsi
	cmp r9, -31
	setae r9b
	or r9b, dil
	je .LBB171_8
	mov rdi, rdx
	jmp .LBB171_11
.LBB171_8:
	mov r9, r8
	and r9, -8
	lea rdi, [rdx + r9]
	lea r10, [rsi + 4*rdx]
	add r10, 16
	lea rdx, [rcx + 4*rdx]
	add rdx, 16
	xor r11d, r11d
	movaps xmm0, xmmword ptr [rip + .LCPI171_0]
.LBB171_9:
	movups xmm1, xmmword ptr [rdx + 4*r11 - 16]
	movups xmm2, xmmword ptr [rdx + 4*r11]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [r10 + 4*r11 - 16], xmm1
	movups xmmword ptr [r10 + 4*r11], xmm2
	add r11, 8
	cmp r9, r11
	jne .LBB171_9
	cmp r8, r9
	je .LBB171_22
.LBB171_11:
	mov r8d, eax
	sub r8d, edi
	mov rdx, rdi
	and r8d, 3
	je .LBB171_14
	mov r9d, -2147483648
	mov rdx, rdi
.LBB171_13:
	mov r10d, dword ptr [rcx + 4*rdx]
	xor r10d, r9d
	mov dword ptr [rsi + 4*rdx], r10d
	inc rdx
	dec r8
	jne .LBB171_13
.LBB171_14:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB171_22
	mov edi, -2147483648
.LBB171_16:
	mov r8d, dword ptr [rcx + 4*rdx]
	xor r8d, edi
	mov dword ptr [rsi + 4*rdx], r8d
	mov r8d, dword ptr [rcx + 4*rdx + 4]
	xor r8d, edi
	mov dword ptr [rsi + 4*rdx + 4], r8d
	mov r8d, dword ptr [rcx + 4*rdx + 8]
	xor r8d, edi
	mov dword ptr [rsi + 4*rdx + 8], r8d
	mov r8d, dword ptr [rcx + 4*rdx + 12]
	xor r8d, edi
	mov dword ptr [rsi + 4*rdx + 12], r8d
	add rdx, 4
	cmp rax, rdx
	jne .LBB171_16
	jmp .LBB171_22
.LBB171_1:
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
	jmp .LBB171_2
