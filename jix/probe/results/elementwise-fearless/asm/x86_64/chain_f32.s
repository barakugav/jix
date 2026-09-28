jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
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
	jne .LBB172_23
.LBB172_1:
	movzx eax, byte ptr [rbx]
	lea rcx, [rip + .LJTI172_0]
	movsxd rax, dword ptr [rcx + 4*rax]
	add rax, rcx
	jmp rax
	mov rax, qword ptr [rsp + 16]
	mov rdx, rax
	and rdx, -32
	je .LBB172_10
	mov r8, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor ecx, ecx
.LBB172_4:
	movups xmm0, xmmword ptr [r8 + 4*rcx]
	movups xmm1, xmmword ptr [r8 + 4*rcx + 16]
	movups xmm2, xmmword ptr [r8 + 4*rcx + 32]
	movups xmm3, xmmword ptr [r8 + 4*rcx + 48]
	movups xmm4, xmmword ptr [r8 + 4*rcx + 64]
	movups xmm5, xmmword ptr [r8 + 4*rcx + 80]
	movups xmm6, xmmword ptr [r8 + 4*rcx + 96]
	movups xmm7, xmmword ptr [r8 + 4*rcx + 112]
	movups xmm8, xmmword ptr [r9 + 4*rcx]
	addps xmm8, xmm0
	movups xmm9, xmmword ptr [r9 + 4*rcx + 16]
	addps xmm9, xmm1
	movups xmm10, xmmword ptr [r9 + 4*rcx + 32]
	addps xmm10, xmm2
	movups xmm11, xmmword ptr [r9 + 4*rcx + 48]
	addps xmm11, xmm3
	movups xmm12, xmmword ptr [r9 + 4*rcx + 64]
	addps xmm12, xmm4
	movups xmm13, xmmword ptr [r9 + 4*rcx + 80]
	addps xmm13, xmm5
	movups xmm14, xmmword ptr [r9 + 4*rcx + 96]
	addps xmm14, xmm6
	movups xmm15, xmmword ptr [r9 + 4*rcx + 112]
	addps xmm15, xmm7
	movups xmm0, xmmword ptr [r10 + 4*rcx]
	mulps xmm0, xmm8
	movups xmm1, xmmword ptr [r10 + 4*rcx + 16]
	mulps xmm1, xmm9
	movups xmm2, xmmword ptr [r10 + 4*rcx + 32]
	mulps xmm2, xmm10
	movups xmm3, xmmword ptr [r10 + 4*rcx + 48]
	mulps xmm3, xmm11
	movups xmm4, xmmword ptr [r10 + 4*rcx + 64]
	mulps xmm4, xmm12
	movups xmm5, xmmword ptr [r10 + 4*rcx + 80]
	mulps xmm5, xmm13
	movups xmm6, xmmword ptr [r10 + 4*rcx + 96]
	mulps xmm6, xmm14
	movups xmm7, xmmword ptr [r10 + 4*rcx + 112]
	mulps xmm7, xmm15
	movups xmm8, xmmword ptr [r11 + 4*rcx]
	subps xmm0, xmm8
	movups xmm8, xmmword ptr [r11 + 4*rcx + 16]
	subps xmm1, xmm8
	movups xmm8, xmmword ptr [r11 + 4*rcx + 32]
	subps xmm2, xmm8
	movups xmm8, xmmword ptr [r11 + 4*rcx + 48]
	subps xmm3, xmm8
	movups xmm8, xmmword ptr [r11 + 4*rcx + 64]
	subps xmm4, xmm8
	movups xmm8, xmmword ptr [r11 + 4*rcx + 80]
	subps xmm5, xmm8
	movups xmm8, xmmword ptr [r11 + 4*rcx + 96]
	subps xmm6, xmm8
	movups xmm8, xmmword ptr [r11 + 4*rcx + 112]
	subps xmm7, xmm8
	movups xmmword ptr [rsi + 4*rcx], xmm0
	movups xmmword ptr [rsi + 4*rcx + 16], xmm1
	movups xmmword ptr [rsi + 4*rcx + 32], xmm2
	movups xmmword ptr [rsi + 4*rcx + 48], xmm3
	movups xmmword ptr [rsi + 4*rcx + 64], xmm4
	movups xmmword ptr [rsi + 4*rcx + 80], xmm5
	movups xmmword ptr [rsi + 4*rcx + 96], xmm6
	movups xmmword ptr [rsi + 4*rcx + 112], xmm7
	add rcx, 32
	cmp rcx, rdx
	jb .LBB172_4
	mov r11, rax
	sub r11, rcx
	jbe .LBB172_9
	jmp .LBB172_11
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>::{closure#3}, ()>
	jmp .LBB172_9
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>::{closure#2}, ()>
	jmp .LBB172_9
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
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>::{closure#1}, ()>
.LBB172_9:
	add rsp, 88
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	ret
.LBB172_10:
	xor ecx, ecx
	mov r11, rax
	sub r11, rcx
	jbe .LBB172_9
.LBB172_11:
	mov rdx, qword ptr [rdi + 136]
	mov r8, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r11, 12
	jb .LBB172_12
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
	je .LBB172_15
.LBB172_12:
	mov r10, rcx
.LBB172_18:
	mov r11d, eax
	sub r11d, r10d
	lea rcx, [r10 + 1]
	test r11b, 1
	je .LBB172_20
	movss xmm0, dword ptr [rdx + 4*r10]
	addss xmm0, dword ptr [r8 + 4*r10]
	mulss xmm0, dword ptr [r9 + 4*r10]
	subss xmm0, dword ptr [rdi + 4*r10]
	movss dword ptr [rsi + 4*r10], xmm0
	mov r10, rcx
.LBB172_20:
	cmp rax, rcx
	je .LBB172_9
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
.LBB172_22:
	movss xmm0, dword ptr [rdx + 4*r9 - 4]
	addss xmm0, dword ptr [r8 + 4*r9 - 4]
	mulss xmm0, dword ptr [rdi + 4*r9 - 4]
	subss xmm0, dword ptr [rsi + 4*r9 - 4]
	movss dword ptr [rcx + 4*r9 - 4], xmm0
	movss xmm0, dword ptr [rdx + 4*r9]
	addss xmm0, dword ptr [r8 + 4*r9]
	mulss xmm0, dword ptr [rdi + 4*r9]
	subss xmm0, dword ptr [rsi + 4*r9]
	movss dword ptr [rcx + 4*r9], xmm0
	add r9, 2
	cmp rax, r9
	jne .LBB172_22
	jmp .LBB172_9
.LBB172_15:
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
.LBB172_16:
	movups xmm0, xmmword ptr [rcx + 4*rbp - 16]
	movups xmm1, xmmword ptr [rcx + 4*rbp]
	movups xmm2, xmmword ptr [r13 + 4*rbp - 16]
	addps xmm2, xmm0
	movups xmm0, xmmword ptr [r13 + 4*rbp]
	addps xmm0, xmm1
	movups xmm1, xmmword ptr [r12 + 4*rbp - 16]
	mulps xmm1, xmm2
	movups xmm2, xmmword ptr [r12 + 4*rbp]
	mulps xmm2, xmm0
	movups xmm0, xmmword ptr [r15 + 4*rbp - 16]
	subps xmm1, xmm0
	movups xmm0, xmmword ptr [r15 + 4*rbp]
	subps xmm2, xmm0
	movups xmmword ptr [r14 + 4*rbp - 16], xmm1
	movups xmmword ptr [r14 + 4*rbp], xmm2
	add rbp, 8
	cmp rbx, rbp
	jne .LBB172_16
	cmp r11, rbx
	je .LBB172_9
	jmp .LBB172_18
.LBB172_23:
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
	jmp .LBB172_1
