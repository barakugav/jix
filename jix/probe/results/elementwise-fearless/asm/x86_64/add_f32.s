jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>:
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
	jne .LBB174_23
.LBB174_1:
	movzx eax, byte ptr [rbx]
	lea rcx, [rip + .LJTI174_0]
	movsxd rax, dword ptr [rcx + 4*rax]
	add rax, rcx
	jmp rax
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -32
	je .LBB174_10
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor r8d, r8d
.LBB174_4:
	movups xmm0, xmmword ptr [rdx + 4*r8]
	movups xmm1, xmmword ptr [rdx + 4*r8 + 16]
	movups xmm2, xmmword ptr [rdx + 4*r8 + 32]
	movups xmm3, xmmword ptr [rdx + 4*r8 + 48]
	movups xmm4, xmmword ptr [rdx + 4*r8 + 64]
	movups xmm5, xmmword ptr [rdx + 4*r8 + 80]
	movups xmm6, xmmword ptr [rdx + 4*r8 + 96]
	movups xmm7, xmmword ptr [rdx + 4*r8 + 112]
	movups xmm8, xmmword ptr [r9 + 4*r8]
	addps xmm8, xmm0
	movups xmm0, xmmword ptr [r9 + 4*r8 + 16]
	addps xmm0, xmm1
	movups xmm1, xmmword ptr [r9 + 4*r8 + 32]
	addps xmm1, xmm2
	movups xmm2, xmmword ptr [r9 + 4*r8 + 48]
	addps xmm2, xmm3
	movups xmm3, xmmword ptr [r9 + 4*r8 + 64]
	addps xmm3, xmm4
	movups xmm4, xmmword ptr [r9 + 4*r8 + 80]
	addps xmm4, xmm5
	movups xmm5, xmmword ptr [r9 + 4*r8 + 96]
	addps xmm5, xmm6
	movups xmm6, xmmword ptr [r9 + 4*r8 + 112]
	addps xmm6, xmm7
	movups xmmword ptr [rsi + 4*r8], xmm8
	movups xmmword ptr [rsi + 4*r8 + 16], xmm0
	movups xmmword ptr [rsi + 4*r8 + 32], xmm1
	movups xmmword ptr [rsi + 4*r8 + 48], xmm2
	movups xmmword ptr [rsi + 4*r8 + 64], xmm3
	movups xmmword ptr [rsi + 4*r8 + 80], xmm4
	movups xmmword ptr [rsi + 4*r8 + 96], xmm5
	movups xmmword ptr [rsi + 4*r8 + 112], xmm6
	add r8, 32
	cmp r8, rcx
	jb .LBB174_4
	mov r9, rax
	sub r9, r8
	jbe .LBB174_9
	jmp .LBB174_11
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>::{closure#3}, ()>
	jmp .LBB174_9
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>::{closure#2}, ()>
	jmp .LBB174_9
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
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>::{closure#1}, ()>
.LBB174_9:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	ret
.LBB174_10:
	xor r8d, r8d
	mov r9, rax
	sub r9, r8
	jbe .LBB174_9
.LBB174_11:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 12
	jb .LBB174_12
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -31
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -31
	setae r10b
	or r10b, dil
	je .LBB174_15
.LBB174_12:
	mov rdi, r8
.LBB174_18:
	mov r9d, eax
	sub r9d, edi
	mov r8, rdi
	and r9d, 3
	je .LBB174_21
	mov r8, rdi
.LBB174_20:
	movss xmm0, dword ptr [rcx + 4*r8]
	addss xmm0, dword ptr [rdx + 4*r8]
	movss dword ptr [rsi + 4*r8], xmm0
	inc r8
	dec r9
	jne .LBB174_20
.LBB174_21:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB174_9
.LBB174_22:
	movss xmm0, dword ptr [rcx + 4*r8]
	addss xmm0, dword ptr [rdx + 4*r8]
	movss dword ptr [rsi + 4*r8], xmm0
	movss xmm0, dword ptr [rcx + 4*r8 + 4]
	addss xmm0, dword ptr [rdx + 4*r8 + 4]
	movss dword ptr [rsi + 4*r8 + 4], xmm0
	movss xmm0, dword ptr [rcx + 4*r8 + 8]
	addss xmm0, dword ptr [rdx + 4*r8 + 8]
	movss dword ptr [rsi + 4*r8 + 8], xmm0
	movss xmm0, dword ptr [rcx + 4*r8 + 12]
	addss xmm0, dword ptr [rdx + 4*r8 + 12]
	movss dword ptr [rsi + 4*r8 + 12], xmm0
	add r8, 4
	cmp rax, r8
	jne .LBB174_22
	jmp .LBB174_9
.LBB174_15:
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
.LBB174_16:
	movups xmm0, xmmword ptr [r8 + 4*r14 - 16]
	movups xmm1, xmmword ptr [r8 + 4*r14]
	movups xmm2, xmmword ptr [rbx + 4*r14 - 16]
	addps xmm2, xmm0
	movups xmm0, xmmword ptr [rbx + 4*r14]
	addps xmm0, xmm1
	movups xmmword ptr [r11 + 4*r14 - 16], xmm2
	movups xmmword ptr [r11 + 4*r14], xmm0
	add r14, 8
	cmp r10, r14
	jne .LBB174_16
	cmp r9, r10
	je .LBB174_9
	jmp .LBB174_18
.LBB174_23:
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
	jmp .LBB174_1
