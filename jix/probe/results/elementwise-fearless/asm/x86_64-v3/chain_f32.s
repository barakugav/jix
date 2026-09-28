jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
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
	jne .LBB172_3
	cmp byte ptr [rbx], 2
	jne .LBB172_4
.LBB172_2:
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>::{closure#1}, ()>
	jmp .LBB172_27
.LBB172_3:
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
	je .LBB172_2
.LBB172_4:
	mov rax, qword ptr [rsp + 8]
	mov rcx, rax
	and rcx, -32
	je .LBB172_8
	mov rdx, qword ptr [rdi + 136]
	mov r8, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor r9d, r9d
.LBB172_6:
	vmovups ymm0, ymmword ptr [rdx + 4*r9]
	vmovups ymm1, ymmword ptr [rdx + 4*r9 + 32]
	vmovups ymm2, ymmword ptr [rdx + 4*r9 + 64]
	vmovups ymm3, ymmword ptr [rdx + 4*r9 + 96]
	vaddps ymm0, ymm0, ymmword ptr [r8 + 4*r9]
	vaddps ymm1, ymm1, ymmword ptr [r8 + 4*r9 + 32]
	vaddps ymm2, ymm2, ymmword ptr [r8 + 4*r9 + 64]
	vaddps ymm3, ymm3, ymmword ptr [r8 + 4*r9 + 96]
	vmulps ymm0, ymm0, ymmword ptr [r10 + 4*r9]
	vmulps ymm1, ymm1, ymmword ptr [r10 + 4*r9 + 32]
	vmulps ymm2, ymm2, ymmword ptr [r10 + 4*r9 + 64]
	vmulps ymm3, ymm3, ymmword ptr [r10 + 4*r9 + 96]
	vsubps ymm0, ymm0, ymmword ptr [r11 + 4*r9]
	vsubps ymm1, ymm1, ymmword ptr [r11 + 4*r9 + 32]
	vsubps ymm2, ymm2, ymmword ptr [r11 + 4*r9 + 64]
	vsubps ymm3, ymm3, ymmword ptr [r11 + 4*r9 + 96]
	vmovups ymmword ptr [rsi + 4*r9], ymm0
	vmovups ymmword ptr [rsi + 4*r9 + 32], ymm1
	vmovups ymmword ptr [rsi + 4*r9 + 64], ymm2
	vmovups ymmword ptr [rsi + 4*r9 + 96], ymm3
	add r9, 32
	cmp r9, rcx
	jb .LBB172_6
	mov r10, rax
	sub r10, r9
	ja .LBB172_9
	jmp .LBB172_27
.LBB172_8:
	xor r9d, r9d
	mov r10, rax
	sub r10, r9
	jbe .LBB172_27
.LBB172_9:
	mov rbp, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r8, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r10, 8
	jb .LBB172_10
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
	je .LBB172_13
.LBB172_10:
	mov r11, r9
.LBB172_22:
	mov r9d, eax
	sub r9d, r11d
	lea rcx, [r11 + 1]
	test r9b, 1
	je .LBB172_24
	vmovss xmm0, dword ptr [rbp + 4*r11]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r11]
	vmulss xmm0, xmm0, dword ptr [r8 + 4*r11]
	vsubss xmm0, xmm0, dword ptr [rdi + 4*r11]
	vmovss dword ptr [rsi + 4*r11], xmm0
	mov r11, rcx
.LBB172_24:
	cmp rax, rcx
	je .LBB172_27
	sub rax, r11
	lea rcx, [rsi + 4*r11]
	add rcx, 4
	lea rsi, [rdi + 4*r11 + 4]
	lea rdi, [r8 + 4*r11]
	add rdi, 4
	lea rdx, [rdx + 4*r11]
	add rdx, 4
	lea r8, [4*r11 + 4]
	add r8, rbp
	xor r9d, r9d
.LBB172_26:
	vmovss xmm0, dword ptr [r8 + 4*r9 - 4]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r9 - 4]
	vmulss xmm0, xmm0, dword ptr [rdi + 4*r9 - 4]
	vsubss xmm0, xmm0, dword ptr [rsi + 4*r9 - 4]
	vmovss dword ptr [rcx + 4*r9 - 4], xmm0
	vmovss xmm0, dword ptr [r8 + 4*r9]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r9]
	vmulss xmm0, xmm0, dword ptr [rdi + 4*r9]
	vsubss xmm0, xmm0, dword ptr [rsi + 4*r9]
	vmovss dword ptr [rcx + 4*r9], xmm0
	add r9, 2
	cmp rax, r9
	jne .LBB172_26
.LBB172_27:
	add rsp, 88
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
.LBB172_13:
	cmp r10, 32
	jae .LBB172_15
	xor ebx, ebx
	jmp .LBB172_19
.LBB172_15:
	mov rbx, r10
	and rbx, -32
	lea r11, [r9 + rbx]
	lea r14, [rsi + 4*r9]
	add r14, 96
	lea r15, [rdi + 4*r9 + 96]
	lea r12, [r8 + 4*r9 + 96]
	lea r13, [rdx + 4*r9]
	add r13, 96
	mov qword ptr [rsp + 16], rbp
	lea rbp, [rbp + 4*r9 + 96]
	xor ecx, ecx
.LBB172_16:
	vmovups ymm0, ymmword ptr [rbp + 4*rcx - 96]
	vmovups ymm1, ymmword ptr [rbp + 4*rcx - 64]
	vmovups ymm2, ymmword ptr [rbp + 4*rcx - 32]
	vmovups ymm3, ymmword ptr [rbp + 4*rcx]
	vaddps ymm0, ymm0, ymmword ptr [r13 + 4*rcx - 96]
	vaddps ymm1, ymm1, ymmword ptr [r13 + 4*rcx - 64]
	vaddps ymm2, ymm2, ymmword ptr [r13 + 4*rcx - 32]
	vaddps ymm3, ymm3, ymmword ptr [r13 + 4*rcx]
	vmulps ymm0, ymm0, ymmword ptr [r12 + 4*rcx - 96]
	vmulps ymm1, ymm1, ymmword ptr [r12 + 4*rcx - 64]
	vmulps ymm2, ymm2, ymmword ptr [r12 + 4*rcx - 32]
	vmulps ymm3, ymm3, ymmword ptr [r12 + 4*rcx]
	vsubps ymm0, ymm0, ymmword ptr [r15 + 4*rcx - 96]
	vsubps ymm1, ymm1, ymmword ptr [r15 + 4*rcx - 64]
	vsubps ymm2, ymm2, ymmword ptr [r15 + 4*rcx - 32]
	vsubps ymm3, ymm3, ymmword ptr [r15 + 4*rcx]
	vmovups ymmword ptr [r14 + 4*rcx - 96], ymm0
	vmovups ymmword ptr [r14 + 4*rcx - 64], ymm1
	vmovups ymmword ptr [r14 + 4*rcx - 32], ymm2
	vmovups ymmword ptr [r14 + 4*rcx], ymm3
	add rcx, 32
	cmp rbx, rcx
	jne .LBB172_16
	cmp r10, rbx
	mov rbp, qword ptr [rsp + 16]
	je .LBB172_27
	test r10b, 24
	je .LBB172_22
.LBB172_19:
	mov rcx, r10
	and rcx, -8
	lea r11, [r9 + rcx]
	lea r14, [rsi + 4*r9]
	lea r15, [rdi + 4*r9]
	lea r12, [r8 + 4*r9]
	lea r13, [rdx + 4*r9]
	lea r9, [4*r9]
	add r9, rbp
.LBB172_20:
	vmovups ymm0, ymmword ptr [r9 + 4*rbx]
	vaddps ymm0, ymm0, ymmword ptr [r13 + 4*rbx]
	vmulps ymm0, ymm0, ymmword ptr [r12 + 4*rbx]
	vsubps ymm0, ymm0, ymmword ptr [r15 + 4*rbx]
	vmovups ymmword ptr [r14 + 4*rbx], ymm0
	add rbx, 8
	cmp rcx, rbx
	jne .LBB172_20
	cmp r10, rcx
	je .LBB172_27
	jmp .LBB172_22
