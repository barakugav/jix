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
	jne .LBB174_3
	cmp byte ptr [rbx], 2
	jne .LBB174_4
.LBB174_2:
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>::{closure#1}, ()>
	jmp .LBB174_27
.LBB174_3:
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
	je .LBB174_2
.LBB174_4:
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -32
	je .LBB174_8
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor r8d, r8d
.LBB174_6:
	vmovups ymm0, ymmword ptr [rdx + 4*r8]
	vmovups ymm1, ymmword ptr [rdx + 4*r8 + 32]
	vmovups ymm2, ymmword ptr [rdx + 4*r8 + 64]
	vmovups ymm3, ymmword ptr [rdx + 4*r8 + 96]
	vaddps ymm0, ymm0, ymmword ptr [r9 + 4*r8]
	vaddps ymm1, ymm1, ymmword ptr [r9 + 4*r8 + 32]
	vaddps ymm2, ymm2, ymmword ptr [r9 + 4*r8 + 64]
	vaddps ymm3, ymm3, ymmword ptr [r9 + 4*r8 + 96]
	vmovups ymmword ptr [rsi + 4*r8], ymm0
	vmovups ymmword ptr [rsi + 4*r8 + 32], ymm1
	vmovups ymmword ptr [rsi + 4*r8 + 64], ymm2
	vmovups ymmword ptr [rsi + 4*r8 + 96], ymm3
	add r8, 32
	cmp r8, rcx
	jb .LBB174_6
	mov r9, rax
	sub r9, r8
	ja .LBB174_9
	jmp .LBB174_27
.LBB174_8:
	xor r8d, r8d
	mov r9, rax
	sub r9, r8
	jbe .LBB174_27
.LBB174_9:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jb .LBB174_10
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -127
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -127
	setae r10b
	or r10b, dil
	je .LBB174_13
.LBB174_10:
	mov rdi, r8
.LBB174_22:
	mov r9d, eax
	sub r9d, edi
	mov r8, rdi
	and r9d, 3
	je .LBB174_25
	mov r8, rdi
.LBB174_24:
	vmovss xmm0, dword ptr [rcx + 4*r8]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r8]
	vmovss dword ptr [rsi + 4*r8], xmm0
	inc r8
	dec r9
	jne .LBB174_24
.LBB174_25:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB174_27
.LBB174_26:
	vmovss xmm0, dword ptr [rcx + 4*r8]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r8]
	vmovss dword ptr [rsi + 4*r8], xmm0
	vmovss xmm0, dword ptr [rcx + 4*r8 + 4]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r8 + 4]
	vmovss dword ptr [rsi + 4*r8 + 4], xmm0
	vmovss xmm0, dword ptr [rcx + 4*r8 + 8]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r8 + 8]
	vmovss dword ptr [rsi + 4*r8 + 8], xmm0
	vmovss xmm0, dword ptr [rcx + 4*r8 + 12]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r8 + 12]
	vmovss dword ptr [rsi + 4*r8 + 12], xmm0
	add r8, 4
	cmp rax, r8
	jne .LBB174_26
.LBB174_27:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	vzeroupper
	ret
.LBB174_13:
	cmp r9, 32
	jae .LBB174_15
	xor r10d, r10d
	jmp .LBB174_19
.LBB174_15:
	mov r10, r9
	and r10, -32
	lea rdi, [r8 + r10]
	lea r11, [rsi + 4*r8]
	add r11, 96
	lea rbx, [rdx + 4*r8]
	add rbx, 96
	lea r14, [rcx + 4*r8]
	add r14, 96
	xor r15d, r15d
.LBB174_16:
	vmovups ymm0, ymmword ptr [r14 + 4*r15 - 96]
	vmovups ymm1, ymmword ptr [r14 + 4*r15 - 64]
	vmovups ymm2, ymmword ptr [r14 + 4*r15 - 32]
	vmovups ymm3, ymmword ptr [r14 + 4*r15]
	vaddps ymm0, ymm0, ymmword ptr [rbx + 4*r15 - 96]
	vaddps ymm1, ymm1, ymmword ptr [rbx + 4*r15 - 64]
	vaddps ymm2, ymm2, ymmword ptr [rbx + 4*r15 - 32]
	vaddps ymm3, ymm3, ymmword ptr [rbx + 4*r15]
	vmovups ymmword ptr [r11 + 4*r15 - 96], ymm0
	vmovups ymmword ptr [r11 + 4*r15 - 64], ymm1
	vmovups ymmword ptr [r11 + 4*r15 - 32], ymm2
	vmovups ymmword ptr [r11 + 4*r15], ymm3
	add r15, 32
	cmp r10, r15
	jne .LBB174_16
	cmp r9, r10
	je .LBB174_27
	test r9b, 24
	je .LBB174_22
.LBB174_19:
	mov r11, r9
	and r11, -8
	lea rdi, [r8 + r11]
	lea rbx, [rsi + 4*r8]
	lea r14, [rdx + 4*r8]
	lea r8, [rcx + 4*r8]
.LBB174_20:
	vmovups ymm0, ymmword ptr [r8 + 4*r10]
	vaddps ymm0, ymm0, ymmword ptr [r14 + 4*r10]
	vmovups ymmword ptr [rbx + 4*r10], ymm0
	add r10, 8
	cmp r11, r10
	jne .LBB174_20
	cmp r9, r11
	je .LBB174_27
	jmp .LBB174_22
