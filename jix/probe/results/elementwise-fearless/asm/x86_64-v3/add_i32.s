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
	jne .LBB178_3
	cmp byte ptr [rbx], 2
	jne .LBB178_4
.LBB178_2:
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>::{closure#1}, ()>
	jmp .LBB178_27
.LBB178_3:
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
	je .LBB178_2
.LBB178_4:
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -32
	je .LBB178_8
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	xor r8d, r8d
.LBB178_6:
	vmovdqu ymm0, ymmword ptr [r9 + 4*r8]
	vmovdqu ymm1, ymmword ptr [r9 + 4*r8 + 32]
	vmovdqu ymm2, ymmword ptr [r9 + 4*r8 + 64]
	vmovdqu ymm3, ymmword ptr [r9 + 4*r8 + 96]
	vpaddd ymm0, ymm0, ymmword ptr [rdx + 4*r8]
	vpaddd ymm1, ymm1, ymmword ptr [rdx + 4*r8 + 32]
	vpaddd ymm2, ymm2, ymmword ptr [rdx + 4*r8 + 64]
	vpaddd ymm3, ymm3, ymmword ptr [rdx + 4*r8 + 96]
	vmovdqu ymmword ptr [rsi + 4*r8], ymm0
	vmovdqu ymmword ptr [rsi + 4*r8 + 32], ymm1
	vmovdqu ymmword ptr [rsi + 4*r8 + 64], ymm2
	vmovdqu ymmword ptr [rsi + 4*r8 + 96], ymm3
	add r8, 32
	cmp r8, rcx
	jb .LBB178_6
	mov r9, rax
	sub r9, r8
	ja .LBB178_9
	jmp .LBB178_27
.LBB178_8:
	xor r8d, r8d
	mov r9, rax
	sub r9, r8
	jbe .LBB178_27
.LBB178_9:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jb .LBB178_10
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -127
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -127
	setae r10b
	or r10b, dil
	je .LBB178_13
.LBB178_10:
	mov rdi, r8
.LBB178_22:
	mov r9d, eax
	sub r9d, edi
	mov r8, rdi
	and r9d, 3
	je .LBB178_25
	mov r8, rdi
.LBB178_24:
	mov r10d, dword ptr [rdx + 4*r8]
	add r10d, dword ptr [rcx + 4*r8]
	mov dword ptr [rsi + 4*r8], r10d
	inc r8
	dec r9
	jne .LBB178_24
.LBB178_25:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB178_27
.LBB178_26:
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
	jne .LBB178_26
.LBB178_27:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	vzeroupper
	ret
.LBB178_13:
	cmp r9, 32
	jae .LBB178_15
	xor r10d, r10d
	jmp .LBB178_19
.LBB178_15:
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
.LBB178_16:
	vmovdqu ymm0, ymmword ptr [rbx + 4*r15 - 96]
	vmovdqu ymm1, ymmword ptr [rbx + 4*r15 - 64]
	vmovdqu ymm2, ymmword ptr [rbx + 4*r15 - 32]
	vmovdqu ymm3, ymmword ptr [rbx + 4*r15]
	vpaddd ymm0, ymm0, ymmword ptr [r14 + 4*r15 - 96]
	vpaddd ymm1, ymm1, ymmword ptr [r14 + 4*r15 - 64]
	vpaddd ymm2, ymm2, ymmword ptr [r14 + 4*r15 - 32]
	vpaddd ymm3, ymm3, ymmword ptr [r14 + 4*r15]
	vmovdqu ymmword ptr [r11 + 4*r15 - 96], ymm0
	vmovdqu ymmword ptr [r11 + 4*r15 - 64], ymm1
	vmovdqu ymmword ptr [r11 + 4*r15 - 32], ymm2
	vmovdqu ymmword ptr [r11 + 4*r15], ymm3
	add r15, 32
	cmp r10, r15
	jne .LBB178_16
	cmp r9, r10
	je .LBB178_27
	test r9b, 24
	je .LBB178_22
.LBB178_19:
	mov r11, r9
	and r11, -8
	lea rdi, [r8 + r11]
	lea rbx, [rsi + 4*r8]
	lea r14, [rdx + 4*r8]
	lea r8, [rcx + 4*r8]
.LBB178_20:
	vmovdqu ymm0, ymmword ptr [r14 + 4*r10]
	vpaddd ymm0, ymm0, ymmword ptr [r8 + 4*r10]
	vmovdqu ymmword ptr [rbx + 4*r10], ymm0
	add r10, 8
	cmp r11, r10
	jne .LBB178_20
	cmp r9, r11
	je .LBB178_27
	jmp .LBB178_22
