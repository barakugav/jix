jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	sub rsp, 24
	mov rax, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov ecx, dword ptr [rax + 8]
	test ecx, ecx
	jne .LBB176_1
	mov rax, r8
	and rax, -32
	je .LBB176_3
.LBB176_18:
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor r9d, r9d
.LBB176_19:
	vmovdqu64 zmm0, zmmword ptr [rdx + 4*r9]
	vmovdqu64 zmm1, zmmword ptr [rdx + 4*r9 + 64]
	vpaddd zmm0, zmm0, zmmword ptr [rcx + 4*r9]
	vpaddd zmm1, zmm1, zmmword ptr [rcx + 4*r9 + 64]
	vpmulld zmm0, zmm0, zmmword ptr [r10 + 4*r9]
	vpmulld zmm1, zmm1, zmmword ptr [r10 + 4*r9 + 64]
	vpsubd zmm0, zmm0, zmmword ptr [r11 + 4*r9]
	vpsubd zmm1, zmm1, zmmword ptr [r11 + 4*r9 + 64]
	vmovdqu64 zmmword ptr [rsi + 4*r9], zmm0
	vmovdqu64 zmmword ptr [rsi + 4*r9 + 64], zmm1
	add r9, 32
	cmp r9, rax
	jb .LBB176_19
	mov rbp, r8
	sub rbp, r9
	ja .LBB176_5
	jmp .LBB176_17
.LBB176_1:
	mov qword ptr [rsp + 8], rax
	lea rcx, [rsp + 8]
	mov qword ptr [rsp + 16], rcx
	add rax, 8
	lea rcx, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.18]
	lea r9, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.20]
	lea rdx, [rsp + 16]
	mov rbx, rdi
	mov rdi, rax
	mov r14, rsi
	mov esi, 1
	mov r15, r8
	mov r8, r9
	call qword ptr [rip + <std::sys::sync::once::futex::Once>::call@GOTPCREL]
	mov rdi, rbx
	mov rsi, r14
	mov r8, r15
	mov rax, r8
	and rax, -32
	jne .LBB176_18
.LBB176_3:
	xor r9d, r9d
	mov rbp, r8
	sub rbp, r9
	jbe .LBB176_17
.LBB176_5:
	mov rax, qword ptr [rdi + 136]
	mov rcx, qword ptr [rdi + 288]
	mov rdx, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp rbp, 8
	jae .LBB176_7
	mov r11, r9
	jmp .LBB176_16
.LBB176_7:
	cmp rbp, 64
	jae .LBB176_9
	xor ebx, ebx
	jmp .LBB176_13
.LBB176_9:
	mov qword ptr [rsp], rbp
	mov rbx, rbp
	and rbx, -64
	lea r11, [r9 + rbx]
	lea r14, [rsi + 4*r9]
	add r14, 192
	lea r15, [rdi + 4*r9 + 192]
	lea r12, [rdx + 4*r9]
	add r12, 192
	lea r13, [rcx + 4*r9]
	add r13, 192
	lea rbp, [rax + 4*r9]
	add rbp, 192
	xor r10d, r10d
.LBB176_10:
	vmovdqu64 zmm0, zmmword ptr [r13 + 4*r10 - 192]
	vmovdqu64 zmm1, zmmword ptr [r13 + 4*r10 - 128]
	vmovdqu64 zmm2, zmmword ptr [r13 + 4*r10 - 64]
	vmovdqu64 zmm3, zmmword ptr [r13 + 4*r10]
	vpaddd zmm0, zmm0, zmmword ptr [rbp + 4*r10 - 192]
	vpaddd zmm1, zmm1, zmmword ptr [rbp + 4*r10 - 128]
	vpaddd zmm2, zmm2, zmmword ptr [rbp + 4*r10 - 64]
	vpaddd zmm3, zmm3, zmmword ptr [rbp + 4*r10]
	vpmulld zmm0, zmm0, zmmword ptr [r12 + 4*r10 - 192]
	vpmulld zmm1, zmm1, zmmword ptr [r12 + 4*r10 - 128]
	vpmulld zmm2, zmm2, zmmword ptr [r12 + 4*r10 - 64]
	vpmulld zmm3, zmm3, zmmword ptr [r12 + 4*r10]
	vpsubd zmm0, zmm0, zmmword ptr [r15 + 4*r10 - 192]
	vpsubd zmm1, zmm1, zmmword ptr [r15 + 4*r10 - 128]
	vpsubd zmm2, zmm2, zmmword ptr [r15 + 4*r10 - 64]
	vpsubd zmm3, zmm3, zmmword ptr [r15 + 4*r10]
	vmovdqu64 zmmword ptr [r14 + 4*r10 - 192], zmm0
	vmovdqu64 zmmword ptr [r14 + 4*r10 - 128], zmm1
	vmovdqu64 zmmword ptr [r14 + 4*r10 - 64], zmm2
	vmovdqu64 zmmword ptr [r14 + 4*r10], zmm3
	add r10, 64
	cmp rbx, r10
	jne .LBB176_10
	mov rbp, qword ptr [rsp]
	cmp rbp, rbx
	je .LBB176_17
	test bpl, 56
	je .LBB176_16
.LBB176_13:
	mov r10, rbp
	and r10, -8
	lea r11, [r9 + r10]
	lea r14, [rsi + 4*r9]
	lea r15, [rdi + 4*r9]
	lea r12, [rdx + 4*r9]
	lea r13, [rcx + 4*r9]
	lea r9, [rax + 4*r9]
.LBB176_14:
	vmovdqu ymm0, ymmword ptr [r13 + 4*rbx]
	vpaddd ymm0, ymm0, ymmword ptr [r9 + 4*rbx]
	vpmulld ymm0, ymm0, ymmword ptr [r12 + 4*rbx]
	vpsubd ymm0, ymm0, ymmword ptr [r15 + 4*rbx]
	vmovdqu ymmword ptr [r14 + 4*rbx], ymm0
	add rbx, 8
	cmp r10, rbx
	jne .LBB176_14
	cmp rbp, r10
	je .LBB176_17
.LBB176_16:
	mov r9d, dword ptr [rcx + 4*r11]
	add r9d, dword ptr [rax + 4*r11]
	imul r9d, dword ptr [rdx + 4*r11]
	sub r9d, dword ptr [rdi + 4*r11]
	mov dword ptr [rsi + 4*r11], r9d
	inc r11
	cmp r8, r11
	jne .LBB176_16
.LBB176_17:
	add rsp, 24
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
