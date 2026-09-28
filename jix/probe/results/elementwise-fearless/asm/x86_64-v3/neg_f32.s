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
	cmp byte ptr [rbx], 2
	jne .LBB171_4
.LBB171_3:
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>::{closure#1}, ()>
	jmp .LBB171_24
.LBB171_1:
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
	je .LBB171_3
.LBB171_4:
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -32
	je .LBB171_5
	mov r8, qword ptr [rdi + 136]
	xor edx, edx
	vbroadcastss ymm0, dword ptr [rip + .LCPI171_0]
.LBB171_26:
	vxorps ymm1, ymm0, ymmword ptr [r8 + 4*rdx]
	vxorps ymm2, ymm0, ymmword ptr [r8 + 4*rdx + 32]
	vxorps ymm3, ymm0, ymmword ptr [r8 + 4*rdx + 64]
	vxorps ymm4, ymm0, ymmword ptr [r8 + 4*rdx + 96]
	vmovups ymmword ptr [rsi + 4*rdx], ymm1
	vmovups ymmword ptr [rsi + 4*rdx + 32], ymm2
	vmovups ymmword ptr [rsi + 4*rdx + 64], ymm3
	vmovups ymmword ptr [rsi + 4*rdx + 96], ymm4
	add rdx, 32
	cmp rdx, rcx
	jb .LBB171_26
	mov r8, rax
	sub r8, rdx
	ja .LBB171_7
	jmp .LBB171_24
.LBB171_5:
	xor edx, edx
	mov r8, rax
	sub r8, rdx
	jbe .LBB171_24
.LBB171_7:
	mov rcx, qword ptr [rdi + 136]
	cmp r8, 8
	setb dil
	mov r9, rcx
	sub r9, rsi
	cmp r9, -127
	setae r9b
	or r9b, dil
	je .LBB171_9
	mov rdi, rdx
	jmp .LBB171_18
.LBB171_9:
	cmp r8, 32
	jae .LBB171_11
	xor r9d, r9d
	jmp .LBB171_15
.LBB171_11:
	mov r9, r8
	and r9, -32
	lea rdi, [rdx + r9]
	lea r10, [rsi + 4*rdx]
	add r10, 96
	lea r11, [rcx + 4*rdx]
	add r11, 96
	xor ebx, ebx
	vbroadcastss ymm0, dword ptr [rip + .LCPI171_0]
.LBB171_12:
	vxorps ymm1, ymm0, ymmword ptr [r11 + 4*rbx - 96]
	vxorps ymm2, ymm0, ymmword ptr [r11 + 4*rbx - 64]
	vxorps ymm3, ymm0, ymmword ptr [r11 + 4*rbx - 32]
	vxorps ymm4, ymm0, ymmword ptr [r11 + 4*rbx]
	vmovups ymmword ptr [r10 + 4*rbx - 96], ymm1
	vmovups ymmword ptr [r10 + 4*rbx - 64], ymm2
	vmovups ymmword ptr [r10 + 4*rbx - 32], ymm3
	vmovups ymmword ptr [r10 + 4*rbx], ymm4
	add rbx, 32
	cmp r9, rbx
	jne .LBB171_12
	cmp r8, r9
	je .LBB171_24
	test r8b, 24
	je .LBB171_18
.LBB171_15:
	mov r10, r8
	and r10, -8
	lea rdi, [rdx + r10]
	lea r11, [rsi + 4*rdx]
	lea rdx, [rcx + 4*rdx]
	vbroadcastss ymm0, dword ptr [rip + .LCPI171_0]
.LBB171_16:
	vxorps ymm1, ymm0, ymmword ptr [rdx + 4*r9]
	vmovups ymmword ptr [r11 + 4*r9], ymm1
	add r9, 8
	cmp r10, r9
	jne .LBB171_16
	cmp r8, r10
	je .LBB171_24
.LBB171_18:
	mov r8d, eax
	sub r8d, edi
	mov rdx, rdi
	and r8d, 3
	je .LBB171_21
	mov r9d, -2147483648
	mov rdx, rdi
.LBB171_20:
	mov r10d, dword ptr [rcx + 4*rdx]
	xor r10d, r9d
	mov dword ptr [rsi + 4*rdx], r10d
	inc rdx
	dec r8
	jne .LBB171_20
.LBB171_21:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB171_24
	mov edi, -2147483648
.LBB171_23:
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
	jne .LBB171_23
.LBB171_24:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	vzeroupper
	ret
