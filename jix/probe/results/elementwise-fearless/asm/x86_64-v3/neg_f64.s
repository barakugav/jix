jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
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
	jne .LBB167_1
	cmp byte ptr [rbx], 2
	jne .LBB167_4
.LBB167_3:
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>::{closure#1}, ()>
	jmp .LBB167_24
.LBB167_1:
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
	je .LBB167_3
.LBB167_4:
	mov rax, qword ptr [rsp + 16]
	mov rcx, rax
	and rcx, -16
	je .LBB167_5
	mov r8, qword ptr [rdi + 136]
	xor edx, edx
	vbroadcastsd ymm0, qword ptr [rip + .LCPI167_0]
.LBB167_26:
	vxorps ymm1, ymm0, ymmword ptr [r8 + 8*rdx]
	vxorps ymm2, ymm0, ymmword ptr [r8 + 8*rdx + 32]
	vxorps ymm3, ymm0, ymmword ptr [r8 + 8*rdx + 64]
	vxorps ymm4, ymm0, ymmword ptr [r8 + 8*rdx + 96]
	vmovups ymmword ptr [rsi + 8*rdx], ymm1
	vmovups ymmword ptr [rsi + 8*rdx + 32], ymm2
	vmovups ymmword ptr [rsi + 8*rdx + 64], ymm3
	vmovups ymmword ptr [rsi + 8*rdx + 96], ymm4
	add rdx, 16
	cmp rdx, rcx
	jb .LBB167_26
	mov r8, rax
	sub r8, rdx
	ja .LBB167_7
	jmp .LBB167_24
.LBB167_5:
	xor edx, edx
	mov r8, rax
	sub r8, rdx
	jbe .LBB167_24
.LBB167_7:
	mov rcx, qword ptr [rdi + 136]
	cmp r8, 4
	setb dil
	mov r9, rcx
	sub r9, rsi
	cmp r9, -127
	setae r9b
	or r9b, dil
	je .LBB167_9
	mov rdi, rdx
	jmp .LBB167_18
.LBB167_9:
	cmp r8, 16
	jae .LBB167_11
	xor r9d, r9d
	jmp .LBB167_15
.LBB167_11:
	mov r9, r8
	and r9, -16
	lea rdi, [rdx + r9]
	lea r10, [rsi + 8*rdx]
	add r10, 96
	lea r11, [rcx + 8*rdx]
	add r11, 96
	xor ebx, ebx
	vbroadcastsd ymm0, qword ptr [rip + .LCPI167_0]
.LBB167_12:
	vxorps ymm1, ymm0, ymmword ptr [r11 + 8*rbx - 96]
	vxorps ymm2, ymm0, ymmword ptr [r11 + 8*rbx - 64]
	vxorps ymm3, ymm0, ymmword ptr [r11 + 8*rbx - 32]
	vxorps ymm4, ymm0, ymmword ptr [r11 + 8*rbx]
	vmovups ymmword ptr [r10 + 8*rbx - 96], ymm1
	vmovups ymmword ptr [r10 + 8*rbx - 64], ymm2
	vmovups ymmword ptr [r10 + 8*rbx - 32], ymm3
	vmovups ymmword ptr [r10 + 8*rbx], ymm4
	add rbx, 16
	cmp r9, rbx
	jne .LBB167_12
	cmp r8, r9
	je .LBB167_24
	test r8b, 12
	je .LBB167_18
.LBB167_15:
	mov r10, r8
	and r10, -4
	lea rdi, [rdx + r10]
	lea r11, [rsi + 8*rdx]
	lea rdx, [rcx + 8*rdx]
	vbroadcastsd ymm0, qword ptr [rip + .LCPI167_0]
.LBB167_16:
	vxorps ymm1, ymm0, ymmword ptr [rdx + 8*r9]
	vmovups ymmword ptr [r11 + 8*r9], ymm1
	add r9, 4
	cmp r10, r9
	jne .LBB167_16
	cmp r8, r10
	je .LBB167_24
.LBB167_18:
	mov r8d, eax
	sub r8d, edi
	mov rdx, rdi
	and r8d, 3
	je .LBB167_21
	movabs r9, -9223372036854775808
	mov rdx, rdi
.LBB167_20:
	mov r10, qword ptr [rcx + 8*rdx]
	xor r10, r9
	mov qword ptr [rsi + 8*rdx], r10
	inc rdx
	dec r8
	jne .LBB167_20
.LBB167_21:
	sub rdi, rax
	cmp rdi, -4
	ja .LBB167_24
	movabs rdi, -9223372036854775808
.LBB167_23:
	mov r8, qword ptr [rcx + 8*rdx]
	xor r8, rdi
	mov qword ptr [rsi + 8*rdx], r8
	mov r8, qword ptr [rcx + 8*rdx + 8]
	xor r8, rdi
	mov qword ptr [rsi + 8*rdx + 8], r8
	mov r8, qword ptr [rcx + 8*rdx + 16]
	xor r8, rdi
	mov qword ptr [rsi + 8*rdx + 16], r8
	mov r8, qword ptr [rcx + 8*rdx + 24]
	xor r8, rdi
	mov qword ptr [rsi + 8*rdx + 24], r8
	add rdx, 4
	cmp rax, rdx
	jne .LBB167_23
.LBB167_24:
	add rsp, 88
	pop rbx
	pop r12
	pop r14
	pop r15
	vzeroupper
	ret
