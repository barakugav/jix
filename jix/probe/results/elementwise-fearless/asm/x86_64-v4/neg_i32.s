jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
	push r15
	push r14
	push rbx
	sub rsp, 16
	mov rax, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov ecx, dword ptr [rax + 8]
	test ecx, ecx
	jne .LBB175_1
	cmp r8, 32
	jae .LBB175_4
.LBB175_3:
	xor eax, eax
	jmp .LBB175_8
.LBB175_1:
	mov qword ptr [rsp], rax
	mov rcx, rsp
	mov qword ptr [rsp + 8], rcx
	add rax, 8
	lea rcx, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.18]
	lea r9, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.20]
	lea rdx, [rsp + 8]
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
	cmp r8, 32
	jb .LBB175_3
.LBB175_4:
	mov rcx, qword ptr [rdi + 136]
	lea rdx, [r8 - 32]
	mov r9, rdx
	shr r9, 5
	je .LBB175_5
	inc r9
	and r9, -2
	xor eax, eax
	vpxor xmm0, xmm0, xmm0
.LBB175_23:
	vpsubd zmm1, zmm0, zmmword ptr [rcx + 4*rax]
	vpsubd zmm2, zmm0, zmmword ptr [rcx + 4*rax + 64]
	vmovdqu64 zmmword ptr [rsi + 4*rax], zmm1
	vmovdqu64 zmmword ptr [rsi + 4*rax + 64], zmm2
	vpsubd zmm1, zmm0, zmmword ptr [rcx + 4*rax + 128]
	vpsubd zmm2, zmm0, zmmword ptr [rcx + 4*rax + 192]
	vmovdqu64 zmmword ptr [rsi + 4*rax + 128], zmm1
	vmovdqu64 zmmword ptr [rsi + 4*rax + 192], zmm2
	add rax, 64
	add r9, -2
	jne .LBB175_23
	test dl, 32
	je .LBB175_7
	jmp .LBB175_8
.LBB175_5:
	xor eax, eax
.LBB175_7:
	vpxor xmm0, xmm0, xmm0
	vpsubd zmm1, zmm0, zmmword ptr [rcx + 4*rax]
	vpsubd zmm0, zmm0, zmmword ptr [rcx + 4*rax + 64]
	vmovdqu64 zmmword ptr [rsi + 4*rax], zmm1
	vmovdqu64 zmmword ptr [rsi + 4*rax + 64], zmm0
	add rax, 32
.LBB175_8:
	mov rdx, r8
	sub rdx, rax
	jbe .LBB175_21
	mov rcx, qword ptr [rdi + 136]
	cmp rdx, 8
	jae .LBB175_11
	mov rdi, rax
	jmp .LBB175_20
.LBB175_11:
	cmp rdx, 64
	jae .LBB175_13
	xor r9d, r9d
	jmp .LBB175_17
.LBB175_13:
	mov r9, rdx
	and r9, -64
	lea rdi, [rax + r9]
	lea r10, [rsi + 4*rax]
	add r10, 192
	lea r11, [rcx + 4*rax]
	add r11, 192
	xor ebx, ebx
	vpxor xmm0, xmm0, xmm0
.LBB175_14:
	vpsubd zmm1, zmm0, zmmword ptr [r11 + 4*rbx - 192]
	vpsubd zmm2, zmm0, zmmword ptr [r11 + 4*rbx - 128]
	vpsubd zmm3, zmm0, zmmword ptr [r11 + 4*rbx - 64]
	vpsubd zmm4, zmm0, zmmword ptr [r11 + 4*rbx]
	vmovdqu64 zmmword ptr [r10 + 4*rbx - 192], zmm1
	vmovdqu64 zmmword ptr [r10 + 4*rbx - 128], zmm2
	vmovdqu64 zmmword ptr [r10 + 4*rbx - 64], zmm3
	vmovdqu64 zmmword ptr [r10 + 4*rbx], zmm4
	add rbx, 64
	cmp r9, rbx
	jne .LBB175_14
	cmp rdx, r9
	je .LBB175_21
	test dl, 56
	je .LBB175_20
.LBB175_17:
	mov r10, rdx
	and r10, -8
	lea rdi, [rax + r10]
	lea r11, [rsi + 4*rax]
	lea rax, [rcx + 4*rax]
	vpxor xmm0, xmm0, xmm0
.LBB175_18:
	vpsubd ymm1, ymm0, ymmword ptr [rax + 4*r9]
	vmovdqu ymmword ptr [r11 + 4*r9], ymm1
	add r9, 8
	cmp r10, r9
	jne .LBB175_18
	cmp rdx, r10
	je .LBB175_21
.LBB175_20:
	xor eax, eax
	sub eax, dword ptr [rcx + 4*rdi]
	mov dword ptr [rsi + 4*rdi], eax
	inc rdi
	cmp r8, rdi
	jne .LBB175_20
.LBB175_21:
	add rsp, 16
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
