jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
	push r15
	push r14
	push rbx
	sub rsp, 16
	mov rax, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov ecx, dword ptr [rax + 8]
	test ecx, ecx
	jne .LBB171_1
	cmp r8, 32
	jae .LBB171_4
.LBB171_3:
	xor eax, eax
	jmp .LBB171_8
.LBB171_1:
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
	jb .LBB171_3
.LBB171_4:
	mov rcx, qword ptr [rdi + 136]
	lea rdx, [r8 - 32]
	mov r9, rdx
	shr r9, 5
	je .LBB171_5
	inc r9
	and r9, -2
	xor eax, eax
	vbroadcastss zmm0, dword ptr [rip + .LCPI171_0]
.LBB171_24:
	vxorps zmm1, zmm0, zmmword ptr [rcx + 4*rax]
	vxorps zmm2, zmm0, zmmword ptr [rcx + 4*rax + 64]
	vmovups zmmword ptr [rsi + 4*rax], zmm1
	vmovups zmmword ptr [rsi + 4*rax + 64], zmm2
	vxorps zmm1, zmm0, zmmword ptr [rcx + 4*rax + 128]
	vxorps zmm2, zmm0, zmmword ptr [rcx + 4*rax + 192]
	vmovups zmmword ptr [rsi + 4*rax + 128], zmm1
	vmovups zmmword ptr [rsi + 4*rax + 192], zmm2
	add rax, 64
	add r9, -2
	jne .LBB171_24
	test dl, 32
	je .LBB171_7
	jmp .LBB171_8
.LBB171_5:
	xor eax, eax
.LBB171_7:
	vbroadcastss zmm0, dword ptr [rip + .LCPI171_0]
	vxorps zmm1, zmm0, zmmword ptr [rcx + 4*rax]
	vxorps zmm0, zmm0, zmmword ptr [rcx + 4*rax + 64]
	vmovups zmmword ptr [rsi + 4*rax], zmm1
	vmovups zmmword ptr [rsi + 4*rax + 64], zmm0
	add rax, 32
.LBB171_8:
	mov rdx, r8
	sub rdx, rax
	jbe .LBB171_22
	mov rcx, qword ptr [rdi + 136]
	cmp rdx, 8
	jae .LBB171_11
	mov rdi, rax
	jmp .LBB171_20
.LBB171_11:
	cmp rdx, 64
	jae .LBB171_13
	xor r9d, r9d
	jmp .LBB171_17
.LBB171_13:
	mov r9, rdx
	and r9, -64
	lea rdi, [rax + r9]
	lea r10, [rsi + 4*rax]
	add r10, 192
	lea r11, [rcx + 4*rax]
	add r11, 192
	xor ebx, ebx
	vbroadcastss zmm0, dword ptr [rip + .LCPI171_0]
.LBB171_14:
	vxorps zmm1, zmm0, zmmword ptr [r11 + 4*rbx - 192]
	vxorps zmm2, zmm0, zmmword ptr [r11 + 4*rbx - 128]
	vxorps zmm3, zmm0, zmmword ptr [r11 + 4*rbx - 64]
	vxorps zmm4, zmm0, zmmword ptr [r11 + 4*rbx]
	vmovups zmmword ptr [r10 + 4*rbx - 192], zmm1
	vmovups zmmword ptr [r10 + 4*rbx - 128], zmm2
	vmovups zmmword ptr [r10 + 4*rbx - 64], zmm3
	vmovups zmmword ptr [r10 + 4*rbx], zmm4
	add rbx, 64
	cmp r9, rbx
	jne .LBB171_14
	cmp rdx, r9
	je .LBB171_22
	test dl, 56
	je .LBB171_20
.LBB171_17:
	mov r10, rdx
	and r10, -8
	lea rdi, [rax + r10]
	lea r11, [rsi + 4*rax]
	lea rax, [rcx + 4*rax]
	vbroadcastss ymm0, dword ptr [rip + .LCPI171_0]
.LBB171_18:
	vxorps ymm1, ymm0, ymmword ptr [rax + 4*r9]
	vmovups ymmword ptr [r11 + 4*r9], ymm1
	add r9, 8
	cmp r10, r9
	jne .LBB171_18
	cmp rdx, r10
	je .LBB171_22
.LBB171_20:
	mov eax, -2147483648
.LBB171_21:
	mov edx, dword ptr [rcx + 4*rdi]
	xor edx, eax
	mov dword ptr [rsi + 4*rdi], edx
	inc rdi
	cmp r8, rdi
	jne .LBB171_21
.LBB171_22:
	add rsp, 16
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
