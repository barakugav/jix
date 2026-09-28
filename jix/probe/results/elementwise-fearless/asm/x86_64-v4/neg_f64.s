jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
	push r15
	push r14
	push rbx
	sub rsp, 16
	mov rax, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov ecx, dword ptr [rax + 8]
	test ecx, ecx
	jne .LBB167_1
	cmp r8, 16
	jae .LBB167_4
.LBB167_3:
	xor eax, eax
	jmp .LBB167_8
.LBB167_1:
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
	cmp r8, 16
	jb .LBB167_3
.LBB167_4:
	mov rcx, qword ptr [rdi + 136]
	lea rdx, [r8 - 16]
	mov r9, rdx
	shr r9, 4
	je .LBB167_5
	inc r9
	and r9, -2
	xor eax, eax
	vbroadcastsd zmm0, qword ptr [rip + .LCPI167_0]
.LBB167_24:
	vxorps zmm1, zmm0, zmmword ptr [rcx + 8*rax]
	vxorps zmm2, zmm0, zmmword ptr [rcx + 8*rax + 64]
	vmovups zmmword ptr [rsi + 8*rax], zmm1
	vmovups zmmword ptr [rsi + 8*rax + 64], zmm2
	vxorps zmm1, zmm0, zmmword ptr [rcx + 8*rax + 128]
	vxorps zmm2, zmm0, zmmword ptr [rcx + 8*rax + 192]
	vmovups zmmword ptr [rsi + 8*rax + 128], zmm1
	vmovups zmmword ptr [rsi + 8*rax + 192], zmm2
	add rax, 32
	add r9, -2
	jne .LBB167_24
	test dl, 16
	je .LBB167_7
	jmp .LBB167_8
.LBB167_5:
	xor eax, eax
.LBB167_7:
	vbroadcastsd zmm0, qword ptr [rip + .LCPI167_0]
	vxorps zmm1, zmm0, zmmword ptr [rcx + 8*rax]
	vxorps zmm0, zmm0, zmmword ptr [rcx + 8*rax + 64]
	vmovups zmmword ptr [rsi + 8*rax], zmm1
	vmovups zmmword ptr [rsi + 8*rax + 64], zmm0
	add rax, 16
.LBB167_8:
	mov rdx, r8
	sub rdx, rax
	jbe .LBB167_22
	mov rcx, qword ptr [rdi + 136]
	cmp rdx, 8
	jae .LBB167_11
	mov rdi, rax
	jmp .LBB167_20
.LBB167_11:
	cmp rdx, 32
	jae .LBB167_13
	xor r9d, r9d
	jmp .LBB167_17
.LBB167_13:
	mov r9, rdx
	and r9, -32
	lea rdi, [rax + r9]
	lea r10, [rsi + 8*rax]
	add r10, 192
	lea r11, [rcx + 8*rax]
	add r11, 192
	xor ebx, ebx
	vbroadcastsd zmm0, qword ptr [rip + .LCPI167_0]
.LBB167_14:
	vxorps zmm1, zmm0, zmmword ptr [r11 + 8*rbx - 192]
	vxorps zmm2, zmm0, zmmword ptr [r11 + 8*rbx - 128]
	vxorps zmm3, zmm0, zmmword ptr [r11 + 8*rbx - 64]
	vxorps zmm4, zmm0, zmmword ptr [r11 + 8*rbx]
	vmovups zmmword ptr [r10 + 8*rbx - 192], zmm1
	vmovups zmmword ptr [r10 + 8*rbx - 128], zmm2
	vmovups zmmword ptr [r10 + 8*rbx - 64], zmm3
	vmovups zmmword ptr [r10 + 8*rbx], zmm4
	add rbx, 32
	cmp r9, rbx
	jne .LBB167_14
	cmp rdx, r9
	je .LBB167_22
	test dl, 24
	je .LBB167_20
.LBB167_17:
	mov r10, rdx
	and r10, -8
	lea rdi, [rax + r10]
	lea r11, [rsi + 8*rax]
	lea rax, [rcx + 8*rax]
	vbroadcastsd zmm0, qword ptr [rip + .LCPI167_0]
.LBB167_18:
	vxorps zmm1, zmm0, zmmword ptr [rax + 8*r9]
	vmovups zmmword ptr [r11 + 8*r9], zmm1
	add r9, 8
	cmp r10, r9
	jne .LBB167_18
	cmp rdx, r10
	je .LBB167_22
.LBB167_20:
	movabs rax, -9223372036854775808
.LBB167_21:
	mov rdx, qword ptr [rcx + 8*rdi]
	xor rdx, rax
	mov qword ptr [rsi + 8*rdi], rdx
	inc rdi
	cmp r8, rdi
	jne .LBB167_21
.LBB167_22:
	add rsp, 16
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
