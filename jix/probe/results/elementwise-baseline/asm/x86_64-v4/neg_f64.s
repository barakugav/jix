jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
	push rbx
	mov rcx, r8
	and rcx, -16
	je .LBB127_1
	mov rdx, qword ptr [rdi + 136]
	lea r9, [r8 - 16]
	cmp r9, 112
	jae .LBB127_7
	xor eax, eax
	jmp .LBB127_10
.LBB127_1:
	xor eax, eax
	jmp .LBB127_2
.LBB127_7:
	shr r9, 4
	inc r9
	mov r10, r9
	and r10, -8
	mov rax, r10
	shl rax, 4
	vbroadcastsd zmm0, qword ptr [rip + .LCPI127_1]
	vmovapd zmm1, zmmword ptr [rip + .LCPI127_0]
	vpbroadcastq zmm2, qword ptr [rip + .LCPI127_2]
	mov r11, r10
.LBB127_8:
	kxnorb k1, k0, k0
	vxorpd xmm3, xmm3, xmm3
	vgatherqpd zmm3 {k1}, qword ptr [rdx + 8*zmm1]
	kxnorb k1, k0, k0
	vxorpd xmm4, xmm4, xmm4
	vgatherqpd zmm4 {k1}, qword ptr [rdx + 8*zmm1 + 8]
	kxnorb k1, k0, k0
	vxorpd xmm5, xmm5, xmm5
	vgatherqpd zmm5 {k1}, qword ptr [rdx + 8*zmm1 + 16]
	kxnorb k1, k0, k0
	vxorpd xmm6, xmm6, xmm6
	vgatherqpd zmm6 {k1}, qword ptr [rdx + 8*zmm1 + 24]
	kxnorb k1, k0, k0
	vxorpd xmm7, xmm7, xmm7
	vgatherqpd zmm7 {k1}, qword ptr [rdx + 8*zmm1 + 32]
	kxnorb k1, k0, k0
	vxorpd xmm8, xmm8, xmm8
	vgatherqpd zmm8 {k1}, qword ptr [rdx + 8*zmm1 + 40]
	kxnorb k1, k0, k0
	vxorpd xmm9, xmm9, xmm9
	vgatherqpd zmm9 {k1}, qword ptr [rdx + 8*zmm1 + 48]
	kxnorb k1, k0, k0
	vxorpd xmm10, xmm10, xmm10
	vgatherqpd zmm10 {k1}, qword ptr [rdx + 8*zmm1 + 56]
	kxnorb k1, k0, k0
	vxorpd xmm11, xmm11, xmm11
	vgatherqpd zmm11 {k1}, qword ptr [rdx + 8*zmm1 + 64]
	kxnorb k1, k0, k0
	vxorpd xmm12, xmm12, xmm12
	vgatherqpd zmm12 {k1}, qword ptr [rdx + 8*zmm1 + 72]
	kxnorb k1, k0, k0
	vxorpd xmm13, xmm13, xmm13
	vgatherqpd zmm13 {k1}, qword ptr [rdx + 8*zmm1 + 80]
	kxnorb k1, k0, k0
	vxorpd xmm14, xmm14, xmm14
	vgatherqpd zmm14 {k1}, qword ptr [rdx + 8*zmm1 + 88]
	kxnorb k1, k0, k0
	vxorpd xmm15, xmm15, xmm15
	vgatherqpd zmm15 {k1}, qword ptr [rdx + 8*zmm1 + 96]
	kxnorb k1, k0, k0
	vxorpd xmm16, xmm16, xmm16
	vgatherqpd zmm16 {k1}, qword ptr [rdx + 8*zmm1 + 104]
	kxnorb k1, k0, k0
	vxorpd xmm17, xmm17, xmm17
	vgatherqpd zmm17 {k1}, qword ptr [rdx + 8*zmm1 + 112]
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [rdx + 8*zmm1 + 120]
	vxorpd zmm3, zmm3, zmm0
	vxorpd zmm4, zmm4, zmm0
	vxorpd zmm5, zmm5, zmm0
	vxorpd zmm6, zmm6, zmm0
	vxorpd zmm7, zmm7, zmm0
	vxorpd zmm8, zmm8, zmm0
	vxorpd zmm9, zmm9, zmm0
	vxorpd zmm10, zmm10, zmm0
	vxorpd zmm11, zmm11, zmm0
	vxorpd zmm12, zmm12, zmm0
	vxorpd zmm13, zmm13, zmm0
	vxorpd zmm14, zmm14, zmm0
	vxorpd zmm15, zmm15, zmm0
	vxorpd zmm16, zmm16, zmm0
	vxorpd zmm17, zmm17, zmm0
	vxorpd zmm18, zmm18, zmm0
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1] {k1}, zmm3
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 8] {k1}, zmm4
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 16] {k1}, zmm5
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 24] {k1}, zmm6
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 32] {k1}, zmm7
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 40] {k1}, zmm8
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 48] {k1}, zmm9
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 56] {k1}, zmm10
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 64] {k1}, zmm11
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 72] {k1}, zmm12
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 80] {k1}, zmm13
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 88] {k1}, zmm14
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 96] {k1}, zmm15
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 104] {k1}, zmm16
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 112] {k1}, zmm17
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm1 + 120] {k1}, zmm18
	vpaddq zmm1, zmm1, zmm2
	add r11, -8
	jne .LBB127_8
	cmp r9, r10
	je .LBB127_2
.LBB127_10:
	vbroadcastsd zmm0, qword ptr [rip + .LCPI127_1]
.LBB127_11:
	vxorpd zmm1, zmm0, zmmword ptr [rdx + 8*rax]
	vxorpd zmm2, zmm0, zmmword ptr [rdx + 8*rax + 64]
	vmovupd zmmword ptr [rsi + 8*rax], zmm1
	vmovupd zmmword ptr [rsi + 8*rax + 64], zmm2
	add rax, 16
	cmp rax, rcx
	jb .LBB127_11
.LBB127_2:
	mov r9, r8
	sub r9, rax
	jbe .LBB127_27
	mov rcx, qword ptr [rdi + 136]
	cmp r9, 8
	setb dl
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -255
	setae dil
	or dil, dl
	je .LBB127_12
	mov rdx, rax
	jmp .LBB127_21
.LBB127_12:
	cmp r9, 32
	jae .LBB127_14
	xor edi, edi
	jmp .LBB127_18
.LBB127_14:
	mov rdi, r9
	and rdi, -32
	lea rdx, [rax + rdi]
	lea r10, [rsi + 8*rax]
	add r10, 192
	lea r11, [rcx + 8*rax]
	add r11, 192
	xor ebx, ebx
	vbroadcastsd zmm0, qword ptr [rip + .LCPI127_1]
.LBB127_15:
	vxorpd zmm1, zmm0, zmmword ptr [r11 + 8*rbx - 192]
	vxorpd zmm2, zmm0, zmmword ptr [r11 + 8*rbx - 128]
	vxorpd zmm3, zmm0, zmmword ptr [r11 + 8*rbx - 64]
	vxorpd zmm4, zmm0, zmmword ptr [r11 + 8*rbx]
	vmovupd zmmword ptr [r10 + 8*rbx - 192], zmm1
	vmovupd zmmword ptr [r10 + 8*rbx - 128], zmm2
	vmovupd zmmword ptr [r10 + 8*rbx - 64], zmm3
	vmovupd zmmword ptr [r10 + 8*rbx], zmm4
	add rbx, 32
	cmp rdi, rbx
	jne .LBB127_15
	cmp r9, rdi
	je .LBB127_27
	test r9b, 24
	je .LBB127_21
.LBB127_18:
	mov r10, r9
	and r10, -8
	lea rdx, [rax + r10]
	lea r11, [rsi + 8*rax]
	lea rax, [rcx + 8*rax]
	vbroadcastsd zmm0, qword ptr [rip + .LCPI127_1]
.LBB127_19:
	vxorpd zmm1, zmm0, zmmword ptr [rax + 8*rdi]
	vmovupd zmmword ptr [r11 + 8*rdi], zmm1
	add rdi, 8
	cmp r10, rdi
	jne .LBB127_19
	cmp r9, r10
	je .LBB127_27
.LBB127_21:
	mov edi, r8d
	sub edi, edx
	mov rax, rdx
	and edi, 3
	je .LBB127_24
	movabs r9, -9223372036854775808
	mov rax, rdx
.LBB127_23:
	mov r10, qword ptr [rcx + 8*rax]
	xor r10, r9
	mov qword ptr [rsi + 8*rax], r10
	inc rax
	dec rdi
	jne .LBB127_23
.LBB127_24:
	sub rdx, r8
	cmp rdx, -4
	ja .LBB127_27
	movabs rdx, -9223372036854775808
.LBB127_26:
	mov rdi, qword ptr [rcx + 8*rax]
	xor rdi, rdx
	mov qword ptr [rsi + 8*rax], rdi
	mov rdi, qword ptr [rcx + 8*rax + 8]
	xor rdi, rdx
	mov qword ptr [rsi + 8*rax + 8], rdi
	mov rdi, qword ptr [rcx + 8*rax + 16]
	xor rdi, rdx
	mov qword ptr [rsi + 8*rax + 16], rdi
	mov rdi, qword ptr [rcx + 8*rax + 24]
	xor rdi, rdx
	mov qword ptr [rsi + 8*rax + 24], rdi
	add rax, 4
	cmp r8, rax
	jne .LBB127_26
.LBB127_27:
	pop rbx
	vzeroupper
	ret
