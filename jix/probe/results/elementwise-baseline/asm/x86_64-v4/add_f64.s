jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	push r15
	push r14
	push rbx
	mov rcx, r8
	and rcx, -16
	je .LBB130_3
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	lea r10, [r8 - 16]
	cmp r10, 112
	jae .LBB130_4
	xor eax, eax
	jmp .LBB130_7
.LBB130_3:
	xor eax, eax
	jmp .LBB130_8
.LBB130_4:
	shr r10, 4
	inc r10
	mov r11, r10
	and r11, -8
	mov rax, r11
	shl rax, 4
	vmovapd zmm0, zmmword ptr [rip + .LCPI130_0]
	vpbroadcastq zmm1, qword ptr [rip + .LCPI130_1]
	mov rbx, r11
.LBB130_5:
	kxnorb k1, k0, k0
	vxorpd xmm3, xmm3, xmm3
	vgatherqpd zmm3 {k1}, qword ptr [rdx + 8*zmm0]
	kxnorb k1, k0, k0
	vxorpd xmm4, xmm4, xmm4
	vgatherqpd zmm4 {k1}, qword ptr [rdx + 8*zmm0 + 8]
	kxnorb k1, k0, k0
	vxorpd xmm5, xmm5, xmm5
	vgatherqpd zmm5 {k1}, qword ptr [rdx + 8*zmm0 + 16]
	kxnorb k1, k0, k0
	vxorpd xmm6, xmm6, xmm6
	vgatherqpd zmm6 {k1}, qword ptr [rdx + 8*zmm0 + 24]
	kxnorb k1, k0, k0
	vxorpd xmm7, xmm7, xmm7
	vgatherqpd zmm7 {k1}, qword ptr [rdx + 8*zmm0 + 32]
	kxnorb k1, k0, k0
	vxorpd xmm8, xmm8, xmm8
	vgatherqpd zmm8 {k1}, qword ptr [rdx + 8*zmm0 + 40]
	kxnorb k1, k0, k0
	vxorpd xmm9, xmm9, xmm9
	vgatherqpd zmm9 {k1}, qword ptr [rdx + 8*zmm0 + 48]
	kxnorb k1, k0, k0
	vxorpd xmm10, xmm10, xmm10
	vgatherqpd zmm10 {k1}, qword ptr [rdx + 8*zmm0 + 56]
	kxnorb k1, k0, k0
	vxorpd xmm11, xmm11, xmm11
	vgatherqpd zmm11 {k1}, qword ptr [rdx + 8*zmm0 + 64]
	kxnorb k1, k0, k0
	vxorpd xmm12, xmm12, xmm12
	vgatherqpd zmm12 {k1}, qword ptr [rdx + 8*zmm0 + 72]
	kxnorb k1, k0, k0
	vxorpd xmm13, xmm13, xmm13
	vgatherqpd zmm13 {k1}, qword ptr [rdx + 8*zmm0 + 80]
	kxnorb k1, k0, k0
	vxorpd xmm14, xmm14, xmm14
	vgatherqpd zmm14 {k1}, qword ptr [rdx + 8*zmm0 + 88]
	kxnorb k1, k0, k0
	vxorpd xmm15, xmm15, xmm15
	vgatherqpd zmm15 {k1}, qword ptr [rdx + 8*zmm0 + 96]
	kxnorb k1, k0, k0
	vxorpd xmm16, xmm16, xmm16
	vgatherqpd zmm16 {k1}, qword ptr [rdx + 8*zmm0 + 104]
	kxnorb k1, k0, k0
	vxorpd xmm17, xmm17, xmm17
	vgatherqpd zmm17 {k1}, qword ptr [rdx + 8*zmm0 + 112]
	kxnorb k1, k0, k0
	vxorpd xmm2, xmm2, xmm2
	vgatherqpd zmm2 {k1}, qword ptr [rdx + 8*zmm0 + 120]
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0]
	vaddpd zmm3, zmm3, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 8]
	vaddpd zmm4, zmm4, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 16]
	vaddpd zmm5, zmm5, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 24]
	vaddpd zmm6, zmm6, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 32]
	vaddpd zmm7, zmm7, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 40]
	vaddpd zmm8, zmm8, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 48]
	vaddpd zmm9, zmm9, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 56]
	vaddpd zmm10, zmm10, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 64]
	vaddpd zmm11, zmm11, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 72]
	vaddpd zmm12, zmm12, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 80]
	vaddpd zmm13, zmm13, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 88]
	vaddpd zmm14, zmm14, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 96]
	vaddpd zmm15, zmm15, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 104]
	vaddpd zmm16, zmm16, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 112]
	vaddpd zmm17, zmm17, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r9 + 8*zmm0 + 120]
	vaddpd zmm2, zmm2, zmm18
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0] {k1}, zmm3
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 8] {k1}, zmm4
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 16] {k1}, zmm5
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 24] {k1}, zmm6
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 32] {k1}, zmm7
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 40] {k1}, zmm8
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 48] {k1}, zmm9
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 56] {k1}, zmm10
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 64] {k1}, zmm11
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 72] {k1}, zmm12
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 80] {k1}, zmm13
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 88] {k1}, zmm14
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 96] {k1}, zmm15
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 104] {k1}, zmm16
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 112] {k1}, zmm17
	kxnorb k1, k0, k0
	vscatterqpd qword ptr [rsi + 8*zmm0 + 120] {k1}, zmm2
	vpaddq zmm0, zmm0, zmm1
	add rbx, -8
	jne .LBB130_5
	cmp r10, r11
	je .LBB130_8
.LBB130_7:
	vmovupd zmm0, zmmword ptr [rdx + 8*rax]
	vmovupd zmm1, zmmword ptr [rdx + 8*rax + 64]
	vaddpd zmm0, zmm0, zmmword ptr [r9 + 8*rax]
	vaddpd zmm1, zmm1, zmmword ptr [r9 + 8*rax + 64]
	vmovupd zmmword ptr [rsi + 8*rax], zmm0
	vmovupd zmmword ptr [rsi + 8*rax + 64], zmm1
	add rax, 16
	cmp rax, rcx
	jb .LBB130_7
.LBB130_8:
	mov r9, r8
	sub r9, rax
	jbe .LBB130_27
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jb .LBB130_10
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -255
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -255
	setae r10b
	or r10b, dil
	je .LBB130_13
.LBB130_10:
	mov rdi, rax
.LBB130_22:
	mov r9d, r8d
	sub r9d, edi
	mov rax, rdi
	and r9d, 3
	je .LBB130_25
	mov rax, rdi
.LBB130_24:
	vmovsd xmm0, qword ptr [rcx + 8*rax]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*rax]
	vmovsd qword ptr [rsi + 8*rax], xmm0
	inc rax
	dec r9
	jne .LBB130_24
.LBB130_25:
	sub rdi, r8
	cmp rdi, -4
	ja .LBB130_27
.LBB130_26:
	vmovsd xmm0, qword ptr [rcx + 8*rax]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*rax]
	vmovsd qword ptr [rsi + 8*rax], xmm0
	vmovsd xmm0, qword ptr [rcx + 8*rax + 8]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*rax + 8]
	vmovsd qword ptr [rsi + 8*rax + 8], xmm0
	vmovsd xmm0, qword ptr [rcx + 8*rax + 16]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*rax + 16]
	vmovsd qword ptr [rsi + 8*rax + 16], xmm0
	vmovsd xmm0, qword ptr [rcx + 8*rax + 24]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*rax + 24]
	vmovsd qword ptr [rsi + 8*rax + 24], xmm0
	add rax, 4
	cmp r8, rax
	jne .LBB130_26
.LBB130_27:
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
.LBB130_13:
	cmp r9, 32
	jae .LBB130_15
	xor r10d, r10d
	jmp .LBB130_19
.LBB130_15:
	mov r10, r9
	and r10, -32
	lea rdi, [rax + r10]
	lea r11, [rsi + 8*rax]
	add r11, 192
	lea rbx, [rdx + 8*rax]
	add rbx, 192
	lea r14, [rcx + 8*rax]
	add r14, 192
	xor r15d, r15d
.LBB130_16:
	vmovupd zmm0, zmmword ptr [r14 + 8*r15 - 192]
	vmovupd zmm1, zmmword ptr [r14 + 8*r15 - 128]
	vmovupd zmm2, zmmword ptr [r14 + 8*r15 - 64]
	vmovupd zmm3, zmmword ptr [r14 + 8*r15]
	vaddpd zmm0, zmm0, zmmword ptr [rbx + 8*r15 - 192]
	vaddpd zmm1, zmm1, zmmword ptr [rbx + 8*r15 - 128]
	vaddpd zmm2, zmm2, zmmword ptr [rbx + 8*r15 - 64]
	vaddpd zmm3, zmm3, zmmword ptr [rbx + 8*r15]
	vmovupd zmmword ptr [r11 + 8*r15 - 192], zmm0
	vmovupd zmmword ptr [r11 + 8*r15 - 128], zmm1
	vmovupd zmmword ptr [r11 + 8*r15 - 64], zmm2
	vmovupd zmmword ptr [r11 + 8*r15], zmm3
	add r15, 32
	cmp r10, r15
	jne .LBB130_16
	cmp r9, r10
	je .LBB130_27
	test r9b, 24
	je .LBB130_22
.LBB130_19:
	mov r11, r9
	and r11, -8
	lea rdi, [rax + r11]
	lea rbx, [rsi + 8*rax]
	lea r14, [rdx + 8*rax]
	lea rax, [rcx + 8*rax]
.LBB130_20:
	vmovupd zmm0, zmmword ptr [rax + 8*r10]
	vaddpd zmm0, zmm0, zmmword ptr [r14 + 8*r10]
	vmovupd zmmword ptr [rbx + 8*r10], zmm0
	add r10, 8
	cmp r11, r10
	jne .LBB130_20
	cmp r9, r11
	je .LBB130_27
	jmp .LBB130_22
