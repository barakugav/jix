jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	mov rcx, r8
	and rcx, -16
	je .LBB128_3
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	lea rbx, [r8 - 16]
	cmp rbx, 112
	jae .LBB128_4
	xor eax, eax
	jmp .LBB128_7
.LBB128_3:
	xor eax, eax
	jmp .LBB128_8
.LBB128_4:
	shr rbx, 4
	inc rbx
	mov r14, rbx
	and r14, -8
	mov rax, r14
	shl rax, 4
	vmovapd zmm0, zmmword ptr [rip + .LCPI128_0]
	vpbroadcastq zmm1, qword ptr [rip + .LCPI128_1]
	mov r15, r14
.LBB128_5:
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
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0]
	vmulpd zmm3, zmm3, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 8]
	vmulpd zmm4, zmm4, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 16]
	vmulpd zmm5, zmm5, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 24]
	vmulpd zmm6, zmm6, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 32]
	vmulpd zmm7, zmm7, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 40]
	vmulpd zmm8, zmm8, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 48]
	vmulpd zmm9, zmm9, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 56]
	vmulpd zmm10, zmm10, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 64]
	vmulpd zmm11, zmm11, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 72]
	vmulpd zmm12, zmm12, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 80]
	vmulpd zmm13, zmm13, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 88]
	vmulpd zmm14, zmm14, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 96]
	vmulpd zmm15, zmm15, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 104]
	vmulpd zmm16, zmm16, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 112]
	vmulpd zmm17, zmm17, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r10 + 8*zmm0 + 120]
	vmulpd zmm2, zmm2, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0]
	vsubpd zmm3, zmm3, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 8]
	vsubpd zmm4, zmm4, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 16]
	vsubpd zmm5, zmm5, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 24]
	vsubpd zmm6, zmm6, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 32]
	vsubpd zmm7, zmm7, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 40]
	vsubpd zmm8, zmm8, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 48]
	vsubpd zmm9, zmm9, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 56]
	vsubpd zmm10, zmm10, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 64]
	vsubpd zmm11, zmm11, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 72]
	vsubpd zmm12, zmm12, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 80]
	vsubpd zmm13, zmm13, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 88]
	vsubpd zmm14, zmm14, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 96]
	vsubpd zmm15, zmm15, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 104]
	vsubpd zmm16, zmm16, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 112]
	vsubpd zmm17, zmm17, zmm18
	kxnorb k1, k0, k0
	vxorpd xmm18, xmm18, xmm18
	vgatherqpd zmm18 {k1}, qword ptr [r11 + 8*zmm0 + 120]
	vsubpd zmm2, zmm2, zmm18
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
	add r15, -8
	jne .LBB128_5
	cmp rbx, r14
	je .LBB128_8
.LBB128_7:
	vmovupd zmm0, zmmword ptr [rdx + 8*rax]
	vmovupd zmm1, zmmword ptr [rdx + 8*rax + 64]
	vaddpd zmm0, zmm0, zmmword ptr [r9 + 8*rax]
	vmulpd zmm0, zmm0, zmmword ptr [r10 + 8*rax]
	vsubpd zmm0, zmm0, zmmword ptr [r11 + 8*rax]
	vaddpd zmm1, zmm1, zmmword ptr [r9 + 8*rax + 64]
	vmulpd zmm1, zmm1, zmmword ptr [r10 + 8*rax + 64]
	vsubpd zmm1, zmm1, zmmword ptr [r11 + 8*rax + 64]
	vmovupd zmmword ptr [rsi + 8*rax], zmm0
	vmovupd zmmword ptr [rsi + 8*rax + 64], zmm1
	add rax, 16
	cmp rax, rcx
	jb .LBB128_7
.LBB128_8:
	mov r10, r8
	sub r10, rax
	jbe .LBB128_27
	mov rbp, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r10, 8
	jb .LBB128_10
	mov rcx, rbp
	sub rcx, rsi
	cmp rcx, -255
	setae cl
	mov r11, rdx
	sub r11, rsi
	cmp r11, -255
	setae r11b
	or r11b, cl
	mov rcx, r9
	sub rcx, rsi
	cmp rcx, -255
	setae cl
	mov rbx, rdi
	sub rbx, rsi
	cmp rbx, -255
	setae bl
	or bl, cl
	or bl, r11b
	je .LBB128_13
.LBB128_10:
	mov r11, rax
.LBB128_22:
	mov ecx, r8d
	sub ecx, r11d
	lea rax, [r11 + 1]
	test cl, 1
	je .LBB128_24
	vmovsd xmm0, qword ptr [rbp + 8*r11]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r11]
	vmulsd xmm0, xmm0, qword ptr [r9 + 8*r11]
	vsubsd xmm0, xmm0, qword ptr [rdi + 8*r11]
	vmovsd qword ptr [rsi + 8*r11], xmm0
	mov r11, rax
.LBB128_24:
	cmp r8, rax
	je .LBB128_27
	sub r8, r11
	lea rax, [rsi + 8*r11]
	add rax, 8
	lea rcx, [rdi + 8*r11 + 8]
	lea rsi, [r9 + 8*r11]
	add rsi, 8
	lea rdx, [rdx + 8*r11]
	add rdx, 8
	lea rdi, [8*r11 + 8]
	add rdi, rbp
	xor r9d, r9d
.LBB128_26:
	vmovsd xmm0, qword ptr [rdi + 8*r9 - 8]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r9 - 8]
	vmulsd xmm0, xmm0, qword ptr [rsi + 8*r9 - 8]
	vsubsd xmm0, xmm0, qword ptr [rcx + 8*r9 - 8]
	vmovsd qword ptr [rax + 8*r9 - 8], xmm0
	vmovsd xmm0, qword ptr [rdi + 8*r9]
	vaddsd xmm0, xmm0, qword ptr [rdx + 8*r9]
	vmulsd xmm0, xmm0, qword ptr [rsi + 8*r9]
	vsubsd xmm0, xmm0, qword ptr [rcx + 8*r9]
	vmovsd qword ptr [rax + 8*r9], xmm0
	add r9, 2
	cmp r8, r9
	jne .LBB128_26
.LBB128_27:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
.LBB128_13:
	cmp r10, 32
	jae .LBB128_15
	xor ebx, ebx
	jmp .LBB128_19
.LBB128_15:
	mov rbx, r10
	and rbx, -32
	lea r11, [rax + rbx]
	lea r14, [rsi + 8*rax]
	add r14, 192
	lea r15, [rdi + 8*rax + 192]
	lea r12, [r9 + 8*rax + 192]
	lea r13, [rdx + 8*rax]
	add r13, 192
	mov qword ptr [rsp - 8], rbp
	lea rbp, [rbp + 8*rax + 192]
	xor ecx, ecx
.LBB128_16:
	vmovupd zmm0, zmmword ptr [rbp + 8*rcx - 192]
	vmovupd zmm1, zmmword ptr [rbp + 8*rcx - 128]
	vmovupd zmm2, zmmword ptr [rbp + 8*rcx - 64]
	vmovupd zmm3, zmmword ptr [rbp + 8*rcx]
	vaddpd zmm0, zmm0, zmmword ptr [r13 + 8*rcx - 192]
	vaddpd zmm1, zmm1, zmmword ptr [r13 + 8*rcx - 128]
	vaddpd zmm2, zmm2, zmmword ptr [r13 + 8*rcx - 64]
	vaddpd zmm3, zmm3, zmmword ptr [r13 + 8*rcx]
	vmulpd zmm0, zmm0, zmmword ptr [r12 + 8*rcx - 192]
	vmulpd zmm1, zmm1, zmmword ptr [r12 + 8*rcx - 128]
	vmulpd zmm2, zmm2, zmmword ptr [r12 + 8*rcx - 64]
	vmulpd zmm3, zmm3, zmmword ptr [r12 + 8*rcx]
	vsubpd zmm0, zmm0, zmmword ptr [r15 + 8*rcx - 192]
	vsubpd zmm1, zmm1, zmmword ptr [r15 + 8*rcx - 128]
	vsubpd zmm2, zmm2, zmmword ptr [r15 + 8*rcx - 64]
	vsubpd zmm3, zmm3, zmmword ptr [r15 + 8*rcx]
	vmovupd zmmword ptr [r14 + 8*rcx - 192], zmm0
	vmovupd zmmword ptr [r14 + 8*rcx - 128], zmm1
	vmovupd zmmword ptr [r14 + 8*rcx - 64], zmm2
	vmovupd zmmword ptr [r14 + 8*rcx], zmm3
	add rcx, 32
	cmp rbx, rcx
	jne .LBB128_16
	cmp r10, rbx
	mov rbp, qword ptr [rsp - 8]
	je .LBB128_27
	test r10b, 24
	je .LBB128_22
.LBB128_19:
	mov rcx, r10
	and rcx, -8
	lea r11, [rax + rcx]
	lea r14, [rsi + 8*rax]
	lea r15, [rdi + 8*rax]
	lea r12, [r9 + 8*rax]
	lea r13, [rdx + 8*rax]
	lea rax, [8*rax]
	add rax, rbp
.LBB128_20:
	vmovupd zmm0, zmmword ptr [rax + 8*rbx]
	vaddpd zmm0, zmm0, zmmword ptr [r13 + 8*rbx]
	vmulpd zmm0, zmm0, zmmword ptr [r12 + 8*rbx]
	vsubpd zmm0, zmm0, zmmword ptr [r15 + 8*rbx]
	vmovupd zmmword ptr [r14 + 8*rbx], zmm0
	add rbx, 8
	cmp rcx, rbx
	jne .LBB128_20
	cmp r10, rcx
	je .LBB128_27
	jmp .LBB128_22
