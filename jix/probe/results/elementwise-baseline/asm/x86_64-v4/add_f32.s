jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>:
	push r15
	push r14
	push rbx
	sub rsp, 992
	mov rcx, r8
	and rcx, -32
	je .LBB146_3
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	lea rax, [r8 - 32]
	cmp rax, 224
	jae .LBB146_4
	xor eax, eax
	jmp .LBB146_13
.LBB146_3:
	xor eax, eax
	jmp .LBB146_14
.LBB146_4:
	mov r10, rax
	shr r10, 5
	inc r10
	movabs r11, 1152921504606846960
	cmp rax, 480
	jae .LBB146_9
	xor ebx, ebx
	xor eax, eax
.LBB146_6:
	add r11, 8
	and r11, r10
	vpbroadcastq zmm0, rax
	mov rax, r11
	vpaddq zmm0, zmm0, zmmword ptr [rip + .LCPI146_1]
	shl rax, 5
	sub rbx, r11
.LBB146_7:
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0]
	kxnorb k1, k0, k0
	vxorps xmm5, xmm5, xmm5
	vgatherqps ymm5 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	kxnorb k1, k0, k0
	vxorps xmm8, xmm8, xmm8
	vgatherqps ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	kxnorb k1, k0, k0
	vxorps xmm11, xmm11, xmm11
	vgatherqps ymm11 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	kxnorb k1, k0, k0
	vxorps xmm14, xmm14, xmm14
	vgatherqps ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	kxnorb k1, k0, k0
	vxorps xmm17, xmm17, xmm17
	vgatherqps ymm17 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	kxnorb k1, k0, k0
	vxorps xmm20, xmm20, xmm20
	vgatherqps ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	kxnorb k1, k0, k0
	vxorps xmm23, xmm23, xmm23
	vgatherqps ymm23 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	kxnorb k1, k0, k0
	vxorps xmm26, xmm26, xmm26
	vgatherqps ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	kxnorb k1, k0, k0
	vxorps xmm29, xmm29, xmm29
	vgatherqps ymm29 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	kxnorb k1, k0, k0
	vxorps xmm31, xmm31, xmm31
	vgatherqps ymm31 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	kxnorb k1, k0, k0
	vxorps xmm30, xmm30, xmm30
	vgatherqps ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	kxnorb k1, k0, k0
	vxorps xmm28, xmm28, xmm28
	vgatherqps ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	kxnorb k1, k0, k0
	vxorps xmm27, xmm27, xmm27
	vgatherqps ymm27 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	kxnorb k1, k0, k0
	vxorps xmm25, xmm25, xmm25
	vgatherqps ymm25 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	kxnorb k1, k0, k0
	vxorps xmm24, xmm24, xmm24
	vgatherqps ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	kxnorb k1, k0, k0
	vxorps xmm22, xmm22, xmm22
	vgatherqps ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	kxnorb k1, k0, k0
	vxorps xmm21, xmm21, xmm21
	vgatherqps ymm21 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	kxnorb k1, k0, k0
	vxorps xmm19, xmm19, xmm19
	vgatherqps ymm19 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	kxnorb k1, k0, k0
	vxorps xmm18, xmm18, xmm18
	vgatherqps ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	kxnorb k1, k0, k0
	vxorps xmm16, xmm16, xmm16
	vgatherqps ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	kxnorb k1, k0, k0
	vxorps xmm15, xmm15, xmm15
	vgatherqps ymm15 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	kxnorb k1, k0, k0
	vxorps xmm13, xmm13, xmm13
	vgatherqps ymm13 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	kxnorb k1, k0, k0
	vxorps xmm12, xmm12, xmm12
	vgatherqps ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	kxnorb k1, k0, k0
	vxorps xmm10, xmm10, xmm10
	vgatherqps ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	kxnorb k1, k0, k0
	vxorps xmm9, xmm9, xmm9
	vgatherqps ymm9 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	kxnorb k1, k0, k0
	vxorps xmm7, xmm7, xmm7
	vgatherqps ymm7 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	kxnorb k1, k0, k0
	vxorps xmm4, xmm4, xmm4
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	vmovups ymmword ptr [rsp - 32], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0]
	vaddps ymm2, ymm3, ymm2
	vmovups ymmword ptr [rsp], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 4]
	vaddps ymm2, ymm5, ymm2
	vmovups ymmword ptr [rsp - 64], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 8]
	vaddps ymm8, ymm8, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 12]
	vaddps ymm11, ymm11, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 16]
	vaddps ymm14, ymm14, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 20]
	vaddps ymm17, ymm17, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 24]
	vaddps ymm20, ymm20, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 28]
	vaddps ymm23, ymm23, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 32]
	vaddps ymm26, ymm26, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 36]
	vaddps ymm29, ymm29, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 40]
	vaddps ymm3, ymm1, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 44]
	vaddps ymm31, ymm31, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 48]
	vaddps ymm30, ymm30, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 52]
	vaddps ymm28, ymm28, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 56]
	vaddps ymm27, ymm27, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 60]
	vaddps ymm25, ymm25, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 64]
	vaddps ymm24, ymm24, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 68]
	vaddps ymm22, ymm22, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 72]
	vaddps ymm21, ymm21, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 76]
	vaddps ymm19, ymm19, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 80]
	vaddps ymm18, ymm18, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 84]
	vaddps ymm16, ymm16, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 88]
	vaddps ymm15, ymm15, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 92]
	vaddps ymm13, ymm13, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 96]
	vaddps ymm12, ymm12, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 100]
	vaddps ymm10, ymm10, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 104]
	vaddps ymm9, ymm9, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 108]
	vaddps ymm7, ymm7, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 112]
	vaddps ymm6, ymm6, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 116]
	vaddps ymm4, ymm4, ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 120]
	kxnorb k1, k0, k0
	vaddps ymm2, ymm2, ymmword ptr [rsp - 32]
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	kxnorb k1, k0, k0
	vxorps xmm5, xmm5, xmm5
	vgatherqps ymm5 {k1}, dword ptr [r9 + 4*zmm0 + 124]
	vaddps ymm1, ymm1, ymm5
	kxnorb k1, k0, k0
	vmovups ymm5, ymmword ptr [rsp]
	vscatterqps dword ptr [rsi + 4*zmm0] {k1}, ymm5
	kxnorb k1, k0, k0
	vmovups ymm5, ymmword ptr [rsp - 64]
	vscatterqps dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm5
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm8
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm11
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm14
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm17
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm20
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm23
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm26
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm29
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm31
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm30
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm28
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm27
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm25
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm24
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm22
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm21
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm19
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm18
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm16
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm15
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm13
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm12
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm10
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm9
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm7
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm6
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm4
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm2
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm1
	vpaddq zmm0, zmm0, qword ptr [rip + .LCPI146_3]{1to8}
	add rbx, 8
	jne .LBB146_7
	cmp r10, r11
	jne .LBB146_13
	jmp .LBB146_14
.LBB146_9:
	mov rbx, r10
	and rbx, r11
	mov rax, rbx
	shl rax, 5
	vmovaps zmm0, zmmword ptr [rip + .LCPI146_0]
	vmovaps zmm1, zmmword ptr [rip + .LCPI146_1]
	mov r14, rbx
.LBB146_10:
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0]
	kxnorb k1, k0, k0
	vxorps xmm31, xmm31, xmm31
	vgatherqps ymm31 {k1}, dword ptr [rdx + 4*zmm1]
	kxnorb k1, k0, k0
	vxorps xmm30, xmm30, xmm30
	vgatherqps ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	kxnorb k1, k0, k0
	vxorps xmm29, xmm29, xmm29
	vgatherqps ymm29 {k1}, dword ptr [rdx + 4*zmm1 + 4]
	kxnorb k1, k0, k0
	vxorps xmm28, xmm28, xmm28
	vgatherqps ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	kxnorb k1, k0, k0
	vxorps xmm27, xmm27, xmm27
	vgatherqps ymm27 {k1}, dword ptr [rdx + 4*zmm1 + 8]
	kxnorb k1, k0, k0
	vxorps xmm26, xmm26, xmm26
	vgatherqps ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	kxnorb k1, k0, k0
	vxorps xmm25, xmm25, xmm25
	vgatherqps ymm25 {k1}, dword ptr [rdx + 4*zmm1 + 12]
	kxnorb k1, k0, k0
	vxorps xmm24, xmm24, xmm24
	vgatherqps ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	kxnorb k1, k0, k0
	vxorps xmm23, xmm23, xmm23
	vgatherqps ymm23 {k1}, dword ptr [rdx + 4*zmm1 + 16]
	kxnorb k1, k0, k0
	vxorps xmm22, xmm22, xmm22
	vgatherqps ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	kxnorb k1, k0, k0
	vxorps xmm21, xmm21, xmm21
	vgatherqps ymm21 {k1}, dword ptr [rdx + 4*zmm1 + 20]
	kxnorb k1, k0, k0
	vxorps xmm20, xmm20, xmm20
	vgatherqps ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	kxnorb k1, k0, k0
	vxorps xmm19, xmm19, xmm19
	vgatherqps ymm19 {k1}, dword ptr [rdx + 4*zmm1 + 24]
	kxnorb k1, k0, k0
	vxorps xmm18, xmm18, xmm18
	vgatherqps ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	kxnorb k1, k0, k0
	vxorps xmm17, xmm17, xmm17
	vgatherqps ymm17 {k1}, dword ptr [rdx + 4*zmm1 + 28]
	kxnorb k1, k0, k0
	vxorps xmm16, xmm16, xmm16
	vgatherqps ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	kxnorb k1, k0, k0
	vxorps xmm15, xmm15, xmm15
	vgatherqps ymm15 {k1}, dword ptr [rdx + 4*zmm1 + 32]
	kxnorb k1, k0, k0
	vxorps xmm14, xmm14, xmm14
	vgatherqps ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	kxnorb k1, k0, k0
	vxorps xmm13, xmm13, xmm13
	vgatherqps ymm13 {k1}, dword ptr [rdx + 4*zmm1 + 36]
	kxnorb k1, k0, k0
	vxorps xmm12, xmm12, xmm12
	vgatherqps ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	kxnorb k1, k0, k0
	vxorps xmm11, xmm11, xmm11
	vgatherqps ymm11 {k1}, dword ptr [rdx + 4*zmm1 + 40]
	kxnorb k1, k0, k0
	vxorps xmm10, xmm10, xmm10
	vgatherqps ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	kxnorb k1, k0, k0
	vxorps xmm9, xmm9, xmm9
	vgatherqps ymm9 {k1}, dword ptr [rdx + 4*zmm1 + 44]
	kxnorb k1, k0, k0
	vxorps xmm8, xmm8, xmm8
	vgatherqps ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	kxnorb k1, k0, k0
	vxorps xmm7, xmm7, xmm7
	vgatherqps ymm7 {k1}, dword ptr [rdx + 4*zmm1 + 48]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	kxnorb k1, k0, k0
	vxorps xmm5, xmm5, xmm5
	vgatherqps ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 52]
	kxnorb k1, k0, k0
	vxorps xmm4, xmm4, xmm4
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm1 + 56]
	vmovups ymmword ptr [rsp - 128], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	vmovups ymmword ptr [rsp - 96], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0]
	vaddps ymm2, ymm2, ymm3
	vmovups ymmword ptr [rsp], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1]
	vaddps ymm2, ymm31, ymm2
	vmovups ymmword ptr [rsp - 64], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 4]
	vaddps ymm2, ymm30, ymm2
	vmovups ymmword ptr [rsp - 32], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 4]
	vaddps ymm2, ymm29, ymm2
	vmovups ymmword ptr [rsp + 928], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 8]
	vaddps ymm2, ymm28, ymm2
	vmovups ymmword ptr [rsp + 960], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 8]
	vaddps ymm2, ymm27, ymm2
	vmovups ymmword ptr [rsp + 864], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 12]
	vaddps ymm2, ymm26, ymm2
	vmovups ymmword ptr [rsp + 896], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 12]
	vaddps ymm2, ymm25, ymm2
	vmovups ymmword ptr [rsp + 800], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 16]
	vaddps ymm2, ymm24, ymm2
	vmovups ymmword ptr [rsp + 832], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 16]
	vaddps ymm2, ymm23, ymm2
	vmovups ymmword ptr [rsp + 736], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 20]
	vaddps ymm2, ymm22, ymm2
	vmovups ymmword ptr [rsp + 768], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 20]
	vaddps ymm2, ymm21, ymm2
	vmovups ymmword ptr [rsp + 672], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 24]
	vaddps ymm2, ymm20, ymm2
	vmovups ymmword ptr [rsp + 704], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 24]
	vaddps ymm2, ymm19, ymm2
	vmovups ymmword ptr [rsp + 608], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 28]
	vaddps ymm2, ymm18, ymm2
	vmovups ymmword ptr [rsp + 640], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 28]
	vaddps ymm2, ymm17, ymm2
	vmovups ymmword ptr [rsp + 544], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 32]
	vaddps ymm2, ymm16, ymm2
	vmovups ymmword ptr [rsp + 576], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 32]
	vaddps ymm2, ymm15, ymm2
	vmovups ymmword ptr [rsp + 480], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 36]
	vaddps ymm2, ymm14, ymm2
	vmovups ymmword ptr [rsp + 512], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 36]
	vaddps ymm2, ymm13, ymm2
	vmovups ymmword ptr [rsp + 416], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 40]
	vaddps ymm2, ymm12, ymm2
	vmovups ymmword ptr [rsp + 448], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 40]
	vaddps ymm2, ymm11, ymm2
	vmovups ymmword ptr [rsp + 352], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 44]
	vaddps ymm2, ymm10, ymm2
	vmovups ymmword ptr [rsp + 384], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 44]
	vaddps ymm2, ymm9, ymm2
	vmovups ymmword ptr [rsp + 288], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 48]
	vaddps ymm2, ymm8, ymm2
	vmovups ymmword ptr [rsp + 320], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 48]
	vaddps ymm2, ymm7, ymm2
	vmovups ymmword ptr [rsp + 224], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 52]
	vaddps ymm2, ymm6, ymm2
	vmovups ymmword ptr [rsp + 256], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 52]
	vaddps ymm2, ymm5, ymm2
	vmovups ymmword ptr [rsp + 160], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 56]
	vaddps ymm2, ymm4, ymm2
	vmovups ymmword ptr [rsp + 192], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 56]
	vaddps ymm2, ymm2, ymmword ptr [rsp - 128]
	vmovups ymmword ptr [rsp - 128], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 60]
	kxnorb k1, k0, k0
	vaddps ymm2, ymm2, ymmword ptr [rsp - 96]
	vmovups ymmword ptr [rsp - 96], ymm2
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 60]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 60]
	vaddps ymm2, ymm2, ymm3
	vmovups ymmword ptr [rsp + 128], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 64]
	vaddps ymm2, ymm2, ymm3
	vmovups ymmword ptr [rsp + 96], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 64]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 64]
	vaddps ymm2, ymm2, ymm3
	vmovups ymmword ptr [rsp + 64], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 68]
	vaddps ymm2, ymm2, ymm3
	vmovups ymmword ptr [rsp + 32], ymm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 68]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 68]
	vaddps ymm30, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 72]
	vaddps ymm29, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 72]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 72]
	vaddps ymm28, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 76]
	vaddps ymm27, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 76]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 76]
	vaddps ymm26, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 80]
	vaddps ymm25, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 80]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 80]
	vaddps ymm24, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 84]
	vaddps ymm23, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 84]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 84]
	vaddps ymm22, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 88]
	vaddps ymm21, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 88]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 88]
	vaddps ymm20, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 92]
	vaddps ymm19, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 92]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 92]
	vaddps ymm18, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 96]
	vaddps ymm17, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 96]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 96]
	vaddps ymm16, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 100]
	vaddps ymm15, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 100]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 100]
	vaddps ymm14, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 104]
	vaddps ymm13, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 104]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 104]
	vaddps ymm12, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 108]
	vaddps ymm11, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 108]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 108]
	vaddps ymm10, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 112]
	vaddps ymm9, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 112]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 112]
	vaddps ymm8, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 116]
	vaddps ymm7, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 116]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 116]
	vaddps ymm6, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 120]
	vaddps ymm5, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 120]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 120]
	vaddps ymm4, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 124]
	vaddps ymm3, ymm2, ymm3
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 124]
	kxnorb k1, k0, k0
	vxorps xmm31, xmm31, xmm31
	vgatherqps ymm31 {k1}, dword ptr [r9 + 4*zmm1 + 124]
	vaddps ymm2, ymm2, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp - 64]
	vscatterqps dword ptr [rsi + 4*zmm1] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp]
	vscatterqps dword ptr [rsi + 4*zmm0] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 928]
	vscatterqps dword ptr [rsi + 4*zmm1 + 4] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp - 32]
	vscatterqps dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 864]
	vscatterqps dword ptr [rsi + 4*zmm1 + 8] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 960]
	vscatterqps dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 800]
	vscatterqps dword ptr [rsi + 4*zmm1 + 12] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 896]
	vscatterqps dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 736]
	vscatterqps dword ptr [rsi + 4*zmm1 + 16] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 832]
	vscatterqps dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 672]
	vscatterqps dword ptr [rsi + 4*zmm1 + 20] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 768]
	vscatterqps dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 608]
	vscatterqps dword ptr [rsi + 4*zmm1 + 24] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 704]
	vscatterqps dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 544]
	vscatterqps dword ptr [rsi + 4*zmm1 + 28] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 640]
	vscatterqps dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 480]
	vscatterqps dword ptr [rsi + 4*zmm1 + 32] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 576]
	vscatterqps dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 416]
	vscatterqps dword ptr [rsi + 4*zmm1 + 36] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 512]
	vscatterqps dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 352]
	vscatterqps dword ptr [rsi + 4*zmm1 + 40] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 448]
	vscatterqps dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 288]
	vscatterqps dword ptr [rsi + 4*zmm1 + 44] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 384]
	vscatterqps dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 224]
	vscatterqps dword ptr [rsi + 4*zmm1 + 48] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 320]
	vscatterqps dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 160]
	vscatterqps dword ptr [rsi + 4*zmm1 + 52] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 256]
	vscatterqps dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp - 128]
	vscatterqps dword ptr [rsi + 4*zmm1 + 56] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 192]
	vscatterqps dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 128]
	vscatterqps dword ptr [rsi + 4*zmm1 + 60] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp - 96]
	vscatterqps dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 64]
	vscatterqps dword ptr [rsi + 4*zmm1 + 64] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovups ymm31, ymmword ptr [rsp + 96]
	vscatterqps dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm31
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 68] {k1}, ymm30
	kxnorb k1, k0, k0
	vmovups ymm30, ymmword ptr [rsp + 32]
	vscatterqps dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm30
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 72] {k1}, ymm28
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm29
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 76] {k1}, ymm26
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm27
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 80] {k1}, ymm24
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm25
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 84] {k1}, ymm22
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm23
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 88] {k1}, ymm20
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm21
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 92] {k1}, ymm18
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm19
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 96] {k1}, ymm16
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm17
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 100] {k1}, ymm14
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm15
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 104] {k1}, ymm12
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm13
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 108] {k1}, ymm10
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm11
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 112] {k1}, ymm8
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm9
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 116] {k1}, ymm6
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm7
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 120] {k1}, ymm4
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm5
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 124] {k1}, ymm2
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm3
	vpbroadcastq zmm2, qword ptr [rip + .LCPI146_2]
	vpaddq zmm1, zmm1, zmm2
	vpaddq zmm0, zmm0, zmm2
	add r14, -16
	jne .LBB146_10
	cmp r10, rbx
	je .LBB146_14
	test r10b, 8
	jne .LBB146_6
.LBB146_13:
	vmovups zmm0, zmmword ptr [rdx + 4*rax]
	vmovups zmm1, zmmword ptr [rdx + 4*rax + 64]
	vaddps zmm0, zmm0, zmmword ptr [r9 + 4*rax]
	vaddps zmm1, zmm1, zmmword ptr [r9 + 4*rax + 64]
	vmovups zmmword ptr [rsi + 4*rax], zmm0
	vmovups zmmword ptr [rsi + 4*rax + 64], zmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB146_13
.LBB146_14:
	mov r9, r8
	sub r9, rax
	jbe .LBB146_33
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jb .LBB146_16
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -255
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -255
	setae r10b
	or r10b, dil
	je .LBB146_19
.LBB146_16:
	mov rdi, rax
.LBB146_28:
	mov r9d, r8d
	sub r9d, edi
	mov rax, rdi
	and r9d, 3
	je .LBB146_31
	mov rax, rdi
.LBB146_30:
	vmovss xmm0, dword ptr [rcx + 4*rax]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*rax]
	vmovss dword ptr [rsi + 4*rax], xmm0
	inc rax
	dec r9
	jne .LBB146_30
.LBB146_31:
	sub rdi, r8
	cmp rdi, -4
	ja .LBB146_33
.LBB146_32:
	vmovss xmm0, dword ptr [rcx + 4*rax]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*rax]
	vmovss dword ptr [rsi + 4*rax], xmm0
	vmovss xmm0, dword ptr [rcx + 4*rax + 4]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*rax + 4]
	vmovss dword ptr [rsi + 4*rax + 4], xmm0
	vmovss xmm0, dword ptr [rcx + 4*rax + 8]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*rax + 8]
	vmovss dword ptr [rsi + 4*rax + 8], xmm0
	vmovss xmm0, dword ptr [rcx + 4*rax + 12]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*rax + 12]
	vmovss dword ptr [rsi + 4*rax + 12], xmm0
	add rax, 4
	cmp r8, rax
	jne .LBB146_32
.LBB146_33:
	add rsp, 992
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
.LBB146_19:
	cmp r9, 64
	jae .LBB146_21
	xor r10d, r10d
	jmp .LBB146_25
.LBB146_21:
	mov r10, r9
	and r10, -64
	lea rdi, [rax + r10]
	lea r11, [rsi + 4*rax]
	add r11, 192
	lea rbx, [rdx + 4*rax]
	add rbx, 192
	lea r14, [rcx + 4*rax]
	add r14, 192
	xor r15d, r15d
.LBB146_22:
	vmovups zmm0, zmmword ptr [r14 + 4*r15 - 192]
	vmovups zmm1, zmmword ptr [r14 + 4*r15 - 128]
	vmovups zmm2, zmmword ptr [r14 + 4*r15 - 64]
	vmovups zmm3, zmmword ptr [r14 + 4*r15]
	vaddps zmm0, zmm0, zmmword ptr [rbx + 4*r15 - 192]
	vaddps zmm1, zmm1, zmmword ptr [rbx + 4*r15 - 128]
	vaddps zmm2, zmm2, zmmword ptr [rbx + 4*r15 - 64]
	vaddps zmm3, zmm3, zmmword ptr [rbx + 4*r15]
	vmovups zmmword ptr [r11 + 4*r15 - 192], zmm0
	vmovups zmmword ptr [r11 + 4*r15 - 128], zmm1
	vmovups zmmword ptr [r11 + 4*r15 - 64], zmm2
	vmovups zmmword ptr [r11 + 4*r15], zmm3
	add r15, 64
	cmp r10, r15
	jne .LBB146_22
	cmp r9, r10
	je .LBB146_33
	test r9b, 56
	je .LBB146_28
.LBB146_25:
	mov r11, r9
	and r11, -8
	lea rdi, [rax + r11]
	lea rbx, [rsi + 4*rax]
	lea r14, [rdx + 4*rax]
	lea rax, [rcx + 4*rax]
.LBB146_26:
	vmovups ymm0, ymmword ptr [rax + 4*r10]
	vaddps ymm0, ymm0, ymmword ptr [r14 + 4*r10]
	vmovups ymmword ptr [rbx + 4*r10], ymm0
	add r10, 8
	cmp r11, r10
	jne .LBB146_26
	cmp r9, r11
	je .LBB146_33
	jmp .LBB146_28
