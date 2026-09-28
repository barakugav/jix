jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
	push rbx
	sub rsp, 1024
	mov rcx, r8
	and rcx, -32
	je .LBB143_1
	mov rdx, qword ptr [rdi + 136]
	lea rax, [r8 - 32]
	cmp rax, 224
	jae .LBB143_7
	xor eax, eax
	jmp .LBB143_16
.LBB143_1:
	xor eax, eax
	jmp .LBB143_2
.LBB143_7:
	mov r9, rax
	shr r9, 5
	inc r9
	movabs r10, 1152921504606846960
	cmp rax, 480
	jae .LBB143_9
	xor r11d, r11d
	xor eax, eax
.LBB143_13:
	add r10, 8
	and r10, r9
	vpbroadcastq zmm0, rax
	mov rax, r10
	shl rax, 5
	vpaddq zmm0, zmm0, zmmword ptr [rip + .LCPI143_1]
	sub r11, r10
	vbroadcastss ymm3, dword ptr [rip + .LCPI143_2]
.LBB143_14:
	vxorps xmm2, xmm2, xmm2
	kxnorb k1, k0, k0
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0]
	vxorps xmm1, xmm1, xmm1
	kxnorb k1, k0, k0
	vgatherqps ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	vmovups ymmword ptr [rsp - 128], ymm1
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	vxorps xmm5, xmm5, xmm5
	kxnorb k1, k0, k0
	vgatherqps ymm5 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	vxorps xmm6, xmm6, xmm6
	kxnorb k1, k0, k0
	vgatherqps ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	vxorps xmm7, xmm7, xmm7
	kxnorb k1, k0, k0
	vgatherqps ymm7 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	vxorps xmm8, xmm8, xmm8
	kxnorb k1, k0, k0
	vgatherqps ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	vxorps xmm9, xmm9, xmm9
	kxnorb k1, k0, k0
	vgatherqps ymm9 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	vxorps xmm10, xmm10, xmm10
	kxnorb k1, k0, k0
	vgatherqps ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	vxorps xmm11, xmm11, xmm11
	kxnorb k1, k0, k0
	vgatherqps ymm11 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	vxorps xmm12, xmm12, xmm12
	kxnorb k1, k0, k0
	vgatherqps ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	vxorps xmm13, xmm13, xmm13
	kxnorb k1, k0, k0
	vgatherqps ymm13 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	vxorps xmm14, xmm14, xmm14
	kxnorb k1, k0, k0
	vgatherqps ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	vxorps xmm15, xmm15, xmm15
	kxnorb k1, k0, k0
	vgatherqps ymm15 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	vxorps xmm16, xmm16, xmm16
	kxnorb k1, k0, k0
	vgatherqps ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	vxorps xmm17, xmm17, xmm17
	kxnorb k1, k0, k0
	vgatherqps ymm17 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	vxorps xmm18, xmm18, xmm18
	kxnorb k1, k0, k0
	vgatherqps ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	vxorps xmm19, xmm19, xmm19
	kxnorb k1, k0, k0
	vgatherqps ymm19 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	vxorps xmm20, xmm20, xmm20
	kxnorb k1, k0, k0
	vgatherqps ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	vxorps xmm21, xmm21, xmm21
	kxnorb k1, k0, k0
	vgatherqps ymm21 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	vxorps xmm22, xmm22, xmm22
	kxnorb k1, k0, k0
	vgatherqps ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	vxorps xmm23, xmm23, xmm23
	kxnorb k1, k0, k0
	vgatherqps ymm23 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	vxorps xmm24, xmm24, xmm24
	kxnorb k1, k0, k0
	vgatherqps ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	vxorps xmm25, xmm25, xmm25
	kxnorb k1, k0, k0
	vgatherqps ymm25 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	vxorps xmm26, xmm26, xmm26
	kxnorb k1, k0, k0
	vgatherqps ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	vxorps xmm27, xmm27, xmm27
	kxnorb k1, k0, k0
	vgatherqps ymm27 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	vxorps xmm28, xmm28, xmm28
	kxnorb k1, k0, k0
	vgatherqps ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	vxorps xmm29, xmm29, xmm29
	kxnorb k1, k0, k0
	vgatherqps ymm29 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	vxorps xmm30, xmm30, xmm30
	kxnorb k1, k0, k0
	vgatherqps ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	vxorps xmm31, xmm31, xmm31
	kxnorb k1, k0, k0
	vgatherqps ymm31 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	vxorps xmm1, xmm1, xmm1
	kxnorb k1, k0, k0
	vgatherqps ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	vmovups ymmword ptr [rsp - 96], ymm1
	vxorps xmm1, xmm1, xmm1
	kxnorb k1, k0, k0
	vgatherqps ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	vxorps ymm2, ymm2, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0] {k1}, ymm2
	vxorps ymm2, ymm3, ymmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm2
	vxorps ymm2, ymm4, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm2
	vxorps ymm2, ymm5, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm2
	vxorps ymm2, ymm6, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm2
	vxorps ymm2, ymm7, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm2
	vxorps ymm2, ymm8, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm2
	vxorps ymm2, ymm9, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm2
	vxorps ymm2, ymm10, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm2
	vxorps ymm2, ymm11, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm2
	vxorps ymm2, ymm12, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm2
	vxorps ymm2, ymm13, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm2
	vxorps ymm2, ymm14, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm2
	vxorps ymm2, ymm15, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm2
	vxorps ymm2, ymm16, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm2
	vxorps ymm2, ymm17, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm2
	vxorps ymm2, ymm18, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm2
	vxorps ymm2, ymm19, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm2
	vxorps ymm2, ymm20, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm2
	vxorps ymm2, ymm21, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm2
	vxorps ymm2, ymm22, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm2
	vxorps ymm2, ymm23, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm2
	vxorps ymm2, ymm24, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm2
	vxorps ymm2, ymm25, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm2
	vxorps ymm2, ymm26, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm2
	vxorps ymm2, ymm27, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm2
	vxorps ymm2, ymm28, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm2
	vxorps ymm2, ymm29, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm2
	vxorps ymm2, ymm30, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm2
	vxorps ymm2, ymm31, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm2
	vxorps ymm2, ymm3, ymmword ptr [rsp - 96]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm2
	vxorps ymm2, ymm1, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm2
	vpaddq zmm0, zmm0, qword ptr [rip + .LCPI143_4]{1to8}
	add r11, 8
	jne .LBB143_14
	cmp r9, r10
	jne .LBB143_16
	jmp .LBB143_2
.LBB143_9:
	mov r11, r9
	and r11, r10
	mov rax, r11
	shl rax, 5
	vmovaps zmm0, zmmword ptr [rip + .LCPI143_0]
	vbroadcastss ymm1, dword ptr [rip + .LCPI143_2]
	vmovaps zmm2, zmmword ptr [rip + .LCPI143_1]
	mov rbx, r11
.LBB143_10:
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0]
	vmovups ymmword ptr [rsp - 96], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm2]
	vmovups ymmword ptr [rsp + 32], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	vmovups ymmword ptr [rsp - 128], ymm3
	kxnorb k1, k0, k0
	vxorps xmm4, xmm4, xmm4
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm2 + 4]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	vmovups ymmword ptr [rsp + 992], ymm3
	kxnorb k1, k0, k0
	vxorps xmm5, xmm5, xmm5
	vgatherqps ymm5 {k1}, dword ptr [rdx + 4*zmm2 + 8]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	vmovups ymmword ptr [rsp + 960], ymm3
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [rdx + 4*zmm2 + 12]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	vmovups ymmword ptr [rsp + 928], ymm3
	kxnorb k1, k0, k0
	vxorps xmm7, xmm7, xmm7
	vgatherqps ymm7 {k1}, dword ptr [rdx + 4*zmm2 + 16]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	vmovups ymmword ptr [rsp + 896], ymm3
	kxnorb k1, k0, k0
	vxorps xmm8, xmm8, xmm8
	vgatherqps ymm8 {k1}, dword ptr [rdx + 4*zmm2 + 20]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	vmovups ymmword ptr [rsp + 864], ymm3
	kxnorb k1, k0, k0
	vxorps xmm9, xmm9, xmm9
	vgatherqps ymm9 {k1}, dword ptr [rdx + 4*zmm2 + 24]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	vmovups ymmword ptr [rsp + 832], ymm3
	kxnorb k1, k0, k0
	vxorps xmm10, xmm10, xmm10
	vgatherqps ymm10 {k1}, dword ptr [rdx + 4*zmm2 + 28]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	vmovups ymmword ptr [rsp + 800], ymm3
	kxnorb k1, k0, k0
	vxorps xmm11, xmm11, xmm11
	vgatherqps ymm11 {k1}, dword ptr [rdx + 4*zmm2 + 32]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	vmovups ymmword ptr [rsp + 768], ymm3
	kxnorb k1, k0, k0
	vxorps xmm12, xmm12, xmm12
	vgatherqps ymm12 {k1}, dword ptr [rdx + 4*zmm2 + 36]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	vmovups ymmword ptr [rsp + 736], ymm3
	kxnorb k1, k0, k0
	vxorps xmm13, xmm13, xmm13
	vgatherqps ymm13 {k1}, dword ptr [rdx + 4*zmm2 + 40]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	vmovups ymmword ptr [rsp + 704], ymm3
	kxnorb k1, k0, k0
	vxorps xmm14, xmm14, xmm14
	vgatherqps ymm14 {k1}, dword ptr [rdx + 4*zmm2 + 44]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	vmovups ymmword ptr [rsp + 672], ymm3
	kxnorb k1, k0, k0
	vxorps xmm15, xmm15, xmm15
	vgatherqps ymm15 {k1}, dword ptr [rdx + 4*zmm2 + 48]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	vmovups ymmword ptr [rsp + 640], ymm3
	kxnorb k1, k0, k0
	vxorps xmm16, xmm16, xmm16
	vgatherqps ymm16 {k1}, dword ptr [rdx + 4*zmm2 + 52]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	vmovups ymmword ptr [rsp + 608], ymm3
	kxnorb k1, k0, k0
	vxorps xmm17, xmm17, xmm17
	vgatherqps ymm17 {k1}, dword ptr [rdx + 4*zmm2 + 56]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	vmovups ymmword ptr [rsp + 576], ymm3
	kxnorb k1, k0, k0
	vxorps xmm18, xmm18, xmm18
	vgatherqps ymm18 {k1}, dword ptr [rdx + 4*zmm2 + 60]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	vmovups ymmword ptr [rsp + 544], ymm3
	kxnorb k1, k0, k0
	vxorps xmm19, xmm19, xmm19
	vgatherqps ymm19 {k1}, dword ptr [rdx + 4*zmm2 + 64]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	vmovups ymmword ptr [rsp + 512], ymm3
	kxnorb k1, k0, k0
	vxorps xmm20, xmm20, xmm20
	vgatherqps ymm20 {k1}, dword ptr [rdx + 4*zmm2 + 68]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	vmovups ymmword ptr [rsp + 480], ymm3
	kxnorb k1, k0, k0
	vxorps xmm21, xmm21, xmm21
	vgatherqps ymm21 {k1}, dword ptr [rdx + 4*zmm2 + 72]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	vmovups ymmword ptr [rsp + 448], ymm3
	kxnorb k1, k0, k0
	vxorps xmm22, xmm22, xmm22
	vgatherqps ymm22 {k1}, dword ptr [rdx + 4*zmm2 + 76]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	vmovups ymmword ptr [rsp + 416], ymm3
	kxnorb k1, k0, k0
	vxorps xmm23, xmm23, xmm23
	vgatherqps ymm23 {k1}, dword ptr [rdx + 4*zmm2 + 80]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	vmovups ymmword ptr [rsp + 384], ymm3
	kxnorb k1, k0, k0
	vxorps xmm24, xmm24, xmm24
	vgatherqps ymm24 {k1}, dword ptr [rdx + 4*zmm2 + 84]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	vmovups ymmword ptr [rsp + 352], ymm3
	kxnorb k1, k0, k0
	vxorps xmm25, xmm25, xmm25
	vgatherqps ymm25 {k1}, dword ptr [rdx + 4*zmm2 + 88]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	vmovups ymmword ptr [rsp + 320], ymm3
	kxnorb k1, k0, k0
	vxorps xmm26, xmm26, xmm26
	vgatherqps ymm26 {k1}, dword ptr [rdx + 4*zmm2 + 92]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	vmovups ymmword ptr [rsp + 288], ymm3
	kxnorb k1, k0, k0
	vxorps xmm27, xmm27, xmm27
	vgatherqps ymm27 {k1}, dword ptr [rdx + 4*zmm2 + 96]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	vmovups ymmword ptr [rsp + 256], ymm3
	kxnorb k1, k0, k0
	vxorps xmm28, xmm28, xmm28
	vgatherqps ymm28 {k1}, dword ptr [rdx + 4*zmm2 + 100]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	vmovups ymmword ptr [rsp + 224], ymm3
	kxnorb k1, k0, k0
	vxorps xmm29, xmm29, xmm29
	vgatherqps ymm29 {k1}, dword ptr [rdx + 4*zmm2 + 104]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	vmovups ymmword ptr [rsp + 192], ymm3
	kxnorb k1, k0, k0
	vxorps xmm30, xmm30, xmm30
	vgatherqps ymm30 {k1}, dword ptr [rdx + 4*zmm2 + 108]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	vmovups ymmword ptr [rsp + 160], ymm3
	kxnorb k1, k0, k0
	vxorps xmm31, xmm31, xmm31
	vgatherqps ymm31 {k1}, dword ptr [rdx + 4*zmm2 + 112]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	vmovups ymmword ptr [rsp + 128], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm2 + 116]
	vmovups ymmword ptr [rsp], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	vmovups ymmword ptr [rsp + 96], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm2 + 120]
	vmovups ymmword ptr [rsp - 32], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	vmovups ymmword ptr [rsp + 64], ymm3
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm2 + 124]
	vmovups ymmword ptr [rsp - 64], ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 32]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp - 96]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0] {k1}, ymm3
	vxorps ymm3, ymm4, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 4] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm3
	vxorps ymm3, ymm5, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 8] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 992]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm3
	vxorps ymm3, ymm6, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 12] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 960]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm3
	vxorps ymm3, ymm7, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 16] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 928]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm3
	vxorps ymm3, ymm8, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 20] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 896]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm3
	vxorps ymm3, ymm9, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 24] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 864]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm3
	vxorps ymm3, ymm10, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 28] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 832]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm3
	vxorps ymm3, ymm11, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 32] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 800]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm3
	vxorps ymm3, ymm12, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 36] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 768]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm3
	vxorps ymm3, ymm13, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 40] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 736]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm3
	vxorps ymm3, ymm14, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 44] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 704]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm3
	vxorps ymm3, ymm15, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 48] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 672]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm3
	vxorps ymm3, ymm16, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 52] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 640]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm3
	vxorps ymm3, ymm17, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 56] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 608]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm3
	vxorps ymm3, ymm18, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 60] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 576]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm3
	vxorps ymm3, ymm19, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 64] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 544]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm3
	vxorps ymm3, ymm20, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 68] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 512]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm3
	vxorps ymm3, ymm21, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 72] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 480]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm3
	vxorps ymm3, ymm22, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 76] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 448]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm3
	vxorps ymm3, ymm23, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 80] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 416]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm3
	vxorps ymm3, ymm24, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 84] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 384]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm3
	vxorps ymm3, ymm25, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 88] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 352]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm3
	vxorps ymm3, ymm26, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 92] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 320]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm3
	vxorps ymm3, ymm27, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 96] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 288]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm3
	vxorps ymm3, ymm28, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 100] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 256]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm3
	vxorps ymm3, ymm29, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 104] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 224]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm3
	vxorps ymm3, ymm30, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 108] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 192]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm3
	vxorps ymm3, ymm31, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 112] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 160]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 116] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 128]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp - 32]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 120] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 96]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp - 64]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm2 + 124] {k1}, ymm3
	vxorps ymm3, ymm1, ymmword ptr [rsp + 64]
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm3
	vpbroadcastq zmm3, qword ptr [rip + .LCPI143_3]
	vpaddq zmm2, zmm2, zmm3
	vpaddq zmm0, zmm0, zmm3
	add rbx, -16
	jne .LBB143_10
	cmp r9, r11
	je .LBB143_2
	test r9b, 8
	jne .LBB143_13
.LBB143_16:
	vbroadcastss zmm0, dword ptr [rip + .LCPI143_2]
.LBB143_17:
	vxorps zmm1, zmm0, zmmword ptr [rdx + 4*rax]
	vxorps zmm2, zmm0, zmmword ptr [rdx + 4*rax + 64]
	vmovups zmmword ptr [rsi + 4*rax], zmm1
	vmovups zmmword ptr [rsi + 4*rax + 64], zmm2
	add rax, 32
	cmp rax, rcx
	jb .LBB143_17
.LBB143_2:
	mov r9, r8
	sub r9, rax
	jbe .LBB143_33
	mov rcx, qword ptr [rdi + 136]
	cmp r9, 8
	setb dl
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -255
	setae dil
	or dil, dl
	je .LBB143_18
	mov rdx, rax
	jmp .LBB143_27
.LBB143_18:
	cmp r9, 64
	jae .LBB143_20
	xor edi, edi
	jmp .LBB143_24
.LBB143_20:
	mov rdi, r9
	and rdi, -64
	lea rdx, [rax + rdi]
	lea r10, [rsi + 4*rax]
	add r10, 192
	lea r11, [rcx + 4*rax]
	add r11, 192
	xor ebx, ebx
	vbroadcastss zmm0, dword ptr [rip + .LCPI143_2]
.LBB143_21:
	vxorps zmm1, zmm0, zmmword ptr [r11 + 4*rbx - 192]
	vxorps zmm2, zmm0, zmmword ptr [r11 + 4*rbx - 128]
	vxorps zmm3, zmm0, zmmword ptr [r11 + 4*rbx - 64]
	vxorps zmm4, zmm0, zmmword ptr [r11 + 4*rbx]
	vmovups zmmword ptr [r10 + 4*rbx - 192], zmm1
	vmovups zmmword ptr [r10 + 4*rbx - 128], zmm2
	vmovups zmmword ptr [r10 + 4*rbx - 64], zmm3
	vmovups zmmword ptr [r10 + 4*rbx], zmm4
	add rbx, 64
	cmp rdi, rbx
	jne .LBB143_21
	cmp r9, rdi
	je .LBB143_33
	test r9b, 56
	je .LBB143_27
.LBB143_24:
	mov r10, r9
	and r10, -8
	lea rdx, [rax + r10]
	lea r11, [rsi + 4*rax]
	lea rax, [rcx + 4*rax]
	vbroadcastss ymm0, dword ptr [rip + .LCPI143_2]
.LBB143_25:
	vxorps ymm1, ymm0, ymmword ptr [rax + 4*rdi]
	vmovups ymmword ptr [r11 + 4*rdi], ymm1
	add rdi, 8
	cmp r10, rdi
	jne .LBB143_25
	cmp r9, r10
	je .LBB143_33
.LBB143_27:
	mov edi, r8d
	sub edi, edx
	mov rax, rdx
	and edi, 3
	je .LBB143_30
	mov r9d, -2147483648
	mov rax, rdx
.LBB143_29:
	mov r10d, dword ptr [rcx + 4*rax]
	xor r10d, r9d
	mov dword ptr [rsi + 4*rax], r10d
	inc rax
	dec rdi
	jne .LBB143_29
.LBB143_30:
	sub rdx, r8
	cmp rdx, -4
	ja .LBB143_33
	mov edx, -2147483648
.LBB143_32:
	mov edi, dword ptr [rcx + 4*rax]
	xor edi, edx
	mov dword ptr [rsi + 4*rax], edi
	mov edi, dword ptr [rcx + 4*rax + 4]
	xor edi, edx
	mov dword ptr [rsi + 4*rax + 4], edi
	mov edi, dword ptr [rcx + 4*rax + 8]
	xor edi, edx
	mov dword ptr [rsi + 4*rax + 8], edi
	mov edi, dword ptr [rcx + 4*rax + 12]
	xor edi, edx
	mov dword ptr [rsi + 4*rax + 12], edi
	add rax, 4
	cmp r8, rax
	jne .LBB143_32
.LBB143_33:
	add rsp, 1024
	pop rbx
	vzeroupper
	ret
