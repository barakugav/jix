jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
	push rbx
	sub rsp, 128
	mov rcx, r8
	and rcx, -32
	je .LBB159_1
	mov rdx, qword ptr [rdi + 136]
	lea rax, [r8 - 32]
	cmp rax, 224
	jae .LBB159_7
	xor eax, eax
	jmp .LBB159_16
.LBB159_1:
	xor eax, eax
	jmp .LBB159_2
.LBB159_7:
	mov r9, rax
	shr r9, 5
	inc r9
	movabs r10, 1152921504606846960
	cmp rax, 480
	jae .LBB159_9
	xor r11d, r11d
	xor eax, eax
.LBB159_13:
	add r10, 8
	and r10, r9
	vpbroadcastq zmm0, rax
	mov rax, r10
	shl rax, 5
	vpaddq zmm0, zmm0, zmmword ptr [rip + .LCPI159_1]
	sub r11, r10
	vpxor xmm3, xmm3, xmm3
.LBB159_14:
	vpxor xmm2, xmm2, xmm2
	kxnorb k1, k0, k0
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0]
	vpxor xmm1, xmm1, xmm1
	kxnorb k1, k0, k0
	vpgatherqd ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	vmovdqu ymmword ptr [rsp - 128], ymm1
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	vpxor xmm5, xmm5, xmm5
	kxnorb k1, k0, k0
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	vpxor xmm6, xmm6, xmm6
	kxnorb k1, k0, k0
	vpgatherqd ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	vpxor xmm7, xmm7, xmm7
	kxnorb k1, k0, k0
	vpgatherqd ymm7 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	vpxor xmm8, xmm8, xmm8
	kxnorb k1, k0, k0
	vpgatherqd ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	vpxor xmm9, xmm9, xmm9
	kxnorb k1, k0, k0
	vpgatherqd ymm9 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	vpxor xmm10, xmm10, xmm10
	kxnorb k1, k0, k0
	vpgatherqd ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	vpxor xmm11, xmm11, xmm11
	kxnorb k1, k0, k0
	vpgatherqd ymm11 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	vpxor xmm12, xmm12, xmm12
	kxnorb k1, k0, k0
	vpgatherqd ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	vpxor xmm13, xmm13, xmm13
	kxnorb k1, k0, k0
	vpgatherqd ymm13 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	vpxor xmm14, xmm14, xmm14
	kxnorb k1, k0, k0
	vpgatherqd ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	vpxor xmm15, xmm15, xmm15
	kxnorb k1, k0, k0
	vpgatherqd ymm15 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	vpxord xmm16, xmm16, xmm16
	kxnorb k1, k0, k0
	vpgatherqd ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	vpxord xmm17, xmm17, xmm17
	kxnorb k1, k0, k0
	vpgatherqd ymm17 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	vpxord xmm18, xmm18, xmm18
	kxnorb k1, k0, k0
	vpgatherqd ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	vpxord xmm19, xmm19, xmm19
	kxnorb k1, k0, k0
	vpgatherqd ymm19 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	vpxord xmm20, xmm20, xmm20
	kxnorb k1, k0, k0
	vpgatherqd ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	vpxord xmm21, xmm21, xmm21
	kxnorb k1, k0, k0
	vpgatherqd ymm21 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	vpxord xmm22, xmm22, xmm22
	kxnorb k1, k0, k0
	vpgatherqd ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	vpxord xmm23, xmm23, xmm23
	kxnorb k1, k0, k0
	vpgatherqd ymm23 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	vpxord xmm24, xmm24, xmm24
	kxnorb k1, k0, k0
	vpgatherqd ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	vpxord xmm25, xmm25, xmm25
	kxnorb k1, k0, k0
	vpgatherqd ymm25 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	vpxord xmm26, xmm26, xmm26
	kxnorb k1, k0, k0
	vpgatherqd ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	vpxord xmm27, xmm27, xmm27
	kxnorb k1, k0, k0
	vpgatherqd ymm27 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	vpxord xmm28, xmm28, xmm28
	kxnorb k1, k0, k0
	vpgatherqd ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	vpxord xmm29, xmm29, xmm29
	kxnorb k1, k0, k0
	vpgatherqd ymm29 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	vpxord xmm30, xmm30, xmm30
	kxnorb k1, k0, k0
	vpgatherqd ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	vpxord xmm31, xmm31, xmm31
	kxnorb k1, k0, k0
	vpgatherqd ymm31 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	vpxor xmm1, xmm1, xmm1
	kxnorb k1, k0, k0
	vpgatherqd ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	vmovdqu ymmword ptr [rsp - 64], ymm1
	vpxor xmm1, xmm1, xmm1
	kxnorb k1, k0, k0
	vpgatherqd ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	vpsubd ymm2, ymm3, ymm2
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0] {k1}, ymm2
	vpsubd ymm2, ymm3, ymmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm4
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm5
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm6
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm7
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm8
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm9
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm10
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm11
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm12
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm13
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm14
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm15
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm16
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm17
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm18
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm19
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm20
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm21
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm22
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm23
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm24
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm25
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm26
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm27
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm28
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm29
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm30
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm31
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm2
	vpsubd ymm2, ymm3, ymmword ptr [rsp - 64]
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm2
	vpsubd ymm2, ymm3, ymm1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm2
	vpaddq zmm0, zmm0, qword ptr [rip + .LCPI159_3]{1to8}
	add r11, 8
	jne .LBB159_14
	cmp r9, r10
	jne .LBB159_16
	jmp .LBB159_2
.LBB159_9:
	mov r11, r9
	and r11, r10
	mov rax, r11
	shl rax, 5
	vmovdqa64 zmm0, zmmword ptr [rip + .LCPI159_0]
	vmovdqa64 zmm1, zmmword ptr [rip + .LCPI159_1]
	vpxor xmm2, xmm2, xmm2
	mov rbx, r11
.LBB159_10:
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm0]
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1]
	vinserti64x4 zmm3, zmm4, ymm3, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 4]
	vinserti64x4 zmm4, zmm5, ymm4, 1
	vmovdqu64 zmmword ptr [rsp - 64], zmm4
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 8]
	vinserti64x4 zmm4, zmm5, ymm4, 1
	vmovdqu64 zmmword ptr [rsp - 128], zmm4
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 12]
	vinserti64x4 zmm4, zmm5, ymm4, 1
	vmovdqu64 zmmword ptr [rsp + 64], zmm4
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 16]
	vinserti64x4 zmm4, zmm5, ymm4, 1
	vmovdqu64 zmmword ptr [rsp], zmm4
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 20]
	vinserti64x4 zmm8, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 24]
	vinserti64x4 zmm9, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 28]
	vinserti64x4 zmm10, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 32]
	vinserti64x4 zmm11, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 36]
	vinserti64x4 zmm12, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 40]
	vinserti64x4 zmm13, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 44]
	vinserti64x4 zmm14, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 48]
	vinserti64x4 zmm15, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 52]
	vinserti64x4 zmm16, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 56]
	vinserti64x4 zmm17, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 60]
	vinserti64x4 zmm18, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 64]
	vinserti64x4 zmm19, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 68]
	vinserti64x4 zmm20, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 72]
	vinserti64x4 zmm21, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 76]
	vinserti64x4 zmm22, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 80]
	vinserti64x4 zmm23, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 84]
	vinserti64x4 zmm24, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 88]
	vinserti64x4 zmm25, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 92]
	vinserti64x4 zmm26, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 96]
	vinserti64x4 zmm27, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 100]
	vinserti64x4 zmm28, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 104]
	vinserti64x4 zmm29, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 108]
	vinserti64x4 zmm30, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 112]
	vinserti64x4 zmm31, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 116]
	vinserti64x4 zmm6, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 120]
	vinserti64x4 zmm5, zmm5, ymm4, 1
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	kxnorb k1, k0, k0
	vpxor xmm7, xmm7, xmm7
	vpgatherqd ymm7 {k1}, dword ptr [rdx + 4*zmm1 + 124]
	vinserti64x4 zmm4, zmm7, ymm4, 1
	vpsubd zmm3, zmm2, zmm3
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0] {k1}, ymm3
	vpsubd zmm3, zmm2, zmmword ptr [rsp - 64]
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 4] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm3
	vpsubd zmm3, zmm2, zmmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 8] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm3
	vpsubd zmm3, zmm2, zmmword ptr [rsp + 64]
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 12] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm3
	vpsubd zmm3, zmm2, zmmword ptr [rsp]
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 16] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm8
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 20] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm9
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 24] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm10
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 28] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm11
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 32] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm12
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 36] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm13
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 40] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm14
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 44] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm15
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 48] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm16
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 52] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm17
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 56] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm18
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 60] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm19
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 64] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm20
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 68] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm21
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 72] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm22
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 76] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm23
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 80] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm24
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 84] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm25
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 88] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm26
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 92] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm27
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 96] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm28
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 100] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm29
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 104] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm30
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 108] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm31
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 112] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm6
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 116] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm5
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 120] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm3
	vpsubd zmm3, zmm2, zmm4
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 124] {k1}, ymm3
	vextracti64x4 ymm3, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm3
	vpbroadcastq zmm3, qword ptr [rip + .LCPI159_2]
	vpaddq zmm1, zmm1, zmm3
	vpaddq zmm0, zmm0, zmm3
	add rbx, -16
	jne .LBB159_10
	cmp r9, r11
	je .LBB159_2
	test r9b, 8
	jne .LBB159_13
.LBB159_16:
	vpxor xmm0, xmm0, xmm0
.LBB159_17:
	vpsubd zmm1, zmm0, zmmword ptr [rdx + 4*rax]
	vpsubd zmm2, zmm0, zmmword ptr [rdx + 4*rax + 64]
	vmovdqu64 zmmword ptr [rsi + 4*rax], zmm1
	vmovdqu64 zmmword ptr [rsi + 4*rax + 64], zmm2
	add rax, 32
	cmp rax, rcx
	jb .LBB159_17
.LBB159_2:
	mov r9, r8
	sub r9, rax
	jbe .LBB159_32
	mov rcx, qword ptr [rdi + 136]
	cmp r9, 8
	setb dl
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -255
	setae dil
	or dil, dl
	je .LBB159_18
	mov rdx, rax
	jmp .LBB159_27
.LBB159_18:
	cmp r9, 64
	jae .LBB159_20
	xor edi, edi
	jmp .LBB159_24
.LBB159_20:
	mov rdi, r9
	and rdi, -64
	lea rdx, [rax + rdi]
	lea r10, [rsi + 4*rax]
	add r10, 192
	lea r11, [rcx + 4*rax]
	add r11, 192
	xor ebx, ebx
	vpxor xmm0, xmm0, xmm0
.LBB159_21:
	vpsubd zmm1, zmm0, zmmword ptr [r11 + 4*rbx - 192]
	vpsubd zmm2, zmm0, zmmword ptr [r11 + 4*rbx - 128]
	vpsubd zmm3, zmm0, zmmword ptr [r11 + 4*rbx - 64]
	vpsubd zmm4, zmm0, zmmword ptr [r11 + 4*rbx]
	vmovdqu64 zmmword ptr [r10 + 4*rbx - 192], zmm1
	vmovdqu64 zmmword ptr [r10 + 4*rbx - 128], zmm2
	vmovdqu64 zmmword ptr [r10 + 4*rbx - 64], zmm3
	vmovdqu64 zmmword ptr [r10 + 4*rbx], zmm4
	add rbx, 64
	cmp rdi, rbx
	jne .LBB159_21
	cmp r9, rdi
	je .LBB159_32
	test r9b, 56
	je .LBB159_27
.LBB159_24:
	mov r10, r9
	and r10, -8
	lea rdx, [rax + r10]
	lea r11, [rsi + 4*rax]
	lea rax, [rcx + 4*rax]
	vpxor xmm0, xmm0, xmm0
.LBB159_25:
	vpsubd ymm1, ymm0, ymmword ptr [rax + 4*rdi]
	vmovdqu ymmword ptr [r11 + 4*rdi], ymm1
	add rdi, 8
	cmp r10, rdi
	jne .LBB159_25
	cmp r9, r10
	je .LBB159_32
.LBB159_27:
	mov edi, r8d
	sub edi, edx
	mov rax, rdx
	and edi, 3
	je .LBB159_30
	mov rax, rdx
.LBB159_29:
	xor r9d, r9d
	sub r9d, dword ptr [rcx + 4*rax]
	mov dword ptr [rsi + 4*rax], r9d
	inc rax
	dec rdi
	jne .LBB159_29
.LBB159_30:
	sub rdx, r8
	cmp rdx, -4
	ja .LBB159_32
.LBB159_31:
	xor edx, edx
	sub edx, dword ptr [rcx + 4*rax]
	mov dword ptr [rsi + 4*rax], edx
	xor edx, edx
	sub edx, dword ptr [rcx + 4*rax + 4]
	mov dword ptr [rsi + 4*rax + 4], edx
	xor edx, edx
	sub edx, dword ptr [rcx + 4*rax + 8]
	mov dword ptr [rsi + 4*rax + 8], edx
	xor edx, edx
	sub edx, dword ptr [rcx + 4*rax + 12]
	mov dword ptr [rsi + 4*rax + 12], edx
	add rax, 4
	cmp r8, rax
	jne .LBB159_31
.LBB159_32:
	add rsp, 128
	pop rbx
	vzeroupper
	ret
