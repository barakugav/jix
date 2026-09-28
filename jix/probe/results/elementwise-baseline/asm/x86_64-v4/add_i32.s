jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
	push r15
	push r14
	push rbx
	sub rsp, 992
	mov rcx, r8
	and rcx, -32
	je .LBB162_3
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	lea rax, [r8 - 32]
	cmp rax, 224
	jae .LBB162_4
	xor eax, eax
	jmp .LBB162_13
.LBB162_3:
	xor eax, eax
	jmp .LBB162_14
.LBB162_4:
	mov r10, rax
	shr r10, 5
	inc r10
	movabs r11, 1152921504606846960
	cmp rax, 480
	jae .LBB162_9
	xor ebx, ebx
	xor eax, eax
.LBB162_6:
	add r11, 8
	and r11, r10
	vpbroadcastq zmm0, rax
	mov rax, r11
	vpaddq zmm0, zmm0, zmmword ptr [rip + .LCPI162_1]
	shl rax, 5
	sub rbx, r11
.LBB162_7:
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm0]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	kxnorb k1, k0, k0
	vpxor xmm8, xmm8, xmm8
	vpgatherqd ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	kxnorb k1, k0, k0
	vpxor xmm11, xmm11, xmm11
	vpgatherqd ymm11 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	kxnorb k1, k0, k0
	vpxor xmm14, xmm14, xmm14
	vpgatherqd ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	kxnorb k1, k0, k0
	vpxord xmm17, xmm17, xmm17
	vpgatherqd ymm17 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	kxnorb k1, k0, k0
	vpxord xmm20, xmm20, xmm20
	vpgatherqd ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	kxnorb k1, k0, k0
	vpxord xmm23, xmm23, xmm23
	vpgatherqd ymm23 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	kxnorb k1, k0, k0
	vpxord xmm26, xmm26, xmm26
	vpgatherqd ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	kxnorb k1, k0, k0
	vpxord xmm29, xmm29, xmm29
	vpgatherqd ymm29 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	kxnorb k1, k0, k0
	vpxord xmm31, xmm31, xmm31
	vpgatherqd ymm31 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	kxnorb k1, k0, k0
	vpxord xmm30, xmm30, xmm30
	vpgatherqd ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	kxnorb k1, k0, k0
	vpxord xmm28, xmm28, xmm28
	vpgatherqd ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	kxnorb k1, k0, k0
	vpxord xmm27, xmm27, xmm27
	vpgatherqd ymm27 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	kxnorb k1, k0, k0
	vpxord xmm25, xmm25, xmm25
	vpgatherqd ymm25 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	kxnorb k1, k0, k0
	vpxord xmm24, xmm24, xmm24
	vpgatherqd ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	kxnorb k1, k0, k0
	vpxord xmm22, xmm22, xmm22
	vpgatherqd ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	kxnorb k1, k0, k0
	vpxord xmm21, xmm21, xmm21
	vpgatherqd ymm21 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	kxnorb k1, k0, k0
	vpxord xmm19, xmm19, xmm19
	vpgatherqd ymm19 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	kxnorb k1, k0, k0
	vpxord xmm18, xmm18, xmm18
	vpgatherqd ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	kxnorb k1, k0, k0
	vpxord xmm16, xmm16, xmm16
	vpgatherqd ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	kxnorb k1, k0, k0
	vpxor xmm15, xmm15, xmm15
	vpgatherqd ymm15 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	kxnorb k1, k0, k0
	vpxor xmm13, xmm13, xmm13
	vpgatherqd ymm13 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	kxnorb k1, k0, k0
	vpxor xmm12, xmm12, xmm12
	vpgatherqd ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	kxnorb k1, k0, k0
	vpxor xmm10, xmm10, xmm10
	vpgatherqd ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	kxnorb k1, k0, k0
	vpxor xmm9, xmm9, xmm9
	vpgatherqd ymm9 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	kxnorb k1, k0, k0
	vpxor xmm7, xmm7, xmm7
	vpgatherqd ymm7 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	vmovdqu ymmword ptr [rsp - 32], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0]
	vpaddd ymm2, ymm2, ymm3
	vmovdqu ymmword ptr [rsp], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 4]
	vpaddd ymm2, ymm2, ymm5
	vmovdqu ymmword ptr [rsp - 64], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 8]
	vpaddd ymm8, ymm8, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 12]
	vpaddd ymm11, ymm11, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 16]
	vpaddd ymm14, ymm14, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 20]
	vpaddd ymm17, ymm2, ymm17
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 24]
	vpaddd ymm20, ymm2, ymm20
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 28]
	vpaddd ymm23, ymm2, ymm23
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 32]
	vpaddd ymm26, ymm2, ymm26
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 36]
	vpaddd ymm29, ymm2, ymm29
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 40]
	vpaddd ymm3, ymm2, ymm1
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 44]
	vpaddd ymm31, ymm2, ymm31
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 48]
	vpaddd ymm30, ymm2, ymm30
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 52]
	vpaddd ymm28, ymm2, ymm28
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 56]
	vpaddd ymm27, ymm2, ymm27
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 60]
	vpaddd ymm25, ymm2, ymm25
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 64]
	vpaddd ymm24, ymm2, ymm24
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 68]
	vpaddd ymm22, ymm2, ymm22
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 72]
	vpaddd ymm21, ymm2, ymm21
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 76]
	vpaddd ymm19, ymm2, ymm19
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 80]
	vpaddd ymm18, ymm2, ymm18
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 84]
	vpaddd ymm16, ymm2, ymm16
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 88]
	vpaddd ymm15, ymm15, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 92]
	vpaddd ymm13, ymm13, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 96]
	vpaddd ymm12, ymm12, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 100]
	vpaddd ymm10, ymm10, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 104]
	vpaddd ymm9, ymm9, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 108]
	vpaddd ymm7, ymm2, ymm7
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 112]
	vpaddd ymm6, ymm2, ymm6
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 116]
	vpaddd ymm4, ymm2, ymm4
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 120]
	kxnorb k1, k0, k0
	vpaddd ymm2, ymm2, ymmword ptr [rsp - 32]
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [r9 + 4*zmm0 + 124]
	vpaddd ymm1, ymm5, ymm1
	kxnorb k1, k0, k0
	vmovdqu ymm5, ymmword ptr [rsp]
	vpscatterqd dword ptr [rsi + 4*zmm0] {k1}, ymm5
	kxnorb k1, k0, k0
	vmovdqu ymm5, ymmword ptr [rsp - 64]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm5
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm8
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm11
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm14
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm17
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm20
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm23
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm26
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm29
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm3
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm31
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm30
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm28
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm27
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm25
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm24
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm22
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm21
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm19
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm18
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm16
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm15
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm13
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm12
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm10
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm9
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm7
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm6
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm4
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm2
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm1
	vpaddq zmm0, zmm0, qword ptr [rip + .LCPI162_3]{1to8}
	add rbx, 8
	jne .LBB162_7
	cmp r10, r11
	jne .LBB162_13
	jmp .LBB162_14
.LBB162_9:
	mov rbx, r10
	and rbx, r11
	mov rax, rbx
	shl rax, 5
	vmovdqa64 zmm0, zmmword ptr [rip + .LCPI162_0]
	vmovdqa64 zmm1, zmmword ptr [rip + .LCPI162_1]
	mov r14, rbx
.LBB162_10:
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0]
	kxnorb k1, k0, k0
	vpxord xmm31, xmm31, xmm31
	vpgatherqd ymm31 {k1}, dword ptr [rdx + 4*zmm1]
	kxnorb k1, k0, k0
	vpxord xmm30, xmm30, xmm30
	vpgatherqd ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	kxnorb k1, k0, k0
	vpxord xmm29, xmm29, xmm29
	vpgatherqd ymm29 {k1}, dword ptr [rdx + 4*zmm1 + 4]
	kxnorb k1, k0, k0
	vpxord xmm28, xmm28, xmm28
	vpgatherqd ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	kxnorb k1, k0, k0
	vpxord xmm27, xmm27, xmm27
	vpgatherqd ymm27 {k1}, dword ptr [rdx + 4*zmm1 + 8]
	kxnorb k1, k0, k0
	vpxord xmm26, xmm26, xmm26
	vpgatherqd ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	kxnorb k1, k0, k0
	vpxord xmm25, xmm25, xmm25
	vpgatherqd ymm25 {k1}, dword ptr [rdx + 4*zmm1 + 12]
	kxnorb k1, k0, k0
	vpxord xmm24, xmm24, xmm24
	vpgatherqd ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	kxnorb k1, k0, k0
	vpxord xmm23, xmm23, xmm23
	vpgatherqd ymm23 {k1}, dword ptr [rdx + 4*zmm1 + 16]
	kxnorb k1, k0, k0
	vpxord xmm22, xmm22, xmm22
	vpgatherqd ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	kxnorb k1, k0, k0
	vpxord xmm21, xmm21, xmm21
	vpgatherqd ymm21 {k1}, dword ptr [rdx + 4*zmm1 + 20]
	kxnorb k1, k0, k0
	vpxord xmm20, xmm20, xmm20
	vpgatherqd ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	kxnorb k1, k0, k0
	vpxord xmm19, xmm19, xmm19
	vpgatherqd ymm19 {k1}, dword ptr [rdx + 4*zmm1 + 24]
	kxnorb k1, k0, k0
	vpxord xmm18, xmm18, xmm18
	vpgatherqd ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	kxnorb k1, k0, k0
	vpxord xmm17, xmm17, xmm17
	vpgatherqd ymm17 {k1}, dword ptr [rdx + 4*zmm1 + 28]
	kxnorb k1, k0, k0
	vpxord xmm16, xmm16, xmm16
	vpgatherqd ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	kxnorb k1, k0, k0
	vpxor xmm15, xmm15, xmm15
	vpgatherqd ymm15 {k1}, dword ptr [rdx + 4*zmm1 + 32]
	kxnorb k1, k0, k0
	vpxor xmm14, xmm14, xmm14
	vpgatherqd ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	kxnorb k1, k0, k0
	vpxor xmm13, xmm13, xmm13
	vpgatherqd ymm13 {k1}, dword ptr [rdx + 4*zmm1 + 36]
	kxnorb k1, k0, k0
	vpxor xmm12, xmm12, xmm12
	vpgatherqd ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	kxnorb k1, k0, k0
	vpxor xmm11, xmm11, xmm11
	vpgatherqd ymm11 {k1}, dword ptr [rdx + 4*zmm1 + 40]
	kxnorb k1, k0, k0
	vpxor xmm10, xmm10, xmm10
	vpgatherqd ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	kxnorb k1, k0, k0
	vpxor xmm9, xmm9, xmm9
	vpgatherqd ymm9 {k1}, dword ptr [rdx + 4*zmm1 + 44]
	kxnorb k1, k0, k0
	vpxor xmm8, xmm8, xmm8
	vpgatherqd ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	kxnorb k1, k0, k0
	vpxor xmm7, xmm7, xmm7
	vpgatherqd ymm7 {k1}, dword ptr [rdx + 4*zmm1 + 48]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm1 + 52]
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm1 + 56]
	vmovdqu ymmword ptr [rsp - 128], ymm3
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	vmovdqu ymmword ptr [rsp - 96], ymm3
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0]
	vpaddd ymm2, ymm3, ymm2
	vmovdqu ymmword ptr [rsp], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1]
	vpaddd ymm2, ymm2, ymm31
	vmovdqu ymmword ptr [rsp - 64], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 4]
	vpaddd ymm2, ymm2, ymm30
	vmovdqu ymmword ptr [rsp - 32], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 4]
	vpaddd ymm2, ymm2, ymm29
	vmovdqu ymmword ptr [rsp + 928], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 8]
	vpaddd ymm2, ymm2, ymm28
	vmovdqu ymmword ptr [rsp + 960], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 8]
	vpaddd ymm2, ymm2, ymm27
	vmovdqu ymmword ptr [rsp + 864], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 12]
	vpaddd ymm2, ymm2, ymm26
	vmovdqu ymmword ptr [rsp + 896], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 12]
	vpaddd ymm2, ymm2, ymm25
	vmovdqu ymmword ptr [rsp + 800], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 16]
	vpaddd ymm2, ymm2, ymm24
	vmovdqu ymmword ptr [rsp + 832], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 16]
	vpaddd ymm2, ymm2, ymm23
	vmovdqu ymmword ptr [rsp + 736], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 20]
	vpaddd ymm2, ymm2, ymm22
	vmovdqu ymmword ptr [rsp + 768], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 20]
	vpaddd ymm2, ymm2, ymm21
	vmovdqu ymmword ptr [rsp + 672], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 24]
	vpaddd ymm2, ymm2, ymm20
	vmovdqu ymmword ptr [rsp + 704], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 24]
	vpaddd ymm2, ymm2, ymm19
	vmovdqu ymmword ptr [rsp + 608], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 28]
	vpaddd ymm2, ymm2, ymm18
	vmovdqu ymmword ptr [rsp + 640], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 28]
	vpaddd ymm2, ymm2, ymm17
	vmovdqu ymmword ptr [rsp + 544], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 32]
	vpaddd ymm2, ymm2, ymm16
	vmovdqu ymmword ptr [rsp + 576], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 32]
	vpaddd ymm2, ymm15, ymm2
	vmovdqu ymmword ptr [rsp + 480], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 36]
	vpaddd ymm2, ymm14, ymm2
	vmovdqu ymmword ptr [rsp + 512], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 36]
	vpaddd ymm2, ymm13, ymm2
	vmovdqu ymmword ptr [rsp + 416], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 40]
	vpaddd ymm2, ymm12, ymm2
	vmovdqu ymmword ptr [rsp + 448], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 40]
	vpaddd ymm2, ymm11, ymm2
	vmovdqu ymmword ptr [rsp + 352], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 44]
	vpaddd ymm2, ymm10, ymm2
	vmovdqu ymmword ptr [rsp + 384], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 44]
	vpaddd ymm2, ymm9, ymm2
	vmovdqu ymmword ptr [rsp + 288], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 48]
	vpaddd ymm2, ymm8, ymm2
	vmovdqu ymmword ptr [rsp + 320], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 48]
	vpaddd ymm2, ymm2, ymm7
	vmovdqu ymmword ptr [rsp + 224], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 52]
	vpaddd ymm2, ymm2, ymm6
	vmovdqu ymmword ptr [rsp + 256], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 52]
	vpaddd ymm2, ymm2, ymm5
	vmovdqu ymmword ptr [rsp + 160], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 56]
	vpaddd ymm2, ymm2, ymm4
	vmovdqu ymmword ptr [rsp + 192], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 56]
	vpaddd ymm2, ymm2, ymmword ptr [rsp - 128]
	vmovdqu ymmword ptr [rsp - 128], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 60]
	kxnorb k1, k0, k0
	vpaddd ymm2, ymm2, ymmword ptr [rsp - 96]
	vmovdqu ymmword ptr [rsp - 96], ymm2
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 60]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 60]
	vpaddd ymm2, ymm3, ymm2
	vmovdqu ymmword ptr [rsp + 128], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 64]
	vpaddd ymm2, ymm3, ymm2
	vmovdqu ymmword ptr [rsp + 96], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 64]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 64]
	vpaddd ymm2, ymm3, ymm2
	vmovdqu ymmword ptr [rsp + 64], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 68]
	vpaddd ymm2, ymm3, ymm2
	vmovdqu ymmword ptr [rsp + 32], ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 68]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 68]
	vpaddd ymm30, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 72]
	vpaddd ymm29, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 72]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 72]
	vpaddd ymm28, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 76]
	vpaddd ymm27, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 76]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 76]
	vpaddd ymm26, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 80]
	vpaddd ymm25, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 80]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 80]
	vpaddd ymm24, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 84]
	vpaddd ymm23, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 84]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 84]
	vpaddd ymm22, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 88]
	vpaddd ymm21, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 88]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 88]
	vpaddd ymm20, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 92]
	vpaddd ymm19, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 92]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 92]
	vpaddd ymm18, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 96]
	vpaddd ymm17, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 96]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 96]
	vpaddd ymm16, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 100]
	vpaddd ymm15, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 100]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 100]
	vpaddd ymm14, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 104]
	vpaddd ymm13, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 104]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 104]
	vpaddd ymm12, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 108]
	vpaddd ymm11, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 108]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 108]
	vpaddd ymm10, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 112]
	vpaddd ymm9, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 112]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 112]
	vpaddd ymm8, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 116]
	vpaddd ymm7, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 116]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 116]
	vpaddd ymm6, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 120]
	vpaddd ymm5, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 120]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 120]
	vpaddd ymm4, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm0 + 124]
	vpaddd ymm3, ymm3, ymm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm1 + 124]
	kxnorb k1, k0, k0
	vpxord xmm31, xmm31, xmm31
	vpgatherqd ymm31 {k1}, dword ptr [r9 + 4*zmm1 + 124]
	vpaddd ymm2, ymm31, ymm2
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp - 64]
	vpscatterqd dword ptr [rsi + 4*zmm1] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp]
	vpscatterqd dword ptr [rsi + 4*zmm0] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 928]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 4] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp - 32]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 864]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 8] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 960]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 800]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 12] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 896]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 736]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 16] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 832]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 672]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 20] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 768]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 608]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 24] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 704]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 544]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 28] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 640]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 480]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 32] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 576]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 416]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 36] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 512]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 352]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 40] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 448]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 288]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 44] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 384]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 224]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 48] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 320]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 160]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 52] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 256]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp - 128]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 56] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 192]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 128]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 60] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp - 96]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 64]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 64] {k1}, ymm31
	kxnorb k1, k0, k0
	vmovdqu64 ymm31, ymmword ptr [rsp + 96]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm31
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 68] {k1}, ymm30
	kxnorb k1, k0, k0
	vmovdqu64 ymm30, ymmword ptr [rsp + 32]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm30
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 72] {k1}, ymm28
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm29
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 76] {k1}, ymm26
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm27
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 80] {k1}, ymm24
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm25
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 84] {k1}, ymm22
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm23
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 88] {k1}, ymm20
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm21
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 92] {k1}, ymm18
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm19
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 96] {k1}, ymm16
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm17
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 100] {k1}, ymm14
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm15
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 104] {k1}, ymm12
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm13
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 108] {k1}, ymm10
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm11
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 112] {k1}, ymm8
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm9
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 116] {k1}, ymm6
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm7
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 120] {k1}, ymm4
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm5
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 124] {k1}, ymm2
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm3
	vpbroadcastq zmm2, qword ptr [rip + .LCPI162_2]
	vpaddq zmm1, zmm1, zmm2
	vpaddq zmm0, zmm0, zmm2
	add r14, -16
	jne .LBB162_10
	cmp r10, rbx
	je .LBB162_14
	test r10b, 8
	jne .LBB162_6
.LBB162_13:
	vmovdqu64 zmm0, zmmword ptr [r9 + 4*rax]
	vmovdqu64 zmm1, zmmword ptr [r9 + 4*rax + 64]
	vpaddd zmm0, zmm0, zmmword ptr [rdx + 4*rax]
	vpaddd zmm1, zmm1, zmmword ptr [rdx + 4*rax + 64]
	vmovdqu64 zmmword ptr [rsi + 4*rax], zmm0
	vmovdqu64 zmmword ptr [rsi + 4*rax + 64], zmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB162_13
.LBB162_14:
	mov r9, r8
	sub r9, rax
	jbe .LBB162_33
	mov rcx, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	cmp r9, 8
	jb .LBB162_16
	mov rdi, rcx
	sub rdi, rsi
	cmp rdi, -255
	setae dil
	mov r10, rdx
	sub r10, rsi
	cmp r10, -255
	setae r10b
	or r10b, dil
	je .LBB162_19
.LBB162_16:
	mov rdi, rax
.LBB162_28:
	mov r9d, r8d
	sub r9d, edi
	mov rax, rdi
	and r9d, 3
	je .LBB162_31
	mov rax, rdi
.LBB162_30:
	mov r10d, dword ptr [rdx + 4*rax]
	add r10d, dword ptr [rcx + 4*rax]
	mov dword ptr [rsi + 4*rax], r10d
	inc rax
	dec r9
	jne .LBB162_30
.LBB162_31:
	sub rdi, r8
	cmp rdi, -4
	ja .LBB162_33
.LBB162_32:
	mov edi, dword ptr [rdx + 4*rax]
	add edi, dword ptr [rcx + 4*rax]
	mov dword ptr [rsi + 4*rax], edi
	mov edi, dword ptr [rdx + 4*rax + 4]
	add edi, dword ptr [rcx + 4*rax + 4]
	mov dword ptr [rsi + 4*rax + 4], edi
	mov edi, dword ptr [rdx + 4*rax + 8]
	add edi, dword ptr [rcx + 4*rax + 8]
	mov dword ptr [rsi + 4*rax + 8], edi
	mov edi, dword ptr [rdx + 4*rax + 12]
	add edi, dword ptr [rcx + 4*rax + 12]
	mov dword ptr [rsi + 4*rax + 12], edi
	add rax, 4
	cmp r8, rax
	jne .LBB162_32
.LBB162_33:
	add rsp, 992
	pop rbx
	pop r14
	pop r15
	vzeroupper
	ret
.LBB162_19:
	cmp r9, 64
	jae .LBB162_21
	xor r10d, r10d
	jmp .LBB162_25
.LBB162_21:
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
.LBB162_22:
	vmovdqu64 zmm0, zmmword ptr [rbx + 4*r15 - 192]
	vmovdqu64 zmm1, zmmword ptr [rbx + 4*r15 - 128]
	vmovdqu64 zmm2, zmmword ptr [rbx + 4*r15 - 64]
	vmovdqu64 zmm3, zmmword ptr [rbx + 4*r15]
	vpaddd zmm0, zmm0, zmmword ptr [r14 + 4*r15 - 192]
	vpaddd zmm1, zmm1, zmmword ptr [r14 + 4*r15 - 128]
	vpaddd zmm2, zmm2, zmmword ptr [r14 + 4*r15 - 64]
	vpaddd zmm3, zmm3, zmmword ptr [r14 + 4*r15]
	vmovdqu64 zmmword ptr [r11 + 4*r15 - 192], zmm0
	vmovdqu64 zmmword ptr [r11 + 4*r15 - 128], zmm1
	vmovdqu64 zmmword ptr [r11 + 4*r15 - 64], zmm2
	vmovdqu64 zmmword ptr [r11 + 4*r15], zmm3
	add r15, 64
	cmp r10, r15
	jne .LBB162_22
	cmp r9, r10
	je .LBB162_33
	test r9b, 56
	je .LBB162_28
.LBB162_25:
	mov r11, r9
	and r11, -8
	lea rdi, [rax + r11]
	lea rbx, [rsi + 4*rax]
	lea r14, [rdx + 4*rax]
	lea rax, [rcx + 4*rax]
.LBB162_26:
	vmovdqu ymm0, ymmword ptr [r14 + 4*r10]
	vpaddd ymm0, ymm0, ymmword ptr [rax + 4*r10]
	vmovdqu ymmword ptr [rbx + 4*r10], ymm0
	add r10, 8
	cmp r11, r10
	jne .LBB162_26
	cmp r9, r11
	je .LBB162_33
	jmp .LBB162_28
