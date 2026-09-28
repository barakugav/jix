jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	sub rsp, 648
	mov rcx, r8
	and rcx, -32
	je .LBB160_3
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	lea rax, [r8 - 32]
	cmp rax, 224
	jae .LBB160_4
	xor eax, eax
	jmp .LBB160_13
.LBB160_3:
	xor eax, eax
	jmp .LBB160_14
.LBB160_4:
	mov rbx, rax
	shr rbx, 5
	inc rbx
	movabs r14, 1152921504606846960
	cmp rax, 480
	jae .LBB160_9
	xor r15d, r15d
	xor eax, eax
.LBB160_6:
	add r14, 8
	and r14, rbx
	vpbroadcastq zmm0, rax
	mov rax, r14
	vpaddq zmm0, zmm0, zmmword ptr [rip + .LCPI160_1]
	shl rax, 5
	sub r15, r14
.LBB160_7:
	kxnorb k1, k0, k0
	vpxord xmm31, xmm31, xmm31
	vpgatherqd ymm31 {k1}, dword ptr [rdx + 4*zmm0]
	kxnorb k1, k0, k0
	vpxord xmm30, xmm30, xmm30
	vpgatherqd ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	kxnorb k1, k0, k0
	vpxord xmm29, xmm29, xmm29
	vpgatherqd ymm29 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	kxnorb k1, k0, k0
	vpxord xmm28, xmm28, xmm28
	vpgatherqd ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	kxnorb k1, k0, k0
	vpxord xmm27, xmm27, xmm27
	vpgatherqd ymm27 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	kxnorb k1, k0, k0
	vpxord xmm26, xmm26, xmm26
	vpgatherqd ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	kxnorb k1, k0, k0
	vpxord xmm25, xmm25, xmm25
	vpgatherqd ymm25 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	kxnorb k1, k0, k0
	vpxord xmm24, xmm24, xmm24
	vpgatherqd ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	kxnorb k1, k0, k0
	vpxord xmm23, xmm23, xmm23
	vpgatherqd ymm23 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	kxnorb k1, k0, k0
	vpxord xmm22, xmm22, xmm22
	vpgatherqd ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	kxnorb k1, k0, k0
	vpxord xmm21, xmm21, xmm21
	vpgatherqd ymm21 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	kxnorb k1, k0, k0
	vpxord xmm20, xmm20, xmm20
	vpgatherqd ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	kxnorb k1, k0, k0
	vpxord xmm19, xmm19, xmm19
	vpgatherqd ymm19 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	kxnorb k1, k0, k0
	vpxord xmm18, xmm18, xmm18
	vpgatherqd ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	kxnorb k1, k0, k0
	vpxord xmm17, xmm17, xmm17
	vpgatherqd ymm17 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	kxnorb k1, k0, k0
	vpxord xmm16, xmm16, xmm16
	vpgatherqd ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	kxnorb k1, k0, k0
	vpxor xmm15, xmm15, xmm15
	vpgatherqd ymm15 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	kxnorb k1, k0, k0
	vpxor xmm14, xmm14, xmm14
	vpgatherqd ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	kxnorb k1, k0, k0
	vpxor xmm13, xmm13, xmm13
	vpgatherqd ymm13 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	kxnorb k1, k0, k0
	vpxor xmm12, xmm12, xmm12
	vpgatherqd ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	kxnorb k1, k0, k0
	vpxor xmm11, xmm11, xmm11
	vpgatherqd ymm11 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	kxnorb k1, k0, k0
	vpxor xmm10, xmm10, xmm10
	vpgatherqd ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	kxnorb k1, k0, k0
	vpxor xmm9, xmm9, xmm9
	vpgatherqd ymm9 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	kxnorb k1, k0, k0
	vpxor xmm8, xmm8, xmm8
	vpgatherqd ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	kxnorb k1, k0, k0
	vpxor xmm7, xmm7, xmm7
	vpgatherqd ymm7 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	vmovdqu ymmword ptr [rsp], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0]
	vpaddd ymm31, ymm1, ymm31
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 4]
	vpaddd ymm30, ymm1, ymm30
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 8]
	vpaddd ymm29, ymm1, ymm29
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 12]
	vpaddd ymm28, ymm1, ymm28
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 16]
	vpaddd ymm27, ymm1, ymm27
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 20]
	vpaddd ymm26, ymm1, ymm26
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 24]
	vpaddd ymm25, ymm1, ymm25
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 28]
	vpaddd ymm24, ymm1, ymm24
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 32]
	vpaddd ymm23, ymm1, ymm23
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 36]
	vpaddd ymm22, ymm1, ymm22
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 40]
	vpaddd ymm21, ymm1, ymm21
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 44]
	vpaddd ymm20, ymm1, ymm20
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 48]
	vpaddd ymm19, ymm1, ymm19
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 52]
	vpaddd ymm18, ymm1, ymm18
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 56]
	vpaddd ymm17, ymm1, ymm17
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 60]
	vpaddd ymm1, ymm1, ymm16
	vmovdqu ymmword ptr [rsp - 128], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 64]
	vpaddd ymm1, ymm15, ymm1
	vmovdqu ymmword ptr [rsp + 224], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 68]
	vpaddd ymm1, ymm14, ymm1
	vmovdqu ymmword ptr [rsp + 160], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 72]
	vpaddd ymm1, ymm13, ymm1
	vmovdqu ymmword ptr [rsp - 64], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 76]
	vpaddd ymm1, ymm12, ymm1
	vmovdqu ymmword ptr [rsp + 96], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 80]
	vpaddd ymm1, ymm11, ymm1
	vmovdqu ymmword ptr [rsp + 32], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 84]
	vpaddd ymm1, ymm10, ymm1
	vmovdqu ymmword ptr [rsp + 576], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 88]
	vpaddd ymm1, ymm9, ymm1
	vmovdqu ymmword ptr [rsp + 544], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 92]
	vpaddd ymm1, ymm8, ymm1
	vmovdqu ymmword ptr [rsp + 512], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 96]
	vpaddd ymm1, ymm1, ymm7
	vmovdqu ymmword ptr [rsp + 480], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 100]
	vpaddd ymm1, ymm1, ymm6
	vmovdqu ymmword ptr [rsp + 448], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 104]
	vpaddd ymm1, ymm1, ymm5
	vmovdqu ymmword ptr [rsp + 416], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 108]
	vpaddd ymm1, ymm1, ymm4
	vmovdqu ymmword ptr [rsp + 384], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 112]
	vpaddd ymm1, ymm1, ymm3
	vmovdqu ymmword ptr [rsp + 352], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 116]
	vpaddd ymm1, ymm1, ymm2
	vmovdqu ymmword ptr [rsp + 320], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 120]
	kxnorb k1, k0, k0
	vpaddd ymm1, ymm1, ymmword ptr [rsp]
	vmovdqu ymmword ptr [rsp], ymm1
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 124]
	vpaddd ymm1, ymm2, ymm1
	vmovdqu ymmword ptr [rsp + 288], ymm1
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm0]
	vpmulld ymm2, ymm2, ymm31
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r10 + 4*zmm0 + 4]
	vpmulld ymm3, ymm3, ymm30
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [r10 + 4*zmm0 + 8]
	vpmulld ymm4, ymm4, ymm29
	kxnorb k1, k0, k0
	vpxor xmm5, xmm5, xmm5
	vpgatherqd ymm5 {k1}, dword ptr [r10 + 4*zmm0 + 12]
	vpmulld ymm5, ymm5, ymm28
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r10 + 4*zmm0 + 16]
	vpmulld ymm6, ymm6, ymm27
	kxnorb k1, k0, k0
	vpxor xmm7, xmm7, xmm7
	vpgatherqd ymm7 {k1}, dword ptr [r10 + 4*zmm0 + 20]
	vpmulld ymm7, ymm7, ymm26
	kxnorb k1, k0, k0
	vpxor xmm8, xmm8, xmm8
	vpgatherqd ymm8 {k1}, dword ptr [r10 + 4*zmm0 + 24]
	vpmulld ymm8, ymm8, ymm25
	kxnorb k1, k0, k0
	vpxor xmm9, xmm9, xmm9
	vpgatherqd ymm9 {k1}, dword ptr [r10 + 4*zmm0 + 28]
	vpmulld ymm9, ymm9, ymm24
	kxnorb k1, k0, k0
	vpxor xmm10, xmm10, xmm10
	vpgatherqd ymm10 {k1}, dword ptr [r10 + 4*zmm0 + 32]
	vpmulld ymm10, ymm10, ymm23
	kxnorb k1, k0, k0
	vpxor xmm11, xmm11, xmm11
	vpgatherqd ymm11 {k1}, dword ptr [r10 + 4*zmm0 + 36]
	vpmulld ymm11, ymm11, ymm22
	kxnorb k1, k0, k0
	vpxor xmm12, xmm12, xmm12
	vpgatherqd ymm12 {k1}, dword ptr [r10 + 4*zmm0 + 40]
	vpmulld ymm12, ymm12, ymm21
	kxnorb k1, k0, k0
	vpxor xmm13, xmm13, xmm13
	vpgatherqd ymm13 {k1}, dword ptr [r10 + 4*zmm0 + 44]
	vpmulld ymm13, ymm13, ymm20
	kxnorb k1, k0, k0
	vpxor xmm14, xmm14, xmm14
	vpgatherqd ymm14 {k1}, dword ptr [r10 + 4*zmm0 + 48]
	vpmulld ymm14, ymm14, ymm19
	kxnorb k1, k0, k0
	vpxor xmm15, xmm15, xmm15
	vpgatherqd ymm15 {k1}, dword ptr [r10 + 4*zmm0 + 52]
	vpmulld ymm15, ymm15, ymm18
	kxnorb k1, k0, k0
	vpxord xmm16, xmm16, xmm16
	vpgatherqd ymm16 {k1}, dword ptr [r10 + 4*zmm0 + 56]
	vpmulld ymm16, ymm16, ymm17
	kxnorb k1, k0, k0
	vpxord xmm17, xmm17, xmm17
	vpgatherqd ymm17 {k1}, dword ptr [r10 + 4*zmm0 + 60]
	vpmulld ymm17, ymm17, ymmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vpxord xmm18, xmm18, xmm18
	vpgatherqd ymm18 {k1}, dword ptr [r10 + 4*zmm0 + 64]
	vpmulld ymm18, ymm18, ymmword ptr [rsp + 224]
	kxnorb k1, k0, k0
	vpxord xmm19, xmm19, xmm19
	vpgatherqd ymm19 {k1}, dword ptr [r10 + 4*zmm0 + 68]
	vpmulld ymm19, ymm19, ymmword ptr [rsp + 160]
	kxnorb k1, k0, k0
	vpxord xmm20, xmm20, xmm20
	vpgatherqd ymm20 {k1}, dword ptr [r10 + 4*zmm0 + 72]
	vpmulld ymm20, ymm20, ymmword ptr [rsp - 64]
	kxnorb k1, k0, k0
	vpxord xmm21, xmm21, xmm21
	vpgatherqd ymm21 {k1}, dword ptr [r10 + 4*zmm0 + 76]
	vpmulld ymm21, ymm21, ymmword ptr [rsp + 96]
	kxnorb k1, k0, k0
	vpxord xmm22, xmm22, xmm22
	vpgatherqd ymm22 {k1}, dword ptr [r10 + 4*zmm0 + 80]
	vpmulld ymm22, ymm22, ymmword ptr [rsp + 32]
	kxnorb k1, k0, k0
	vpxord xmm23, xmm23, xmm23
	vpgatherqd ymm23 {k1}, dword ptr [r10 + 4*zmm0 + 84]
	vpmulld ymm23, ymm23, ymmword ptr [rsp + 576]
	kxnorb k1, k0, k0
	vpxord xmm24, xmm24, xmm24
	vpgatherqd ymm24 {k1}, dword ptr [r10 + 4*zmm0 + 88]
	vpmulld ymm24, ymm24, ymmword ptr [rsp + 544]
	kxnorb k1, k0, k0
	vpxord xmm25, xmm25, xmm25
	vpgatherqd ymm25 {k1}, dword ptr [r10 + 4*zmm0 + 92]
	vpmulld ymm25, ymm25, ymmword ptr [rsp + 512]
	kxnorb k1, k0, k0
	vpxord xmm26, xmm26, xmm26
	vpgatherqd ymm26 {k1}, dword ptr [r10 + 4*zmm0 + 96]
	vpmulld ymm26, ymm26, ymmword ptr [rsp + 480]
	kxnorb k1, k0, k0
	vpxord xmm27, xmm27, xmm27
	vpgatherqd ymm27 {k1}, dword ptr [r10 + 4*zmm0 + 100]
	vpmulld ymm27, ymm27, ymmword ptr [rsp + 448]
	kxnorb k1, k0, k0
	vpxord xmm28, xmm28, xmm28
	vpgatherqd ymm28 {k1}, dword ptr [r10 + 4*zmm0 + 104]
	vpmulld ymm28, ymm28, ymmword ptr [rsp + 416]
	kxnorb k1, k0, k0
	vpxord xmm29, xmm29, xmm29
	vpgatherqd ymm29 {k1}, dword ptr [r10 + 4*zmm0 + 108]
	vpmulld ymm29, ymm29, ymmword ptr [rsp + 384]
	kxnorb k1, k0, k0
	vpxord xmm30, xmm30, xmm30
	vpgatherqd ymm30 {k1}, dword ptr [r10 + 4*zmm0 + 112]
	vpmulld ymm30, ymm30, ymmword ptr [rsp + 352]
	kxnorb k1, k0, k0
	vpxord xmm31, xmm31, xmm31
	vpgatherqd ymm31 {k1}, dword ptr [r10 + 4*zmm0 + 116]
	vpmulld ymm31, ymm31, ymmword ptr [rsp + 320]
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r10 + 4*zmm0 + 120]
	vpmulld ymm1, ymm1, ymmword ptr [rsp]
	vmovdqu ymmword ptr [rsp - 128], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r10 + 4*zmm0 + 124]
	vpmulld ymm1, ymm1, ymmword ptr [rsp + 288]
	vmovdqu ymmword ptr [rsp + 224], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0]
	vpsubd ymm1, ymm2, ymm1
	vmovdqu ymmword ptr [rsp + 160], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 4]
	vpsubd ymm1, ymm3, ymm1
	vmovdqu ymmword ptr [rsp - 64], ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 8]
	vpsubd ymm4, ymm4, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 12]
	vpsubd ymm5, ymm5, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 16]
	vpsubd ymm6, ymm6, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 20]
	vpsubd ymm7, ymm7, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 24]
	vpsubd ymm8, ymm8, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 28]
	vpsubd ymm9, ymm9, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 32]
	vpsubd ymm10, ymm10, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 36]
	vpsubd ymm11, ymm11, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 40]
	vpsubd ymm12, ymm12, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 44]
	vpsubd ymm13, ymm13, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 48]
	vpsubd ymm14, ymm14, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 52]
	vpsubd ymm15, ymm15, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 56]
	vpsubd ymm16, ymm16, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 60]
	vpsubd ymm17, ymm17, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 64]
	vpsubd ymm18, ymm18, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 68]
	vpsubd ymm19, ymm19, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 72]
	vpsubd ymm20, ymm20, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 76]
	vpsubd ymm21, ymm21, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 80]
	vpsubd ymm22, ymm22, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 84]
	vpsubd ymm23, ymm23, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 88]
	vpsubd ymm24, ymm24, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 92]
	vpsubd ymm25, ymm25, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 96]
	vpsubd ymm26, ymm26, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 100]
	vpsubd ymm27, ymm27, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 104]
	vpsubd ymm28, ymm28, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 108]
	vpsubd ymm29, ymm29, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 112]
	vpsubd ymm30, ymm30, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 116]
	vpsubd ymm31, ymm31, ymm1
	kxnorb k1, k0, k0
	vpxor xmm1, xmm1, xmm1
	vpgatherqd ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 120]
	vmovdqu ymm2, ymmword ptr [rsp - 128]
	vpsubd ymm1, ymm2, ymm1
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm0 + 124]
	vmovdqu ymm3, ymmword ptr [rsp + 224]
	vpsubd ymm2, ymm3, ymm2
	kxnorb k1, k0, k0
	vmovdqu ymm3, ymmword ptr [rsp + 160]
	vpscatterqd dword ptr [rsi + 4*zmm0] {k1}, ymm3
	kxnorb k1, k0, k0
	vmovdqu ymm3, ymmword ptr [rsp - 64]
	vpscatterqd dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm3
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm4
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm5
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm6
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm7
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm8
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm9
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm10
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm11
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm12
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm13
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm14
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm15
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm16
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm17
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm18
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm19
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm20
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm21
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm22
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm23
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm24
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm25
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm26
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm27
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm28
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm29
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm30
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm31
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm2
	vpaddq zmm0, zmm0, qword ptr [rip + .LCPI160_3]{1to8}
	add r15, 8
	jne .LBB160_7
	cmp rbx, r14
	jne .LBB160_13
	jmp .LBB160_14
.LBB160_9:
	mov r15, rbx
	and r15, r14
	mov rax, r15
	shl rax, 5
	vmovdqa64 zmm7, zmmword ptr [rip + .LCPI160_0]
	vmovdqa64 zmm1, zmmword ptr [rip + .LCPI160_1]
	mov r12, r15
.LBB160_10:
	vpxor xmm2, xmm2, xmm2
	kxnorb k1, k0, k0
	vpgatherqd ymm2 {k1}, dword ptr [rdx + 4*zmm7]
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm1]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 4]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 4]
	vinserti64x4 zmm5, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 8]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 8]
	vinserti64x4 zmm8, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 12]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 12]
	vinserti64x4 zmm9, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 16]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 16]
	vinserti64x4 zmm10, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 20]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 20]
	vinserti64x4 zmm11, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 24]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 24]
	vinserti64x4 zmm12, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 28]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 28]
	vinserti64x4 zmm13, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 32]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 32]
	vinserti64x4 zmm14, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 36]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 36]
	vinserti64x4 zmm15, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 40]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 40]
	vinserti64x4 zmm16, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 44]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 44]
	vinserti64x4 zmm17, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 48]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 48]
	vinserti64x4 zmm18, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 52]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 52]
	vinserti64x4 zmm19, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 56]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 56]
	vinserti64x4 zmm20, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 60]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 60]
	vinserti64x4 zmm21, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 64]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 64]
	vinserti64x4 zmm22, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 68]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 68]
	vinserti64x4 zmm23, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 72]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 72]
	vinserti64x4 zmm24, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 76]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 76]
	vinserti64x4 zmm25, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 80]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 80]
	vinserti64x4 zmm26, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 84]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 84]
	vinserti64x4 zmm27, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 88]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 88]
	vinserti64x4 zmm28, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 92]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 92]
	vinserti64x4 zmm29, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 96]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 96]
	vinserti64x4 zmm30, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 100]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 100]
	vinserti64x4 zmm31, zmm4, ymm3, 1
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 104]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 104]
	vinserti64x4 zmm0, zmm4, ymm3, 1
	vmovdqu64 zmmword ptr [rsp + 96], zmm0
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 108]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 108]
	vinserti64x4 zmm0, zmm4, ymm3, 1
	vmovdqu64 zmmword ptr [rsp - 64], zmm0
	vpxor xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vpgatherqd ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 112]
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 112]
	vinserti64x4 zmm3, zmm4, ymm3, 1
	vpxor xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vpgatherqd ymm4 {k1}, dword ptr [rdx + 4*zmm7 + 116]
	vpxor xmm0, xmm0, xmm0
	kxnorb k1, k0, k0
	vpgatherqd ymm0 {k1}, dword ptr [rdx + 4*zmm1 + 116]
	vinserti64x4 zmm4, zmm0, ymm4, 1
	vpxor xmm0, xmm0, xmm0
	kxnorb k1, k0, k0
	vpgatherqd ymm0 {k1}, dword ptr [rdx + 4*zmm7 + 120]
	vpxor xmm6, xmm6, xmm6
	kxnorb k1, k0, k0
	vpgatherqd ymm6 {k1}, dword ptr [rdx + 4*zmm1 + 120]
	vinserti64x4 zmm0, zmm6, ymm0, 1
	vmovdqu64 zmmword ptr [rsp + 32], zmm0
	vpxor xmm0, xmm0, xmm0
	kxnorb k1, k0, k0
	vpgatherqd ymm0 {k1}, dword ptr [rdx + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [rdx + 4*zmm1 + 124]
	vinserti64x4 zmm0, zmm6, ymm0, 1
	vmovdqu64 zmmword ptr [rsp - 128], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r9 + 4*zmm7]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm1]
	vinserti64x4 zmm0, zmm6, ymm0, 1
	vpaddd zmm0, zmm0, zmm2
	vmovdqu64 zmmword ptr [rsp + 224], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 4]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 4]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpaddd zmm5, zmm0, zmm5
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 8]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 8]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpaddd zmm0, zmm0, zmm8
	vmovdqu64 zmmword ptr [rsp + 160], zmm0
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r9 + 4*zmm7 + 12]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm1 + 12]
	vinserti64x4 zmm2, zmm6, ymm2, 1
	vpaddd zmm2, zmm2, zmm9
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 16]
	kxnorb k1, k0, k0
	vpxor xmm8, xmm8, xmm8
	vpgatherqd ymm8 {k1}, dword ptr [r9 + 4*zmm1 + 16]
	vinserti64x4 zmm6, zmm8, ymm6, 1
	vpaddd zmm8, zmm6, zmm10
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 20]
	kxnorb k1, k0, k0
	vpxor xmm9, xmm9, xmm9
	vpgatherqd ymm9 {k1}, dword ptr [r9 + 4*zmm1 + 20]
	vinserti64x4 zmm6, zmm9, ymm6, 1
	vpaddd zmm9, zmm6, zmm11
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 24]
	kxnorb k1, k0, k0
	vpxor xmm10, xmm10, xmm10
	vpgatherqd ymm10 {k1}, dword ptr [r9 + 4*zmm1 + 24]
	vinserti64x4 zmm6, zmm10, ymm6, 1
	vpaddd zmm10, zmm6, zmm12
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 28]
	kxnorb k1, k0, k0
	vpxor xmm11, xmm11, xmm11
	vpgatherqd ymm11 {k1}, dword ptr [r9 + 4*zmm1 + 28]
	vinserti64x4 zmm6, zmm11, ymm6, 1
	vpaddd zmm11, zmm6, zmm13
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 32]
	kxnorb k1, k0, k0
	vpxor xmm12, xmm12, xmm12
	vpgatherqd ymm12 {k1}, dword ptr [r9 + 4*zmm1 + 32]
	vinserti64x4 zmm6, zmm12, ymm6, 1
	vpaddd zmm12, zmm6, zmm14
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 36]
	kxnorb k1, k0, k0
	vpxor xmm13, xmm13, xmm13
	vpgatherqd ymm13 {k1}, dword ptr [r9 + 4*zmm1 + 36]
	vinserti64x4 zmm6, zmm13, ymm6, 1
	vpaddd zmm13, zmm6, zmm15
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 40]
	kxnorb k1, k0, k0
	vpxor xmm14, xmm14, xmm14
	vpgatherqd ymm14 {k1}, dword ptr [r9 + 4*zmm1 + 40]
	vinserti64x4 zmm6, zmm14, ymm6, 1
	vpaddd zmm14, zmm6, zmm16
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 44]
	kxnorb k1, k0, k0
	vpxor xmm15, xmm15, xmm15
	vpgatherqd ymm15 {k1}, dword ptr [r9 + 4*zmm1 + 44]
	vinserti64x4 zmm6, zmm15, ymm6, 1
	vpaddd zmm15, zmm6, zmm17
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 48]
	kxnorb k1, k0, k0
	vpxord xmm16, xmm16, xmm16
	vpgatherqd ymm16 {k1}, dword ptr [r9 + 4*zmm1 + 48]
	vinserti64x4 zmm6, zmm16, ymm6, 1
	vpaddd zmm16, zmm6, zmm18
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 52]
	kxnorb k1, k0, k0
	vpxord xmm17, xmm17, xmm17
	vpgatherqd ymm17 {k1}, dword ptr [r9 + 4*zmm1 + 52]
	vinserti64x4 zmm6, zmm17, ymm6, 1
	vpaddd zmm17, zmm6, zmm19
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 56]
	kxnorb k1, k0, k0
	vpxord xmm18, xmm18, xmm18
	vpgatherqd ymm18 {k1}, dword ptr [r9 + 4*zmm1 + 56]
	vinserti64x4 zmm6, zmm18, ymm6, 1
	vpaddd zmm18, zmm6, zmm20
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 60]
	kxnorb k1, k0, k0
	vpxord xmm19, xmm19, xmm19
	vpgatherqd ymm19 {k1}, dword ptr [r9 + 4*zmm1 + 60]
	vinserti64x4 zmm6, zmm19, ymm6, 1
	vpaddd zmm19, zmm6, zmm21
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 64]
	kxnorb k1, k0, k0
	vpxord xmm20, xmm20, xmm20
	vpgatherqd ymm20 {k1}, dword ptr [r9 + 4*zmm1 + 64]
	vinserti64x4 zmm6, zmm20, ymm6, 1
	vpaddd zmm20, zmm6, zmm22
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 68]
	kxnorb k1, k0, k0
	vpxord xmm21, xmm21, xmm21
	vpgatherqd ymm21 {k1}, dword ptr [r9 + 4*zmm1 + 68]
	vinserti64x4 zmm6, zmm21, ymm6, 1
	vpaddd zmm21, zmm6, zmm23
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 72]
	kxnorb k1, k0, k0
	vpxord xmm22, xmm22, xmm22
	vpgatherqd ymm22 {k1}, dword ptr [r9 + 4*zmm1 + 72]
	vinserti64x4 zmm6, zmm22, ymm6, 1
	vpaddd zmm22, zmm6, zmm24
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 76]
	kxnorb k1, k0, k0
	vpxord xmm23, xmm23, xmm23
	vpgatherqd ymm23 {k1}, dword ptr [r9 + 4*zmm1 + 76]
	vinserti64x4 zmm6, zmm23, ymm6, 1
	vpaddd zmm23, zmm6, zmm25
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 80]
	kxnorb k1, k0, k0
	vpxord xmm24, xmm24, xmm24
	vpgatherqd ymm24 {k1}, dword ptr [r9 + 4*zmm1 + 80]
	vinserti64x4 zmm6, zmm24, ymm6, 1
	vpaddd zmm24, zmm6, zmm26
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 84]
	kxnorb k1, k0, k0
	vpxord xmm25, xmm25, xmm25
	vpgatherqd ymm25 {k1}, dword ptr [r9 + 4*zmm1 + 84]
	vinserti64x4 zmm6, zmm25, ymm6, 1
	vpaddd zmm25, zmm6, zmm27
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 88]
	kxnorb k1, k0, k0
	vpxord xmm26, xmm26, xmm26
	vpgatherqd ymm26 {k1}, dword ptr [r9 + 4*zmm1 + 88]
	vinserti64x4 zmm6, zmm26, ymm6, 1
	vpaddd zmm26, zmm6, zmm28
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 92]
	kxnorb k1, k0, k0
	vpxord xmm27, xmm27, xmm27
	vpgatherqd ymm27 {k1}, dword ptr [r9 + 4*zmm1 + 92]
	vinserti64x4 zmm6, zmm27, ymm6, 1
	vpaddd zmm27, zmm6, zmm29
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 96]
	kxnorb k1, k0, k0
	vpxord xmm28, xmm28, xmm28
	vpgatherqd ymm28 {k1}, dword ptr [r9 + 4*zmm1 + 96]
	vinserti64x4 zmm6, zmm28, ymm6, 1
	vpaddd zmm28, zmm6, zmm30
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 100]
	kxnorb k1, k0, k0
	vpxord xmm29, xmm29, xmm29
	vpgatherqd ymm29 {k1}, dword ptr [r9 + 4*zmm1 + 100]
	vinserti64x4 zmm6, zmm29, ymm6, 1
	vpaddd zmm29, zmm6, zmm31
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 104]
	kxnorb k1, k0, k0
	vpxord xmm30, xmm30, xmm30
	vpgatherqd ymm30 {k1}, dword ptr [r9 + 4*zmm1 + 104]
	vinserti64x4 zmm6, zmm30, ymm6, 1
	vpaddd zmm30, zmm6, zmmword ptr [rsp + 96]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 108]
	kxnorb k1, k0, k0
	vpxord xmm31, xmm31, xmm31
	vpgatherqd ymm31 {k1}, dword ptr [r9 + 4*zmm1 + 108]
	vinserti64x4 zmm6, zmm31, ymm6, 1
	vpaddd zmm31, zmm6, zmmword ptr [rsp - 64]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 112]
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r9 + 4*zmm1 + 112]
	vinserti64x4 zmm0, zmm0, ymm6, 1
	vpaddd zmm0, zmm0, zmm3
	vmovdqu64 zmmword ptr [rsp - 64], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 116]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 116]
	vinserti64x4 zmm0, zmm3, ymm0, 1
	vpaddd zmm0, zmm0, zmm4
	vmovdqu64 zmmword ptr [rsp + 96], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 120]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 120]
	vinserti64x4 zmm0, zmm3, ymm0, 1
	vpaddd zmm0, zmm0, zmmword ptr [rsp + 32]
	vmovdqu64 zmmword ptr [rsp + 32], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 124]
	vinserti64x4 zmm0, zmm3, ymm0, 1
	vpaddd zmm3, zmm0, zmmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7]
	kxnorb k1, k0, k0
	vpxor xmm4, xmm4, xmm4
	vpgatherqd ymm4 {k1}, dword ptr [r10 + 4*zmm1]
	vinserti64x4 zmm0, zmm4, ymm0, 1
	vpmulld zmm4, zmm0, zmmword ptr [rsp + 224]
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 4]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r10 + 4*zmm1 + 4]
	vinserti64x4 zmm0, zmm6, ymm0, 1
	vpmulld zmm5, zmm0, zmm5
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 8]
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r10 + 4*zmm1 + 8]
	vinserti64x4 zmm0, zmm6, ymm0, 1
	vpmulld zmm0, zmm0, zmmword ptr [rsp + 160]
	vmovdqu64 zmmword ptr [rsp - 128], zmm0
	kxnorb k1, k0, k0
	vpxor xmm6, xmm6, xmm6
	vpgatherqd ymm6 {k1}, dword ptr [r10 + 4*zmm7 + 12]
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm1 + 12]
	vinserti64x4 zmm0, zmm0, ymm6, 1
	vpmulld zmm6, zmm0, zmm2
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 16]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 16]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm8, zmm0, zmm8
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 20]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 20]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm9, zmm0, zmm9
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 24]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 24]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm10, zmm0, zmm10
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 28]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 28]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm11, zmm0, zmm11
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 32]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 32]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm12, zmm0, zmm12
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 36]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 36]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm13, zmm0, zmm13
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 40]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 40]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm14, zmm0, zmm14
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 44]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 44]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm15, zmm0, zmm15
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 48]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 48]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm16, zmm0, zmm16
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 52]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 52]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm17, zmm0, zmm17
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 56]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 56]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm18, zmm0, zmm18
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 60]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 60]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm19, zmm0, zmm19
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 64]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 64]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm20, zmm0, zmm20
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 68]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 68]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm21, zmm0, zmm21
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 72]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 72]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm22, zmm0, zmm22
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 76]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 76]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm23, zmm0, zmm23
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 80]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 80]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm24, zmm0, zmm24
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 84]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 84]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm25, zmm0, zmm25
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 88]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 88]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm26, zmm0, zmm26
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 92]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 92]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm27, zmm0, zmm27
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 96]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 96]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm28, zmm0, zmm28
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 100]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 100]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm29, zmm0, zmm29
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 104]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 104]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm30, zmm0, zmm30
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 108]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 108]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm31, zmm0, zmm31
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 112]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 112]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm0, zmm0, zmmword ptr [rsp - 64]
	vmovdqu64 zmmword ptr [rsp + 224], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 116]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 116]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm0, zmm0, zmmword ptr [rsp + 96]
	vmovdqu64 zmmword ptr [rsp + 160], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 120]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 120]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm0, zmm0, zmmword ptr [rsp + 32]
	vmovdqu64 zmmword ptr [rsp - 64], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 124]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpmulld zmm0, zmm0, zmm3
	vmovdqu64 zmmword ptr [rsp + 96], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r11 + 4*zmm7]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm1]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpsubd zmm0, zmm4, zmm0
	vmovdqu64 zmmword ptr [rsp + 32], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r11 + 4*zmm7 + 4]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm1 + 4]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vpsubd zmm0, zmm5, zmm0
	vmovdqu64 zmmword ptr [rsp + 576], zmm0
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r11 + 4*zmm7 + 8]
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm1 + 8]
	vinserti64x4 zmm0, zmm2, ymm0, 1
	vmovdqu64 zmm2, zmmword ptr [rsp - 128]
	vpsubd zmm0, zmm2, zmm0
	vmovdqu64 zmmword ptr [rsp - 128], zmm0
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 12]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 12]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm6, zmm6, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 16]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 16]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm8, zmm8, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 20]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 20]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm9, zmm9, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 24]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 24]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm10, zmm10, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 28]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 28]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm11, zmm11, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 32]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 32]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm12, zmm12, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 36]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 36]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm13, zmm13, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 40]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 40]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm14, zmm14, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 44]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 44]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm15, zmm15, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 48]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 48]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm16, zmm16, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 52]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 52]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm17, zmm17, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 56]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 56]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm18, zmm18, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 60]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 60]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm19, zmm19, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 64]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 64]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm20, zmm20, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 68]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 68]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm21, zmm21, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 72]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 72]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm22, zmm22, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 76]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 76]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm23, zmm23, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 80]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 80]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm24, zmm24, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 84]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 84]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm25, zmm25, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 88]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 88]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm26, zmm26, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 92]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 92]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm27, zmm27, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 96]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 96]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm28, zmm28, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 100]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 100]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm29, zmm29, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 104]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 104]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm30, zmm30, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 108]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 108]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vpsubd zmm31, zmm31, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 112]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 112]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vmovdqu64 zmm0, zmmword ptr [rsp + 224]
	vpsubd zmm5, zmm0, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 116]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 116]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vmovdqu64 zmm0, zmmword ptr [rsp + 160]
	vpsubd zmm4, zmm0, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 120]
	kxnorb k1, k0, k0
	vpxor xmm3, xmm3, xmm3
	vpgatherqd ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 120]
	vinserti64x4 zmm2, zmm3, ymm2, 1
	vmovdqu64 zmm0, zmmword ptr [rsp - 64]
	vpsubd zmm3, zmm0, zmm2
	kxnorb k1, k0, k0
	vpxor xmm2, xmm2, xmm2
	vpgatherqd ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vpxor xmm0, xmm0, xmm0
	vpgatherqd ymm0 {k1}, dword ptr [r11 + 4*zmm1 + 124]
	vinserti64x4 zmm0, zmm0, ymm2, 1
	vmovdqu64 zmm2, zmmword ptr [rsp + 96]
	vpsubd zmm2, zmm2, zmm0
	kxnorb k1, k0, k0
	vmovdqu64 zmm0, zmmword ptr [rsp + 32]
	vpscatterqd dword ptr [rsi + 4*zmm1] {k1}, ymm0
	vextracti64x4 ymm0, zmm0, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7] {k1}, ymm0
	kxnorb k1, k0, k0
	vmovdqu64 zmm0, zmmword ptr [rsp + 576]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 4] {k1}, ymm0
	vextracti64x4 ymm0, zmm0, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 4] {k1}, ymm0
	kxnorb k1, k0, k0
	vmovdqu64 zmm0, zmmword ptr [rsp - 128]
	vpscatterqd dword ptr [rsi + 4*zmm1 + 8] {k1}, ymm0
	vextracti64x4 ymm0, zmm0, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 8] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 12] {k1}, ymm6
	vextracti64x4 ymm0, zmm6, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 12] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 16] {k1}, ymm8
	vextracti64x4 ymm0, zmm8, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 16] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 20] {k1}, ymm9
	vextracti64x4 ymm0, zmm9, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 20] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 24] {k1}, ymm10
	vextracti64x4 ymm0, zmm10, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 24] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 28] {k1}, ymm11
	vextracti64x4 ymm0, zmm11, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 28] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 32] {k1}, ymm12
	vextracti64x4 ymm0, zmm12, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 32] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 36] {k1}, ymm13
	vextracti64x4 ymm0, zmm13, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 36] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 40] {k1}, ymm14
	vextracti64x4 ymm0, zmm14, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 40] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 44] {k1}, ymm15
	vextracti64x4 ymm0, zmm15, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 44] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 48] {k1}, ymm16
	vextracti64x4 ymm0, zmm16, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 48] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 52] {k1}, ymm17
	vextracti64x4 ymm0, zmm17, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 52] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 56] {k1}, ymm18
	vextracti64x4 ymm0, zmm18, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 56] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 60] {k1}, ymm19
	vextracti64x4 ymm0, zmm19, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 60] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 64] {k1}, ymm20
	vextracti64x4 ymm0, zmm20, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 64] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 68] {k1}, ymm21
	vextracti64x4 ymm0, zmm21, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 68] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 72] {k1}, ymm22
	vextracti64x4 ymm0, zmm22, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 72] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 76] {k1}, ymm23
	vextracti64x4 ymm0, zmm23, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 76] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 80] {k1}, ymm24
	vextracti64x4 ymm0, zmm24, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 80] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 84] {k1}, ymm25
	vextracti64x4 ymm0, zmm25, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 84] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 88] {k1}, ymm26
	vextracti64x4 ymm0, zmm26, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 88] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 92] {k1}, ymm27
	vextracti64x4 ymm0, zmm27, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 92] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 96] {k1}, ymm28
	vextracti64x4 ymm0, zmm28, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 96] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 100] {k1}, ymm29
	vextracti64x4 ymm0, zmm29, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 100] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 104] {k1}, ymm30
	vextracti64x4 ymm0, zmm30, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 104] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 108] {k1}, ymm31
	vextracti64x4 ymm0, zmm31, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 108] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 112] {k1}, ymm5
	vextracti64x4 ymm0, zmm5, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 112] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 116] {k1}, ymm4
	vextracti64x4 ymm0, zmm4, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 116] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 120] {k1}, ymm3
	vextracti64x4 ymm0, zmm3, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 120] {k1}, ymm0
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm1 + 124] {k1}, ymm2
	vextracti64x4 ymm0, zmm2, 1
	kxnorb k1, k0, k0
	vpscatterqd dword ptr [rsi + 4*zmm7 + 124] {k1}, ymm0
	vpbroadcastq zmm0, qword ptr [rip + .LCPI160_2]
	vpaddq zmm1, zmm1, zmm0
	vpaddq zmm7, zmm7, zmm0
	add r12, -16
	jne .LBB160_10
	cmp rbx, r15
	je .LBB160_14
	test bl, 8
	jne .LBB160_6
.LBB160_13:
	vmovdqu64 zmm0, zmmword ptr [r9 + 4*rax]
	vmovdqu64 zmm1, zmmword ptr [r9 + 4*rax + 64]
	vpaddd zmm0, zmm0, zmmword ptr [rdx + 4*rax]
	vpmulld zmm0, zmm0, zmmword ptr [r10 + 4*rax]
	vpsubd zmm0, zmm0, zmmword ptr [r11 + 4*rax]
	vpaddd zmm1, zmm1, zmmword ptr [rdx + 4*rax + 64]
	vpmulld zmm1, zmm1, zmmword ptr [r10 + 4*rax + 64]
	vpsubd zmm1, zmm1, zmmword ptr [r11 + 4*rax + 64]
	vmovdqu64 zmmword ptr [rsi + 4*rax], zmm0
	vmovdqu64 zmmword ptr [rsi + 4*rax + 64], zmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB160_13
.LBB160_14:
	mov r10, r8
	sub r10, rax
	jbe .LBB160_33
	mov rbp, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r10, 8
	jb .LBB160_16
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
	je .LBB160_19
.LBB160_16:
	mov r11, rax
.LBB160_28:
	mov ecx, r8d
	sub ecx, r11d
	lea rax, [r11 + 1]
	test cl, 1
	je .LBB160_30
	mov ecx, dword ptr [rdx + 4*r11]
	add ecx, dword ptr [rbp + 4*r11]
	imul ecx, dword ptr [r9 + 4*r11]
	sub ecx, dword ptr [rdi + 4*r11]
	mov dword ptr [rsi + 4*r11], ecx
	mov r11, rax
.LBB160_30:
	cmp r8, rax
	je .LBB160_33
	sub r8, r11
	lea rax, [rsi + 4*r11]
	add rax, 4
	lea rsi, [rdi + 4*r11 + 4]
	lea rdi, [r9 + 4*r11]
	add rdi, 4
	lea rcx, [rdx + 4*r11]
	add rcx, 4
	lea rdx, [4*r11 + 4]
	add rdx, rbp
	xor r10d, r10d
.LBB160_32:
	mov r9d, dword ptr [rcx + 4*r10 - 4]
	add r9d, dword ptr [rdx + 4*r10 - 4]
	imul r9d, dword ptr [rdi + 4*r10 - 4]
	sub r9d, dword ptr [rsi + 4*r10 - 4]
	mov dword ptr [rax + 4*r10 - 4], r9d
	mov r9d, dword ptr [rcx + 4*r10]
	add r9d, dword ptr [rdx + 4*r10]
	imul r9d, dword ptr [rdi + 4*r10]
	sub r9d, dword ptr [rsi + 4*r10]
	mov dword ptr [rax + 4*r10], r9d
	add r10, 2
	cmp r8, r10
	jne .LBB160_32
.LBB160_33:
	add rsp, 648
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
.LBB160_19:
	cmp r10, 64
	jae .LBB160_21
	xor ebx, ebx
	jmp .LBB160_25
.LBB160_21:
	mov rbx, r10
	and rbx, -64
	lea r11, [rax + rbx]
	lea r14, [rsi + 4*rax]
	add r14, 192
	lea r15, [rdi + 4*rax + 192]
	lea r12, [r9 + 4*rax + 192]
	lea r13, [rdx + 4*rax]
	add r13, 192
	mov qword ptr [rsp - 128], rbp
	lea rbp, [rbp + 4*rax + 192]
	xor ecx, ecx
.LBB160_22:
	vmovdqu64 zmm0, zmmword ptr [r13 + 4*rcx - 192]
	vmovdqu64 zmm1, zmmword ptr [r13 + 4*rcx - 128]
	vmovdqu64 zmm2, zmmword ptr [r13 + 4*rcx - 64]
	vmovdqu64 zmm3, zmmword ptr [r13 + 4*rcx]
	vpaddd zmm0, zmm0, zmmword ptr [rbp + 4*rcx - 192]
	vpaddd zmm1, zmm1, zmmword ptr [rbp + 4*rcx - 128]
	vpaddd zmm2, zmm2, zmmword ptr [rbp + 4*rcx - 64]
	vpaddd zmm3, zmm3, zmmword ptr [rbp + 4*rcx]
	vpmulld zmm0, zmm0, zmmword ptr [r12 + 4*rcx - 192]
	vpmulld zmm1, zmm1, zmmword ptr [r12 + 4*rcx - 128]
	vpmulld zmm2, zmm2, zmmword ptr [r12 + 4*rcx - 64]
	vpmulld zmm3, zmm3, zmmword ptr [r12 + 4*rcx]
	vpsubd zmm0, zmm0, zmmword ptr [r15 + 4*rcx - 192]
	vpsubd zmm1, zmm1, zmmword ptr [r15 + 4*rcx - 128]
	vpsubd zmm2, zmm2, zmmword ptr [r15 + 4*rcx - 64]
	vpsubd zmm3, zmm3, zmmword ptr [r15 + 4*rcx]
	vmovdqu64 zmmword ptr [r14 + 4*rcx - 192], zmm0
	vmovdqu64 zmmword ptr [r14 + 4*rcx - 128], zmm1
	vmovdqu64 zmmword ptr [r14 + 4*rcx - 64], zmm2
	vmovdqu64 zmmword ptr [r14 + 4*rcx], zmm3
	add rcx, 64
	cmp rbx, rcx
	jne .LBB160_22
	cmp r10, rbx
	mov rbp, qword ptr [rsp - 128]
	je .LBB160_33
	test r10b, 56
	je .LBB160_28
.LBB160_25:
	mov rcx, r10
	and rcx, -8
	lea r11, [rax + rcx]
	lea r14, [rsi + 4*rax]
	lea r15, [rdi + 4*rax]
	lea r12, [r9 + 4*rax]
	lea r13, [rdx + 4*rax]
	lea rax, [4*rax]
	add rax, rbp
.LBB160_26:
	vmovdqu ymm0, ymmword ptr [r13 + 4*rbx]
	vpaddd ymm0, ymm0, ymmword ptr [rax + 4*rbx]
	vpmulld ymm0, ymm0, ymmword ptr [r12 + 4*rbx]
	vpsubd ymm0, ymm0, ymmword ptr [r15 + 4*rbx]
	vmovdqu ymmword ptr [r14 + 4*rbx], ymm0
	add rbx, 8
	cmp rcx, rbx
	jne .LBB160_26
	cmp r10, rcx
	je .LBB160_33
	jmp .LBB160_28
