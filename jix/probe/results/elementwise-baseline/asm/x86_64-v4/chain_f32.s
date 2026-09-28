jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	sub rsp, 648
	mov rcx, r8
	and rcx, -32
	je .LBB144_3
	mov rdx, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	lea rax, [r8 - 32]
	cmp rax, 224
	jae .LBB144_4
	xor eax, eax
	jmp .LBB144_13
.LBB144_3:
	xor eax, eax
	jmp .LBB144_14
.LBB144_4:
	mov rbx, rax
	shr rbx, 5
	inc rbx
	movabs r14, 1152921504606846960
	cmp rax, 480
	jae .LBB144_9
	xor r15d, r15d
	xor eax, eax
.LBB144_6:
	add r14, 8
	and r14, rbx
	vpbroadcastq zmm0, rax
	mov rax, r14
	vpaddq zmm0, zmm0, zmmword ptr [rip + .LCPI144_1]
	shl rax, 5
	sub r15, r14
.LBB144_7:
	kxnorb k1, k0, k0
	vxorps xmm31, xmm31, xmm31
	vgatherqps ymm31 {k1}, dword ptr [rdx + 4*zmm0]
	kxnorb k1, k0, k0
	vxorps xmm30, xmm30, xmm30
	vgatherqps ymm30 {k1}, dword ptr [rdx + 4*zmm0 + 4]
	kxnorb k1, k0, k0
	vxorps xmm29, xmm29, xmm29
	vgatherqps ymm29 {k1}, dword ptr [rdx + 4*zmm0 + 8]
	kxnorb k1, k0, k0
	vxorps xmm28, xmm28, xmm28
	vgatherqps ymm28 {k1}, dword ptr [rdx + 4*zmm0 + 12]
	kxnorb k1, k0, k0
	vxorps xmm27, xmm27, xmm27
	vgatherqps ymm27 {k1}, dword ptr [rdx + 4*zmm0 + 16]
	kxnorb k1, k0, k0
	vxorps xmm26, xmm26, xmm26
	vgatherqps ymm26 {k1}, dword ptr [rdx + 4*zmm0 + 20]
	kxnorb k1, k0, k0
	vxorps xmm25, xmm25, xmm25
	vgatherqps ymm25 {k1}, dword ptr [rdx + 4*zmm0 + 24]
	kxnorb k1, k0, k0
	vxorps xmm24, xmm24, xmm24
	vgatherqps ymm24 {k1}, dword ptr [rdx + 4*zmm0 + 28]
	kxnorb k1, k0, k0
	vxorps xmm23, xmm23, xmm23
	vgatherqps ymm23 {k1}, dword ptr [rdx + 4*zmm0 + 32]
	kxnorb k1, k0, k0
	vxorps xmm22, xmm22, xmm22
	vgatherqps ymm22 {k1}, dword ptr [rdx + 4*zmm0 + 36]
	kxnorb k1, k0, k0
	vxorps xmm21, xmm21, xmm21
	vgatherqps ymm21 {k1}, dword ptr [rdx + 4*zmm0 + 40]
	kxnorb k1, k0, k0
	vxorps xmm20, xmm20, xmm20
	vgatherqps ymm20 {k1}, dword ptr [rdx + 4*zmm0 + 44]
	kxnorb k1, k0, k0
	vxorps xmm19, xmm19, xmm19
	vgatherqps ymm19 {k1}, dword ptr [rdx + 4*zmm0 + 48]
	kxnorb k1, k0, k0
	vxorps xmm18, xmm18, xmm18
	vgatherqps ymm18 {k1}, dword ptr [rdx + 4*zmm0 + 52]
	kxnorb k1, k0, k0
	vxorps xmm17, xmm17, xmm17
	vgatherqps ymm17 {k1}, dword ptr [rdx + 4*zmm0 + 56]
	kxnorb k1, k0, k0
	vxorps xmm16, xmm16, xmm16
	vgatherqps ymm16 {k1}, dword ptr [rdx + 4*zmm0 + 60]
	kxnorb k1, k0, k0
	vxorps xmm15, xmm15, xmm15
	vgatherqps ymm15 {k1}, dword ptr [rdx + 4*zmm0 + 64]
	kxnorb k1, k0, k0
	vxorps xmm14, xmm14, xmm14
	vgatherqps ymm14 {k1}, dword ptr [rdx + 4*zmm0 + 68]
	kxnorb k1, k0, k0
	vxorps xmm13, xmm13, xmm13
	vgatherqps ymm13 {k1}, dword ptr [rdx + 4*zmm0 + 72]
	kxnorb k1, k0, k0
	vxorps xmm12, xmm12, xmm12
	vgatherqps ymm12 {k1}, dword ptr [rdx + 4*zmm0 + 76]
	kxnorb k1, k0, k0
	vxorps xmm11, xmm11, xmm11
	vgatherqps ymm11 {k1}, dword ptr [rdx + 4*zmm0 + 80]
	kxnorb k1, k0, k0
	vxorps xmm10, xmm10, xmm10
	vgatherqps ymm10 {k1}, dword ptr [rdx + 4*zmm0 + 84]
	kxnorb k1, k0, k0
	vxorps xmm9, xmm9, xmm9
	vgatherqps ymm9 {k1}, dword ptr [rdx + 4*zmm0 + 88]
	kxnorb k1, k0, k0
	vxorps xmm8, xmm8, xmm8
	vgatherqps ymm8 {k1}, dword ptr [rdx + 4*zmm0 + 92]
	kxnorb k1, k0, k0
	vxorps xmm7, xmm7, xmm7
	vgatherqps ymm7 {k1}, dword ptr [rdx + 4*zmm0 + 96]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [rdx + 4*zmm0 + 100]
	kxnorb k1, k0, k0
	vxorps xmm5, xmm5, xmm5
	vgatherqps ymm5 {k1}, dword ptr [rdx + 4*zmm0 + 104]
	kxnorb k1, k0, k0
	vxorps xmm4, xmm4, xmm4
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm0 + 108]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm0 + 112]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm0 + 116]
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 120]
	vmovups ymmword ptr [rsp], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0]
	vaddps ymm31, ymm31, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 4]
	vaddps ymm30, ymm30, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 8]
	vaddps ymm29, ymm29, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 12]
	vaddps ymm28, ymm28, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 16]
	vaddps ymm27, ymm27, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 20]
	vaddps ymm26, ymm26, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 24]
	vaddps ymm25, ymm25, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 28]
	vaddps ymm24, ymm24, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 32]
	vaddps ymm23, ymm23, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 36]
	vaddps ymm22, ymm22, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 40]
	vaddps ymm21, ymm21, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 44]
	vaddps ymm20, ymm20, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 48]
	vaddps ymm19, ymm19, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 52]
	vaddps ymm18, ymm18, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 56]
	vaddps ymm17, ymm17, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 60]
	vaddps ymm1, ymm16, ymm1
	vmovups ymmword ptr [rsp - 128], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 64]
	vaddps ymm1, ymm15, ymm1
	vmovups ymmword ptr [rsp + 224], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 68]
	vaddps ymm1, ymm14, ymm1
	vmovups ymmword ptr [rsp + 160], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 72]
	vaddps ymm1, ymm13, ymm1
	vmovups ymmword ptr [rsp - 64], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 76]
	vaddps ymm1, ymm12, ymm1
	vmovups ymmword ptr [rsp + 96], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 80]
	vaddps ymm1, ymm11, ymm1
	vmovups ymmword ptr [rsp + 32], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 84]
	vaddps ymm1, ymm10, ymm1
	vmovups ymmword ptr [rsp + 576], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 88]
	vaddps ymm1, ymm9, ymm1
	vmovups ymmword ptr [rsp + 544], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 92]
	vaddps ymm1, ymm8, ymm1
	vmovups ymmword ptr [rsp + 512], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 96]
	vaddps ymm1, ymm7, ymm1
	vmovups ymmword ptr [rsp + 480], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 100]
	vaddps ymm1, ymm6, ymm1
	vmovups ymmword ptr [rsp + 448], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 104]
	vaddps ymm1, ymm5, ymm1
	vmovups ymmword ptr [rsp + 416], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 108]
	vaddps ymm1, ymm4, ymm1
	vmovups ymmword ptr [rsp + 384], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 112]
	vaddps ymm1, ymm3, ymm1
	vmovups ymmword ptr [rsp + 352], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 116]
	vaddps ymm1, ymm2, ymm1
	vmovups ymmword ptr [rsp + 320], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r9 + 4*zmm0 + 120]
	kxnorb k1, k0, k0
	vaddps ymm1, ymm1, ymmword ptr [rsp]
	vmovups ymmword ptr [rsp], ymm1
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [rdx + 4*zmm0 + 124]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm0 + 124]
	vaddps ymm1, ymm1, ymm2
	vmovups ymmword ptr [rsp + 288], ymm1
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm0]
	vmulps ymm2, ymm31, ymm2
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r10 + 4*zmm0 + 4]
	vmulps ymm3, ymm30, ymm3
	kxnorb k1, k0, k0
	vxorps xmm4, xmm4, xmm4
	vgatherqps ymm4 {k1}, dword ptr [r10 + 4*zmm0 + 8]
	vmulps ymm4, ymm29, ymm4
	kxnorb k1, k0, k0
	vxorps xmm5, xmm5, xmm5
	vgatherqps ymm5 {k1}, dword ptr [r10 + 4*zmm0 + 12]
	vmulps ymm5, ymm28, ymm5
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r10 + 4*zmm0 + 16]
	vmulps ymm6, ymm27, ymm6
	kxnorb k1, k0, k0
	vxorps xmm7, xmm7, xmm7
	vgatherqps ymm7 {k1}, dword ptr [r10 + 4*zmm0 + 20]
	vmulps ymm7, ymm26, ymm7
	kxnorb k1, k0, k0
	vxorps xmm8, xmm8, xmm8
	vgatherqps ymm8 {k1}, dword ptr [r10 + 4*zmm0 + 24]
	vmulps ymm8, ymm25, ymm8
	kxnorb k1, k0, k0
	vxorps xmm9, xmm9, xmm9
	vgatherqps ymm9 {k1}, dword ptr [r10 + 4*zmm0 + 28]
	vmulps ymm9, ymm24, ymm9
	kxnorb k1, k0, k0
	vxorps xmm10, xmm10, xmm10
	vgatherqps ymm10 {k1}, dword ptr [r10 + 4*zmm0 + 32]
	vmulps ymm10, ymm23, ymm10
	kxnorb k1, k0, k0
	vxorps xmm11, xmm11, xmm11
	vgatherqps ymm11 {k1}, dword ptr [r10 + 4*zmm0 + 36]
	vmulps ymm11, ymm22, ymm11
	kxnorb k1, k0, k0
	vxorps xmm12, xmm12, xmm12
	vgatherqps ymm12 {k1}, dword ptr [r10 + 4*zmm0 + 40]
	vmulps ymm12, ymm21, ymm12
	kxnorb k1, k0, k0
	vxorps xmm13, xmm13, xmm13
	vgatherqps ymm13 {k1}, dword ptr [r10 + 4*zmm0 + 44]
	vmulps ymm13, ymm20, ymm13
	kxnorb k1, k0, k0
	vxorps xmm14, xmm14, xmm14
	vgatherqps ymm14 {k1}, dword ptr [r10 + 4*zmm0 + 48]
	vmulps ymm14, ymm19, ymm14
	kxnorb k1, k0, k0
	vxorps xmm15, xmm15, xmm15
	vgatherqps ymm15 {k1}, dword ptr [r10 + 4*zmm0 + 52]
	vmulps ymm15, ymm18, ymm15
	kxnorb k1, k0, k0
	vxorps xmm16, xmm16, xmm16
	vgatherqps ymm16 {k1}, dword ptr [r10 + 4*zmm0 + 56]
	vmulps ymm16, ymm17, ymm16
	kxnorb k1, k0, k0
	vxorps xmm17, xmm17, xmm17
	vgatherqps ymm17 {k1}, dword ptr [r10 + 4*zmm0 + 60]
	vmulps ymm17, ymm17, ymmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vxorps xmm18, xmm18, xmm18
	vgatherqps ymm18 {k1}, dword ptr [r10 + 4*zmm0 + 64]
	vmulps ymm18, ymm18, ymmword ptr [rsp + 224]
	kxnorb k1, k0, k0
	vxorps xmm19, xmm19, xmm19
	vgatherqps ymm19 {k1}, dword ptr [r10 + 4*zmm0 + 68]
	vmulps ymm19, ymm19, ymmword ptr [rsp + 160]
	kxnorb k1, k0, k0
	vxorps xmm20, xmm20, xmm20
	vgatherqps ymm20 {k1}, dword ptr [r10 + 4*zmm0 + 72]
	vmulps ymm20, ymm20, ymmword ptr [rsp - 64]
	kxnorb k1, k0, k0
	vxorps xmm21, xmm21, xmm21
	vgatherqps ymm21 {k1}, dword ptr [r10 + 4*zmm0 + 76]
	vmulps ymm21, ymm21, ymmword ptr [rsp + 96]
	kxnorb k1, k0, k0
	vxorps xmm22, xmm22, xmm22
	vgatherqps ymm22 {k1}, dword ptr [r10 + 4*zmm0 + 80]
	vmulps ymm22, ymm22, ymmword ptr [rsp + 32]
	kxnorb k1, k0, k0
	vxorps xmm23, xmm23, xmm23
	vgatherqps ymm23 {k1}, dword ptr [r10 + 4*zmm0 + 84]
	vmulps ymm23, ymm23, ymmword ptr [rsp + 576]
	kxnorb k1, k0, k0
	vxorps xmm24, xmm24, xmm24
	vgatherqps ymm24 {k1}, dword ptr [r10 + 4*zmm0 + 88]
	vmulps ymm24, ymm24, ymmword ptr [rsp + 544]
	kxnorb k1, k0, k0
	vxorps xmm25, xmm25, xmm25
	vgatherqps ymm25 {k1}, dword ptr [r10 + 4*zmm0 + 92]
	vmulps ymm25, ymm25, ymmword ptr [rsp + 512]
	kxnorb k1, k0, k0
	vxorps xmm26, xmm26, xmm26
	vgatherqps ymm26 {k1}, dword ptr [r10 + 4*zmm0 + 96]
	vmulps ymm26, ymm26, ymmword ptr [rsp + 480]
	kxnorb k1, k0, k0
	vxorps xmm27, xmm27, xmm27
	vgatherqps ymm27 {k1}, dword ptr [r10 + 4*zmm0 + 100]
	vmulps ymm27, ymm27, ymmword ptr [rsp + 448]
	kxnorb k1, k0, k0
	vxorps xmm28, xmm28, xmm28
	vgatherqps ymm28 {k1}, dword ptr [r10 + 4*zmm0 + 104]
	vmulps ymm28, ymm28, ymmword ptr [rsp + 416]
	kxnorb k1, k0, k0
	vxorps xmm29, xmm29, xmm29
	vgatherqps ymm29 {k1}, dword ptr [r10 + 4*zmm0 + 108]
	vmulps ymm29, ymm29, ymmword ptr [rsp + 384]
	kxnorb k1, k0, k0
	vxorps xmm30, xmm30, xmm30
	vgatherqps ymm30 {k1}, dword ptr [r10 + 4*zmm0 + 112]
	vmulps ymm30, ymm30, ymmword ptr [rsp + 352]
	kxnorb k1, k0, k0
	vxorps xmm31, xmm31, xmm31
	vgatherqps ymm31 {k1}, dword ptr [r10 + 4*zmm0 + 116]
	vmulps ymm31, ymm31, ymmword ptr [rsp + 320]
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r10 + 4*zmm0 + 120]
	vmulps ymm1, ymm1, ymmword ptr [rsp]
	vmovups ymmword ptr [rsp - 128], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r10 + 4*zmm0 + 124]
	vmulps ymm1, ymm1, ymmword ptr [rsp + 288]
	vmovups ymmword ptr [rsp + 224], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0]
	vsubps ymm1, ymm2, ymm1
	vmovups ymmword ptr [rsp + 160], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 4]
	vsubps ymm1, ymm3, ymm1
	vmovups ymmword ptr [rsp - 64], ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 8]
	vsubps ymm4, ymm4, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 12]
	vsubps ymm5, ymm5, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 16]
	vsubps ymm6, ymm6, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 20]
	vsubps ymm7, ymm7, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 24]
	vsubps ymm8, ymm8, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 28]
	vsubps ymm9, ymm9, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 32]
	vsubps ymm10, ymm10, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 36]
	vsubps ymm11, ymm11, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 40]
	vsubps ymm12, ymm12, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 44]
	vsubps ymm13, ymm13, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 48]
	vsubps ymm14, ymm14, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 52]
	vsubps ymm15, ymm15, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 56]
	vsubps ymm16, ymm16, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 60]
	vsubps ymm17, ymm17, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 64]
	vsubps ymm18, ymm18, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 68]
	vsubps ymm19, ymm19, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 72]
	vsubps ymm20, ymm20, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 76]
	vsubps ymm21, ymm21, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 80]
	vsubps ymm22, ymm22, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 84]
	vsubps ymm23, ymm23, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 88]
	vsubps ymm24, ymm24, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 92]
	vsubps ymm25, ymm25, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 96]
	vsubps ymm26, ymm26, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 100]
	vsubps ymm27, ymm27, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 104]
	vsubps ymm28, ymm28, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 108]
	vsubps ymm29, ymm29, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 112]
	vsubps ymm30, ymm30, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 116]
	vsubps ymm31, ymm31, ymm1
	kxnorb k1, k0, k0
	vxorps xmm1, xmm1, xmm1
	vgatherqps ymm1 {k1}, dword ptr [r11 + 4*zmm0 + 120]
	vmovups ymm2, ymmword ptr [rsp - 128]
	vsubps ymm1, ymm2, ymm1
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm0 + 124]
	vmovups ymm3, ymmword ptr [rsp + 224]
	vsubps ymm2, ymm3, ymm2
	kxnorb k1, k0, k0
	vmovups ymm3, ymmword ptr [rsp + 160]
	vscatterqps dword ptr [rsi + 4*zmm0] {k1}, ymm3
	kxnorb k1, k0, k0
	vmovups ymm3, ymmword ptr [rsp - 64]
	vscatterqps dword ptr [rsi + 4*zmm0 + 4] {k1}, ymm3
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 8] {k1}, ymm4
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 12] {k1}, ymm5
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 16] {k1}, ymm6
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 20] {k1}, ymm7
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 24] {k1}, ymm8
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 28] {k1}, ymm9
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 32] {k1}, ymm10
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 36] {k1}, ymm11
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 40] {k1}, ymm12
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 44] {k1}, ymm13
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 48] {k1}, ymm14
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 52] {k1}, ymm15
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 56] {k1}, ymm16
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 60] {k1}, ymm17
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 64] {k1}, ymm18
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 68] {k1}, ymm19
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 72] {k1}, ymm20
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 76] {k1}, ymm21
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 80] {k1}, ymm22
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 84] {k1}, ymm23
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 88] {k1}, ymm24
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 92] {k1}, ymm25
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 96] {k1}, ymm26
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 100] {k1}, ymm27
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 104] {k1}, ymm28
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 108] {k1}, ymm29
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 112] {k1}, ymm30
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 116] {k1}, ymm31
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 120] {k1}, ymm1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm0 + 124] {k1}, ymm2
	vpaddq zmm0, zmm0, qword ptr [rip + .LCPI144_3]{1to8}
	add r15, 8
	jne .LBB144_7
	cmp rbx, r14
	jne .LBB144_13
	jmp .LBB144_14
.LBB144_9:
	mov r15, rbx
	and r15, r14
	mov rax, r15
	shl rax, 5
	vmovaps zmm7, zmmword ptr [rip + .LCPI144_0]
	vmovaps zmm1, zmmword ptr [rip + .LCPI144_1]
	mov r12, r15
.LBB144_10:
	vxorps xmm2, xmm2, xmm2
	kxnorb k1, k0, k0
	vgatherqps ymm2 {k1}, dword ptr [rdx + 4*zmm7]
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm1]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 4]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 4]
	vinsertf64x4 zmm5, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 8]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 8]
	vinsertf64x4 zmm8, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 12]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 12]
	vinsertf64x4 zmm9, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 16]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 16]
	vinsertf64x4 zmm10, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 20]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 20]
	vinsertf64x4 zmm11, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 24]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 24]
	vinsertf64x4 zmm12, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 28]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 28]
	vinsertf64x4 zmm13, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 32]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 32]
	vinsertf64x4 zmm14, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 36]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 36]
	vinsertf64x4 zmm15, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 40]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 40]
	vinsertf64x4 zmm16, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 44]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 44]
	vinsertf64x4 zmm17, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 48]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 48]
	vinsertf64x4 zmm18, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 52]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 52]
	vinsertf64x4 zmm19, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 56]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 56]
	vinsertf64x4 zmm20, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 60]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 60]
	vinsertf64x4 zmm21, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 64]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 64]
	vinsertf64x4 zmm22, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 68]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 68]
	vinsertf64x4 zmm23, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 72]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 72]
	vinsertf64x4 zmm24, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 76]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 76]
	vinsertf64x4 zmm25, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 80]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 80]
	vinsertf64x4 zmm26, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 84]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 84]
	vinsertf64x4 zmm27, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 88]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 88]
	vinsertf64x4 zmm28, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 92]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 92]
	vinsertf64x4 zmm29, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 96]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 96]
	vinsertf64x4 zmm30, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 100]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 100]
	vinsertf64x4 zmm31, zmm4, ymm3, 1
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 104]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 104]
	vinsertf64x4 zmm0, zmm4, ymm3, 1
	vmovups zmmword ptr [rsp + 96], zmm0
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 108]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 108]
	vinsertf64x4 zmm0, zmm4, ymm3, 1
	vmovups zmmword ptr [rsp - 64], zmm0
	vxorps xmm3, xmm3, xmm3
	kxnorb k1, k0, k0
	vgatherqps ymm3 {k1}, dword ptr [rdx + 4*zmm7 + 112]
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm1 + 112]
	vinsertf64x4 zmm3, zmm4, ymm3, 1
	vxorps xmm4, xmm4, xmm4
	kxnorb k1, k0, k0
	vgatherqps ymm4 {k1}, dword ptr [rdx + 4*zmm7 + 116]
	vxorps xmm0, xmm0, xmm0
	kxnorb k1, k0, k0
	vgatherqps ymm0 {k1}, dword ptr [rdx + 4*zmm1 + 116]
	vinsertf64x4 zmm4, zmm0, ymm4, 1
	vxorps xmm0, xmm0, xmm0
	kxnorb k1, k0, k0
	vgatherqps ymm0 {k1}, dword ptr [rdx + 4*zmm7 + 120]
	vxorps xmm6, xmm6, xmm6
	kxnorb k1, k0, k0
	vgatherqps ymm6 {k1}, dword ptr [rdx + 4*zmm1 + 120]
	vinsertf64x4 zmm0, zmm6, ymm0, 1
	vmovups zmmword ptr [rsp + 32], zmm0
	vxorps xmm0, xmm0, xmm0
	kxnorb k1, k0, k0
	vgatherqps ymm0 {k1}, dword ptr [rdx + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [rdx + 4*zmm1 + 124]
	vinsertf64x4 zmm0, zmm6, ymm0, 1
	vmovups zmmword ptr [rsp - 128], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r9 + 4*zmm7]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm1]
	vinsertf64x4 zmm0, zmm6, ymm0, 1
	vaddps zmm0, zmm2, zmm0
	vmovups zmmword ptr [rsp + 224], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 4]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 4]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vaddps zmm5, zmm5, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 8]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm1 + 8]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vaddps zmm0, zmm8, zmm0
	vmovups zmmword ptr [rsp + 160], zmm0
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r9 + 4*zmm7 + 12]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm1 + 12]
	vinsertf64x4 zmm2, zmm6, ymm2, 1
	vaddps zmm2, zmm9, zmm2
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 16]
	kxnorb k1, k0, k0
	vxorps xmm8, xmm8, xmm8
	vgatherqps ymm8 {k1}, dword ptr [r9 + 4*zmm1 + 16]
	vinsertf64x4 zmm6, zmm8, ymm6, 1
	vaddps zmm8, zmm10, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 20]
	kxnorb k1, k0, k0
	vxorps xmm9, xmm9, xmm9
	vgatherqps ymm9 {k1}, dword ptr [r9 + 4*zmm1 + 20]
	vinsertf64x4 zmm6, zmm9, ymm6, 1
	vaddps zmm9, zmm11, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 24]
	kxnorb k1, k0, k0
	vxorps xmm10, xmm10, xmm10
	vgatherqps ymm10 {k1}, dword ptr [r9 + 4*zmm1 + 24]
	vinsertf64x4 zmm6, zmm10, ymm6, 1
	vaddps zmm10, zmm12, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 28]
	kxnorb k1, k0, k0
	vxorps xmm11, xmm11, xmm11
	vgatherqps ymm11 {k1}, dword ptr [r9 + 4*zmm1 + 28]
	vinsertf64x4 zmm6, zmm11, ymm6, 1
	vaddps zmm11, zmm13, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 32]
	kxnorb k1, k0, k0
	vxorps xmm12, xmm12, xmm12
	vgatherqps ymm12 {k1}, dword ptr [r9 + 4*zmm1 + 32]
	vinsertf64x4 zmm6, zmm12, ymm6, 1
	vaddps zmm12, zmm14, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 36]
	kxnorb k1, k0, k0
	vxorps xmm13, xmm13, xmm13
	vgatherqps ymm13 {k1}, dword ptr [r9 + 4*zmm1 + 36]
	vinsertf64x4 zmm6, zmm13, ymm6, 1
	vaddps zmm13, zmm15, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 40]
	kxnorb k1, k0, k0
	vxorps xmm14, xmm14, xmm14
	vgatherqps ymm14 {k1}, dword ptr [r9 + 4*zmm1 + 40]
	vinsertf64x4 zmm6, zmm14, ymm6, 1
	vaddps zmm14, zmm16, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 44]
	kxnorb k1, k0, k0
	vxorps xmm15, xmm15, xmm15
	vgatherqps ymm15 {k1}, dword ptr [r9 + 4*zmm1 + 44]
	vinsertf64x4 zmm6, zmm15, ymm6, 1
	vaddps zmm15, zmm17, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 48]
	kxnorb k1, k0, k0
	vxorps xmm16, xmm16, xmm16
	vgatherqps ymm16 {k1}, dword ptr [r9 + 4*zmm1 + 48]
	vinsertf64x4 zmm6, zmm16, ymm6, 1
	vaddps zmm16, zmm18, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 52]
	kxnorb k1, k0, k0
	vxorps xmm17, xmm17, xmm17
	vgatherqps ymm17 {k1}, dword ptr [r9 + 4*zmm1 + 52]
	vinsertf64x4 zmm6, zmm17, ymm6, 1
	vaddps zmm17, zmm19, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 56]
	kxnorb k1, k0, k0
	vxorps xmm18, xmm18, xmm18
	vgatherqps ymm18 {k1}, dword ptr [r9 + 4*zmm1 + 56]
	vinsertf64x4 zmm6, zmm18, ymm6, 1
	vaddps zmm18, zmm20, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 60]
	kxnorb k1, k0, k0
	vxorps xmm19, xmm19, xmm19
	vgatherqps ymm19 {k1}, dword ptr [r9 + 4*zmm1 + 60]
	vinsertf64x4 zmm6, zmm19, ymm6, 1
	vaddps zmm19, zmm21, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 64]
	kxnorb k1, k0, k0
	vxorps xmm20, xmm20, xmm20
	vgatherqps ymm20 {k1}, dword ptr [r9 + 4*zmm1 + 64]
	vinsertf64x4 zmm6, zmm20, ymm6, 1
	vaddps zmm20, zmm22, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 68]
	kxnorb k1, k0, k0
	vxorps xmm21, xmm21, xmm21
	vgatherqps ymm21 {k1}, dword ptr [r9 + 4*zmm1 + 68]
	vinsertf64x4 zmm6, zmm21, ymm6, 1
	vaddps zmm21, zmm23, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 72]
	kxnorb k1, k0, k0
	vxorps xmm22, xmm22, xmm22
	vgatherqps ymm22 {k1}, dword ptr [r9 + 4*zmm1 + 72]
	vinsertf64x4 zmm6, zmm22, ymm6, 1
	vaddps zmm22, zmm24, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 76]
	kxnorb k1, k0, k0
	vxorps xmm23, xmm23, xmm23
	vgatherqps ymm23 {k1}, dword ptr [r9 + 4*zmm1 + 76]
	vinsertf64x4 zmm6, zmm23, ymm6, 1
	vaddps zmm23, zmm25, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 80]
	kxnorb k1, k0, k0
	vxorps xmm24, xmm24, xmm24
	vgatherqps ymm24 {k1}, dword ptr [r9 + 4*zmm1 + 80]
	vinsertf64x4 zmm6, zmm24, ymm6, 1
	vaddps zmm24, zmm26, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 84]
	kxnorb k1, k0, k0
	vxorps xmm25, xmm25, xmm25
	vgatherqps ymm25 {k1}, dword ptr [r9 + 4*zmm1 + 84]
	vinsertf64x4 zmm6, zmm25, ymm6, 1
	vaddps zmm25, zmm27, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 88]
	kxnorb k1, k0, k0
	vxorps xmm26, xmm26, xmm26
	vgatherqps ymm26 {k1}, dword ptr [r9 + 4*zmm1 + 88]
	vinsertf64x4 zmm6, zmm26, ymm6, 1
	vaddps zmm26, zmm28, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 92]
	kxnorb k1, k0, k0
	vxorps xmm27, xmm27, xmm27
	vgatherqps ymm27 {k1}, dword ptr [r9 + 4*zmm1 + 92]
	vinsertf64x4 zmm6, zmm27, ymm6, 1
	vaddps zmm27, zmm29, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 96]
	kxnorb k1, k0, k0
	vxorps xmm28, xmm28, xmm28
	vgatherqps ymm28 {k1}, dword ptr [r9 + 4*zmm1 + 96]
	vinsertf64x4 zmm6, zmm28, ymm6, 1
	vaddps zmm28, zmm30, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 100]
	kxnorb k1, k0, k0
	vxorps xmm29, xmm29, xmm29
	vgatherqps ymm29 {k1}, dword ptr [r9 + 4*zmm1 + 100]
	vinsertf64x4 zmm6, zmm29, ymm6, 1
	vaddps zmm29, zmm31, zmm6
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 104]
	kxnorb k1, k0, k0
	vxorps xmm30, xmm30, xmm30
	vgatherqps ymm30 {k1}, dword ptr [r9 + 4*zmm1 + 104]
	vinsertf64x4 zmm6, zmm30, ymm6, 1
	vaddps zmm30, zmm6, zmmword ptr [rsp + 96]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 108]
	kxnorb k1, k0, k0
	vxorps xmm31, xmm31, xmm31
	vgatherqps ymm31 {k1}, dword ptr [r9 + 4*zmm1 + 108]
	vinsertf64x4 zmm6, zmm31, ymm6, 1
	vaddps zmm31, zmm6, zmmword ptr [rsp - 64]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r9 + 4*zmm7 + 112]
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r9 + 4*zmm1 + 112]
	vinsertf64x4 zmm0, zmm0, ymm6, 1
	vaddps zmm0, zmm3, zmm0
	vmovups zmmword ptr [rsp - 64], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 116]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 116]
	vinsertf64x4 zmm0, zmm3, ymm0, 1
	vaddps zmm0, zmm4, zmm0
	vmovups zmmword ptr [rsp + 96], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 120]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 120]
	vinsertf64x4 zmm0, zmm3, ymm0, 1
	vaddps zmm0, zmm0, zmmword ptr [rsp + 32]
	vmovups zmmword ptr [rsp + 32], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r9 + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r9 + 4*zmm1 + 124]
	vinsertf64x4 zmm0, zmm3, ymm0, 1
	vaddps zmm3, zmm0, zmmword ptr [rsp - 128]
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7]
	kxnorb k1, k0, k0
	vxorps xmm4, xmm4, xmm4
	vgatherqps ymm4 {k1}, dword ptr [r10 + 4*zmm1]
	vinsertf64x4 zmm0, zmm4, ymm0, 1
	vmulps zmm4, zmm0, zmmword ptr [rsp + 224]
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 4]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r10 + 4*zmm1 + 4]
	vinsertf64x4 zmm0, zmm6, ymm0, 1
	vmulps zmm5, zmm5, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 8]
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r10 + 4*zmm1 + 8]
	vinsertf64x4 zmm0, zmm6, ymm0, 1
	vmulps zmm0, zmm0, zmmword ptr [rsp + 160]
	vmovups zmmword ptr [rsp - 128], zmm0
	kxnorb k1, k0, k0
	vxorps xmm6, xmm6, xmm6
	vgatherqps ymm6 {k1}, dword ptr [r10 + 4*zmm7 + 12]
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm1 + 12]
	vinsertf64x4 zmm0, zmm0, ymm6, 1
	vmulps zmm6, zmm2, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 16]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 16]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm8, zmm8, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 20]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 20]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm9, zmm9, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 24]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 24]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm10, zmm10, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 28]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 28]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm11, zmm11, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 32]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 32]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm12, zmm12, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 36]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 36]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm13, zmm13, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 40]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 40]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm14, zmm14, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 44]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 44]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm15, zmm15, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 48]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 48]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm16, zmm16, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 52]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 52]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm17, zmm17, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 56]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 56]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm18, zmm18, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 60]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 60]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm19, zmm19, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 64]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 64]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm20, zmm20, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 68]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 68]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm21, zmm21, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 72]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 72]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm22, zmm22, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 76]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 76]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm23, zmm23, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 80]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 80]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm24, zmm24, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 84]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 84]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm25, zmm25, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 88]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 88]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm26, zmm26, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 92]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 92]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm27, zmm27, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 96]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 96]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm28, zmm28, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 100]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 100]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm29, zmm29, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 104]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 104]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm30, zmm30, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 108]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 108]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm31, zmm31, zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 112]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 112]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm0, zmm0, zmmword ptr [rsp - 64]
	vmovups zmmword ptr [rsp + 224], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 116]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 116]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm0, zmm0, zmmword ptr [rsp + 96]
	vmovups zmmword ptr [rsp + 160], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 120]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 120]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm0, zmm0, zmmword ptr [rsp + 32]
	vmovups zmmword ptr [rsp - 64], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r10 + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r10 + 4*zmm1 + 124]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmulps zmm0, zmm3, zmm0
	vmovups zmmword ptr [rsp + 96], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r11 + 4*zmm7]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm1]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vsubps zmm0, zmm4, zmm0
	vmovups zmmword ptr [rsp + 32], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r11 + 4*zmm7 + 4]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm1 + 4]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vsubps zmm0, zmm5, zmm0
	vmovups zmmword ptr [rsp + 576], zmm0
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r11 + 4*zmm7 + 8]
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm1 + 8]
	vinsertf64x4 zmm0, zmm2, ymm0, 1
	vmovups zmm2, zmmword ptr [rsp - 128]
	vsubps zmm0, zmm2, zmm0
	vmovups zmmword ptr [rsp - 128], zmm0
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 12]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 12]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm6, zmm6, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 16]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 16]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm8, zmm8, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 20]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 20]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm9, zmm9, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 24]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 24]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm10, zmm10, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 28]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 28]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm11, zmm11, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 32]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 32]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm12, zmm12, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 36]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 36]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm13, zmm13, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 40]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 40]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm14, zmm14, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 44]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 44]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm15, zmm15, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 48]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 48]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm16, zmm16, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 52]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 52]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm17, zmm17, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 56]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 56]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm18, zmm18, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 60]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 60]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm19, zmm19, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 64]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 64]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm20, zmm20, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 68]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 68]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm21, zmm21, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 72]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 72]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm22, zmm22, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 76]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 76]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm23, zmm23, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 80]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 80]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm24, zmm24, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 84]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 84]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm25, zmm25, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 88]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 88]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm26, zmm26, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 92]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 92]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm27, zmm27, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 96]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 96]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm28, zmm28, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 100]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 100]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm29, zmm29, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 104]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 104]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm30, zmm30, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 108]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 108]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vsubps zmm31, zmm31, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 112]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 112]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vmovups zmm0, zmmword ptr [rsp + 224]
	vsubps zmm5, zmm0, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 116]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 116]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vmovups zmm0, zmmword ptr [rsp + 160]
	vsubps zmm4, zmm0, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 120]
	kxnorb k1, k0, k0
	vxorps xmm3, xmm3, xmm3
	vgatherqps ymm3 {k1}, dword ptr [r11 + 4*zmm1 + 120]
	vinsertf64x4 zmm2, zmm3, ymm2, 1
	vmovups zmm0, zmmword ptr [rsp - 64]
	vsubps zmm3, zmm0, zmm2
	kxnorb k1, k0, k0
	vxorps xmm2, xmm2, xmm2
	vgatherqps ymm2 {k1}, dword ptr [r11 + 4*zmm7 + 124]
	kxnorb k1, k0, k0
	vxorps xmm0, xmm0, xmm0
	vgatherqps ymm0 {k1}, dword ptr [r11 + 4*zmm1 + 124]
	vinsertf64x4 zmm0, zmm0, ymm2, 1
	vmovups zmm2, zmmword ptr [rsp + 96]
	vsubps zmm2, zmm2, zmm0
	kxnorb k1, k0, k0
	vmovups zmm0, zmmword ptr [rsp + 32]
	vscatterqps dword ptr [rsi + 4*zmm1] {k1}, ymm0
	vextractf64x4 ymm0, zmm0, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7] {k1}, ymm0
	kxnorb k1, k0, k0
	vmovups zmm0, zmmword ptr [rsp + 576]
	vscatterqps dword ptr [rsi + 4*zmm1 + 4] {k1}, ymm0
	vextractf64x4 ymm0, zmm0, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 4] {k1}, ymm0
	kxnorb k1, k0, k0
	vmovups zmm0, zmmword ptr [rsp - 128]
	vscatterqps dword ptr [rsi + 4*zmm1 + 8] {k1}, ymm0
	vextractf64x4 ymm0, zmm0, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 8] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 12] {k1}, ymm6
	vextractf64x4 ymm0, zmm6, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 12] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 16] {k1}, ymm8
	vextractf64x4 ymm0, zmm8, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 16] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 20] {k1}, ymm9
	vextractf64x4 ymm0, zmm9, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 20] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 24] {k1}, ymm10
	vextractf64x4 ymm0, zmm10, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 24] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 28] {k1}, ymm11
	vextractf64x4 ymm0, zmm11, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 28] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 32] {k1}, ymm12
	vextractf64x4 ymm0, zmm12, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 32] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 36] {k1}, ymm13
	vextractf64x4 ymm0, zmm13, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 36] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 40] {k1}, ymm14
	vextractf64x4 ymm0, zmm14, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 40] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 44] {k1}, ymm15
	vextractf64x4 ymm0, zmm15, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 44] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 48] {k1}, ymm16
	vextractf64x4 ymm0, zmm16, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 48] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 52] {k1}, ymm17
	vextractf64x4 ymm0, zmm17, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 52] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 56] {k1}, ymm18
	vextractf64x4 ymm0, zmm18, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 56] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 60] {k1}, ymm19
	vextractf64x4 ymm0, zmm19, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 60] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 64] {k1}, ymm20
	vextractf64x4 ymm0, zmm20, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 64] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 68] {k1}, ymm21
	vextractf64x4 ymm0, zmm21, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 68] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 72] {k1}, ymm22
	vextractf64x4 ymm0, zmm22, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 72] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 76] {k1}, ymm23
	vextractf64x4 ymm0, zmm23, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 76] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 80] {k1}, ymm24
	vextractf64x4 ymm0, zmm24, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 80] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 84] {k1}, ymm25
	vextractf64x4 ymm0, zmm25, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 84] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 88] {k1}, ymm26
	vextractf64x4 ymm0, zmm26, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 88] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 92] {k1}, ymm27
	vextractf64x4 ymm0, zmm27, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 92] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 96] {k1}, ymm28
	vextractf64x4 ymm0, zmm28, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 96] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 100] {k1}, ymm29
	vextractf64x4 ymm0, zmm29, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 100] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 104] {k1}, ymm30
	vextractf64x4 ymm0, zmm30, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 104] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 108] {k1}, ymm31
	vextractf64x4 ymm0, zmm31, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 108] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 112] {k1}, ymm5
	vextractf64x4 ymm0, zmm5, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 112] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 116] {k1}, ymm4
	vextractf64x4 ymm0, zmm4, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 116] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 120] {k1}, ymm3
	vextractf64x4 ymm0, zmm3, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 120] {k1}, ymm0
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm1 + 124] {k1}, ymm2
	vextractf64x4 ymm0, zmm2, 1
	kxnorb k1, k0, k0
	vscatterqps dword ptr [rsi + 4*zmm7 + 124] {k1}, ymm0
	vpbroadcastq zmm0, qword ptr [rip + .LCPI144_2]
	vpaddq zmm1, zmm1, zmm0
	vpaddq zmm7, zmm7, zmm0
	add r12, -16
	jne .LBB144_10
	cmp rbx, r15
	je .LBB144_14
	test bl, 8
	jne .LBB144_6
.LBB144_13:
	vmovups zmm0, zmmword ptr [rdx + 4*rax]
	vmovups zmm1, zmmword ptr [rdx + 4*rax + 64]
	vaddps zmm0, zmm0, zmmword ptr [r9 + 4*rax]
	vmulps zmm0, zmm0, zmmword ptr [r10 + 4*rax]
	vsubps zmm0, zmm0, zmmword ptr [r11 + 4*rax]
	vaddps zmm1, zmm1, zmmword ptr [r9 + 4*rax + 64]
	vmulps zmm1, zmm1, zmmword ptr [r10 + 4*rax + 64]
	vsubps zmm1, zmm1, zmmword ptr [r11 + 4*rax + 64]
	vmovups zmmword ptr [rsi + 4*rax], zmm0
	vmovups zmmword ptr [rsi + 4*rax + 64], zmm1
	add rax, 32
	cmp rax, rcx
	jb .LBB144_13
.LBB144_14:
	mov r10, r8
	sub r10, rax
	jbe .LBB144_33
	mov rbp, qword ptr [rdi + 136]
	mov rdx, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r10, 8
	jb .LBB144_16
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
	je .LBB144_19
.LBB144_16:
	mov r11, rax
.LBB144_28:
	mov ecx, r8d
	sub ecx, r11d
	lea rax, [r11 + 1]
	test cl, 1
	je .LBB144_30
	vmovss xmm0, dword ptr [rbp + 4*r11]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r11]
	vmulss xmm0, xmm0, dword ptr [r9 + 4*r11]
	vsubss xmm0, xmm0, dword ptr [rdi + 4*r11]
	vmovss dword ptr [rsi + 4*r11], xmm0
	mov r11, rax
.LBB144_30:
	cmp r8, rax
	je .LBB144_33
	sub r8, r11
	lea rax, [rsi + 4*r11]
	add rax, 4
	lea rcx, [rdi + 4*r11 + 4]
	lea rsi, [r9 + 4*r11]
	add rsi, 4
	lea rdx, [rdx + 4*r11]
	add rdx, 4
	lea rdi, [4*r11 + 4]
	add rdi, rbp
	xor r9d, r9d
.LBB144_32:
	vmovss xmm0, dword ptr [rdi + 4*r9 - 4]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r9 - 4]
	vmulss xmm0, xmm0, dword ptr [rsi + 4*r9 - 4]
	vsubss xmm0, xmm0, dword ptr [rcx + 4*r9 - 4]
	vmovss dword ptr [rax + 4*r9 - 4], xmm0
	vmovss xmm0, dword ptr [rdi + 4*r9]
	vaddss xmm0, xmm0, dword ptr [rdx + 4*r9]
	vmulss xmm0, xmm0, dword ptr [rsi + 4*r9]
	vsubss xmm0, xmm0, dword ptr [rcx + 4*r9]
	vmovss dword ptr [rax + 4*r9], xmm0
	add r9, 2
	cmp r8, r9
	jne .LBB144_32
.LBB144_33:
	add rsp, 648
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	vzeroupper
	ret
.LBB144_19:
	cmp r10, 64
	jae .LBB144_21
	xor ebx, ebx
	jmp .LBB144_25
.LBB144_21:
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
.LBB144_22:
	vmovups zmm0, zmmword ptr [rbp + 4*rcx - 192]
	vmovups zmm1, zmmword ptr [rbp + 4*rcx - 128]
	vmovups zmm2, zmmword ptr [rbp + 4*rcx - 64]
	vmovups zmm3, zmmword ptr [rbp + 4*rcx]
	vaddps zmm0, zmm0, zmmword ptr [r13 + 4*rcx - 192]
	vaddps zmm1, zmm1, zmmword ptr [r13 + 4*rcx - 128]
	vaddps zmm2, zmm2, zmmword ptr [r13 + 4*rcx - 64]
	vaddps zmm3, zmm3, zmmword ptr [r13 + 4*rcx]
	vmulps zmm0, zmm0, zmmword ptr [r12 + 4*rcx - 192]
	vmulps zmm1, zmm1, zmmword ptr [r12 + 4*rcx - 128]
	vmulps zmm2, zmm2, zmmword ptr [r12 + 4*rcx - 64]
	vmulps zmm3, zmm3, zmmword ptr [r12 + 4*rcx]
	vsubps zmm0, zmm0, zmmword ptr [r15 + 4*rcx - 192]
	vsubps zmm1, zmm1, zmmword ptr [r15 + 4*rcx - 128]
	vsubps zmm2, zmm2, zmmword ptr [r15 + 4*rcx - 64]
	vsubps zmm3, zmm3, zmmword ptr [r15 + 4*rcx]
	vmovups zmmword ptr [r14 + 4*rcx - 192], zmm0
	vmovups zmmword ptr [r14 + 4*rcx - 128], zmm1
	vmovups zmmword ptr [r14 + 4*rcx - 64], zmm2
	vmovups zmmword ptr [r14 + 4*rcx], zmm3
	add rcx, 64
	cmp rbx, rcx
	jne .LBB144_22
	cmp r10, rbx
	mov rbp, qword ptr [rsp - 128]
	je .LBB144_33
	test r10b, 56
	je .LBB144_28
.LBB144_25:
	mov rcx, r10
	and rcx, -8
	lea r11, [rax + rcx]
	lea r14, [rsi + 4*rax]
	lea r15, [rdi + 4*rax]
	lea r12, [r9 + 4*rax]
	lea r13, [rdx + 4*rax]
	lea rax, [4*rax]
	add rax, rbp
.LBB144_26:
	vmovups ymm0, ymmword ptr [rax + 4*rbx]
	vaddps ymm0, ymm0, ymmword ptr [r13 + 4*rbx]
	vmulps ymm0, ymm0, ymmword ptr [r12 + 4*rbx]
	vsubps ymm0, ymm0, ymmword ptr [r15 + 4*rbx]
	vmovups ymmword ptr [r14 + 4*rbx], ymm0
	add rbx, 8
	cmp rcx, rbx
	jne .LBB144_26
	cmp r10, rcx
	je .LBB144_33
	jmp .LBB144_28
