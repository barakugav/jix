jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 140
	call .L165$pb
.L165$pb:
	pop ebx
.Ltmp8840:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp8840-.L165$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB165_1
.LBB165_2:
	movzx edx, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 168]
	mov eax, dword ptr [esp + 164]
	mov ebp, dword ptr [esp + 160]
	mov edx, dword ptr [ebx + 4*edx + .LJTI165_0@GOTOFF]
	add edx, ebx
	jmp edx
	mov edi, dword ptr [esp + 176]
	mov dword ptr [esp + 16], edi
	and edi, -32
	je .LBB165_4
	mov ecx, ebp
	mov ebx, dword ptr [ebp + 68]
	mov ebp, dword ptr [ebp + 144]
	xor esi, esi
	mov edx, dword ptr [ecx + 224]
	mov ecx, dword ptr [ecx + 304]
.LBB165_15:
	movups xmm2, xmmword ptr [ebx + 4*esi]
	movups xmm1, xmmword ptr [ebp + 4*esi]
	movups xmm3, xmmword ptr [ebx + 4*esi + 16]
	movups xmm4, xmmword ptr [ebx + 4*esi + 32]
	movups xmm5, xmmword ptr [ebx + 4*esi + 48]
	movups xmm6, xmmword ptr [ebx + 4*esi + 64]
	movups xmm7, xmmword ptr [ebx + 4*esi + 80]
	movups xmm0, xmmword ptr [ebx + 4*esi + 96]
	addps xmm1, xmm2
	movups xmm2, xmmword ptr [ebp + 4*esi + 16]
	addps xmm2, xmm3
	movups xmm3, xmmword ptr [ebp + 4*esi + 32]
	addps xmm3, xmm4
	movups xmm4, xmmword ptr [ebp + 4*esi + 48]
	addps xmm4, xmm5
	movups xmm5, xmmword ptr [ebp + 4*esi + 64]
	addps xmm5, xmm6
	movups xmm6, xmmword ptr [ebp + 4*esi + 80]
	addps xmm6, xmm7
	movups xmm7, xmmword ptr [ebp + 4*esi + 96]
	addps xmm7, xmm0
	movups xmm0, xmmword ptr [ebx + 4*esi + 112]
	movaps xmmword ptr [esp + 48], xmm0
	movups xmm0, xmmword ptr [ebp + 4*esi + 112]
	addps xmm0, xmmword ptr [esp + 48]
	movaps xmmword ptr [esp + 112], xmm0
	movups xmm0, xmmword ptr [edx + 4*esi]
	mulps xmm0, xmm1
	movaps xmmword ptr [esp + 48], xmm0
	movups xmm0, xmmword ptr [edx + 4*esi + 16]
	movaps xmm1, xmmword ptr [esp + 48]
	mulps xmm0, xmm2
	movups xmm2, xmmword ptr [edx + 4*esi + 32]
	movaps xmmword ptr [esp + 96], xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi]
	mulps xmm2, xmm3
	movups xmm3, xmmword ptr [edx + 4*esi + 48]
	subps xmm1, xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi + 16]
	movaps xmmword ptr [esp + 48], xmm1
	movaps xmm1, xmmword ptr [esp + 96]
	mulps xmm3, xmm4
	movups xmm4, xmmword ptr [edx + 4*esi + 64]
	subps xmm1, xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi + 32]
	mulps xmm4, xmm5
	movups xmm5, xmmword ptr [edx + 4*esi + 80]
	subps xmm2, xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi + 48]
	mulps xmm5, xmm6
	movups xmm6, xmmword ptr [edx + 4*esi + 96]
	subps xmm3, xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi + 64]
	mulps xmm6, xmm7
	movups xmm7, xmmword ptr [edx + 4*esi + 112]
	subps xmm4, xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi + 80]
	mulps xmm7, xmmword ptr [esp + 112]
	subps xmm5, xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi + 96]
	subps xmm6, xmm0
	movups xmm0, xmmword ptr [ecx + 4*esi + 112]
	subps xmm7, xmm0
	movaps xmm0, xmmword ptr [esp + 48]
	movups xmmword ptr [eax + 4*esi], xmm0
	movups xmmword ptr [eax + 4*esi + 16], xmm1
	movups xmmword ptr [eax + 4*esi + 32], xmm2
	movups xmmword ptr [eax + 4*esi + 48], xmm3
	movups xmmword ptr [eax + 4*esi + 64], xmm4
	movups xmmword ptr [eax + 4*esi + 80], xmm5
	movups xmmword ptr [eax + 4*esi + 96], xmm6
	movups xmmword ptr [eax + 4*esi + 112], xmm7
	add esi, 32
	cmp esi, edi
	jb .LBB165_15
	mov ecx, dword ptr [esp + 16]
	mov ebp, ecx
	sub ebp, esi
	ja .LBB165_6
	jmp .LBB165_19
	mov dword ptr [esp + 20], eax
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 172]
	lea edx, [esp + 176]
	mov dword ptr [esp + 28], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ebp
	mov dword ptr [esp + 40], ecx
	lea ecx, [esp + 20]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>::{closure#3}, ()>
	jmp .LBB165_19
	mov dword ptr [esp + 20], eax
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 172]
	lea edx, [esp + 176]
	mov dword ptr [esp + 28], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ebp
	mov dword ptr [esp + 40], ecx
	lea ecx, [esp + 20]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>::{closure#2}, ()>
	jmp .LBB165_19
	mov dword ptr [esp + 20], eax
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 172]
	lea edx, [esp + 176]
	mov dword ptr [esp + 28], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ebp
	mov dword ptr [esp + 40], ecx
	lea ecx, [esp + 20]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>::{closure#1}, ()>
	jmp .LBB165_19
.LBB165_4:
	xor esi, esi
	mov ecx, dword ptr [esp + 16]
	mov ebp, ecx
	sub ebp, esi
	jbe .LBB165_19
.LBB165_6:
	mov ebx, dword ptr [esp + 160]
	cmp ebp, 12
	mov edx, dword ptr [ebx + 68]
	mov edi, dword ptr [ebx + 224]
	mov dword ptr [esp + 12], edx
	mov edx, dword ptr [ebx + 144]
	mov ebx, dword ptr [ebx + 304]
	jae .LBB165_8
	mov ebp, esi
	mov esi, dword ptr [esp + 12]
	jmp .LBB165_13
.LBB165_8:
	mov dword ptr [esp + 44], ebp
	mov ebp, dword ptr [esp + 12]
	sub ebp, eax
	cmp ebp, -15
	mov ebp, edx
	setae cl
	sub ebp, eax
	cmp ebp, -15
	mov ebp, edi
	setae ch
	sub ebp, eax
	or ch, cl
	cmp ebp, -15
	mov ebp, ebx
	setae byte ptr [esp + 48]
	sub ebp, eax
	cmp ebp, -15
	setae cl
	or cl, byte ptr [esp + 48]
	or cl, ch
	je .LBB165_10
	mov ebp, esi
	mov ecx, dword ptr [esp + 16]
	mov esi, dword ptr [esp + 12]
	jmp .LBB165_13
.LBB165_10:
	mov ecx, dword ptr [esp + 44]
	mov dword ptr [esp + 76], edx
	and ecx, -4
	mov dword ptr [esp + 48], ecx
	lea ebp, [esi + ecx]
	lea ecx, [eax + 4*esi]
	mov dword ptr [esp + 112], ecx
	lea ecx, [ebx + 4*esi]
	mov dword ptr [esp + 96], ecx
	lea ecx, [edi + 4*esi]
	mov dword ptr [esp + 88], ecx
	lea ecx, [edx + 4*esi]
	mov dword ptr [esp + 84], ecx
	mov ecx, dword ptr [esp + 12]
	lea ecx, [ecx + 4*esi]
	xor esi, esi
	mov dword ptr [esp + 80], ecx
	mov ecx, dword ptr [esp + 16]
.LBB165_11:
	mov edx, dword ptr [esp + 80]
	movups xmm0, xmmword ptr [edx + 4*esi]
	mov edx, dword ptr [esp + 84]
	movups xmm1, xmmword ptr [edx + 4*esi]
	mov edx, dword ptr [esp + 88]
	movups xmm3, xmmword ptr [edx + 4*esi]
	mov edx, dword ptr [esp + 96]
	addps xmm1, xmm0
	movups xmm2, xmmword ptr [edx + 4*esi]
	mov edx, dword ptr [esp + 112]
	mulps xmm3, xmm1
	subps xmm3, xmm2
	movups xmmword ptr [edx + 4*esi], xmm3
	add esi, 4
	cmp dword ptr [esp + 48], esi
	jne .LBB165_11
	mov esi, dword ptr [esp + 48]
	mov edx, dword ptr [esp + 76]
	cmp dword ptr [esp + 44], esi
	mov esi, dword ptr [esp + 12]
	je .LBB165_19
.LBB165_13:
	movss xmm0, dword ptr [esi + 4*ebp]
	addss xmm0, dword ptr [edx + 4*ebp]
	mulss xmm0, dword ptr [edi + 4*ebp]
	subss xmm0, dword ptr [ebx + 4*ebp]
	movss dword ptr [eax + 4*ebp], xmm0
	inc ebp
	cmp ecx, ebp
	jne .LBB165_13
.LBB165_19:
	add esp, 140
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB165_1:
	lea eax, [esp + 92]
	mov dword ptr [esp + 92], esi
	mov dword ptr [esp + 20], eax
	sub esp, 12
	lea eax, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.20@GOTOFF]
	lea ecx, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.18@GOTOFF]
	lea edx, [esp + 32]
	push eax
	push ecx
	push edx
	push 1
	push esi
	call <std::sys::sync::once::futex::Once>::call@PLT
	add esp, 32
	jmp .LBB165_2
