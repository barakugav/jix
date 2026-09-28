jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 92
	call .L160$pb
.L160$pb:
	pop ebx
.Ltmp8446:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp8446-.L160$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB160_1
.LBB160_2:
	movzx esi, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 120]
	mov eax, dword ptr [esp + 116]
	mov edx, dword ptr [esp + 112]
	mov esi, dword ptr [ebx + 4*esi + .LJTI160_0@GOTOFF]
	add esi, ebx
	jmp esi
	mov ecx, dword ptr [esp + 128]
	mov dword ptr [esp + 36], ebx
	mov edi, ecx
	and edi, -16
	je .LBB160_4
	movaps xmm0, xmmword ptr [ebx + .LCPI160_0@GOTOFF]
	mov ebp, dword ptr [edx + 68]
	xor esi, esi
	movaps xmmword ptr [esp + 48], xmm0
.LBB160_14:
	movups xmm0, xmmword ptr [ebp + 8*esi + 112]
	movups xmm1, xmmword ptr [ebp + 8*esi]
	movups xmm7, xmmword ptr [ebp + 8*esi + 16]
	movups xmm6, xmmword ptr [ebp + 8*esi + 32]
	movups xmm5, xmmword ptr [ebp + 8*esi + 48]
	movups xmm4, xmmword ptr [ebp + 8*esi + 64]
	movups xmm3, xmmword ptr [ebp + 8*esi + 80]
	movups xmm2, xmmword ptr [ebp + 8*esi + 96]
	movaps xmmword ptr [esp + 64], xmm0
	movaps xmm0, xmmword ptr [esp + 48]
	xorps xmm1, xmm0
	xorps xmm7, xmm0
	xorps xmm6, xmm0
	xorps xmm5, xmm0
	xorps xmm4, xmm0
	xorps xmm3, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [eax + 8*esi], xmm1
	movups xmmword ptr [eax + 8*esi + 16], xmm7
	movups xmmword ptr [eax + 8*esi + 32], xmm6
	movups xmmword ptr [eax + 8*esi + 48], xmm5
	movups xmmword ptr [eax + 8*esi + 64], xmm4
	movups xmmword ptr [eax + 8*esi + 80], xmm3
	movups xmmword ptr [eax + 8*esi + 96], xmm2
	movaps xmm2, xmmword ptr [esp + 64]
	xorps xmm2, xmm0
	movups xmmword ptr [eax + 8*esi + 112], xmm2
	add esi, 16
	cmp esi, edi
	jb .LBB160_14
	mov ebp, ecx
	sub ebp, esi
	ja .LBB160_6
	jmp .LBB160_18
	mov dword ptr [esp + 12], eax
	mov dword ptr [esp + 16], ecx
	lea ecx, [esp + 124]
	lea esi, [esp + 128]
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 24], esi
	mov dword ptr [esp + 28], edx
	mov dword ptr [esp + 32], ecx
	lea ecx, [esp + 12]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>::{closure#3}, ()>
	jmp .LBB160_18
	mov dword ptr [esp + 12], eax
	mov dword ptr [esp + 16], ecx
	lea ecx, [esp + 124]
	lea esi, [esp + 128]
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 24], esi
	mov dword ptr [esp + 28], edx
	mov dword ptr [esp + 32], ecx
	lea ecx, [esp + 12]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>::{closure#2}, ()>
	jmp .LBB160_18
	mov dword ptr [esp + 12], eax
	mov dword ptr [esp + 16], ecx
	lea ecx, [esp + 124]
	lea esi, [esp + 128]
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 24], esi
	mov dword ptr [esp + 28], edx
	mov dword ptr [esp + 32], ecx
	lea ecx, [esp + 12]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>::{closure#1}, ()>
	jmp .LBB160_18
.LBB160_4:
	xor esi, esi
	mov ebp, ecx
	sub ebp, esi
	jbe .LBB160_18
.LBB160_6:
	mov ebx, dword ptr [edx + 68]
	cmp ebp, 6
	mov dword ptr [esp + 64], ecx
	setb cl
	mov edx, ebx
	sub edx, eax
	cmp edx, -31
	setae dl
	or dl, cl
	je .LBB160_8
	mov edi, esi
	jmp .LBB160_11
.LBB160_8:
	mov dword ptr [esp + 40], ebp
	and ebp, -4
	lea ecx, [eax + 8*esi + 16]
	mov dword ptr [esp + 48], ebx
	xor edx, edx
	lea edi, [esi + ebp]
	lea esi, [ebx + 8*esi + 16]
	mov ebx, dword ptr [esp + 36]
	movaps xmm0, xmmword ptr [ebx + .LCPI160_0@GOTOFF]
.LBB160_9:
	movups xmm1, xmmword ptr [esi + 8*edx - 16]
	movups xmm2, xmmword ptr [esi + 8*edx]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [ecx + 8*edx - 16], xmm1
	movups xmmword ptr [ecx + 8*edx], xmm2
	add edx, 4
	cmp ebp, edx
	jne .LBB160_9
	cmp dword ptr [esp + 40], ebp
	mov ebx, dword ptr [esp + 48]
	je .LBB160_18
.LBB160_11:
	mov ecx, dword ptr [esp + 36]
	movaps xmm0, xmmword ptr [ecx + .LCPI160_0@GOTOFF]
	mov ecx, dword ptr [esp + 64]
.LBB160_12:
	movsd xmm1, qword ptr [ebx + 8*edi]
	xorps xmm1, xmm0
	movlps qword ptr [eax + 8*edi], xmm1
	inc edi
	cmp ecx, edi
	jne .LBB160_12
.LBB160_18:
	add esp, 92
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB160_1:
	lea eax, [esp + 44]
	mov dword ptr [esp + 44], esi
	mov dword ptr [esp + 12], eax
	sub esp, 12
	lea eax, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.20@GOTOFF]
	lea ecx, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.18@GOTOFF]
	lea edx, [esp + 24]
	push eax
	push ecx
	push edx
	push 1
	push esi
	call <std::sys::sync::once::futex::Once>::call@PLT
	add esp, 32
	jmp .LBB160_2
