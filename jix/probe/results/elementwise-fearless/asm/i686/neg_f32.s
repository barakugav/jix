jix::storage::elementwise_pipeline::inner_loop_contiguous::<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 76
	call .L164$pb
.L164$pb:
	pop ebx
.Ltmp8783:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp8783-.L164$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB164_1
.LBB164_2:
	movzx esi, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 104]
	mov eax, dword ptr [esp + 100]
	mov edx, dword ptr [esp + 96]
	mov esi, dword ptr [ebx + 4*esi + .LJTI164_0@GOTOFF]
	add esi, ebx
	jmp esi
	mov edi, dword ptr [esp + 112]
	mov dword ptr [esp + 40], edi
	and edi, -32
	je .LBB164_4
	mov ebp, dword ptr [edx + 68]
	xor esi, esi
.LBB164_14:
	movups xmm0, xmmword ptr [ebp + 4*esi + 112]
	movups xmm1, xmmword ptr [ebp + 4*esi]
	movups xmm7, xmmword ptr [ebp + 4*esi + 16]
	movups xmm6, xmmword ptr [ebp + 4*esi + 32]
	movups xmm5, xmmword ptr [ebp + 4*esi + 48]
	movups xmm4, xmmword ptr [ebp + 4*esi + 64]
	movups xmm3, xmmword ptr [ebp + 4*esi + 80]
	movups xmm2, xmmword ptr [ebp + 4*esi + 96]
	movaps xmmword ptr [esp + 48], xmm0
	movaps xmm0, xmmword ptr [ebx + .LCPI164_0@GOTOFF]
	xorps xmm1, xmm0
	xorps xmm7, xmm0
	xorps xmm6, xmm0
	xorps xmm5, xmm0
	xorps xmm4, xmm0
	xorps xmm3, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [eax + 4*esi], xmm1
	movups xmmword ptr [eax + 4*esi + 16], xmm7
	movups xmmword ptr [eax + 4*esi + 32], xmm6
	movups xmmword ptr [eax + 4*esi + 48], xmm5
	movups xmmword ptr [eax + 4*esi + 64], xmm4
	movups xmmword ptr [eax + 4*esi + 80], xmm3
	movups xmmword ptr [eax + 4*esi + 96], xmm2
	movaps xmm2, xmmword ptr [esp + 48]
	xorps xmm2, xmm0
	movups xmmword ptr [eax + 4*esi + 112], xmm2
	add esi, 32
	cmp esi, edi
	jb .LBB164_14
	mov ebp, dword ptr [esp + 40]
	mov ecx, ebp
	sub ecx, esi
	ja .LBB164_6
	jmp .LBB164_18
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 15]
	mov dword ptr [esp + 28], esi
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>::{closure#3}, ()>
	jmp .LBB164_18
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 15]
	mov dword ptr [esp + 28], esi
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>::{closure#2}, ()>
	jmp .LBB164_18
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 15]
	mov dword ptr [esp + 28], esi
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>::{closure#1}, ()>
	jmp .LBB164_18
.LBB164_4:
	xor esi, esi
	mov ebp, dword ptr [esp + 40]
	mov ecx, ebp
	sub ecx, esi
	jbe .LBB164_18
.LBB164_6:
	mov edx, dword ptr [edx + 68]
	cmp ecx, 8
	mov dword ptr [esp + 48], ecx
	setb cl
	mov edi, edx
	sub edi, eax
	cmp edi, -31
	setae ch
	or ch, cl
	je .LBB164_8
	mov edi, esi
	jmp .LBB164_11
.LBB164_8:
	mov ebp, dword ptr [esp + 48]
	movaps xmm0, xmmword ptr [ebx + .LCPI164_0@GOTOFF]
	lea ecx, [eax + 4*esi + 16]
	xor ebx, ebx
	and ebp, -8
	lea edi, [esi + ebp]
	lea esi, [edx + 4*esi + 16]
.LBB164_9:
	movups xmm1, xmmword ptr [esi + 4*ebx - 16]
	movups xmm2, xmmword ptr [esi + 4*ebx]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [ecx + 4*ebx - 16], xmm1
	movups xmmword ptr [ecx + 4*ebx], xmm2
	add ebx, 8
	cmp ebp, ebx
	jne .LBB164_9
	cmp dword ptr [esp + 48], ebp
	mov ebp, dword ptr [esp + 40]
	je .LBB164_18
.LBB164_11:
	mov ecx, -2147483648
.LBB164_12:
	mov esi, dword ptr [edx + 4*edi]
	xor esi, ecx
	mov dword ptr [eax + 4*edi], esi
	inc edi
	cmp ebp, edi
	jne .LBB164_12
.LBB164_18:
	add esp, 76
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB164_1:
	lea eax, [esp + 44]
	mov dword ptr [esp + 44], esi
	mov dword ptr [esp + 16], eax
	sub esp, 12
	lea eax, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.20@GOTOFF]
	lea ecx, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.18@GOTOFF]
	lea edx, [esp + 28]
	push eax
	push ecx
	push edx
	push 1
	push esi
	call <std::sys::sync::once::futex::Once>::call@PLT
	add esp, 32
	jmp .LBB164_2
