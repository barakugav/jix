jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 172
	call .L169$pb
.L169$pb:
	pop ebx
.Ltmp9178:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp9178-.L169$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB169_1
.LBB169_2:
	movzx edx, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 200]
	mov eax, dword ptr [esp + 196]
	mov ebp, dword ptr [esp + 192]
	mov edx, dword ptr [ebx + 4*edx + .LJTI169_0@GOTOFF]
	add edx, ebx
	jmp edx
	mov edi, dword ptr [esp + 208]
	mov dword ptr [esp + 44], edi
	and edi, -32
	je .LBB169_4
	mov ecx, ebp
	mov ebx, dword ptr [ebp + 68]
	mov ebp, dword ptr [ebp + 144]
	xor esi, esi
	mov edx, dword ptr [ecx + 224]
	mov ecx, dword ptr [ecx + 304]
.LBB169_14:
	movdqu xmm1, xmmword ptr [ebx + 4*esi]
	movdqu xmm0, xmmword ptr [ebp + 4*esi]
	movdqu xmm3, xmmword ptr [ebx + 4*esi + 16]
	movdqu xmm5, xmmword ptr [ebx + 4*esi + 32]
	movdqu xmm7, xmmword ptr [ebx + 4*esi + 48]
	movdqu xmm6, xmmword ptr [ebx + 4*esi + 64]
	movdqu xmm4, xmmword ptr [ebx + 4*esi + 80]
	movdqu xmm2, xmmword ptr [ebx + 4*esi + 96]
	paddd xmm0, xmm1
	movdqu xmm1, xmmword ptr [ebp + 4*esi + 16]
	paddd xmm1, xmm3
	movdqu xmm3, xmmword ptr [ebp + 4*esi + 32]
	paddd xmm3, xmm5
	movdqu xmm5, xmmword ptr [ebp + 4*esi + 48]
	paddd xmm5, xmm7
	movdqu xmm7, xmmword ptr [ebp + 4*esi + 64]
	paddd xmm7, xmm6
	movdqu xmm6, xmmword ptr [ebp + 4*esi + 80]
	paddd xmm6, xmm4
	movdqu xmm4, xmmword ptr [ebp + 4*esi + 96]
	paddd xmm4, xmm2
	movdqu xmm2, xmmword ptr [ebx + 4*esi + 112]
	movdqa xmmword ptr [esp + 80], xmm4
	movdqu xmm4, xmmword ptr [ebp + 4*esi + 112]
	paddd xmm4, xmm2
	movdqu xmm2, xmmword ptr [edx + 4*esi]
	movdqa xmmword ptr [esp + 144], xmm4
	pshufd xmm4, xmm2, 245
	pmuludq xmm2, xmm0
	pshufd xmm0, xmm0, 245
	pmuludq xmm0, xmm4
	pshufd xmm2, xmm2, 232
	pshufd xmm4, xmm1, 245
	pshufd xmm0, xmm0, 232
	punpckldq xmm2, xmm0
	movdqu xmm0, xmmword ptr [edx + 4*esi + 16]
	movdqa xmmword ptr [esp + 128], xmm2
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pmuludq xmm4, xmm2
	pshufd xmm1, xmm0, 232
	pshufd xmm0, xmm4, 232
	pshufd xmm4, xmm5, 245
	punpckldq xmm1, xmm0
	movdqu xmm0, xmmword ptr [edx + 4*esi + 32]
	movdqa xmmword ptr [esp + 112], xmm1
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm3
	pshufd xmm3, xmm3, 245
	pmuludq xmm2, xmm3
	pshufd xmm1, xmm0, 232
	pshufd xmm0, xmm2, 232
	punpckldq xmm1, xmm0
	movdqu xmm0, xmmword ptr [edx + 4*esi + 48]
	movdqa xmm3, xmm1
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm5
	movdqa xmm5, xmmword ptr [esp + 112]
	pmuludq xmm2, xmm4
	pshufd xmm1, xmm0, 232
	pshufd xmm4, xmm7, 245
	pshufd xmm0, xmm2, 232
	punpckldq xmm1, xmm0
	movdqu xmm0, xmmword ptr [edx + 4*esi + 64]
	movdqa xmmword ptr [esp + 96], xmm1
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm7
	pmuludq xmm2, xmm4
	pshufd xmm1, xmm0, 232
	pshufd xmm4, xmm6, 245
	pshufd xmm0, xmm2, 232
	punpckldq xmm1, xmm0
	movdqu xmm0, xmmword ptr [edx + 4*esi + 80]
	movdqa xmm7, xmm1
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm6
	movdqa xmm6, xmmword ptr [esp + 128]
	pmuludq xmm2, xmm4
	pshufd xmm1, xmm0, 232
	pshufd xmm0, xmm2, 232
	punpckldq xmm1, xmm0
	movdqu xmm0, xmmword ptr [edx + 4*esi + 96]
	movdqa xmmword ptr [esp + 48], xmm1
	movdqa xmm1, xmmword ptr [esp + 80]
	pshufd xmm2, xmm0, 245
	pshufd xmm4, xmm1, 245
	pmuludq xmm0, xmm1
	pmuludq xmm2, xmm4
	pshufd xmm1, xmm0, 232
	movdqa xmm4, xmm3
	movdqa xmm3, xmmword ptr [esp + 96]
	pshufd xmm0, xmm2, 232
	movdqu xmm2, xmmword ptr [edx + 4*esi + 112]
	punpckldq xmm1, xmm0
	movdqa xmmword ptr [esp + 80], xmm1
	movdqa xmm1, xmmword ptr [esp + 144]
	pshufd xmm0, xmm2, 245
	pmuludq xmm2, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm0, xmm1
	pshufd xmm2, xmm2, 232
	movdqa xmm1, xmm7
	movdqa xmm7, xmmword ptr [esp + 48]
	pshufd xmm0, xmm0, 232
	punpckldq xmm2, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi]
	psubd xmm6, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi + 16]
	psubd xmm5, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi + 32]
	psubd xmm4, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi + 48]
	psubd xmm3, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi + 64]
	psubd xmm1, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi + 80]
	psubd xmm7, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi + 96]
	movdqa xmmword ptr [esp + 48], xmm7
	movdqa xmm7, xmmword ptr [esp + 80]
	psubd xmm7, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi + 112]
	movdqu xmmword ptr [eax + 4*esi], xmm6
	movdqu xmmword ptr [eax + 4*esi + 16], xmm5
	movdqu xmmword ptr [eax + 4*esi + 32], xmm4
	movdqu xmmword ptr [eax + 4*esi + 48], xmm3
	movdqu xmmword ptr [eax + 4*esi + 64], xmm1
	movdqa xmm1, xmmword ptr [esp + 48]
	psubd xmm2, xmm0
	movdqu xmmword ptr [eax + 4*esi + 80], xmm1
	movdqu xmmword ptr [eax + 4*esi + 96], xmm7
	movdqu xmmword ptr [eax + 4*esi + 112], xmm2
	add esi, 32
	cmp esi, edi
	jb .LBB169_14
	mov ebp, dword ptr [esp + 44]
	sub ebp, esi
	ja .LBB169_6
	jmp .LBB169_18
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 204]
	lea edx, [esp + 208]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 28], edx
	mov dword ptr [esp + 32], ebp
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#3}, ()>
	jmp .LBB169_18
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 204]
	lea edx, [esp + 208]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 28], edx
	mov dword ptr [esp + 32], ebp
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#2}, ()>
	jmp .LBB169_18
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 204]
	lea edx, [esp + 208]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 28], edx
	mov dword ptr [esp + 32], ebp
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>::{closure#1}, ()>
	jmp .LBB169_18
.LBB169_4:
	xor esi, esi
	mov ebp, dword ptr [esp + 44]
	sub ebp, esi
	jbe .LBB169_18
.LBB169_6:
	mov ecx, dword ptr [esp + 192]
	cmp ebp, 16
	mov edx, dword ptr [ecx + 68]
	mov edi, dword ptr [ecx + 224]
	mov ebx, dword ptr [ecx + 304]
	mov dword ptr [esp + 12], edx
	mov edx, dword ptr [ecx + 144]
	jb .LBB169_7
	mov dword ptr [esp + 40], ebp
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
	je .LBB169_9
.LBB169_7:
	mov ebp, esi
	mov esi, dword ptr [esp + 12]
.LBB169_12:
	mov ecx, dword ptr [edx + 4*ebp]
	add ecx, dword ptr [esi + 4*ebp]
	imul ecx, dword ptr [edi + 4*ebp]
	sub ecx, dword ptr [ebx + 4*ebp]
	mov dword ptr [eax + 4*ebp], ecx
	inc ebp
	cmp dword ptr [esp + 44], ebp
	jne .LBB169_12
.LBB169_18:
	add esp, 172
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB169_9:
	mov ecx, dword ptr [esp + 40]
	and ecx, -4
	mov dword ptr [esp + 48], ecx
	lea ebp, [esi + ecx]
	lea ecx, [eax + 4*esi]
	mov dword ptr [esp + 144], ecx
	lea ecx, [ebx + 4*esi]
	mov dword ptr [esp + 128], ecx
	lea ecx, [edi + 4*esi]
	mov dword ptr [esp + 80], ecx
	lea ecx, [edx + 4*esi]
	mov dword ptr [esp + 112], ecx
	mov ecx, dword ptr [esp + 12]
	lea ecx, [ecx + 4*esi]
	xor esi, esi
	mov dword ptr [esp + 96], ecx
.LBB169_10:
	mov ecx, dword ptr [esp + 96]
	movdqu xmm0, xmmword ptr [ecx + 4*esi]
	mov ecx, dword ptr [esp + 112]
	movdqu xmm1, xmmword ptr [ecx + 4*esi]
	mov ecx, dword ptr [esp + 80]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [ecx + 4*esi]
	mov ecx, dword ptr [esp + 128]
	movdqu xmm2, xmmword ptr [ecx + 4*esi]
	pshufd xmm3, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	mov ecx, dword ptr [esp + 144]
	pmuludq xmm1, xmm3
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm1, 232
	punpckldq xmm0, xmm1
	psubd xmm0, xmm2
	movdqu xmmword ptr [ecx + 4*esi], xmm0
	add esi, 4
	cmp dword ptr [esp + 48], esi
	jne .LBB169_10
	mov ecx, dword ptr [esp + 48]
	mov esi, dword ptr [esp + 12]
	cmp dword ptr [esp + 40], ecx
	jne .LBB169_12
	jmp .LBB169_18
.LBB169_1:
	lea eax, [esp + 76]
	mov dword ptr [esp + 76], esi
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
	jmp .LBB169_2
