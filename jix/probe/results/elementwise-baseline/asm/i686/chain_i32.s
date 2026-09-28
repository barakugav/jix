jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::MulKernel, i32, i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::SubKernel, i32, i32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 32
	mov edi, dword ptr [esp + 68]
	mov ecx, dword ptr [esp + 56]
	mov eax, dword ptr [esp + 52]
	and edi, -32
	je .LBB153_1
	mov ebx, dword ptr [eax + 68]
	mov ebp, dword ptr [eax + 144]
	mov esi, dword ptr [eax + 224]
	mov eax, dword ptr [eax + 304]
	xor edx, edx
.LBB153_11:
	movdqu xmm0, xmmword ptr [ebx + 4*edx]
	movdqu xmm1, xmmword ptr [ebp + 4*edx]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	punpckldq xmm0, xmm1
	movdqu xmm1, xmmword ptr [eax + 4*edx]
	psubd xmm0, xmm1
	movdqu xmmword ptr [ecx + 4*edx], xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 16]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 16]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx + 16]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	punpckldq xmm0, xmm1
	movdqu xmm1, xmmword ptr [eax + 4*edx + 16]
	psubd xmm0, xmm1
	movdqu xmmword ptr [ecx + 4*edx + 16], xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 32]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 32]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx + 32]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	punpckldq xmm0, xmm1
	movdqu xmm1, xmmword ptr [eax + 4*edx + 32]
	psubd xmm0, xmm1
	movdqu xmmword ptr [ecx + 4*edx + 32], xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 48]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 48]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx + 48]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	punpckldq xmm0, xmm1
	movdqu xmm1, xmmword ptr [eax + 4*edx + 48]
	psubd xmm0, xmm1
	movdqu xmmword ptr [ecx + 4*edx + 48], xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 64]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 64]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx + 64]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	punpckldq xmm0, xmm1
	movdqu xmm1, xmmword ptr [eax + 4*edx + 64]
	psubd xmm0, xmm1
	movdqu xmmword ptr [ecx + 4*edx + 64], xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 80]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 80]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx + 80]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	punpckldq xmm0, xmm1
	movdqu xmm1, xmmword ptr [eax + 4*edx + 80]
	psubd xmm0, xmm1
	movdqu xmmword ptr [ecx + 4*edx + 80], xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 96]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 96]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx + 96]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	punpckldq xmm0, xmm1
	movdqu xmm1, xmmword ptr [eax + 4*edx + 96]
	psubd xmm0, xmm1
	movdqu xmmword ptr [ecx + 4*edx + 96], xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 112]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 112]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [esi + 4*edx + 112]
	pshufd xmm2, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	pmuludq xmm2, xmm1
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm2, 232
	movdqu xmm2, xmmword ptr [eax + 4*edx + 112]
	punpckldq xmm0, xmm1
	psubd xmm0, xmm2
	movdqu xmmword ptr [ecx + 4*edx + 112], xmm0
	add edx, 32
	cmp edx, edi
	jb .LBB153_11
	mov esi, dword ptr [esp + 68]
	mov ebp, esi
	sub ebp, edx
	ja .LBB153_3
	jmp .LBB153_9
.LBB153_1:
	xor edx, edx
	mov esi, dword ptr [esp + 68]
	mov ebp, esi
	sub ebp, edx
	jbe .LBB153_9
.LBB153_3:
	mov eax, dword ptr [esp + 52]
	cmp ebp, 4
	mov edi, dword ptr [eax + 68]
	mov ebx, dword ptr [eax + 144]
	mov dword ptr [esp + 4], edi
	mov edi, dword ptr [eax + 224]
	mov dword ptr [esp], edi
	mov edi, dword ptr [eax + 304]
	jae .LBB153_6
	mov ebp, edx
	mov edx, dword ptr [esp]
	jmp .LBB153_5
.LBB153_6:
	lea esi, [ecx + 4*edx]
	mov dword ptr [esp + 12], edi
	mov eax, ebp
	mov dword ptr [esp + 8], ebp
	mov dword ptr [esp + 16], ebx
	mov dword ptr [esp + 28], esi
	lea esi, [edi + 4*edx]
	and eax, -4
	mov dword ptr [esp + 24], esi
	mov esi, dword ptr [esp]
	lea ebp, [edx + eax]
	lea edi, [esi + 4*edx]
	lea esi, [ebx + 4*edx]
	mov dword ptr [esp + 20], edi
	mov edi, dword ptr [esp + 4]
	lea edx, [edi + 4*edx]
	xor edi, edi
.LBB153_7:
	movdqu xmm0, xmmword ptr [edx + 4*edi]
	movdqu xmm1, xmmword ptr [esi + 4*edi]
	mov ebx, dword ptr [esp + 20]
	paddd xmm1, xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*edi]
	mov ebx, dword ptr [esp + 24]
	movdqu xmm2, xmmword ptr [ebx + 4*edi]
	pshufd xmm3, xmm0, 245
	pmuludq xmm0, xmm1
	pshufd xmm1, xmm1, 245
	mov ebx, dword ptr [esp + 28]
	pmuludq xmm1, xmm3
	pshufd xmm0, xmm0, 232
	pshufd xmm1, xmm1, 232
	punpckldq xmm0, xmm1
	psubd xmm0, xmm2
	movdqu xmmword ptr [ebx + 4*edi], xmm0
	add edi, 4
	cmp eax, edi
	jne .LBB153_7
	cmp dword ptr [esp + 8], eax
	mov esi, dword ptr [esp + 68]
	mov ebx, dword ptr [esp + 16]
	mov edx, dword ptr [esp]
	mov edi, dword ptr [esp + 12]
	je .LBB153_9
.LBB153_5:
	mov ecx, edx
	mov edx, edi
	mov edi, esi
	mov eax, dword ptr [ebx + 4*ebp]
	mov esi, ebx
	mov ebx, dword ptr [esp + 4]
	add eax, dword ptr [ebx + 4*ebp]
	mov ebx, esi
	mov esi, edi
	mov edi, edx
	mov edx, ecx
	mov ecx, dword ptr [esp + 56]
	imul eax, dword ptr [edx + 4*ebp]
	sub eax, dword ptr [edi + 4*ebp]
	mov dword ptr [ecx + 4*ebp], eax
	inc ebp
	cmp esi, ebp
	jne .LBB153_5
.LBB153_9:
	add esp, 32
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
