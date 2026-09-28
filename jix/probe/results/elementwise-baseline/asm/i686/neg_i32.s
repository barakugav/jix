jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
	push ebp
	push ebx
	push edi
	push esi
	push eax
	mov ebp, dword ptr [esp + 40]
	mov ecx, dword ptr [esp + 28]
	mov esi, dword ptr [esp + 24]
	mov edi, ebp
	and edi, -32
	je .LBB152_1
	mov ebx, dword ptr [esi + 68]
	xor edx, edx
.LBB152_11:
	movdqu xmm0, xmmword ptr [ebx + 4*edx]
	pxor xmm1, xmm1
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx], xmm1
	pxor xmm1, xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 16]
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 16], xmm1
	pxor xmm1, xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 32]
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 32], xmm1
	pxor xmm1, xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 48]
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 48], xmm1
	pxor xmm1, xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 64]
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 64], xmm1
	pxor xmm1, xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 80]
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 80], xmm1
	pxor xmm1, xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 96]
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 96], xmm1
	pxor xmm1, xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 112]
	psubd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 112], xmm1
	add edx, 32
	cmp edx, edi
	jb .LBB152_11
	mov eax, ebp
	sub eax, edx
	ja .LBB152_3
	jmp .LBB152_9
.LBB152_1:
	xor edx, edx
	mov eax, ebp
	sub eax, edx
	jbe .LBB152_9
.LBB152_3:
	mov esi, dword ptr [esi + 68]
	cmp eax, 8
	jae .LBB152_6
	mov edi, edx
	jmp .LBB152_5
.LBB152_6:
	mov ebp, eax
	mov dword ptr [esp], eax
	lea eax, [ecx + 4*edx + 16]
	xor ebx, ebx
	and ebp, -8
	lea edi, [edx + ebp]
	lea edx, [esi + 4*edx + 16]
.LBB152_7:
	movdqu xmm0, xmmword ptr [edx + 4*ebx - 16]
	movdqu xmm1, xmmword ptr [edx + 4*ebx]
	pxor xmm2, xmm2
	psubd xmm2, xmm0
	pxor xmm0, xmm0
	psubd xmm0, xmm1
	movdqu xmmword ptr [eax + 4*ebx - 16], xmm2
	movdqu xmmword ptr [eax + 4*ebx], xmm0
	add ebx, 8
	cmp ebp, ebx
	jne .LBB152_7
	cmp dword ptr [esp], ebp
	mov ebp, dword ptr [esp + 40]
	je .LBB152_9
.LBB152_5:
	xor eax, eax
	sub eax, dword ptr [esi + 4*edi]
	mov dword ptr [ecx + 4*edi], eax
	inc edi
	cmp ebp, edi
	jne .LBB152_5
.LBB152_9:
	add esp, 4
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
