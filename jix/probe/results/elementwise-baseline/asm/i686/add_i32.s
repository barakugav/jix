jix::storage::elementwise_pipeline::inner_loop::<i32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 12
	mov edi, dword ptr [esp + 48]
	mov ecx, dword ptr [esp + 36]
	mov esi, dword ptr [esp + 32]
	and edi, -32
	je .LBB155_1
	mov ebx, dword ptr [esi + 68]
	mov ebp, dword ptr [esi + 144]
	xor edx, edx
.LBB155_11:
	movdqu xmm0, xmmword ptr [ebx + 4*edx]
	movdqu xmm1, xmmword ptr [ebp + 4*edx]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx], xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 16]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 16]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 16], xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 32]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 32]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 32], xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 48]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 48]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 48], xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 64]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 64]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 64], xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 80]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 80]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 80], xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 96]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 96]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 96], xmm1
	movdqu xmm0, xmmword ptr [ebx + 4*edx + 112]
	movdqu xmm1, xmmword ptr [ebp + 4*edx + 112]
	paddd xmm1, xmm0
	movdqu xmmword ptr [ecx + 4*edx + 112], xmm1
	add edx, 32
	cmp edx, edi
	jb .LBB155_11
	mov edi, dword ptr [esp + 48]
	mov eax, edi
	sub eax, edx
	ja .LBB155_3
	jmp .LBB155_9
.LBB155_1:
	xor edx, edx
	mov edi, dword ptr [esp + 48]
	mov eax, edi
	sub eax, edx
	jbe .LBB155_9
.LBB155_3:
	mov ebp, dword ptr [esi + 68]
	mov esi, dword ptr [esi + 144]
	cmp eax, 4
	jae .LBB155_6
	mov ebx, edx
	jmp .LBB155_5
.LBB155_6:
	mov edi, esi
	mov esi, ebp
	mov ebp, eax
	mov dword ptr [esp], eax
	lea eax, [ecx + 4*edx]
	and ebp, -4
	mov dword ptr [esp + 4], edi
	lea edi, [edi + 4*edx]
	mov dword ptr [esp + 8], esi
	lea ebx, [edx + ebp]
	lea edx, [esi + 4*edx]
	xor esi, esi
.LBB155_7:
	movdqu xmm0, xmmword ptr [edx + 4*esi]
	movdqu xmm1, xmmword ptr [edi + 4*esi]
	paddd xmm1, xmm0
	movdqu xmmword ptr [eax + 4*esi], xmm1
	add esi, 4
	cmp ebp, esi
	jne .LBB155_7
	cmp dword ptr [esp], ebp
	mov edi, dword ptr [esp + 48]
	mov ebp, dword ptr [esp + 8]
	mov esi, dword ptr [esp + 4]
	je .LBB155_9
.LBB155_5:
	mov eax, dword ptr [esi + 4*ebx]
	add eax, dword ptr [ebp + 4*ebx]
	mov dword ptr [ecx + 4*ebx], eax
	inc ebx
	cmp edi, ebx
	jne .LBB155_5
.LBB155_9:
	add esp, 12
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
