jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 12
	mov eax, dword ptr [esp + 48]
	mov ecx, dword ptr [esp + 36]
	mov esi, dword ptr [esp + 32]
	mov edi, eax
	and edi, -32
	je .LBB139_1
	mov ebx, dword ptr [esi + 68]
	mov ebp, dword ptr [esi + 144]
	xor edx, edx
.LBB139_11:
	movups xmm0, xmmword ptr [ebx + 4*edx]
	movups xmm1, xmmword ptr [ebp + 4*edx]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx], xmm1
	movups xmm0, xmmword ptr [ebx + 4*edx + 16]
	movups xmm1, xmmword ptr [ebp + 4*edx + 16]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx + 16], xmm1
	movups xmm0, xmmword ptr [ebx + 4*edx + 32]
	movups xmm1, xmmword ptr [ebp + 4*edx + 32]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx + 32], xmm1
	movups xmm0, xmmword ptr [ebx + 4*edx + 48]
	movups xmm1, xmmword ptr [ebp + 4*edx + 48]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx + 48], xmm1
	movups xmm0, xmmword ptr [ebx + 4*edx + 64]
	movups xmm1, xmmword ptr [ebp + 4*edx + 64]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx + 64], xmm1
	movups xmm0, xmmword ptr [ebx + 4*edx + 80]
	movups xmm1, xmmword ptr [ebp + 4*edx + 80]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx + 80], xmm1
	movups xmm0, xmmword ptr [ebx + 4*edx + 96]
	movups xmm1, xmmword ptr [ebp + 4*edx + 96]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx + 96], xmm1
	movups xmm0, xmmword ptr [ebx + 4*edx + 112]
	movups xmm1, xmmword ptr [ebp + 4*edx + 112]
	addps xmm1, xmm0
	movups xmmword ptr [ecx + 4*edx + 112], xmm1
	add edx, 32
	cmp edx, edi
	jb .LBB139_11
	mov ebp, eax
	sub ebp, edx
	ja .LBB139_3
	jmp .LBB139_9
.LBB139_1:
	xor edx, edx
	mov ebp, eax
	sub ebp, edx
	jbe .LBB139_9
.LBB139_3:
	mov edi, dword ptr [esi + 68]
	mov esi, dword ptr [esi + 144]
	cmp ebp, 4
	jae .LBB139_6
	mov ebx, edx
	jmp .LBB139_5
.LBB139_6:
	mov dword ptr [esp], edi
	mov dword ptr [esp + 8], esi
	lea edi, [esi + 4*edx]
	mov dword ptr [esp + 4], ebp
	and ebp, -4
	lea eax, [ecx + 4*edx]
	mov esi, dword ptr [esp]
	lea ebx, [edx + ebp]
	lea edx, [esi + 4*edx]
	xor esi, esi
.LBB139_7:
	movups xmm0, xmmword ptr [edx + 4*esi]
	movups xmm1, xmmword ptr [edi + 4*esi]
	addps xmm1, xmm0
	movups xmmword ptr [eax + 4*esi], xmm1
	add esi, 4
	cmp ebp, esi
	jne .LBB139_7
	cmp dword ptr [esp + 4], ebp
	mov eax, dword ptr [esp + 48]
	mov edi, dword ptr [esp]
	mov esi, dword ptr [esp + 8]
	je .LBB139_9
.LBB139_5:
	movss xmm0, dword ptr [edi + 4*ebx]
	addss xmm0, dword ptr [esi + 4*ebx]
	movss dword ptr [ecx + 4*ebx], xmm0
	inc ebx
	cmp eax, ebx
	jne .LBB139_5
.LBB139_9:
	add esp, 12
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
