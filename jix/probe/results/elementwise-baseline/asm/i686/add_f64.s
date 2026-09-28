jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 12
	mov eax, dword ptr [esp + 48]
	mov ecx, dword ptr [esp + 36]
	mov esi, dword ptr [esp + 32]
	mov edi, eax
	and edi, -16
	je .LBB123_1
	mov ebx, dword ptr [esi + 68]
	mov ebp, dword ptr [esi + 144]
	xor edx, edx
.LBB123_11:
	movupd xmm0, xmmword ptr [ebx + 8*edx]
	movupd xmm1, xmmword ptr [ebp + 8*edx]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx], xmm1
	movupd xmm0, xmmword ptr [ebx + 8*edx + 16]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 16]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx + 16], xmm1
	movupd xmm0, xmmword ptr [ebx + 8*edx + 32]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 32]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx + 32], xmm1
	movupd xmm0, xmmword ptr [ebx + 8*edx + 48]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 48]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx + 48], xmm1
	movupd xmm0, xmmword ptr [ebx + 8*edx + 64]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 64]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx + 64], xmm1
	movupd xmm0, xmmword ptr [ebx + 8*edx + 80]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 80]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx + 80], xmm1
	movupd xmm0, xmmword ptr [ebx + 8*edx + 96]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 96]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx + 96], xmm1
	movupd xmm0, xmmword ptr [ebx + 8*edx + 112]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 112]
	addpd xmm1, xmm0
	movupd xmmword ptr [ecx + 8*edx + 112], xmm1
	add edx, 16
	cmp edx, edi
	jb .LBB123_11
	mov ebp, eax
	sub ebp, edx
	ja .LBB123_3
	jmp .LBB123_9
.LBB123_1:
	xor edx, edx
	mov ebp, eax
	sub ebp, edx
	jbe .LBB123_9
.LBB123_3:
	mov edi, dword ptr [esi + 68]
	mov esi, dword ptr [esi + 144]
	cmp ebp, 2
	jae .LBB123_6
	mov ebx, edx
	jmp .LBB123_5
.LBB123_6:
	mov dword ptr [esp], edi
	mov dword ptr [esp + 8], esi
	lea edi, [esi + 8*edx]
	mov dword ptr [esp + 4], ebp
	and ebp, -2
	lea eax, [ecx + 8*edx]
	mov esi, dword ptr [esp]
	lea ebx, [edx + ebp]
	lea edx, [esi + 8*edx]
	xor esi, esi
.LBB123_7:
	movupd xmm0, xmmword ptr [edx + 8*esi]
	movupd xmm1, xmmword ptr [edi + 8*esi]
	addpd xmm1, xmm0
	movupd xmmword ptr [eax + 8*esi], xmm1
	add esi, 2
	cmp ebp, esi
	jne .LBB123_7
	cmp dword ptr [esp + 4], ebp
	mov eax, dword ptr [esp + 48]
	mov edi, dword ptr [esp]
	mov esi, dword ptr [esp + 8]
	je .LBB123_9
.LBB123_5:
	movsd xmm0, qword ptr [edi + 8*ebx]
	addsd xmm0, qword ptr [esi + 8*ebx]
	movsd qword ptr [ecx + 8*ebx], xmm0
	inc ebx
	cmp eax, ebx
	jne .LBB123_5
.LBB123_9:
	add esp, 12
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
