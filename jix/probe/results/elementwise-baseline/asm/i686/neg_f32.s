jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op1::NegKernel, f32>>:
	push ebp
	push ebx
	push edi
	push esi
	push eax
	mov ebx, dword ptr [esp + 40]
	mov ecx, dword ptr [esp + 28]
	mov edi, dword ptr [esp + 24]
	call .L136$pb
.L136$pb:
	pop eax
.Ltmp5785:
	add eax, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp5785-.L136$pb)
	and ebx, -32
	je .LBB136_1
	mov ebp, dword ptr [edi + 68]
	movaps xmm0, xmmword ptr [eax + .LCPI136_0@GOTOFF]
	xor esi, esi
.LBB136_12:
	movups xmm1, xmmword ptr [ebp + 4*esi]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi], xmm1
	movups xmm1, xmmword ptr [ebp + 4*esi + 16]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi + 16], xmm1
	movups xmm1, xmmword ptr [ebp + 4*esi + 32]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi + 32], xmm1
	movups xmm1, xmmword ptr [ebp + 4*esi + 48]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi + 48], xmm1
	movups xmm1, xmmword ptr [ebp + 4*esi + 64]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi + 64], xmm1
	movups xmm1, xmmword ptr [ebp + 4*esi + 80]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi + 80], xmm1
	movups xmm1, xmmword ptr [ebp + 4*esi + 96]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi + 96], xmm1
	movups xmm1, xmmword ptr [ebp + 4*esi + 112]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 4*esi + 112], xmm1
	add esi, 32
	cmp esi, ebx
	jb .LBB136_12
	mov ebp, dword ptr [esp + 40]
	mov edx, ebp
	sub edx, esi
	ja .LBB136_3
	jmp .LBB136_10
.LBB136_1:
	xor esi, esi
	mov ebp, dword ptr [esp + 40]
	mov edx, ebp
	sub edx, esi
	jbe .LBB136_10
.LBB136_3:
	mov edi, dword ptr [edi + 68]
	cmp edx, 8
	jae .LBB136_5
	mov ebx, esi
	jmp .LBB136_8
.LBB136_5:
	movaps xmm0, xmmword ptr [eax + .LCPI136_0@GOTOFF]
	mov ebp, edx
	mov dword ptr [esp], edx
	lea edx, [ecx + 4*esi + 16]
	xor eax, eax
	and ebp, -8
	lea ebx, [esi + ebp]
	lea esi, [edi + 4*esi + 16]
.LBB136_6:
	movups xmm1, xmmword ptr [esi + 4*eax - 16]
	movups xmm2, xmmword ptr [esi + 4*eax]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [edx + 4*eax - 16], xmm1
	movups xmmword ptr [edx + 4*eax], xmm2
	add eax, 8
	cmp ebp, eax
	jne .LBB136_6
	cmp dword ptr [esp], ebp
	mov ebp, dword ptr [esp + 40]
	je .LBB136_10
.LBB136_8:
	mov eax, -2147483648
.LBB136_9:
	mov edx, dword ptr [edi + 4*ebx]
	xor edx, eax
	mov dword ptr [ecx + 4*ebx], edx
	inc ebx
	cmp ebp, ebx
	jne .LBB136_9
.LBB136_10:
	add esp, 4
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
