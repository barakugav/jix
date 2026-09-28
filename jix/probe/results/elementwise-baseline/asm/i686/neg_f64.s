jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op1::NegKernel, f64>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 12
	mov eax, dword ptr [esp + 48]
	mov ecx, dword ptr [esp + 36]
	mov edi, dword ptr [esp + 32]
	call .L120$pb
.L120$pb:
	pop edx
.Ltmp4967:
	add edx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp4967-.L120$pb)
	mov ebx, eax
	and ebx, -16
	je .LBB120_1
	mov ebp, dword ptr [edi + 68]
	movaps xmm0, xmmword ptr [edx + .LCPI120_0@GOTOFF]
	xor esi, esi
.LBB120_12:
	movups xmm1, xmmword ptr [ebp + 8*esi]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi], xmm1
	movups xmm1, xmmword ptr [ebp + 8*esi + 16]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi + 16], xmm1
	movups xmm1, xmmword ptr [ebp + 8*esi + 32]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi + 32], xmm1
	movups xmm1, xmmword ptr [ebp + 8*esi + 48]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi + 48], xmm1
	movups xmm1, xmmword ptr [ebp + 8*esi + 64]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi + 64], xmm1
	movups xmm1, xmmword ptr [ebp + 8*esi + 80]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi + 80], xmm1
	movups xmm1, xmmword ptr [ebp + 8*esi + 96]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi + 96], xmm1
	movups xmm1, xmmword ptr [ebp + 8*esi + 112]
	xorps xmm1, xmm0
	movups xmmword ptr [ecx + 8*esi + 112], xmm1
	add esi, 16
	cmp esi, ebx
	jb .LBB120_12
	mov ebp, eax
	sub ebp, esi
	ja .LBB120_3
	jmp .LBB120_10
.LBB120_1:
	xor esi, esi
	mov ebp, eax
	sub ebp, esi
	jbe .LBB120_10
.LBB120_3:
	mov edi, dword ptr [edi + 68]
	cmp ebp, 4
	jae .LBB120_5
	mov ebx, esi
	jmp .LBB120_8
.LBB120_5:
	mov dword ptr [esp + 8], ebp
	and ebp, -4
	mov dword ptr [esp], edx
	lea edx, [ecx + 8*esi + 16]
	mov dword ptr [esp + 4], edi
	xor eax, eax
	lea ebx, [esi + ebp]
	lea esi, [edi + 8*esi + 16]
	mov edi, dword ptr [esp]
	movaps xmm0, xmmword ptr [edi + .LCPI120_0@GOTOFF]
.LBB120_6:
	movups xmm1, xmmword ptr [esi + 8*eax - 16]
	movups xmm2, xmmword ptr [esi + 8*eax]
	xorps xmm1, xmm0
	xorps xmm2, xmm0
	movups xmmword ptr [edx + 8*eax - 16], xmm1
	movups xmmword ptr [edx + 8*eax], xmm2
	add eax, 4
	cmp ebp, eax
	jne .LBB120_6
	cmp dword ptr [esp + 8], ebp
	mov eax, dword ptr [esp + 48]
	mov edx, dword ptr [esp]
	mov edi, dword ptr [esp + 4]
	je .LBB120_10
.LBB120_8:
	movaps xmm0, xmmword ptr [edx + .LCPI120_0@GOTOFF]
.LBB120_9:
	movsd xmm1, qword ptr [edi + 8*ebx]
	xorps xmm1, xmm0
	movlps qword ptr [ecx + 8*ebx], xmm1
	inc ebx
	cmp eax, ebx
	jne .LBB120_9
.LBB120_10:
	add esp, 12
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
