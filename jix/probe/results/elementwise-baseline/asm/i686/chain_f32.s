jix::storage::elementwise_pipeline::inner_loop::<f32, 32, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::AddKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::MulKernel, f32, f32>, jix::storage::elementwise_pipeline::OperandTyped<f32>, jix::ops::op2::SubKernel, f32, f32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 36
	mov edi, dword ptr [esp + 72]
	mov ecx, dword ptr [esp + 60]
	mov eax, dword ptr [esp + 56]
	and edi, -32
	je .LBB137_1
	mov ebx, dword ptr [eax + 68]
	mov ebp, dword ptr [eax + 144]
	mov esi, dword ptr [eax + 224]
	mov eax, dword ptr [eax + 304]
	xor edx, edx
.LBB137_11:
	movups xmm0, xmmword ptr [ebx + 4*edx]
	movups xmm1, xmmword ptr [ebp + 4*edx]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [esi + 4*edx]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [eax + 4*edx]
	subps xmm0, xmm1
	movups xmmword ptr [ecx + 4*edx], xmm0
	movups xmm0, xmmword ptr [ebx + 4*edx + 16]
	movups xmm1, xmmword ptr [ebp + 4*edx + 16]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [esi + 4*edx + 16]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [eax + 4*edx + 16]
	subps xmm0, xmm1
	movups xmmword ptr [ecx + 4*edx + 16], xmm0
	movups xmm0, xmmword ptr [ebx + 4*edx + 32]
	movups xmm1, xmmword ptr [ebp + 4*edx + 32]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [esi + 4*edx + 32]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [eax + 4*edx + 32]
	subps xmm0, xmm1
	movups xmmword ptr [ecx + 4*edx + 32], xmm0
	movups xmm0, xmmword ptr [ebx + 4*edx + 48]
	movups xmm1, xmmword ptr [ebp + 4*edx + 48]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [esi + 4*edx + 48]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [eax + 4*edx + 48]
	subps xmm0, xmm1
	movups xmmword ptr [ecx + 4*edx + 48], xmm0
	movups xmm0, xmmword ptr [ebx + 4*edx + 64]
	movups xmm1, xmmword ptr [ebp + 4*edx + 64]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [esi + 4*edx + 64]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [eax + 4*edx + 64]
	subps xmm0, xmm1
	movups xmmword ptr [ecx + 4*edx + 64], xmm0
	movups xmm0, xmmword ptr [ebx + 4*edx + 80]
	movups xmm1, xmmword ptr [ebp + 4*edx + 80]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [esi + 4*edx + 80]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [eax + 4*edx + 80]
	subps xmm0, xmm1
	movups xmmword ptr [ecx + 4*edx + 80], xmm0
	movups xmm0, xmmword ptr [ebx + 4*edx + 96]
	movups xmm1, xmmword ptr [ebp + 4*edx + 96]
	addps xmm1, xmm0
	movups xmm0, xmmword ptr [esi + 4*edx + 96]
	mulps xmm0, xmm1
	movups xmm1, xmmword ptr [eax + 4*edx + 96]
	subps xmm0, xmm1
	movups xmmword ptr [ecx + 4*edx + 96], xmm0
	movups xmm0, xmmword ptr [ebx + 4*edx + 112]
	movups xmm1, xmmword ptr [ebp + 4*edx + 112]
	movups xmm3, xmmword ptr [esi + 4*edx + 112]
	movups xmm2, xmmword ptr [eax + 4*edx + 112]
	addps xmm1, xmm0
	mulps xmm3, xmm1
	subps xmm3, xmm2
	movups xmmword ptr [ecx + 4*edx + 112], xmm3
	add edx, 32
	cmp edx, edi
	jb .LBB137_11
	mov eax, dword ptr [esp + 72]
	mov ebp, eax
	sub ebp, edx
	ja .LBB137_3
	jmp .LBB137_9
.LBB137_1:
	xor edx, edx
	mov eax, dword ptr [esp + 72]
	mov ebp, eax
	sub ebp, edx
	jbe .LBB137_9
.LBB137_3:
	mov esi, dword ptr [esp + 56]
	cmp ebp, 4
	mov edi, dword ptr [esi + 68]
	mov ebx, dword ptr [esi + 144]
	mov esi, dword ptr [esi + 224]
	mov dword ptr [esp], esi
	mov esi, dword ptr [esp + 56]
	mov esi, dword ptr [esi + 304]
	jae .LBB137_6
	mov ebp, edx
	mov edx, dword ptr [esp]
	jmp .LBB137_5
.LBB137_6:
	mov dword ptr [esp + 8], ebp
	and ebp, -4
	mov dword ptr [esp + 12], esi
	lea esi, [esi + 4*edx]
	lea eax, [ecx + 4*edx]
	mov dword ptr [esp + 20], edi
	mov dword ptr [esp + 4], ebp
	mov dword ptr [esp + 28], esi
	mov esi, ebx
	mov ebx, dword ptr [esp]
	mov dword ptr [esp + 32], eax
	add ebp, edx
	mov eax, dword ptr [esp + 4]
	mov dword ptr [esp + 16], esi
	lea esi, [esi + 4*edx]
	lea ebx, [ebx + 4*edx]
	lea edx, [edi + 4*edx]
	xor edi, edi
	mov dword ptr [esp + 24], ebx
.LBB137_7:
	mov ebx, dword ptr [esp + 24]
	movups xmm0, xmmword ptr [edx + 4*edi]
	movups xmm1, xmmword ptr [esi + 4*edi]
	movups xmm3, xmmword ptr [ebx + 4*edi]
	mov ebx, dword ptr [esp + 28]
	addps xmm1, xmm0
	movups xmm2, xmmword ptr [ebx + 4*edi]
	mov ebx, dword ptr [esp + 32]
	mulps xmm3, xmm1
	subps xmm3, xmm2
	movups xmmword ptr [ebx + 4*edi], xmm3
	add edi, 4
	cmp eax, edi
	jne .LBB137_7
	cmp dword ptr [esp + 8], eax
	mov eax, dword ptr [esp + 72]
	mov edi, dword ptr [esp + 20]
	mov ebx, dword ptr [esp + 16]
	mov edx, dword ptr [esp]
	mov esi, dword ptr [esp + 12]
	je .LBB137_9
.LBB137_5:
	movss xmm0, dword ptr [edi + 4*ebp]
	addss xmm0, dword ptr [ebx + 4*ebp]
	mulss xmm0, dword ptr [edx + 4*ebp]
	subss xmm0, dword ptr [esi + 4*ebp]
	movss dword ptr [ecx + 4*ebp], xmm0
	inc ebp
	cmp eax, ebp
	jne .LBB137_5
.LBB137_9:
	add esp, 36
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
