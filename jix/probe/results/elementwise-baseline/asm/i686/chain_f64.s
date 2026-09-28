jix::storage::elementwise_pipeline::inner_loop::<f64, 16, true, true, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 36
	mov edi, dword ptr [esp + 72]
	mov ecx, dword ptr [esp + 60]
	mov eax, dword ptr [esp + 56]
	and edi, -16
	je .LBB121_1
	mov ebx, dword ptr [eax + 68]
	mov ebp, dword ptr [eax + 144]
	mov esi, dword ptr [eax + 224]
	mov eax, dword ptr [eax + 304]
	xor edx, edx
.LBB121_11:
	movupd xmm0, xmmword ptr [ebx + 8*edx]
	movupd xmm1, xmmword ptr [ebp + 8*edx]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [esi + 8*edx]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [eax + 8*edx]
	subpd xmm0, xmm1
	movupd xmmword ptr [ecx + 8*edx], xmm0
	movupd xmm0, xmmword ptr [ebx + 8*edx + 16]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 16]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [esi + 8*edx + 16]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [eax + 8*edx + 16]
	subpd xmm0, xmm1
	movupd xmmword ptr [ecx + 8*edx + 16], xmm0
	movupd xmm0, xmmword ptr [ebx + 8*edx + 32]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 32]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [esi + 8*edx + 32]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [eax + 8*edx + 32]
	subpd xmm0, xmm1
	movupd xmmword ptr [ecx + 8*edx + 32], xmm0
	movupd xmm0, xmmword ptr [ebx + 8*edx + 48]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 48]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [esi + 8*edx + 48]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [eax + 8*edx + 48]
	subpd xmm0, xmm1
	movupd xmmword ptr [ecx + 8*edx + 48], xmm0
	movupd xmm0, xmmword ptr [ebx + 8*edx + 64]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 64]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [esi + 8*edx + 64]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [eax + 8*edx + 64]
	subpd xmm0, xmm1
	movupd xmmword ptr [ecx + 8*edx + 64], xmm0
	movupd xmm0, xmmword ptr [ebx + 8*edx + 80]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 80]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [esi + 8*edx + 80]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [eax + 8*edx + 80]
	subpd xmm0, xmm1
	movupd xmmword ptr [ecx + 8*edx + 80], xmm0
	movupd xmm0, xmmword ptr [ebx + 8*edx + 96]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 96]
	addpd xmm1, xmm0
	movupd xmm0, xmmword ptr [esi + 8*edx + 96]
	mulpd xmm0, xmm1
	movupd xmm1, xmmword ptr [eax + 8*edx + 96]
	subpd xmm0, xmm1
	movupd xmmword ptr [ecx + 8*edx + 96], xmm0
	movupd xmm0, xmmword ptr [ebx + 8*edx + 112]
	movupd xmm1, xmmword ptr [ebp + 8*edx + 112]
	movupd xmm3, xmmword ptr [esi + 8*edx + 112]
	movupd xmm2, xmmword ptr [eax + 8*edx + 112]
	addpd xmm1, xmm0
	mulpd xmm3, xmm1
	subpd xmm3, xmm2
	movupd xmmword ptr [ecx + 8*edx + 112], xmm3
	add edx, 16
	cmp edx, edi
	jb .LBB121_11
	mov eax, dword ptr [esp + 72]
	mov ebp, eax
	sub ebp, edx
	ja .LBB121_3
	jmp .LBB121_9
.LBB121_1:
	xor edx, edx
	mov eax, dword ptr [esp + 72]
	mov ebp, eax
	sub ebp, edx
	jbe .LBB121_9
.LBB121_3:
	mov esi, dword ptr [esp + 56]
	cmp ebp, 2
	mov edi, dword ptr [esi + 68]
	mov ebx, dword ptr [esi + 144]
	mov esi, dword ptr [esi + 224]
	mov dword ptr [esp], esi
	mov esi, dword ptr [esp + 56]
	mov esi, dword ptr [esi + 304]
	jae .LBB121_6
	mov ebp, edx
	mov edx, dword ptr [esp]
	jmp .LBB121_5
.LBB121_6:
	mov dword ptr [esp + 8], ebp
	and ebp, -2
	mov dword ptr [esp + 12], esi
	lea esi, [esi + 8*edx]
	lea eax, [ecx + 8*edx]
	mov dword ptr [esp + 20], edi
	mov dword ptr [esp + 4], ebp
	mov dword ptr [esp + 28], esi
	mov esi, ebx
	mov ebx, dword ptr [esp]
	mov dword ptr [esp + 32], eax
	add ebp, edx
	mov eax, dword ptr [esp + 4]
	mov dword ptr [esp + 16], esi
	lea esi, [esi + 8*edx]
	lea ebx, [ebx + 8*edx]
	lea edx, [edi + 8*edx]
	xor edi, edi
	mov dword ptr [esp + 24], ebx
.LBB121_7:
	mov ebx, dword ptr [esp + 24]
	movupd xmm0, xmmword ptr [edx + 8*edi]
	movupd xmm1, xmmword ptr [esi + 8*edi]
	movupd xmm3, xmmword ptr [ebx + 8*edi]
	mov ebx, dword ptr [esp + 28]
	addpd xmm1, xmm0
	movupd xmm2, xmmword ptr [ebx + 8*edi]
	mov ebx, dword ptr [esp + 32]
	mulpd xmm3, xmm1
	subpd xmm3, xmm2
	movupd xmmword ptr [ebx + 8*edi], xmm3
	add edi, 2
	cmp eax, edi
	jne .LBB121_7
	cmp dword ptr [esp + 8], eax
	mov eax, dword ptr [esp + 72]
	mov edi, dword ptr [esp + 20]
	mov ebx, dword ptr [esp + 16]
	mov edx, dword ptr [esp]
	mov esi, dword ptr [esp + 12]
	je .LBB121_9
.LBB121_5:
	movsd xmm0, qword ptr [edi + 8*ebp]
	addsd xmm0, qword ptr [ebx + 8*ebp]
	mulsd xmm0, qword ptr [edx + 8*ebp]
	subsd xmm0, qword ptr [esi + 8*ebp]
	movsd qword ptr [ecx + 8*ebp], xmm0
	inc ebp
	cmp eax, ebp
	jne .LBB121_5
.LBB121_9:
	add esp, 36
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
