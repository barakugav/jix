jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 76
	call .L168$pb
.L168$pb:
	pop ebx
.Ltmp9119:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp9119-.L168$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB168_1
.LBB168_2:
	movzx esi, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 104]
	mov eax, dword ptr [esp + 100]
	mov edx, dword ptr [esp + 96]
	mov esi, dword ptr [ebx + 4*esi + .LJTI168_0@GOTOFF]
	add esi, ebx
	jmp esi
	mov ebp, dword ptr [esp + 112]
	mov edi, ebp
	and edi, -32
	je .LBB168_4
	mov ebx, dword ptr [edx + 68]
	xor esi, esi
.LBB168_13:
	movdqu xmm1, xmmword ptr [ebx + 4*esi]
	movdqu xmm2, xmmword ptr [ebx + 4*esi + 16]
	movdqu xmm3, xmmword ptr [ebx + 4*esi + 32]
	pxor xmm7, xmm7
	movdqu xmm4, xmmword ptr [ebx + 4*esi + 48]
	movdqu xmm5, xmmword ptr [ebx + 4*esi + 64]
	movdqu xmm6, xmmword ptr [ebx + 4*esi + 80]
	movdqu xmm0, xmmword ptr [ebx + 4*esi + 96]
	psubd xmm7, xmm1
	pxor xmm1, xmm1
	psubd xmm1, xmm2
	pxor xmm2, xmm2
	psubd xmm2, xmm3
	pxor xmm3, xmm3
	psubd xmm3, xmm4
	pxor xmm4, xmm4
	psubd xmm4, xmm5
	pxor xmm5, xmm5
	psubd xmm5, xmm6
	pxor xmm6, xmm6
	psubd xmm6, xmm0
	movdqu xmm0, xmmword ptr [ebx + 4*esi + 112]
	movdqu xmmword ptr [eax + 4*esi], xmm7
	movdqu xmmword ptr [eax + 4*esi + 16], xmm1
	movdqu xmmword ptr [eax + 4*esi + 32], xmm2
	movdqu xmmword ptr [eax + 4*esi + 48], xmm3
	movdqu xmmword ptr [eax + 4*esi + 64], xmm4
	movdqu xmmword ptr [eax + 4*esi + 80], xmm5
	movdqu xmmword ptr [eax + 4*esi + 96], xmm6
	movdqa xmmword ptr [esp + 48], xmm0
	pxor xmm0, xmm0
	psubd xmm0, xmmword ptr [esp + 48]
	movdqu xmmword ptr [eax + 4*esi + 112], xmm0
	add esi, 32
	cmp esi, edi
	jb .LBB168_13
	mov ebx, ebp
	sub ebx, esi
	ja .LBB168_6
	jmp .LBB168_17
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 15]
	mov dword ptr [esp + 28], esi
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>::{closure#3}, ()>
	jmp .LBB168_17
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 15]
	mov dword ptr [esp + 28], esi
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>::{closure#2}, ()>
	jmp .LBB168_17
	mov dword ptr [esp + 16], eax
	mov dword ptr [esp + 20], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 15]
	mov dword ptr [esp + 28], esi
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ecx
	lea ecx, [esp + 16]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op1::Op1<_, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op1Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op1::NegKernel, i32>>::{closure#1}, ()>
	jmp .LBB168_17
.LBB168_4:
	xor esi, esi
	mov ebx, ebp
	sub ebx, esi
	jbe .LBB168_17
.LBB168_6:
	mov edx, dword ptr [edx + 68]
	cmp ebx, 8
	setb cl
	mov edi, edx
	sub edi, eax
	cmp edi, -31
	setae ch
	or ch, cl
	je .LBB168_8
	mov edi, esi
	jmp .LBB168_11
.LBB168_8:
	mov dword ptr [esp + 48], ebp
	mov ebp, ebx
	mov dword ptr [esp + 40], ebx
	lea ebx, [eax + 4*esi + 16]
	xor ecx, ecx
	and ebp, -8
	lea edi, [esi + ebp]
	lea esi, [edx + 4*esi + 16]
.LBB168_9:
	movdqu xmm0, xmmword ptr [esi + 4*ecx - 16]
	movdqu xmm1, xmmword ptr [esi + 4*ecx]
	pxor xmm2, xmm2
	psubd xmm2, xmm0
	pxor xmm0, xmm0
	psubd xmm0, xmm1
	movdqu xmmword ptr [ebx + 4*ecx - 16], xmm2
	movdqu xmmword ptr [ebx + 4*ecx], xmm0
	add ecx, 8
	cmp ebp, ecx
	jne .LBB168_9
	cmp dword ptr [esp + 40], ebp
	mov ebp, dword ptr [esp + 48]
	je .LBB168_17
.LBB168_11:
	xor ecx, ecx
	sub ecx, dword ptr [edx + 4*edi]
	mov dword ptr [eax + 4*edi], ecx
	inc edi
	cmp ebp, edi
	jne .LBB168_11
.LBB168_17:
	add esp, 76
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB168_1:
	lea eax, [esp + 44]
	mov dword ptr [esp + 44], esi
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
	jmp .LBB168_2
