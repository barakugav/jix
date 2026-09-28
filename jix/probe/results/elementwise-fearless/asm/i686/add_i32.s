jix::storage::elementwise_pipeline::inner_loop_contiguous::<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 76
	call .L171$pb
.L171$pb:
	pop ebx
.Ltmp9414:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp9414-.L171$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB171_1
.LBB171_2:
	movzx esi, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 104]
	mov eax, dword ptr [esp + 100]
	mov edx, dword ptr [esp + 96]
	mov esi, dword ptr [ebx + 4*esi + .LJTI171_0@GOTOFF]
	add esi, ebx
	jmp esi
	mov ecx, dword ptr [esp + 112]
	mov edi, ecx
	and edi, -32
	je .LBB171_4
	mov ebx, dword ptr [edx + 68]
	mov ebp, dword ptr [edx + 144]
	xor esi, esi
.LBB171_15:
	movdqu xmm1, xmmword ptr [ebx + 4*esi]
	movdqu xmm2, xmmword ptr [ebp + 4*esi]
	movdqu xmm3, xmmword ptr [ebx + 4*esi + 16]
	movdqu xmm5, xmmword ptr [ebx + 4*esi + 32]
	movdqu xmm7, xmmword ptr [ebx + 4*esi + 48]
	movdqu xmm6, xmmword ptr [ebx + 4*esi + 64]
	movdqu xmm4, xmmword ptr [ebx + 4*esi + 80]
	movdqu xmm0, xmmword ptr [ebx + 4*esi + 96]
	paddd xmm2, xmm1
	movdqu xmm1, xmmword ptr [ebp + 4*esi + 16]
	paddd xmm1, xmm3
	movdqu xmm3, xmmword ptr [ebp + 4*esi + 32]
	paddd xmm3, xmm5
	movdqu xmm5, xmmword ptr [ebp + 4*esi + 48]
	paddd xmm5, xmm7
	movdqu xmm7, xmmword ptr [ebp + 4*esi + 64]
	paddd xmm7, xmm6
	movdqu xmm6, xmmword ptr [ebp + 4*esi + 80]
	paddd xmm6, xmm4
	movdqu xmm4, xmmword ptr [ebp + 4*esi + 96]
	paddd xmm4, xmm0
	movups xmm0, xmmword ptr [ebx + 4*esi + 112]
	movaps xmmword ptr [esp + 48], xmm0
	movdqu xmm0, xmmword ptr [ebp + 4*esi + 112]
	movdqu xmmword ptr [eax + 4*esi], xmm2
	movdqu xmmword ptr [eax + 4*esi + 16], xmm1
	movdqu xmmword ptr [eax + 4*esi + 32], xmm3
	movdqu xmmword ptr [eax + 4*esi + 48], xmm5
	movdqu xmmword ptr [eax + 4*esi + 64], xmm7
	movdqu xmmword ptr [eax + 4*esi + 80], xmm6
	movdqu xmmword ptr [eax + 4*esi + 96], xmm4
	paddd xmm0, xmmword ptr [esp + 48]
	movdqu xmmword ptr [eax + 4*esi + 112], xmm0
	add esi, 32
	cmp esi, edi
	jb .LBB171_15
	mov ebx, ecx
	sub ebx, esi
	ja .LBB171_6
	jmp .LBB171_19
	mov dword ptr [esp + 4], eax
	mov dword ptr [esp + 8], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 12], ecx
	lea ecx, [esp + 3]
	mov dword ptr [esp + 16], esi
	mov dword ptr [esp + 20], edx
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 4]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>::{closure#3}, ()>
	jmp .LBB171_19
	mov dword ptr [esp + 4], eax
	mov dword ptr [esp + 8], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 12], ecx
	lea ecx, [esp + 3]
	mov dword ptr [esp + 16], esi
	mov dword ptr [esp + 20], edx
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 4]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>::{closure#2}, ()>
	jmp .LBB171_19
	mov dword ptr [esp + 4], eax
	mov dword ptr [esp + 8], ecx
	lea ecx, [esp + 108]
	lea esi, [esp + 112]
	mov dword ptr [esp + 12], ecx
	lea ecx, [esp + 3]
	mov dword ptr [esp + 16], esi
	mov dword ptr [esp + 20], edx
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 4]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<i32, 32, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::storage::elementwise_pipeline::OperandTyped<i32>, jix::ops::op2::AddKernel, i32, i32>>::{closure#1}, ()>
	jmp .LBB171_19
.LBB171_4:
	xor esi, esi
	mov ebx, ecx
	sub ebx, esi
	jbe .LBB171_19
.LBB171_6:
	mov edi, dword ptr [edx + 68]
	mov edx, dword ptr [edx + 144]
	cmp ebx, 12
	jae .LBB171_8
	mov ebp, esi
	jmp .LBB171_13
.LBB171_8:
	mov dword ptr [esp + 28], ebx
	mov ebx, edi
	mov ebp, edx
	sub ebx, eax
	cmp ebx, -15
	setae bl
	sub ebp, eax
	cmp ebp, -15
	setae bh
	or bh, bl
	je .LBB171_10
	mov ebp, esi
	jmp .LBB171_13
.LBB171_10:
	mov ebx, dword ptr [esp + 28]
	and ebx, -4
	mov dword ptr [esp + 48], ebx
	lea ebp, [esi + ebx]
	lea ebx, [eax + 4*esi]
	mov dword ptr [esp + 40], ebx
	lea ebx, [edx + 4*esi]
	lea esi, [edi + 4*esi]
	mov dword ptr [esp + 36], ebx
	xor ebx, ebx
	mov dword ptr [esp + 32], esi
.LBB171_11:
	mov esi, dword ptr [esp + 32]
	movdqu xmm0, xmmword ptr [esi + 4*ebx]
	mov esi, dword ptr [esp + 36]
	movdqu xmm1, xmmword ptr [esi + 4*ebx]
	mov esi, dword ptr [esp + 40]
	paddd xmm1, xmm0
	movdqu xmmword ptr [esi + 4*ebx], xmm1
	add ebx, 4
	cmp dword ptr [esp + 48], ebx
	jne .LBB171_11
	mov esi, dword ptr [esp + 48]
	cmp dword ptr [esp + 28], esi
	je .LBB171_19
.LBB171_13:
	mov esi, dword ptr [edx + 4*ebp]
	add esi, dword ptr [edi + 4*ebp]
	mov dword ptr [eax + 4*ebp], esi
	inc ebp
	cmp ecx, ebp
	jne .LBB171_13
.LBB171_19:
	add esp, 76
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB171_1:
	lea eax, [esp + 44]
	mov dword ptr [esp + 44], esi
	mov dword ptr [esp + 4], eax
	sub esp, 12
	lea eax, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.20@GOTOFF]
	lea ecx, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.18@GOTOFF]
	lea edx, [esp + 16]
	push eax
	push ecx
	push edx
	push 1
	push esi
	call <std::sys::sync::once::futex::Once>::call@PLT
	add esp, 32
	jmp .LBB171_2
