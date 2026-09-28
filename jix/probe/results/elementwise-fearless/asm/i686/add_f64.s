jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 76
	call .L163$pb
.L163$pb:
	pop ebx
.Ltmp8712:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp8712-.L163$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB163_1
.LBB163_2:
	movzx esi, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 104]
	mov eax, dword ptr [esp + 100]
	mov edx, dword ptr [esp + 96]
	mov esi, dword ptr [ebx + 4*esi + .LJTI163_0@GOTOFF]
	add esi, ebx
	jmp esi
	mov ecx, dword ptr [esp + 112]
	mov edi, ecx
	and edi, -16
	je .LBB163_4
	mov ebx, dword ptr [edx + 68]
	mov ebp, dword ptr [edx + 144]
	xor esi, esi
.LBB163_15:
	movupd xmm1, xmmword ptr [ebx + 8*esi]
	movupd xmm2, xmmword ptr [ebp + 8*esi]
	movupd xmm3, xmmword ptr [ebx + 8*esi + 16]
	movupd xmm5, xmmword ptr [ebx + 8*esi + 32]
	movupd xmm7, xmmword ptr [ebx + 8*esi + 48]
	movupd xmm6, xmmword ptr [ebx + 8*esi + 64]
	movupd xmm4, xmmword ptr [ebx + 8*esi + 80]
	movupd xmm0, xmmword ptr [ebx + 8*esi + 96]
	addpd xmm2, xmm1
	movupd xmm1, xmmword ptr [ebp + 8*esi + 16]
	addpd xmm1, xmm3
	movupd xmm3, xmmword ptr [ebp + 8*esi + 32]
	addpd xmm3, xmm5
	movupd xmm5, xmmword ptr [ebp + 8*esi + 48]
	addpd xmm5, xmm7
	movupd xmm7, xmmword ptr [ebp + 8*esi + 64]
	addpd xmm7, xmm6
	movupd xmm6, xmmword ptr [ebp + 8*esi + 80]
	addpd xmm6, xmm4
	movupd xmm4, xmmword ptr [ebp + 8*esi + 96]
	addpd xmm4, xmm0
	movups xmm0, xmmword ptr [ebx + 8*esi + 112]
	movaps xmmword ptr [esp + 48], xmm0
	movupd xmm0, xmmword ptr [ebp + 8*esi + 112]
	movupd xmmword ptr [eax + 8*esi], xmm2
	movupd xmmword ptr [eax + 8*esi + 16], xmm1
	movupd xmmword ptr [eax + 8*esi + 32], xmm3
	movupd xmmword ptr [eax + 8*esi + 48], xmm5
	movupd xmmword ptr [eax + 8*esi + 64], xmm7
	movupd xmmword ptr [eax + 8*esi + 80], xmm6
	movupd xmmword ptr [eax + 8*esi + 96], xmm4
	addpd xmm0, xmmword ptr [esp + 48]
	movupd xmmword ptr [eax + 8*esi + 112], xmm0
	add esi, 16
	cmp esi, edi
	jb .LBB163_15
	mov ebx, ecx
	sub ebx, esi
	ja .LBB163_6
	jmp .LBB163_19
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
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>::{closure#3}, ()>
	jmp .LBB163_19
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
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>::{closure#2}, ()>
	jmp .LBB163_19
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
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>>::{closure#1}, ()>
	jmp .LBB163_19
.LBB163_4:
	xor esi, esi
	mov ebx, ecx
	sub ebx, esi
	jbe .LBB163_19
.LBB163_6:
	mov edi, dword ptr [edx + 68]
	mov edx, dword ptr [edx + 144]
	cmp ebx, 10
	jae .LBB163_8
	mov ebp, esi
	jmp .LBB163_13
.LBB163_8:
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
	je .LBB163_10
	mov ebp, esi
	jmp .LBB163_13
.LBB163_10:
	mov ebx, dword ptr [esp + 28]
	and ebx, -2
	mov dword ptr [esp + 48], ebx
	lea ebp, [esi + ebx]
	lea ebx, [eax + 8*esi]
	mov dword ptr [esp + 40], ebx
	lea ebx, [edx + 8*esi]
	lea esi, [edi + 8*esi]
	mov dword ptr [esp + 36], ebx
	xor ebx, ebx
	mov dword ptr [esp + 32], esi
.LBB163_11:
	mov esi, dword ptr [esp + 32]
	movupd xmm0, xmmword ptr [esi + 8*ebx]
	mov esi, dword ptr [esp + 36]
	movupd xmm1, xmmword ptr [esi + 8*ebx]
	mov esi, dword ptr [esp + 40]
	addpd xmm1, xmm0
	movupd xmmword ptr [esi + 8*ebx], xmm1
	add ebx, 2
	cmp dword ptr [esp + 48], ebx
	jne .LBB163_11
	mov esi, dword ptr [esp + 48]
	cmp dword ptr [esp + 28], esi
	je .LBB163_19
.LBB163_13:
	movsd xmm0, qword ptr [edi + 8*ebp]
	addsd xmm0, qword ptr [edx + 8*ebp]
	movsd qword ptr [eax + 8*ebp], xmm0
	inc ebp
	cmp ecx, ebp
	jne .LBB163_13
.LBB163_19:
	add esp, 76
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB163_1:
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
	jmp .LBB163_2
