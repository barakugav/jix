jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	push ebp
	push ebx
	push edi
	push esi
	sub esp, 140
	call .L161$pb
.L161$pb:
	pop ebx
.Ltmp8504:
	add ebx, offset _GLOBAL_OFFSET_TABLE_+(.Ltmp8504-.L161$pb)
	mov esi, dword ptr [ebx + fearless_simd::X86_LEVEL@GOT]
	mov eax, dword ptr [esi]
	test eax, eax
	jne .LBB161_1
.LBB161_2:
	movzx edx, byte ptr [esi + 4]
	mov ecx, dword ptr [esp + 168]
	mov eax, dword ptr [esp + 164]
	mov ebp, dword ptr [esp + 160]
	mov edx, dword ptr [ebx + 4*edx + .LJTI161_0@GOTOFF]
	add edx, ebx
	jmp edx
	mov edi, dword ptr [esp + 176]
	mov dword ptr [esp + 16], edi
	and edi, -16
	je .LBB161_4
	mov ecx, ebp
	mov ebx, dword ptr [ebp + 68]
	mov ebp, dword ptr [ebp + 144]
	xor esi, esi
	mov edx, dword ptr [ecx + 224]
	mov ecx, dword ptr [ecx + 304]
.LBB161_15:
	movupd xmm2, xmmword ptr [ebx + 8*esi]
	movupd xmm1, xmmword ptr [ebp + 8*esi]
	movupd xmm3, xmmword ptr [ebx + 8*esi + 16]
	movupd xmm4, xmmword ptr [ebx + 8*esi + 32]
	movupd xmm5, xmmword ptr [ebx + 8*esi + 48]
	movupd xmm6, xmmword ptr [ebx + 8*esi + 64]
	movupd xmm7, xmmword ptr [ebx + 8*esi + 80]
	movupd xmm0, xmmword ptr [ebx + 8*esi + 96]
	addpd xmm1, xmm2
	movupd xmm2, xmmword ptr [ebp + 8*esi + 16]
	addpd xmm2, xmm3
	movupd xmm3, xmmword ptr [ebp + 8*esi + 32]
	addpd xmm3, xmm4
	movupd xmm4, xmmword ptr [ebp + 8*esi + 48]
	addpd xmm4, xmm5
	movupd xmm5, xmmword ptr [ebp + 8*esi + 64]
	addpd xmm5, xmm6
	movupd xmm6, xmmword ptr [ebp + 8*esi + 80]
	addpd xmm6, xmm7
	movupd xmm7, xmmword ptr [ebp + 8*esi + 96]
	addpd xmm7, xmm0
	movups xmm0, xmmword ptr [ebx + 8*esi + 112]
	movaps xmmword ptr [esp + 48], xmm0
	movupd xmm0, xmmword ptr [ebp + 8*esi + 112]
	addpd xmm0, xmmword ptr [esp + 48]
	movapd xmmword ptr [esp + 112], xmm0
	movupd xmm0, xmmword ptr [edx + 8*esi]
	mulpd xmm0, xmm1
	movapd xmmword ptr [esp + 48], xmm0
	movupd xmm0, xmmword ptr [edx + 8*esi + 16]
	movapd xmm1, xmmword ptr [esp + 48]
	mulpd xmm0, xmm2
	movupd xmm2, xmmword ptr [edx + 8*esi + 32]
	movapd xmmword ptr [esp + 96], xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi]
	mulpd xmm2, xmm3
	movupd xmm3, xmmword ptr [edx + 8*esi + 48]
	subpd xmm1, xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi + 16]
	movapd xmmword ptr [esp + 48], xmm1
	movapd xmm1, xmmword ptr [esp + 96]
	mulpd xmm3, xmm4
	movupd xmm4, xmmword ptr [edx + 8*esi + 64]
	subpd xmm1, xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi + 32]
	mulpd xmm4, xmm5
	movupd xmm5, xmmword ptr [edx + 8*esi + 80]
	subpd xmm2, xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi + 48]
	mulpd xmm5, xmm6
	movupd xmm6, xmmword ptr [edx + 8*esi + 96]
	subpd xmm3, xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi + 64]
	mulpd xmm6, xmm7
	movupd xmm7, xmmword ptr [edx + 8*esi + 112]
	subpd xmm4, xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi + 80]
	mulpd xmm7, xmmword ptr [esp + 112]
	subpd xmm5, xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi + 96]
	subpd xmm6, xmm0
	movupd xmm0, xmmword ptr [ecx + 8*esi + 112]
	subpd xmm7, xmm0
	movapd xmm0, xmmword ptr [esp + 48]
	movupd xmmword ptr [eax + 8*esi], xmm0
	movupd xmmword ptr [eax + 8*esi + 16], xmm1
	movupd xmmword ptr [eax + 8*esi + 32], xmm2
	movupd xmmword ptr [eax + 8*esi + 48], xmm3
	movupd xmmword ptr [eax + 8*esi + 64], xmm4
	movupd xmmword ptr [eax + 8*esi + 80], xmm5
	movupd xmmword ptr [eax + 8*esi + 96], xmm6
	movupd xmmword ptr [eax + 8*esi + 112], xmm7
	add esi, 16
	cmp esi, edi
	jb .LBB161_15
	mov ecx, dword ptr [esp + 16]
	mov ebp, ecx
	sub ebp, esi
	ja .LBB161_6
	jmp .LBB161_19
	mov dword ptr [esp + 20], eax
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 172]
	lea edx, [esp + 176]
	mov dword ptr [esp + 28], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ebp
	mov dword ptr [esp + 40], ecx
	lea ecx, [esp + 20]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>::{closure#3}, ()>
	jmp .LBB161_19
	mov dword ptr [esp + 20], eax
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 172]
	lea edx, [esp + 176]
	mov dword ptr [esp + 28], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ebp
	mov dword ptr [esp + 40], ecx
	lea ecx, [esp + 20]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>::{closure#2}, ()>
	jmp .LBB161_19
	mov dword ptr [esp + 20], eax
	mov dword ptr [esp + 24], ecx
	lea ecx, [esp + 172]
	lea edx, [esp + 176]
	mov dword ptr [esp + 28], ecx
	lea ecx, [esp + 11]
	mov dword ptr [esp + 32], edx
	mov dword ptr [esp + 36], ebp
	mov dword ptr [esp + 40], ecx
	lea ecx, [esp + 20]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>::{closure#1}, ()>
	jmp .LBB161_19
.LBB161_4:
	xor esi, esi
	mov ecx, dword ptr [esp + 16]
	mov ebp, ecx
	sub ebp, esi
	jbe .LBB161_19
.LBB161_6:
	mov ebx, dword ptr [esp + 160]
	cmp ebp, 12
	mov edx, dword ptr [ebx + 68]
	mov edi, dword ptr [ebx + 224]
	mov dword ptr [esp + 12], edx
	mov edx, dword ptr [ebx + 144]
	mov ebx, dword ptr [ebx + 304]
	jae .LBB161_8
	mov ebp, esi
	mov esi, dword ptr [esp + 12]
	jmp .LBB161_13
.LBB161_8:
	mov dword ptr [esp + 44], ebp
	mov ebp, dword ptr [esp + 12]
	sub ebp, eax
	cmp ebp, -15
	mov ebp, edx
	setae cl
	sub ebp, eax
	cmp ebp, -15
	mov ebp, edi
	setae ch
	sub ebp, eax
	or ch, cl
	cmp ebp, -15
	mov ebp, ebx
	setae byte ptr [esp + 48]
	sub ebp, eax
	cmp ebp, -15
	setae cl
	or cl, byte ptr [esp + 48]
	or cl, ch
	je .LBB161_10
	mov ebp, esi
	mov ecx, dword ptr [esp + 16]
	mov esi, dword ptr [esp + 12]
	jmp .LBB161_13
.LBB161_10:
	mov ecx, dword ptr [esp + 44]
	mov dword ptr [esp + 76], edx
	and ecx, -2
	mov dword ptr [esp + 48], ecx
	lea ebp, [esi + ecx]
	lea ecx, [eax + 8*esi]
	mov dword ptr [esp + 112], ecx
	lea ecx, [ebx + 8*esi]
	mov dword ptr [esp + 96], ecx
	lea ecx, [edi + 8*esi]
	mov dword ptr [esp + 88], ecx
	lea ecx, [edx + 8*esi]
	mov dword ptr [esp + 84], ecx
	mov ecx, dword ptr [esp + 12]
	lea ecx, [ecx + 8*esi]
	xor esi, esi
	mov dword ptr [esp + 80], ecx
	mov ecx, dword ptr [esp + 16]
.LBB161_11:
	mov edx, dword ptr [esp + 80]
	movupd xmm0, xmmword ptr [edx + 8*esi]
	mov edx, dword ptr [esp + 84]
	movupd xmm1, xmmword ptr [edx + 8*esi]
	mov edx, dword ptr [esp + 88]
	movupd xmm3, xmmword ptr [edx + 8*esi]
	mov edx, dword ptr [esp + 96]
	addpd xmm1, xmm0
	movupd xmm2, xmmword ptr [edx + 8*esi]
	mov edx, dword ptr [esp + 112]
	mulpd xmm3, xmm1
	subpd xmm3, xmm2
	movupd xmmword ptr [edx + 8*esi], xmm3
	add esi, 2
	cmp dword ptr [esp + 48], esi
	jne .LBB161_11
	mov esi, dword ptr [esp + 48]
	mov edx, dword ptr [esp + 76]
	cmp dword ptr [esp + 44], esi
	mov esi, dword ptr [esp + 12]
	je .LBB161_19
.LBB161_13:
	movsd xmm0, qword ptr [esi + 8*ebp]
	addsd xmm0, qword ptr [edx + 8*ebp]
	mulsd xmm0, qword ptr [edi + 8*ebp]
	subsd xmm0, qword ptr [ebx + 8*ebp]
	movsd qword ptr [eax + 8*ebp], xmm0
	inc ebp
	cmp ecx, ebp
	jne .LBB161_13
.LBB161_19:
	add esp, 140
	pop esi
	pop edi
	pop ebx
	pop ebp
	ret
.LBB161_1:
	lea eax, [esp + 92]
	mov dword ptr [esp + 92], esi
	mov dword ptr [esp + 20], eax
	sub esp, 12
	lea eax, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.20@GOTOFF]
	lea ecx, [ebx + .Lanon.10111737824569b159b69e25ecbcf278.18@GOTOFF]
	lea edx, [esp + 32]
	push eax
	push ecx
	push edx
	push 1
	push esi
	call <std::sys::sync::once::futex::Once>::call@PLT
	add esp, 32
	jmp .LBB161_2
