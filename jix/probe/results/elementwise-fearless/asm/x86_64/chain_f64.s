jix::storage::elementwise_pipeline::inner_loop_contiguous::<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>:
	push rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	sub rsp, 88
	mov qword ptr [rsp + 24], rcx
	mov qword ptr [rsp + 16], r8
	mov rbx, qword ptr [rip + fearless_simd::X86_LEVEL@GOTPCREL]
	mov eax, dword ptr [rbx + 8]
	test eax, eax
	jne .LBB168_23
.LBB168_1:
	movzx eax, byte ptr [rbx]
	lea rcx, [rip + .LJTI168_0]
	movsxd rax, dword ptr [rcx + 4*rax]
	add rax, rcx
	jmp rax
	mov rax, qword ptr [rsp + 16]
	mov rdx, rax
	and rdx, -16
	je .LBB168_10
	mov r8, qword ptr [rdi + 136]
	mov r9, qword ptr [rdi + 288]
	mov r10, qword ptr [rdi + 448]
	mov r11, qword ptr [rdi + 608]
	xor ecx, ecx
.LBB168_4:
	movupd xmm0, xmmword ptr [r8 + 8*rcx]
	movupd xmm1, xmmword ptr [r8 + 8*rcx + 16]
	movupd xmm2, xmmword ptr [r8 + 8*rcx + 32]
	movupd xmm3, xmmword ptr [r8 + 8*rcx + 48]
	movupd xmm4, xmmword ptr [r8 + 8*rcx + 64]
	movupd xmm5, xmmword ptr [r8 + 8*rcx + 80]
	movupd xmm6, xmmword ptr [r8 + 8*rcx + 96]
	movupd xmm7, xmmword ptr [r8 + 8*rcx + 112]
	movupd xmm8, xmmword ptr [r9 + 8*rcx]
	addpd xmm8, xmm0
	movupd xmm9, xmmword ptr [r9 + 8*rcx + 16]
	addpd xmm9, xmm1
	movupd xmm10, xmmword ptr [r9 + 8*rcx + 32]
	addpd xmm10, xmm2
	movupd xmm11, xmmword ptr [r9 + 8*rcx + 48]
	addpd xmm11, xmm3
	movupd xmm12, xmmword ptr [r9 + 8*rcx + 64]
	addpd xmm12, xmm4
	movupd xmm13, xmmword ptr [r9 + 8*rcx + 80]
	addpd xmm13, xmm5
	movupd xmm14, xmmword ptr [r9 + 8*rcx + 96]
	addpd xmm14, xmm6
	movupd xmm15, xmmword ptr [r9 + 8*rcx + 112]
	addpd xmm15, xmm7
	movupd xmm0, xmmword ptr [r10 + 8*rcx]
	mulpd xmm0, xmm8
	movupd xmm1, xmmword ptr [r10 + 8*rcx + 16]
	mulpd xmm1, xmm9
	movupd xmm2, xmmword ptr [r10 + 8*rcx + 32]
	mulpd xmm2, xmm10
	movupd xmm3, xmmword ptr [r10 + 8*rcx + 48]
	mulpd xmm3, xmm11
	movupd xmm4, xmmword ptr [r10 + 8*rcx + 64]
	mulpd xmm4, xmm12
	movupd xmm5, xmmword ptr [r10 + 8*rcx + 80]
	mulpd xmm5, xmm13
	movupd xmm6, xmmword ptr [r10 + 8*rcx + 96]
	mulpd xmm6, xmm14
	movupd xmm7, xmmword ptr [r10 + 8*rcx + 112]
	mulpd xmm7, xmm15
	movupd xmm8, xmmword ptr [r11 + 8*rcx]
	subpd xmm0, xmm8
	movupd xmm8, xmmword ptr [r11 + 8*rcx + 16]
	subpd xmm1, xmm8
	movupd xmm8, xmmword ptr [r11 + 8*rcx + 32]
	subpd xmm2, xmm8
	movupd xmm8, xmmword ptr [r11 + 8*rcx + 48]
	subpd xmm3, xmm8
	movupd xmm8, xmmword ptr [r11 + 8*rcx + 64]
	subpd xmm4, xmm8
	movupd xmm8, xmmword ptr [r11 + 8*rcx + 80]
	subpd xmm5, xmm8
	movupd xmm8, xmmword ptr [r11 + 8*rcx + 96]
	subpd xmm6, xmm8
	movupd xmm8, xmmword ptr [r11 + 8*rcx + 112]
	subpd xmm7, xmm8
	movupd xmmword ptr [rsi + 8*rcx], xmm0
	movupd xmmword ptr [rsi + 8*rcx + 16], xmm1
	movupd xmmword ptr [rsi + 8*rcx + 32], xmm2
	movupd xmmword ptr [rsi + 8*rcx + 48], xmm3
	movupd xmmword ptr [rsi + 8*rcx + 64], xmm4
	movupd xmmword ptr [rsi + 8*rcx + 80], xmm5
	movupd xmmword ptr [rsi + 8*rcx + 96], xmm6
	movupd xmmword ptr [rsi + 8*rcx + 112], xmm7
	add rcx, 16
	cmp rcx, rdx
	jb .LBB168_4
	mov r11, rax
	sub r11, rcx
	jbe .LBB168_9
	jmp .LBB168_11
	mov qword ptr [rsp + 32], rsi
	mov qword ptr [rsp + 40], rdx
	lea rax, [rsp + 24]
	mov qword ptr [rsp + 48], rax
	lea rax, [rsp + 16]
	mov qword ptr [rsp + 56], rax
	mov qword ptr [rsp + 64], rdi
	lea rax, [rsp + 15]
	mov qword ptr [rsp + 72], rax
	lea rdi, [rsp + 32]
	call <fearless_simd::generated::avx512::Avx512 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx512::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>::{closure#3}, ()>
	jmp .LBB168_9
	mov qword ptr [rsp + 32], rsi
	mov qword ptr [rsp + 40], rdx
	lea rax, [rsp + 24]
	mov qword ptr [rsp + 48], rax
	lea rax, [rsp + 16]
	mov qword ptr [rsp + 56], rax
	mov qword ptr [rsp + 64], rdi
	lea rax, [rsp + 15]
	mov qword ptr [rsp + 72], rax
	lea rdi, [rsp + 32]
	call <fearless_simd::generated::avx2::Avx2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_avx2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>::{closure#2}, ()>
	jmp .LBB168_9
	mov qword ptr [rsp + 32], rsi
	mov qword ptr [rsp + 40], rdx
	lea rax, [rsp + 24]
	mov qword ptr [rsp + 48], rax
	lea rax, [rsp + 16]
	mov qword ptr [rsp + 56], rax
	mov qword ptr [rsp + 64], rdi
	lea rax, [rsp + 15]
	mov qword ptr [rsp + 72], rax
	lea rdi, [rsp + 32]
	call <fearless_simd::generated::sse4_2::Sse4_2 as fearless_simd::generated::simd_trait::Simd>::vectorize::vectorize_sse4_2::<jix::storage::elementwise_pipeline::inner_loop_contiguous<f64, 16, <jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<<jix::ops::op2::Op2<_, _, _> as jix::storage::core_trait::ArrayStorage>::read_as_elementwise_pipeline::Op2Pipeline<jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::AddKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::MulKernel, f64, f64>, jix::storage::elementwise_pipeline::OperandTyped<f64>, jix::ops::op2::SubKernel, f64, f64>>::{closure#1}, ()>
.LBB168_9:
	add rsp, 88
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	ret
.LBB168_10:
	xor ecx, ecx
	mov r11, rax
	sub r11, rcx
	jbe .LBB168_9
.LBB168_11:
	mov rdx, qword ptr [rdi + 136]
	mov r8, qword ptr [rdi + 288]
	mov r9, qword ptr [rdi + 448]
	mov rdi, qword ptr [rdi + 608]
	cmp r11, 12
	jb .LBB168_12
	mov r10, rdx
	sub r10, rsi
	cmp r10, -31
	setae r10b
	mov rbx, r8
	sub rbx, rsi
	cmp rbx, -31
	setae bl
	or bl, r10b
	mov r10, r9
	sub r10, rsi
	cmp r10, -31
	setae r10b
	mov r14, rdi
	sub r14, rsi
	cmp r14, -31
	setae bpl
	or bpl, r10b
	or bpl, bl
	je .LBB168_15
.LBB168_12:
	mov r10, rcx
.LBB168_18:
	mov r11d, eax
	sub r11d, r10d
	lea rcx, [r10 + 1]
	test r11b, 1
	je .LBB168_20
	movsd xmm0, qword ptr [rdx + 8*r10]
	addsd xmm0, qword ptr [r8 + 8*r10]
	mulsd xmm0, qword ptr [r9 + 8*r10]
	subsd xmm0, qword ptr [rdi + 8*r10]
	movsd qword ptr [rsi + 8*r10], xmm0
	mov r10, rcx
.LBB168_20:
	cmp rax, rcx
	je .LBB168_9
	sub rax, r10
	lea rcx, [rsi + 8*r10]
	add rcx, 8
	lea rsi, [rdi + 8*r10 + 8]
	lea rdi, [r9 + 8*r10]
	add rdi, 8
	lea r8, [r8 + 8*r10]
	add r8, 8
	lea rdx, [rdx + 8*r10]
	add rdx, 8
	xor r9d, r9d
.LBB168_22:
	movsd xmm0, qword ptr [rdx + 8*r9 - 8]
	addsd xmm0, qword ptr [r8 + 8*r9 - 8]
	mulsd xmm0, qword ptr [rdi + 8*r9 - 8]
	subsd xmm0, qword ptr [rsi + 8*r9 - 8]
	movsd qword ptr [rcx + 8*r9 - 8], xmm0
	movsd xmm0, qword ptr [rdx + 8*r9]
	addsd xmm0, qword ptr [r8 + 8*r9]
	mulsd xmm0, qword ptr [rdi + 8*r9]
	subsd xmm0, qword ptr [rsi + 8*r9]
	movsd qword ptr [rcx + 8*r9], xmm0
	add r9, 2
	cmp rax, r9
	jne .LBB168_22
	jmp .LBB168_9
.LBB168_15:
	mov rbx, r11
	and rbx, -4
	lea r10, [rcx + rbx]
	lea r14, [rsi + 8*rcx]
	add r14, 16
	lea r15, [rdi + 8*rcx + 16]
	lea r12, [r9 + 8*rcx]
	add r12, 16
	lea r13, [r8 + 8*rcx]
	add r13, 16
	lea rcx, [rdx + 8*rcx]
	add rcx, 16
	xor ebp, ebp
.LBB168_16:
	movupd xmm0, xmmword ptr [rcx + 8*rbp - 16]
	movupd xmm1, xmmword ptr [rcx + 8*rbp]
	movupd xmm2, xmmword ptr [r13 + 8*rbp - 16]
	addpd xmm2, xmm0
	movupd xmm0, xmmword ptr [r13 + 8*rbp]
	addpd xmm0, xmm1
	movupd xmm1, xmmword ptr [r12 + 8*rbp - 16]
	mulpd xmm1, xmm2
	movupd xmm2, xmmword ptr [r12 + 8*rbp]
	mulpd xmm2, xmm0
	movupd xmm0, xmmword ptr [r15 + 8*rbp - 16]
	subpd xmm1, xmm0
	movupd xmm0, xmmword ptr [r15 + 8*rbp]
	subpd xmm2, xmm0
	movupd xmmword ptr [r14 + 8*rbp - 16], xmm1
	movupd xmmword ptr [r14 + 8*rbp], xmm2
	add rbp, 4
	cmp rbx, rbp
	jne .LBB168_16
	cmp r11, rbx
	je .LBB168_9
	jmp .LBB168_18
.LBB168_23:
	mov qword ptr [rsp + 80], rbx
	lea rax, [rsp + 80]
	mov qword ptr [rsp + 32], rax
	lea rax, [rbx + 8]
	lea rcx, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.18]
	lea r8, [rip + .Lanon.db56d094e776968c4de1bd5d19159acc.20]
	lea r9, [rsp + 32]
	mov r14, rdi
	mov rdi, rax
	mov r15, rsi
	mov esi, 1
	mov r12, rdx
	mov rdx, r9
	call qword ptr [rip + <std::sys::sync::once::futex::Once>::call@GOTPCREL]
	mov rdx, r12
	mov rdi, r14
	mov rsi, r15
	jmp .LBB168_1
