jix_probe::byte_shuffle::encode_impl::<16, 8>:
	lsr x6, x1, #4
	ands x8, x6, #0x7fffffffffffff8
	b.eq .LBB0_4
	stp x20, x19, [sp, #-16]!
	mov w9, #6
	mov w10, #10
	mov w11, #11
	mov w12, #12
	mov w13, #13
	mov w14, #14
	madd x9, x6, x9, x2
	lsl x19, x6, #3
	movi v0.2d, #0x000000000000ff
	lsl x17, x6, #1
	lsl x3, x6, #2
	sub x7, x2, x6
	madd x10, x6, x10, x2
	sub x4, x19, x6
	add x15, x2, x6
	mov x5, xzr
	add x16, x2, x17
	add x17, x15, x17
	madd x11, x6, x11, x2
	add x18, x2, x3
	add x3, x15, x3
	add x4, x2, x4
	add x20, x0, #64
	madd x12, x6, x12, x2
	madd x13, x6, x13, x2
	madd x14, x6, x14, x2
	add x6, x7, x6, lsl #4
	add x7, x2, x19
	add x19, x15, x19
.LBB0_2:
	ldur s2, [x20, #-48]
	ldp s4, s27, [x20, #-64]
	ldur s1, [x20, #-16]
	ldp s3, s16, [x20, #-32]
	ushll v2.8h, v2.8b, #0
	ushll v7.8h, v4.8b, #0
	ushll v5.8h, v1.8b, #0
	ldur s4, [x20, #-12]
	ushll v6.8h, v3.8b, #0
	ldp s1, s3, [x20]
	ushll v2.4s, v2.4h, #0
	ushll v18.4s, v7.4h, #0
	ushll v5.4s, v5.4h, #0
	ldp s20, s7, [x20, #16]
	ushll v4.8h, v4.8b, #0
	ushll v1.8h, v1.8b, #0
	ushll v17.4s, v6.4h, #0
	ushll v28.8h, v3.8b, #0
	ushll v21.2d, v2.2s, #0
	ushll2 v2.2d, v2.4s, #0
	ushll2 v22.2d, v5.4s, #0
	ushll v20.8h, v20.8b, #0
	ldp s19, s6, [x20, #32]
	ushll v1.4s, v1.4h, #0
	ushll2 v23.2d, v17.4s, #0
	ushll v24.2d, v18.2s, #0
	ushll2 v18.2d, v18.4s, #0
	and v21.16b, v21.16b, v0.16b
	and v2.16b, v2.16b, v0.16b
	ushll v20.4s, v20.4h, #0
	ushll v5.2d, v5.2s, #0
	ushll v17.2d, v17.2s, #0
	ushll v19.8h, v19.8b, #0
	ushll v25.2d, v1.2s, #0
	and v22.16b, v22.16b, v0.16b
	and v23.16b, v23.16b, v0.16b
	sli v18.2d, v2.2d, #8
	sli v24.2d, v21.2d, #8
	ldp s21, s2, [x20, #48]
	ushll2 v26.2d, v20.4s, #0
	ushll2 v1.2d, v1.4s, #0
	and v5.16b, v5.16b, v0.16b
	and v17.16b, v17.16b, v0.16b
	ushll v20.2d, v20.2s, #0
	and v25.16b, v25.16b, v0.16b
	shl v22.2d, v22.2d, #24
	shl v23.2d, v23.2d, #16
	ushll v21.8h, v21.8b, #0
	ushll v19.4s, v19.4h, #0
	and v1.16b, v1.16b, v0.16b
	and v26.16b, v26.16b, v0.16b
	shl v5.2d, v5.2d, #24
	shl v17.2d, v17.2d, #16
	and v20.16b, v20.16b, v0.16b
	shl v25.2d, v25.2d, #32
	orr v22.16b, v23.16b, v22.16b
	ldur s23, [x20, #-44]
	ushll v21.4s, v21.4h, #0
	shl v1.2d, v1.2d, #32
	shl v26.2d, v26.2d, #40
	ushll v16.8h, v16.8b, #0
	orr v5.16b, v17.16b, v5.16b
	ushll2 v17.2d, v19.4s, #0
	ushll v19.2d, v19.2s, #0
	shl v20.2d, v20.2d, #40
	orr v24.16b, v24.16b, v25.16b
	ushll v3.8h, v23.8b, #0
	orr v18.16b, v18.16b, v1.16b
	ldp s25, s1, [x20, #40]
	orr v22.16b, v22.16b, v26.16b
	ushll2 v26.2d, v21.4s, #0
	and v17.16b, v17.16b, v0.16b
	ushll v21.2d, v21.2s, #0
	and v19.16b, v19.16b, v0.16b
	orr v20.16b, v5.16b, v20.16b
	ushll v5.8h, v2.8b, #0
	ushll v2.8h, v25.8b, #0
	ushll v23.4s, v16.4h, #0
	shl v25.2d, v26.2d, #56
	shl v17.2d, v17.2d, #48
	ushll v26.8h, v7.8b, #0
	shl v21.2d, v21.2d, #56
	shl v19.2d, v19.2d, #48
	ushll v3.4s, v3.4h, #0
	ushll v6.8h, v6.8b, #0
	ushll v16.4s, v5.4h, #0
	ushll v1.8h, v1.8b, #0
	orr v17.16b, v18.16b, v17.16b
	orr v18.16b, v22.16b, v25.16b
	ushll v22.4s, v26.4h, #0
	orr v7.16b, v24.16b, v19.16b
	orr v19.16b, v20.16b, v21.16b
	ushll v20.4s, v4.4h, #0
	ushll v21.8h, v27.8b, #0
	ushll v25.2d, v3.2s, #0
	ushll v6.4s, v6.4h, #0
	orr v4.16b, v17.16b, v18.16b
	ushll v17.4s, v28.4h, #0
	ushll2 v3.2d, v3.4s, #0
	orr v18.16b, v7.16b, v19.16b
	ushll v5.2d, v20.2s, #0
	ushll v19.2d, v23.2s, #0
	ushll v21.4s, v21.4h, #0
	ushll2 v20.2d, v20.4s, #0
	ushll2 v23.2d, v23.4s, #0
	str d18, [x2, x5]
	ushll v7.2d, v22.2s, #0
	ushll v24.2d, v17.2s, #0
	and v5.16b, v5.16b, v0.16b
	and v19.16b, v19.16b, v0.16b
	and v25.16b, v25.16b, v0.16b
	ushll v26.2d, v21.2s, #0
	and v20.16b, v20.16b, v0.16b
	and v23.16b, v23.16b, v0.16b
	and v7.16b, v7.16b, v0.16b
	and v24.16b, v24.16b, v0.16b
	and v3.16b, v3.16b, v0.16b
	shl v5.2d, v5.2d, #24
	shl v19.2d, v19.2d, #16
	ushll2 v21.2d, v21.4s, #0
	sli v26.2d, v25.2d, #8
	ushll v25.2d, v6.2s, #0
	shl v20.2d, v20.2d, #24
	shl v23.2d, v23.2d, #16
	ushll2 v17.2d, v17.4s, #0
	shl v7.2d, v7.2d, #40
	shl v24.2d, v24.2d, #32
	sli v21.2d, v3.2d, #8
	orr v3.16b, v19.16b, v5.16b
	ldp s19, s5, [x20, #24]
	ushll2 v22.2d, v22.4s, #0
	orr v20.16b, v23.16b, v20.16b
	ushll v23.2d, v16.2s, #0
	and v25.16b, v25.16b, v0.16b
	and v17.16b, v17.16b, v0.16b
	orr v24.16b, v26.16b, v24.16b
	orr v26.16b, v3.16b, v7.16b
	ldp s7, s3, [x20, #8]
	and v22.16b, v22.16b, v0.16b
	shl v23.2d, v23.2d, #56
	shl v25.2d, v25.2d, #48
	ushll2 v27.2d, v6.4s, #0
	shl v17.2d, v17.2d, #32
	mov d18, v18.d[1]
	ushll2 v16.2d, v16.4s, #0
	shl v22.2d, v22.2d, #40
	ushll v19.8h, v19.8b, #0
	ushll v7.8h, v7.8b, #0
	orr v24.16b, v24.16b, v25.16b
	ldp s25, s6, [x20, #-8]
	str d18, [x15, x5]
	ldur s18, [x20, #-24]
	orr v23.16b, v26.16b, v23.16b
	str d4, [x16, x5]
	orr v17.16b, v21.16b, v17.16b
	ldur s21, [x20, #-40]
	and v26.16b, v27.16b, v0.16b
	orr v20.16b, v20.16b, v22.16b
	mov d22, v4.d[1]
	shl v16.2d, v16.2d, #56
	orr v23.16b, v24.16b, v23.16b
	ushll v4.8h, v21.8b, #0
	ldur s21, [x20, #-56]
	ushll v24.8h, v25.8b, #0
	ushll v18.8h, v18.8b, #0
	str d22, [x17, x5]
	shl v26.2d, v26.2d, #48
	ushll v7.4s, v7.4h, #0
	str d23, [x18, x5]
	orr v16.16b, v20.16b, v16.16b
	mov d20, v23.d[1]
	ushll v21.8h, v21.8b, #0
	ushll v23.4s, v4.4h, #0
	ushll v18.4s, v18.4h, #0
	orr v22.16b, v17.16b, v26.16b
	ushll v17.4s, v19.4h, #0
	ushll v19.4s, v24.4h, #0
	str d20, [x3, x5]
	ushll v25.2d, v7.2s, #0
	ushll2 v7.2d, v7.4s, #0
	ushll v24.2d, v23.2s, #0
	ushll2 v23.2d, v23.4s, #0
	ushll v21.4s, v21.4h, #0
	orr v4.16b, v22.16b, v16.16b
	ushll2 v16.2d, v19.4s, #0
	ushll2 v22.2d, v18.4s, #0
	ushll v19.2d, v19.2s, #0
	ushll v18.2d, v18.2s, #0
	and v25.16b, v25.16b, v0.16b
	str d4, [x9, x5]
	and v24.16b, v24.16b, v0.16b
	and v23.16b, v23.16b, v0.16b
	ushll v26.2d, v21.2s, #0
	ushll2 v21.2d, v21.4s, #0
	and v7.16b, v7.16b, v0.16b
	and v19.16b, v19.16b, v0.16b
	and v18.16b, v18.16b, v0.16b
	and v16.16b, v16.16b, v0.16b
	and v22.16b, v22.16b, v0.16b
	ushll v20.2d, v17.2s, #0
	shl v25.2d, v25.2d, #32
	sli v21.2d, v23.2d, #8
	sli v26.2d, v24.2d, #8
	ushll v27.4s, v2.4h, #0
	ldp s24, s23, [x20, #56]
	shl v19.2d, v19.2d, #24
	shl v18.2d, v18.2d, #16
	shl v2.2d, v7.2d, #32
	shl v16.2d, v16.2d, #24
	shl v22.2d, v22.2d, #16
	and v20.16b, v20.16b, v0.16b
	ushll2 v28.2d, v17.4s, #0
	ushll v7.8h, v24.8b, #0
	ldur s24, [x20, #-36]
	ushll v5.8h, v5.8b, #0
	orr v19.16b, v18.16b, v19.16b
	orr v18.16b, v26.16b, v25.16b
	ldur s25, [x20, #-52]
	orr v17.16b, v21.16b, v2.16b
	ushll v21.2d, v27.2s, #0
	ushll2 v26.2d, v27.4s, #0
	ushll v2.4s, v7.4h, #0
	orr v16.16b, v22.16b, v16.16b
	ldur s22, [x20, #-20]
	add x20, x20, #128
	shl v27.2d, v20.2d, #40
	ushll v7.8h, v23.8b, #0
	ushll v20.8h, v24.8b, #0
	and v29.16b, v21.16b, v0.16b
	and v24.16b, v26.16b, v0.16b
	ushll v23.2d, v2.2s, #0
	ushll v21.8h, v6.8b, #0
	ushll v22.8h, v22.8b, #0
	and v28.16b, v28.16b, v0.16b
	orr v19.16b, v19.16b, v27.16b
	ushll v6.8h, v25.8b, #0
	shl v25.2d, v29.2d, #48
	ushll v20.4s, v20.4h, #0
	ushll v27.8h, v3.8b, #0
	shl v23.2d, v23.2d, #56
	shl v3.2d, v24.2d, #48
	ushll v21.4s, v21.4h, #0
	ushll v22.4s, v22.4h, #0
	shl v26.2d, v28.2d, #40
	ushll v6.4s, v6.4h, #0
	orr v18.16b, v18.16b, v25.16b
	ushll2 v24.2d, v20.4s, #0
	ushll v5.4s, v5.4h, #0
	orr v3.16b, v17.16b, v3.16b
	orr v17.16b, v19.16b, v23.16b
	ushll v19.2d, v20.2s, #0
	ushll2 v23.2d, v21.4s, #0
	ushll2 v25.2d, v22.4s, #0
	orr v16.16b, v16.16b, v26.16b
	ushll v20.4s, v27.4h, #0
	ushll v21.2d, v21.2s, #0
	and v24.16b, v24.16b, v0.16b
	ushll2 v26.2d, v6.4s, #0
	ushll v22.2d, v22.2s, #0
	and v19.16b, v19.16b, v0.16b
	ushll v6.2d, v6.2s, #0
	and v23.16b, v23.16b, v0.16b
	and v25.16b, v25.16b, v0.16b
	ushll v1.4s, v1.4h, #0
	and v21.16b, v21.16b, v0.16b
	ushll v7.4s, v7.4h, #0
	sli v26.2d, v24.2d, #8
	ushll v24.2d, v20.2s, #0
	and v22.16b, v22.16b, v0.16b
	sli v6.2d, v19.2d, #8
	ushll v19.2d, v5.2s, #0
	shl v23.2d, v23.2d, #24
	shl v25.2d, v25.2d, #16
	ushll2 v20.2d, v20.4s, #0
	ushll2 v5.2d, v5.4s, #0
	and v24.16b, v24.16b, v0.16b
	shl v21.2d, v21.2d, #24
	shl v22.2d, v22.2d, #16
	and v19.16b, v19.16b, v0.16b
	ushll2 v2.2d, v2.4s, #0
	orr v23.16b, v25.16b, v23.16b
	ushll v25.2d, v1.2s, #0
	and v20.16b, v20.16b, v0.16b
	ushll2 v1.2d, v1.4s, #0
	and v5.16b, v5.16b, v0.16b
	shl v24.2d, v24.2d, #32
	orr v21.16b, v22.16b, v21.16b
	ushll v22.2d, v7.2s, #0
	shl v19.2d, v19.2d, #40
	and v25.16b, v25.16b, v0.16b
	shl v2.2d, v2.2d, #56
	shl v20.2d, v20.2d, #32
	ushll2 v7.2d, v7.4s, #0
	and v1.16b, v1.16b, v0.16b
	shl v5.2d, v5.2d, #40
	orr v6.16b, v6.16b, v24.16b
	orr v19.16b, v21.16b, v19.16b
	shl v21.2d, v22.2d, #56
	shl v22.2d, v25.2d, #48
	orr v2.16b, v16.16b, v2.16b
	orr v16.16b, v18.16b, v17.16b
	orr v17.16b, v26.16b, v20.16b
	orr v5.16b, v23.16b, v5.16b
	shl v7.2d, v7.2d, #56
	shl v1.2d, v1.2d, #48
	orr v18.16b, v19.16b, v21.16b
	mov d19, v4.d[1]
	orr v6.16b, v6.16b, v22.16b
	orr v2.16b, v3.16b, v2.16b
	mov d4, v16.d[1]
	orr v3.16b, v5.16b, v7.16b
	str d19, [x4, x5]
	orr v1.16b, v17.16b, v1.16b
	str d16, [x7, x5]
	orr v5.16b, v6.16b, v18.16b
	mov d6, v2.d[1]
	str d4, [x19, x5]
	str d2, [x10, x5]
	orr v1.16b, v1.16b, v3.16b
	str d6, [x11, x5]
	mov d2, v5.d[1]
	str d5, [x12, x5]
	mov d3, v1.d[1]
	str d2, [x13, x5]
	str d1, [x14, x5]
	str d3, [x6, x5]
	add x5, x5, #8
	cmp x5, x8
	b.lo .LBB0_2
	ldp x20, x19, [sp], #16
	mov w4, #16
	b jix_probe::byte_shuffle::encode_impl_generic
.LBB0_4:
	mov x5, xzr
	mov w4, #16
	b jix_probe::byte_shuffle::encode_impl_generic
