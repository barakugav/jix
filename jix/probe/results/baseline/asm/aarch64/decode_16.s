jix_probe::byte_shuffle::decode_impl::<16, 8>:
	lsr x6, x1, #4
	ands x8, x6, #0x7fffffffffffff8
	b.eq .LBB0_4
	stp d15, d14, [sp, #-112]!
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	str x30, [sp, #64]
	stp x22, x21, [sp, #80]
	stp x20, x19, [sp, #96]
	sub sp, sp, #400
	mov w9, #6
	mov w10, #14
	mov w11, #13
	mov w12, #12
	mov w13, #11
	mov w15, #10
	madd x9, x6, x9, x0
	lsl x20, x6, #3
	lsl x17, x6, #1
	lsl x3, x6, #2
	sub x7, x0, x6
	adrp x19, .LCPI0_0
	madd x10, x6, x10, x0
	sub x4, x20, x6
	add x14, x0, x6
	ldr q0, [x19, :lo12:.LCPI0_0]
	mov x5, xzr
	add x16, x0, x17
	madd x11, x6, x11, x0
	add x17, x14, x17
	add x18, x0, x3
	add x3, x14, x3
	add x4, x0, x4
	add x19, x14, x20
	madd x12, x6, x12, x0
	add x20, x0, x20
	str q0, [sp]
	madd x13, x6, x13, x0
	madd x15, x6, x15, x0
	add x6, x7, x6, lsl #4
	add x7, x2, #64
.LBB0_2:
	ldr d20, [x0, x5]
	add x22, x14, x5
	ldr d16, [x20, x5]
	add x21, x19, x5
	ld1 { v20.d }[1], [x22]
	add x22, x17, x5
	ld1 { v16.d }[1], [x21]
	ldr d21, [x16, x5]
	ldr d17, [x15, x5]
	add x21, x13, x5
	ld1 { v21.d }[1], [x22]
	add x22, x3, x5
	ld1 { v17.d }[1], [x21]
	ldr d22, [x18, x5]
	ldr d18, [x12, x5]
	add x21, x11, x5
	ld1 { v22.d }[1], [x22]
	add x22, x4, x5
	ld1 { v18.d }[1], [x21]
	ldr d23, [x9, x5]
	ldr d19, [x10, x5]
	add x21, x6, x5
	add x5, x5, #8
	ld1 { v23.d }[1], [x22]
	ld1 { v19.d }[1], [x21]
	add x21, sp, #336
	cmp x5, x8
	ushr v3.2d, v23.2d, #8
	ushr v27.2d, v19.2d, #8
	ushr v12.2d, v23.2d, #16
	ushr v26.2d, v18.2d, #8
	ushr v11.2d, v22.2d, #16
	ushr v7.2d, v19.2d, #24
	ushr v6.2d, v18.2d, #24
	ushr v31.2d, v19.2d, #16
	st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x21]
	ushr v3.2d, v19.2d, #32
	add x21, sp, #208
	ushr v30.2d, v18.2d, #16
	ushr v29.2d, v17.2d, #16
	ushr v28.2d, v16.2d, #16
	st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x21]
	add x21, sp, #16
	ushr v3.2d, v19.2d, #48
	ushr v2.2d, v18.2d, #48
	st1 { v24.2d, v25.2d, v26.2d, v27.2d }, [x21]
	add x21, sp, #336
	ld1 { v24.2d, v25.2d, v26.2d, v27.2d }, [x21]
	add x21, sp, #144
	ushr v26.2d, v22.2d, #8
	ushr v25.2d, v21.2d, #8
	st1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	ushr v12.2d, v19.2d, #40
	add x21, sp, #80
	ushr v11.2d, v18.2d, #40
	ushr v24.2d, v20.2d, #8
	st1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x21]
	add x21, sp, #208
	ld1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x21]
	add x21, sp, #272
	ushr v6.2d, v18.2d, #32
	ushr v5.2d, v17.2d, #32
	st1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	add x21, sp, #336
	st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x21]
	add x21, sp, #16
	ld1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x21]
	add x21, sp, #144
	ld1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	ushr v1.2d, v17.2d, #8
	ushr v10.2d, v21.2d, #16
	ushr v0.2d, v16.2d, #8
	st1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	add x21, sp, #80
	ld1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	add x21, sp, #208
	ushr v10.2d, v17.2d, #24
	ushr v9.2d, v16.2d, #24
	st1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x21]
	add x21, sp, #272
	ld1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x21]
	ushr v5.2d, v17.2d, #40
	st1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x21]
	add x21, sp, #336
	ld1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x21]
	ushr v5.2d, v17.2d, #48
	st1 { v4.2d, v5.2d, v6.2d, v7.2d }, [x21]
	add x21, sp, #16
	st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x21]
	add x21, sp, #144
	ushr v3.2d, v19.2d, #56
	ld1 { v5.2d, v6.2d, v7.2d, v8.2d }, [x21]
	add x21, sp, #80
	ushr v2.2d, v18.2d, #56
	ushr v1.2d, v17.2d, #56
	ushr v5.2d, v20.2d, #16
	ushr v0.2d, v16.2d, #56
	st1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	add x21, sp, #208
	ld1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	ushr v9.2d, v16.2d, #32
	st1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	add x21, sp, #272
	ld1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	ushr v9.2d, v16.2d, #40
	st1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	add x21, sp, #336
	ld1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	ushr v9.2d, v16.2d, #48
	st1 { v9.2d, v10.2d, v11.2d, v12.2d }, [x21]
	add x21, sp, #144
	ushr v12.2d, v23.2d, #32
	ushr v11.2d, v22.2d, #32
	ushr v10.2d, v21.2d, #32
	ushr v9.2d, v20.2d, #32
	st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x21]
	add x21, sp, #16
	ldr q0, [sp]
	ld1 { v1.2d, v2.2d, v3.2d, v4.2d }, [x21]
	add x21, sp, #80
	tbl v13.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v0.16b
	ushr v27.2d, v23.2d, #24
	tbl v15.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v0.16b
	ushr v26.2d, v22.2d, #24
	tbl v8.16b, { v5.16b, v6.16b, v7.16b, v8.16b }, v0.16b
	ushr v19.2d, v23.2d, #40
	ushr v25.2d, v21.2d, #24
	tbl v2.16b, { v1.16b, v2.16b, v3.16b, v4.16b }, v0.16b
	tbl v3.16b, { v28.16b, v29.16b, v30.16b, v31.16b }, v0.16b
	ushr v24.2d, v20.2d, #24
	ushr v31.2d, v23.2d, #56
	ushr v7.2d, v23.2d, #48
	ushr v30.2d, v22.2d, #56
	tbl v14.16b, { v20.16b, v21.16b, v22.16b, v23.16b }, v0.16b
	ushr v18.2d, v22.2d, #40
	ushr v29.2d, v21.2d, #56
	ushr v6.2d, v22.2d, #48
	ushr v17.2d, v21.2d, #40
	ushr v28.2d, v20.2d, #56
	tbl v23.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v0.16b
	ld1 { v24.2d, v25.2d, v26.2d, v27.2d }, [x21]
	ushr v5.2d, v21.2d, #48
	add x21, sp, #208
	ushr v16.2d, v20.2d, #40
	ushr v4.2d, v20.2d, #48
	mov v13.d[1], v2.d[0]
	mov v8.d[1], v3.d[0]
	tbl v21.16b, { v9.16b, v10.16b, v11.16b, v12.16b }, v0.16b
	mov v14.d[1], v15.d[0]
	tbl v20.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v0.16b
	ld1 { v24.2d, v25.2d, v26.2d, v27.2d }, [x21]
	add x21, sp, #272
	tbl v2.16b, { v4.16b, v5.16b, v6.16b, v7.16b }, v0.16b
	tbl v16.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v0.16b
	stp q14, q13, [x7, #-64]
	tbl v22.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v0.16b
	ld1 { v24.2d, v25.2d, v26.2d, v27.2d }, [x21]
	add x21, sp, #336
	ld1 { v3.2d, v4.2d, v5.2d, v6.2d }, [x21]
	add x21, sp, #144
	mov v23.d[1], v20.d[0]
	ld1 { v17.2d, v18.2d, v19.2d, v20.2d }, [x21]
	tbl v1.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v0.16b
	mov v21.d[1], v22.d[0]
	tbl v3.16b, { v3.16b, v4.16b, v5.16b, v6.16b }, v0.16b
	tbl v4.16b, { v28.16b, v29.16b, v30.16b, v31.16b }, v0.16b
	stp q8, q23, [x7, #-32]
	tbl v5.16b, { v17.16b, v18.16b, v19.16b, v20.16b }, v0.16b
	mov v16.d[1], v1.d[0]
	mov v2.d[1], v3.d[0]
	mov v4.d[1], v5.d[0]
	stp q21, q16, [x7]
	stp q2, q4, [x7, #32]
	add x7, x7, #128
	b.lo .LBB0_2
	add sp, sp, #400
	ldp x20, x19, [sp, #96]
	ldr x30, [sp, #64]
	ldp x22, x21, [sp, #80]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #112
	mov w4, #16
	b jix_probe::byte_shuffle::decode_impl_generic
.LBB0_4:
	mov x5, xzr
	mov w4, #16
	b jix_probe::byte_shuffle::decode_impl_generic
