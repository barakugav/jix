jix_probe::byte_shuffle::decode_impl::<2, 64>:
	lsr x9, x1, #1
	mov x5, xzr
	ands x8, x9, #0x3fffffffffffffc0
	b.eq .LBB1_3
	add x9, x0, x9
	add x10, x2, #64
.LBB1_2:
	add x11, x0, x5
	add x12, x9, x5
	add x5, x5, #64
	ldp d0, d1, [x11]
	cmp x5, x8
	ldp d2, d3, [x12]
	ldp d4, d5, [x11, #16]
	ldp d6, d7, [x12, #16]
	zip1 v0.16b, v0.16b, v2.16b
	zip1 v1.16b, v1.16b, v3.16b
	ldp d3, d16, [x11, #32]
	zip1 v4.16b, v4.16b, v6.16b
	ldp d2, d6, [x12, #32]
	zip1 v5.16b, v5.16b, v7.16b
	stp q0, q1, [x10, #-64]
	ldp d7, d1, [x12, #48]
	zip1 v2.16b, v3.16b, v2.16b
	ldp d3, d0, [x11, #48]
	zip1 v6.16b, v16.16b, v6.16b
	stp q4, q5, [x10, #-32]
	zip1 v3.16b, v3.16b, v7.16b
	zip1 v0.16b, v0.16b, v1.16b
	stp q2, q6, [x10]
	stp q3, q0, [x10, #32]
	add x10, x10, #128
	b.lo .LBB1_2
.LBB1_3:
	mov w4, #2
	b jix_probe::byte_shuffle::decode_impl_generic
