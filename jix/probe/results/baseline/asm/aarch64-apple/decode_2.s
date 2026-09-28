jix_probe::byte_shuffle::decode_impl::<2, 64>:
Lfunc_begin1:
	lsr x9, x1, #1
	mov x5, #0
	ands x8, x9, #0x3fffffffffffffc0
	b.eq LBB1_3
	add x9, x0, x9
	add x10, x2, #64
LBB1_2:
	add x11, x0, x5
	add x12, x9, x5
	ldp d0, d1, [x11]
	ldp d2, d3, [x12]
	zip1.16b v0, v0, v2
	zip1.16b v1, v1, v3
	stp q0, q1, [x10, #-64]
	ldp d0, d1, [x11, #16]
	ldp d2, d3, [x12, #16]
	zip1.16b v0, v0, v2
	zip1.16b v1, v1, v3
	stp q0, q1, [x10, #-32]
	ldp d0, d1, [x11, #32]
	ldp d2, d3, [x12, #32]
	zip1.16b v0, v0, v2
	zip1.16b v1, v1, v3
	stp q0, q1, [x10]
	ldp d0, d1, [x11, #48]
	ldp d2, d3, [x12, #48]
	zip1.16b v0, v0, v2
	zip1.16b v1, v1, v3
	add x5, x5, #64
	stp q0, q1, [x10, #32]
	add x10, x10, #128
	cmp x5, x8
	b.lo LBB1_2
LBB1_3:
	mov w4, #2
	b jix_probe::byte_shuffle::decode_impl_generic
