jix_probe::byte_shuffle::decode_impl::<16, 8>:
	push {r4, r5, r6, r7, r8, r9, r10, r11, lr}
	sub sp, sp, #148
	mov r11, r2
	movw r2, #65528
	movt r2, #2047
	ands r8, r2, r1, lsr #4
	str r1, [sp, #12]
	str r0, [sp, #8]
	beq .LBB0_5
	add r7, sp, #16
	lsr r5, r1, #4
	mov r9, #0
	mov r10, r0
.LBB0_2:
	mov r0, r10
	mov r1, #0
.LBB0_3:
	ldr r3, [r0, #4]
	add r6, r7, r1
	ldr r2, [r0]
	add r0, r0, r5
	strb r3, [r6, #64]
	lsr r4, r3, #24
	strb r2, [r7, r1]
	add r1, r1, #1
	strb r4, [r6, #112]
	lsr r4, r3, #16
	lsr r3, r3, #8
	cmp r1, #16
	strb r3, [r6, #80]
	lsr r3, r2, #24
	strb r3, [r6, #48]
	lsr r3, r2, #16
	lsr r2, r2, #8
	strb r4, [r6, #96]
	strb r3, [r6, #32]
	strb r2, [r6, #16]
	bne .LBB0_3
	add r0, r11, r9, lsl #4
	mov r1, r7
	mov r2, #128
	bl memcpy
	add r9, r9, #8
	add r10, r10, #8
	cmp r9, r8
	blo .LBB0_2
	b .LBB0_6
.LBB0_5:
	mov r9, #0
.LBB0_6:
	mov r0, #16
	ldr r1, [sp, #12]
	stm sp, {r0, r9}
	mov r2, r11
	ldr r0, [sp, #8]
	bl jix_probe::byte_shuffle::decode_impl_generic
	add sp, sp, #148
	pop {r4, r5, r6, r7, r8, r9, r10, r11, pc}
