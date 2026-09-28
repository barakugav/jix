jix_probe::byte_shuffle::encode_impl::<16, 8>:
	push {r4, r5, r6, r7, r8, r9, r10, r11, lr}
	sub sp, sp, #156
	str r0, [sp, #20]
	movw r0, #65528
	movt r0, #2047
	str r2, [sp, #12]
	ands r0, r0, r1, lsr #4
	str r0, [sp, #16]
	str r1, [sp, #8]
	beq .LBB4_5
	add r8, sp, #24
	lsr r11, r1, #4
	mov r9, #0
	mov r10, r2
.LBB4_2:
	ldr r0, [sp, #20]
	mov r2, #128
	add r1, r0, r9, lsl #4
	mov r0, r8
	bl memcpy
	mov r0, r10
	mov r1, #0
.LBB4_3:
	add r2, r8, r1
	ldrb r12, [r8, r1]
	add r1, r1, #1
	ldrb r3, [r2, #80]
	cmp r1, #16
	ldrb r4, [r2, #64]
	ldrb r6, [r2, #16]
	orr r3, r4, r3, lsl #8
	ldrb r5, [r2, #32]
	ldrb r7, [r2, #48]
	ldrb r4, [r2, #112]
	ldrb r2, [r2, #96]
	orr r2, r3, r2, lsl #16
	orr r2, r2, r4, lsl #24
	str r2, [r0, #4]
	orr r2, r12, r6, lsl #8
	orr r2, r2, r5, lsl #16
	orr r2, r2, r7, lsl #24
	str r2, [r0]
	add r0, r0, r11
	bne .LBB4_3
	add r10, r10, #8
	add r9, r9, #8
	ldr r0, [sp, #16]
	cmp r9, r0
	blo .LBB4_2
	b .LBB4_6
.LBB4_5:
	mov r9, #0
.LBB4_6:
	mov r0, #16
	ldr r1, [sp, #8]
	stm sp, {r0, r9}
	ldr r0, [sp, #20]
	ldr r2, [sp, #12]
	bl jix_probe::byte_shuffle::encode_impl_generic
	add sp, sp, #156
	pop {r4, r5, r6, r7, r8, r9, r10, r11, pc}
