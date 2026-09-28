jix_probe::byte_shuffle::decode_impl_generic:
	push {r4, r5, r6, r7, r8, r9, r11, lr}
	ldr r9, [sp, #32]
	cmp r9, #0
	beq .LBB8_7
	mov r8, r0
	mov r0, r1
	mov r1, r9
	mov r5, r2
	ldr r7, [sp, #36]
	bl __aeabi_uidiv
	cmp r7, r0
	bhs .LBB8_6
	mla r4, r7, r9, r5
.LBB8_3:
	add r2, r8, r7
	mov r3, #0
	mov r5, r9
	mov r1, r4
.LBB8_4:
	mul r6, r3, r0
	add r3, r3, #1
	subs r5, r5, #1
	ldrb r6, [r2, r6]
	strb r6, [r1], #1
	bne .LBB8_4
	add r7, r7, #1
	add r4, r4, r9
	cmp r7, r0
	blo .LBB8_3
.LBB8_6:
	pop {r4, r5, r6, r7, r8, r9, r11, pc}
.LBB8_7:
	ldr r0, .LCPI8_0
.LPC8_0:
	add r0, pc, r0
	bl core::panicking::panic_const::panic_const_div_by_zero
.LCPI8_0:
