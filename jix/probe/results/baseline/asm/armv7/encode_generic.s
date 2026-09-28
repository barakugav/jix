jix_probe::byte_shuffle::encode_impl_generic:
	push {r4, r5, r6, r7, r8, r9, r11, lr}
	ldr r5, [sp, #32]
	cmp r5, #0
	beq .LBB9_7
	mov r6, r0
	mov r0, r1
	mov r1, r5
	mov r9, r2
	ldr r8, [sp, #36]
	bl __aeabi_uidiv
	cmp r8, r0
	bhs .LBB9_6
	mla r6, r8, r5, r6
	mov r2, #0
.LBB9_3:
	mla r3, r2, r0, r9
	mov r1, r6
	mov r7, r8
.LBB9_4:
	ldrb r4, [r1], r5
	strb r4, [r3, r7]
	add r7, r7, #1
	cmp r7, r0
	blo .LBB9_4
	add r2, r2, #1
	add r6, r6, #1
	cmp r2, r5
	bne .LBB9_3
.LBB9_6:
	pop {r4, r5, r6, r7, r8, r9, r11, pc}
.LBB9_7:
	ldr r0, .LCPI9_0
.LPC9_0:
	add r0, pc, r0
	bl core::panicking::panic_const::panic_const_div_by_zero
.LCPI9_0:
