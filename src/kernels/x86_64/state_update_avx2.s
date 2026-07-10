.intel_syntax noprefix
# state_update_avx2.s
# Args: rdi=state, rsi=k, rdx=v, rcx=head_dim, r8=b_vec, xmm0=decay
# r12,r13,r14 dipakai -- WAJIB push/pop.

.global state_update_avx2_core
.text

state_update_avx2_core:
    push r12
    push r13
    push r14
    vbroadcastsd ymm0, xmm0
    xor r9, r9

row_loop:
    cmp r9, rcx
    jge row_done

    mov rax, r9
    lea rax, [rsi + rax*8]
    vbroadcastsd ymm1, [rax]

    mov r12, r9
    imul r12, rcx
    lea r12, [rdi + r12*8]

    mov r13, rdx
    xor r14, r14

col_loop:
    cmp r14, r8
    jge row_next
    vmovupd ymm3, [r12]
    vmovupd ymm4, [r13]
    vmulpd ymm5, ymm3, ymm0
    vmulpd ymm6, ymm1, ymm4
    vaddpd ymm5, ymm5, ymm6
    vmovupd [r12], ymm5
    add r12, 32
    add r13, 32
    add r14, 4
    jmp col_loop

row_next:
    inc r9
    jmp row_loop

row_done:
    pop r14
    pop r13
    pop r12
    ret
