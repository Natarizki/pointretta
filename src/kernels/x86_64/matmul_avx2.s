.intel_syntax noprefix
# matmul_avx2.s
# void matmul_avx2_core(A, B, out, M, K, N_stride, N_compute)
# Args: rdi=A, rsi=B, rdx=out, rcx=M, r8=K, r9=N_stride, [rsp+8]=N_compute
# CATATAN: r12, r13 callee-saved -- WAJIB push/pop kalau dipakai.

.global matmul_avx2_core
.text

matmul_avx2_core:
    mov r10, [rsp+8]
    push r12
    push r13
    xor r11, r11

i_loop:
    cmp r11, rcx
    jge done

    xor r12, r12

j_loop:
    cmp r12, r10
    jge i_next

    vxorpd ymm0, ymm0, ymm0
    xor r13, r13

k_loop:
    cmp r13, r8
    jge store_acc

    mov rax, r11
    imul rax, r8
    add rax, r13
    lea rax, [rdi + rax*8]
    vbroadcastsd ymm1, [rax]

    mov rax, r13
    imul rax, r9
    add rax, r12
    lea rax, [rsi + rax*8]
    vmovupd ymm2, [rax]

    vmulpd ymm3, ymm1, ymm2
    vaddpd ymm0, ymm0, ymm3

    inc r13
    jmp k_loop

store_acc:
    mov rax, r11
    imul rax, r9
    add rax, r12
    lea rax, [rdx + rax*8]
    vmovupd [rax], ymm0

    add r12, 4
    jmp j_loop

i_next:
    inc r11
    jmp i_loop

done:
    pop r13
    pop r12
    ret
