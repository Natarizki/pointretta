.intel_syntax noprefix
# softmax_avx2.s
.global max_avx2_core
.global sum_avx2_core
.global vecscale_avx2_core
.text

max_avx2_core:
    vmovupd ymm0, [rdi]
    add rdi, 32
    mov rax, 4

max_loop:
    cmp rax, rsi
    jge max_finish
    vmovupd ymm1, [rdi]
    vmaxpd ymm0, ymm0, ymm1
    add rdi, 32
    add rax, 4
    jmp max_loop

max_finish:
    vextractf128 xmm1, ymm0, 1
    vmaxpd xmm0, xmm0, xmm1
    vshufpd xmm1, xmm0, xmm0, 1
    vmaxpd xmm0, xmm0, xmm1
    ret

sum_avx2_core:
    vxorpd ymm0, ymm0, ymm0
    xor rax, rax

sum_loop:
    cmp rax, rsi
    jge sum_finish
    vmovupd ymm1, [rdi]
    vaddpd ymm0, ymm0, ymm1
    add rdi, 32
    add rax, 4
    jmp sum_loop

sum_finish:
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0
    ret

vecscale_avx2_core:
    vbroadcastsd ymm1, xmm0
    xor rax, rax

vecscale_loop:
    cmp rax, rdx
    jge vecscale_done
    vmovupd ymm2, [rdi]
    vmulpd ymm3, ymm2, ymm1
    vmovupd [rsi], ymm3
    add rdi, 32
    add rsi, 32
    add rax, 4
    jmp vecscale_loop

vecscale_done:
    ret
