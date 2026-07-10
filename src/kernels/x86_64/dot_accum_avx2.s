.intel_syntax noprefix
# dot_accum_avx2.s
.global dot_avx2_core
.global accum_scale_avx2_core
.text

dot_avx2_core:
    vxorpd ymm0, ymm0, ymm0
    xor rax, rax

dot_loop:
    cmp rax, rdx
    jge dot_finish
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rsi]
    vmulpd ymm3, ymm1, ymm2
    vaddpd ymm0, ymm0, ymm3
    add rdi, 32
    add rsi, 32
    add rax, 4
    jmp dot_loop

dot_finish:
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0
    ret

accum_scale_avx2_core:
    vbroadcastsd ymm3, xmm0
    xor rax, rax

accum_loop:
    cmp rax, rdx
    jge accum_done
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rsi]
    vmulpd ymm4, ymm3, ymm2
    vaddpd ymm1, ymm1, ymm4
    vmovupd [rdi], ymm1
    add rdi, 32
    add rsi, 32
    add rax, 4
    jmp accum_loop

accum_done:
    ret
