.intel_syntax noprefix
# rmsnorm_avx2.s
.global sumsq_avx2_core
.global scale_gain_avx2_core
.text

sumsq_avx2_core:
    vxorpd ymm0, ymm0, ymm0
    xor rax, rax

sumsq_loop:
    cmp rax, rsi
    jge sumsq_finish
    vmovupd ymm1, [rdi]
    vmulpd ymm2, ymm1, ymm1
    vaddpd ymm0, ymm0, ymm2
    add rdi, 32
    add rax, 4
    jmp sumsq_loop

sumsq_finish:
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0
    ret

scale_gain_avx2_core:
    vbroadcastsd ymm3, xmm0
    xor rax, rax

scale_loop:
    cmp rax, rcx
    jge scale_done
    vmovupd ymm1, [rdi]
    vmovupd ymm2, [rsi]
    vmulpd ymm4, ymm1, ymm3
    vmulpd ymm5, ymm4, ymm2
    vmovupd [rdx], ymm5
    add rdi, 32
    add rsi, 32
    add rdx, 32
    add rax, 4
    jmp scale_loop

scale_done:
    ret
