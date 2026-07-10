.intel_syntax noprefix
# vecmul_avx2.s
.global vecmul_avx2_core
.text

vecmul_avx2_core:
    xor rax, rax

vecmul_loop:
    cmp rax, rcx
    jge vecmul_done
    vmovupd ymm0, [rdi]
    vmovupd ymm1, [rsi]
    vmulpd ymm2, ymm0, ymm1
    vmovupd [rdx], ymm2
    add rdi, 32
    add rsi, 32
    add rdx, 32
    add rax, 4
    jmp vecmul_loop

vecmul_done:
    ret
