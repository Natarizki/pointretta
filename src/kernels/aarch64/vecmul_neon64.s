// vecmul_neon64.s
.global vecmul_neon64_core
.text
.align 4

vecmul_neon64_core:
    mov x4, xzr

vecmul_loop:
    cmp x4, x3
    b.ge vecmul_done
    ld1 {v0.2d}, [x0], #16
    ld1 {v1.2d}, [x1], #16
    fmul v2.2d, v0.2d, v1.2d
    st1 {v2.2d}, [x2], #16
    add x4, x4, #2
    b vecmul_loop

vecmul_done:
    ret
