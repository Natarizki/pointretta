// softmax_neon64.s
.global max_neon64_core
.global sum_neon64_core
.global vecscale_neon64_core
.text
.align 4

max_neon64_core:
    ld1 {v0.2d}, [x0], #16
    mov x2, #2

max_loop:
    cmp x2, x1
    b.ge max_finish
    ld1 {v1.2d}, [x0], #16
    fmax v0.2d, v0.2d, v1.2d
    add x2, x2, #2
    b max_loop

max_finish:
    fmaxp d0, v0.2d
    ret

sum_neon64_core:
    movi v0.2d, #0
    mov x2, xzr

sum_loop:
    cmp x2, x1
    b.ge sum_finish
    ld1 {v1.2d}, [x0], #16
    fadd v0.2d, v0.2d, v1.2d
    add x2, x2, #2
    b sum_loop

sum_finish:
    faddp d0, v0.2d
    ret

vecscale_neon64_core:
    dup v1.2d, v0.d[0]
    mov x3, xzr

vecscale_loop:
    cmp x3, x2
    b.ge vecscale_done
    ld1 {v2.2d}, [x0], #16
    fmul v3.2d, v2.2d, v1.2d
    st1 {v3.2d}, [x1], #16
    add x3, x3, #2
    b vecscale_loop

vecscale_done:
    ret
