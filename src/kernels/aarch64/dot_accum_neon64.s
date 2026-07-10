// dot_accum_neon64.s
.global dot_neon64_core
.global accum_scale_neon64_core
.text
.align 4

dot_neon64_core:
    movi v0.2d, #0
    mov x3, xzr

dot_loop:
    cmp x3, x2
    b.ge dot_finish
    ld1 {v1.2d}, [x0], #16
    ld1 {v2.2d}, [x1], #16
    fmla v0.2d, v1.2d, v2.2d
    add x3, x3, #2
    b dot_loop

dot_finish:
    faddp d0, v0.2d
    ret

accum_scale_neon64_core:
    dup v3.2d, v0.d[0]
    mov x4, xzr
    mov x5, x0

accum_loop:
    cmp x4, x2
    b.ge accum_done
    ld1 {v1.2d}, [x5]
    ld1 {v2.2d}, [x1], #16
    fmla v1.2d, v3.2d, v2.2d
    st1 {v1.2d}, [x5], #16
    add x4, x4, #2
    b accum_loop

accum_done:
    ret
