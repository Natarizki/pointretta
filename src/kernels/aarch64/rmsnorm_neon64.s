// rmsnorm_neon64.s
// Dua kernel ASM NEON (AArch64) buat RMSNorm:
//
// 1. double sumsq_neon64_core(const double *x, long n)
//    Hitung sum(x[i]^2) untuk i=0..n-1 (n HARUS genap).
//
// 2. void scale_gain_neon64_core(const double *x, const double *gain,
//                                 double *out, long n, double inv_rms)
//    Hitung out[i] = x[i] * inv_rms * gain[i] untuk i=0..n-1 (n HARUS genap).

.global sumsq_neon64_core
.global scale_gain_neon64_core
.text
.align 4

// ---- sumsq_neon64_core ----
// x0 = x, x1 = n
sumsq_neon64_core:
    movi v0.2d, #0          // accumulator = [0,0]
    mov x2, xzr             // i = 0

sumsq_loop:
    cmp x2, x1
    b.ge sumsq_finish

    ld1 {v1.2d}, [x0], #16  // load x[i:i+2], pointer += 16 byte
    fmla v0.2d, v1.2d, v1.2d
    add x2, x2, #2
    b sumsq_loop

sumsq_finish:
    faddp d0, v0.2d         // d0 = v0[0] + v0[1] (hasil dikembalikan via d0)
    ret

// ---- scale_gain_neon64_core ----
// x0 = x, x1 = gain, x2 = out, x3 = n, d0 = inv_rms
scale_gain_neon64_core:
    dup v3.2d, v0.d[0]      // broadcast inv_rms ke v3.2d
    mov x4, xzr             // i = 0

scale_loop:
    cmp x4, x3
    b.ge scale_done

    ld1 {v1.2d}, [x0], #16  // load x[i:i+2]
    ld1 {v2.2d}, [x1], #16  // load gain[i:i+2]
    fmul v4.2d, v1.2d, v3.2d  // x * inv_rms
    fmul v5.2d, v4.2d, v2.2d  // * gain
    st1 {v5.2d}, [x2], #16    // simpan ke out

    add x4, x4, #2
    b scale_loop

scale_done:
    ret
