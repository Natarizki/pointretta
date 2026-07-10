// matmul_neon64.s
// Kernel ASM NEON (AArch64) untuk matmul double-precision.
// void matmul_neon64_core(const double *A, const double *B, double *out,
//                          long M, long K, long N_stride, long N_compute)
// Menghitung out[i, 0..N_compute) = A[i,:] @ B[:, 0..N_compute) untuk semua baris i,
// TAPI B dan out punya lebar sebenarnya N_stride (dipakai buat hitung address).
// N_compute HARUS genap. (N_stride bisa ganjil/genap -- itu lebar asli matrix).

.global matmul_neon64_core
.text
.align 4

matmul_neon64_core:
    // x0=A, x1=B, x2=out, x3=M, x4=K, x5=N_stride, x6=N_compute
    mov x7, xzr            // i = 0

i_loop:
    cmp x7, x3
    b.ge done

    mov x8, xzr            // j = 0

j_loop:
    cmp x8, x6             // j < N_compute
    b.ge i_next

    movi v0.2d, #0         // acc = [0,0]
    mov x9, xzr            // k = 0

k_loop:
    cmp x9, x4
    b.ge store_acc

    // address A[i*K + k]
    mul x10, x7, x4
    add x10, x10, x9
    lsl x10, x10, #3
    add x11, x0, x10
    ld1r {v1.2d}, [x11]     // broadcast A[i,k]

    // address B[k*N_stride + j]  (2 double contiguous)
    mul x12, x9, x5
    add x12, x12, x8
    lsl x12, x12, #3
    add x13, x1, x12
    ld1 {v2.2d}, [x13]      // load B[k, j:j+2]

    fmla v0.2d, v1.2d, v2.2d

    add x9, x9, #1
    b k_loop

store_acc:
    // address out[i*N_stride + j]
    mul x14, x7, x5
    add x14, x14, x8
    lsl x14, x14, #3
    add x15, x2, x14
    st1 {v0.2d}, [x15]

    add x8, x8, #2
    b j_loop

i_next:
    add x7, x7, #1
    b i_loop

done:
    ret
