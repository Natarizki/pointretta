// state_update_neon64.s
// Kernel ASM NEON (AArch64) buat update state retention (recurrent form).
// void state_update_neon64_core(double *state, const double *k, const double *v,
//                                double decay, long head_dim, long b_even)
//
// Menghitung: state[a][b] = decay*state[a][b] + k[a]*v[b]
// untuk semua baris a (0..head_dim-1), kolom b divektorisasi 2-per-2 (0..b_even).
// b_even HARUS genap (sisa kolom ganjil ditangani wrapper C).

.global state_update_neon64_core
.text
.align 4

state_update_neon64_core:
    // x0=state, x1=k, x2=v, x3=head_dim, x4=b_even, d0=decay
    dup v0.2d, v0.d[0]       // broadcast decay ke v0.2d
    mov x5, xzr              // a = 0

row_loop:
    cmp x5, x3
    b.ge row_done

    // broadcast k[a]
    lsl x9, x5, #3
    add x9, x1, x9
    ld1r {v1.2d}, [x9]

    // alamat awal state[a, 0]
    mul x10, x5, x3
    lsl x10, x10, #3
    add x11, x0, x10

    mov x6, xzr              // b = 0
    mov x12, x2              // pointer v, reset tiap baris
    mov x13, x11             // pointer state baris ini

col_loop:
    cmp x6, x4
    b.ge row_next

    ld1 {v3.2d}, [x13]        // load state[a, b:b+2]
    ld1 {v4.2d}, [x12], #16   // load v[b:b+2], pointer v += 16
    fmul v5.2d, v3.2d, v0.2d  // decay * state
    fmla v5.2d, v1.2d, v4.2d  // += k[a] * v_chunk
    st1 {v5.2d}, [x13], #16   // simpan balik, pointer state += 16

    add x6, x6, #2
    b col_loop

row_next:
    add x5, x5, #1
    b row_loop

row_done:
    ret
