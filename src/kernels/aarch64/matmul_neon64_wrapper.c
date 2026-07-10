// matmul_neon64_wrapper.c
#include "tensor.h"
#include <stdio.h>

extern void matmul_neon64_core(const double *A, const double *B, double *out,
                                long M, long K, long N_stride, long N_compute);

int tensor_matmul_neon(Tensor *out, const Tensor *A, const Tensor *B) {
    if (!out || !A || !B) return -1;
    if (A->ndim != 2 || B->ndim != 2 || out->ndim != 2) return -1;

    long M = A->shape[0];
    long K = A->shape[1];
    long K2 = B->shape[0];
    long N = B->shape[1];

    if (K != K2) return -1;
    if (out->shape[0] != M || out->shape[1] != N) return -1;

    long N_even = (N % 2 == 0) ? N : N - 1;

    if (N_even > 0) {
        matmul_neon64_core(A->data, B->data, out->data, M, K, N, N_even);
    }
    if (N_even < N) {
        long j = N - 1;
        for (long i = 0; i < M; i++) {
            double sum = 0.0;
            for (long k = 0; k < K; k++) {
                sum += A->data[i * K + k] * B->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }

    return 0;
}
