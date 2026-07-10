// test_dispatch.c
#include "tensor.h"
#include "tensor_ops.h"
#include "norm.h"
#include "retention.h"
#include "dispatch.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("Backend aktif: %s\n\n", dispatch_active_backend());

    int all_pass = 1;

    printf("=== Test 1: tensor_matmul_fast vs tensor_matmul ===\n");
    Tensor *A = tensor_create((int[]){5, 7}, 2, 0);
    Tensor *B = tensor_create((int[]){7, 6}, 2, 0);
    Tensor *out_c = tensor_create((int[]){5, 6}, 2, 0);
    Tensor *out_f = tensor_create((int[]){5, 6}, 2, 0);
    tensor_fill_random(A, -1, 1, 1);
    tensor_fill_random(B, -1, 1, 2);
    tensor_matmul(out_c, A, B);
    tensor_matmul_fast(out_f, A, B);
    double d1 = 0;
    for (size_t i = 0; i < out_c->size; i++) { double d = fabs(out_c->data[i] - out_f->data[i]); if (d > d1) d1 = d; }
    printf("Max diff: %.12f | %s\n\n", d1, d1 < 1e-6 ? "LOLOS" : "GAGAL");
    all_pass &= (d1 < 1e-6);

    printf("=== Test 2: rmsnorm_forward_fast vs rmsnorm_forward ===\n");
    Tensor *x = tensor_create((int[]){4, 13}, 2, 0);
    Tensor *gain = tensor_create((int[]){13}, 1, 0);
    Tensor *rout_c = tensor_create((int[]){4, 13}, 2, 0);
    Tensor *rout_f = tensor_create((int[]){4, 13}, 2, 0);
    tensor_fill_random(x, -1, 1, 3);
    tensor_fill_random(gain, 0.5, 1.5, 4);
    rmsnorm_forward(x, gain, rout_c, 1e-6);
    rmsnorm_forward_fast(x, gain, rout_f, 1e-6);
    double d2 = 0;
    for (size_t i = 0; i < rout_c->size; i++) { double d = fabs(rout_c->data[i] - rout_f->data[i]); if (d > d2) d2 = d; }
    printf("Max diff: %.12f | %s\n\n", d2, d2 < 1e-6 ? "LOLOS" : "GAGAL");
    all_pass &= (d2 < 1e-6);

    printf("=== Test 3: retention_parallel_forward_fast vs generic ===\n");
    RetentionConfig cfg = { .seq_len = 15, .dim = 9, .decay = 0.93 };
    Tensor *Q = tensor_create((int[]){15, 9}, 2, 0);
    Tensor *K = tensor_create((int[]){15, 9}, 2, 0);
    Tensor *V = tensor_create((int[]){15, 9}, 2, 0);
    Tensor *pout_c = tensor_create((int[]){15, 9}, 2, 0);
    Tensor *pout_f = tensor_create((int[]){15, 9}, 2, 0);
    tensor_fill_random(Q, -1, 1, 5);
    tensor_fill_random(K, -1, 1, 6);
    tensor_fill_random(V, -1, 1, 7);
    retention_parallel_forward(&cfg, Q, K, V, pout_c);
    retention_parallel_forward_fast(&cfg, Q, K, V, pout_f);
    double d3 = 0;
    for (size_t i = 0; i < pout_c->size; i++) { double d = fabs(pout_c->data[i] - pout_f->data[i]); if (d > d3) d3 = d; }
    printf("Max diff: %.12f | %s\n\n", d3, d3 < 1e-5 ? "LOLOS" : "GAGAL");
    all_pass &= (d3 < 1e-5);

    printf("=== Test 4: retention_recurrent_step_fast vs generic (10 langkah) ===\n");
    Tensor *state_c = retention_state_create(9);
    Tensor *state_f = retention_state_create(9);
    Tensor *q = tensor_create((int[]){9}, 1, 0);
    Tensor *k = tensor_create((int[]){9}, 1, 0);
    Tensor *v = tensor_create((int[]){9}, 1, 0);
    Tensor *ot_c = tensor_create((int[]){9}, 1, 0);
    Tensor *ot_f = tensor_create((int[]){9}, 1, 0);
    double d4 = 0;
    for (int t = 0; t < 10; t++) {
        tensor_fill_random(q, -1, 1, 100 + t);
        tensor_fill_random(k, -1, 1, 200 + t);
        tensor_fill_random(v, -1, 1, 300 + t);
        retention_recurrent_step(&cfg, state_c, q, k, v, ot_c);
        retention_recurrent_step_fast(&cfg, state_f, q, k, v, ot_f);
        for (int d = 0; d < 9; d++) { double diff = fabs(ot_c->data[d] - ot_f->data[d]); if (diff > d4) d4 = diff; }
    }
    printf("Max diff: %.12f | %s\n\n", d4, d4 < 1e-6 ? "LOLOS" : "GAGAL");
    all_pass &= (d4 < 1e-6);

    printf("=== HASIL AKHIR: %s (backend: %s) ===\n",
           all_pass ? "SEMUA TEST LOLOS" : "ADA TEST GAGAL", dispatch_active_backend());

    return all_pass ? 0 : 1;
}
