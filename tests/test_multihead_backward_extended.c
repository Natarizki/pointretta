// test_multihead_backward_extended.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention_multihead.h"
#include "backward_ops.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define SEQ_LEN 6
#define NUM_HEADS 3
#define HEAD_DIM 4
#define FD_EPS 1e-6
#define CHECK_TOL 1e-4

static double compute_loss(MultiHeadRetentionConfig *cfg, Tensor *Qh, Tensor *Kh, Tensor *Vh,
                            Tensor *G, Tensor *scratch_out) {
    retention_multihead_parallel_forward(cfg, Qh, Kh, Vh, scratch_out);
    double loss = 0.0;
    for (size_t i = 0; i < scratch_out->size; i++) loss += scratch_out->data[i] * G->data[i];
    return loss;
}

static void numerical_gradient(MultiHeadRetentionConfig *cfg, Tensor *target,
                                Tensor *Qh, Tensor *Kh, Tensor *Vh, Tensor *G,
                                Tensor *scratch_out, Tensor *grad_numeric) {
    for (size_t i = 0; i < target->size; i++) {
        double orig = target->data[i];
        target->data[i] = orig + FD_EPS;
        double lp = compute_loss(cfg, Qh, Kh, Vh, G, scratch_out);
        target->data[i] = orig - FD_EPS;
        double lm = compute_loss(cfg, Qh, Kh, Vh, G, scratch_out);
        target->data[i] = orig;
        grad_numeric->data[i] = (lp - lm) / (2.0 * FD_EPS);
    }
}

static double max_abs_diff(const Tensor *a, const Tensor *b) {
    double m = 0.0;
    for (size_t i = 0; i < a->size; i++) { double d = fabs(a->data[i] - b->data[i]); if (d > m) m = d; }
    return m;
}

int main(void) {
    double *decays = make_multiscale_decay(NUM_HEADS, 0.7, 0.97);
    MultiHeadRetentionConfig cfg = { SEQ_LEN, NUM_HEADS, HEAD_DIM, decays };

    int shape[] = {NUM_HEADS, SEQ_LEN, HEAD_DIM};
    Tensor *Qh = tensor_create(shape, 3, 0);
    Tensor *Kh = tensor_create(shape, 3, 0);
    Tensor *Vh = tensor_create(shape, 3, 0);
    Tensor *G = tensor_create(shape, 3, 0);
    Tensor *out = tensor_create(shape, 3, 0);

    tensor_fill_random(Qh, -1, 1, 1);
    tensor_fill_random(Kh, -1, 1, 2);
    tensor_fill_random(Vh, -1, 1, 3);
    tensor_fill_random(G, -1, 1, 4);

    Tensor *dQh_a = tensor_create(shape, 3, 0);
    Tensor *dKh_a = tensor_create(shape, 3, 0);
    Tensor *dVh_a = tensor_create(shape, 3, 0);
    retention_multihead_backward(&cfg, Qh, Kh, Vh, G, dQh_a, dKh_a, dVh_a);

    Tensor *dQh_n = tensor_create(shape, 3, 0);
    Tensor *dKh_n = tensor_create(shape, 3, 0);
    Tensor *dVh_n = tensor_create(shape, 3, 0);

    numerical_gradient(&cfg, Qh, Qh, Kh, Vh, G, out, dQh_n);
    numerical_gradient(&cfg, Kh, Qh, Kh, Vh, G, out, dKh_n);
    numerical_gradient(&cfg, Vh, Qh, Kh, Vh, G, out, dVh_n);

    double diff_q = max_abs_diff(dQh_a, dQh_n);
    double diff_k = max_abs_diff(dKh_a, dKh_n);
    double diff_v = max_abs_diff(dVh_a, dVh_n);

    printf("Max diff dQh: %.10f | %s\n", diff_q, diff_q < CHECK_TOL ? "LOLOS" : "GAGAL");
    printf("Max diff dKh: %.10f | %s\n", diff_k, diff_k < CHECK_TOL ? "LOLOS" : "GAGAL");
    printf("Max diff dVh: %.10f | %s\n", diff_v, diff_v < CHECK_TOL ? "LOLOS" : "GAGAL");

    int pass = (diff_q < CHECK_TOL) && (diff_k < CHECK_TOL) && (diff_v < CHECK_TOL);
    printf("\n=== HASIL AKHIR: %s ===\n", pass ? "MULTI-HEAD BACKWARD (Q,K,V) TERVALIDASI LENGKAP" : "GAGAL");

    free(decays);
    tensor_free(Qh); tensor_free(Kh); tensor_free(Vh); tensor_free(G); tensor_free(out);
    tensor_free(dQh_a); tensor_free(dKh_a); tensor_free(dVh_a);
    tensor_free(dQh_n); tensor_free(dKh_n); tensor_free(dVh_n);
    return pass ? 0 : 1;
}
