// test_backward_ops.c
#include "tensor.h"
#include "tensor_ops.h"
#include "norm.h"
#include "ffn.h"
#include "retention_multihead.h"
#include "backward_ops.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define FD_EPS 1e-6
#define CHECK_TOL 1e-4

static double max_abs_diff(const Tensor *a, const Tensor *b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a->size; i++) {
        double diff = fabs(a->data[i] - b->data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

static int test_matmul_backward(void) {
    printf("=== Test 1: Matmul backward ===\n");
    int M = 3, K = 4, N = 2;
    Tensor *A = tensor_create((int[]){M, K}, 2, 0);
    Tensor *B = tensor_create((int[]){K, N}, 2, 0);
    Tensor *G = tensor_create((int[]){M, N}, 2, 0);
    Tensor *out = tensor_create((int[]){M, N}, 2, 0);

    tensor_fill_random(A, -1.0, 1.0, 1);
    tensor_fill_random(B, -1.0, 1.0, 2);
    tensor_fill_random(G, -1.0, 1.0, 3);

    Tensor *dA_analytic = tensor_create((int[]){M, K}, 2, 0);
    Tensor *dB_analytic = tensor_create((int[]){K, N}, 2, 0);
    matmul_backward(G, A, B, dA_analytic, dB_analytic);

    Tensor *dA_numeric = tensor_create((int[]){M, K}, 2, 0);
    Tensor *dB_numeric = tensor_create((int[]){K, N}, 2, 0);

    for (size_t i = 0; i < A->size; i++) {
        double orig = A->data[i];
        A->data[i] = orig + FD_EPS;
        tensor_matmul(out, A, B);
        double loss_p = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_p += out->data[k] * G->data[k];

        A->data[i] = orig - FD_EPS;
        tensor_matmul(out, A, B);
        double loss_m = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_m += out->data[k] * G->data[k];

        A->data[i] = orig;
        dA_numeric->data[i] = (loss_p - loss_m) / (2.0 * FD_EPS);
    }

    for (size_t i = 0; i < B->size; i++) {
        double orig = B->data[i];
        B->data[i] = orig + FD_EPS;
        tensor_matmul(out, A, B);
        double loss_p = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_p += out->data[k] * G->data[k];

        B->data[i] = orig - FD_EPS;
        tensor_matmul(out, A, B);
        double loss_m = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_m += out->data[k] * G->data[k];

        B->data[i] = orig;
        dB_numeric->data[i] = (loss_p - loss_m) / (2.0 * FD_EPS);
    }

    double diff_a = max_abs_diff(dA_analytic, dA_numeric);
    double diff_b = max_abs_diff(dB_analytic, dB_numeric);
    printf("Max diff dA: %.8f, dB: %.8f\n", diff_a, diff_b);
    int pass = (diff_a < CHECK_TOL) && (diff_b < CHECK_TOL);
    printf("Matmul backward: %s\n\n", pass ? "LOLOS" : "GAGAL");

    tensor_free(A); tensor_free(B); tensor_free(G); tensor_free(out);
    tensor_free(dA_analytic); tensor_free(dB_analytic);
    tensor_free(dA_numeric); tensor_free(dB_numeric);
    return pass;
}

static int test_rmsnorm_backward(void) {
    printf("=== Test 2: RMSNorm backward ===\n");
    int seq_len = 3, dim = 4;
    double eps = 1e-6;

    Tensor *x = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *gain = tensor_create((int[]){dim}, 1, 0);
    Tensor *G = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *out = tensor_create((int[]){seq_len, dim}, 2, 0);

    tensor_fill_random(x, -1.0, 1.0, 1);
    tensor_fill_random(gain, 0.5, 1.5, 2);
    tensor_fill_random(G, -1.0, 1.0, 3);

    Tensor *dX_analytic = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *dGain_analytic = tensor_create((int[]){dim}, 1, 0);
    rmsnorm_backward(x, gain, G, eps, dX_analytic, dGain_analytic);

    Tensor *dX_numeric = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *dGain_numeric = tensor_create((int[]){dim}, 1, 0);

    for (size_t i = 0; i < x->size; i++) {
        double orig = x->data[i];
        x->data[i] = orig + FD_EPS;
        rmsnorm_forward(x, gain, out, eps);
        double loss_p = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_p += out->data[k] * G->data[k];

        x->data[i] = orig - FD_EPS;
        rmsnorm_forward(x, gain, out, eps);
        double loss_m = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_m += out->data[k] * G->data[k];

        x->data[i] = orig;
        dX_numeric->data[i] = (loss_p - loss_m) / (2.0 * FD_EPS);
    }

    for (size_t i = 0; i < gain->size; i++) {
        double orig = gain->data[i];
        gain->data[i] = orig + FD_EPS;
        rmsnorm_forward(x, gain, out, eps);
        double loss_p = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_p += out->data[k] * G->data[k];

        gain->data[i] = orig - FD_EPS;
        rmsnorm_forward(x, gain, out, eps);
        double loss_m = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_m += out->data[k] * G->data[k];

        gain->data[i] = orig;
        dGain_numeric->data[i] = (loss_p - loss_m) / (2.0 * FD_EPS);
    }

    double diff_x = max_abs_diff(dX_analytic, dX_numeric);
    double diff_g = max_abs_diff(dGain_analytic, dGain_numeric);
    printf("Max diff dX: %.8f, dGain: %.8f\n", diff_x, diff_g);
    int pass = (diff_x < CHECK_TOL) && (diff_g < CHECK_TOL);
    printf("RMSNorm backward: %s\n\n", pass ? "LOLOS" : "GAGAL");

    tensor_free(x); tensor_free(gain); tensor_free(G); tensor_free(out);
    tensor_free(dX_analytic); tensor_free(dGain_analytic);
    tensor_free(dX_numeric); tensor_free(dGain_numeric);
    return pass;
}

static int test_swiglu_backward(void) {
    printf("=== Test 3: SwiGLU backward ===\n");
    int seq_len = 3, dim = 3, hidden = 4;

    Tensor *x = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *W1 = tensor_create((int[]){dim, hidden}, 2, 0);
    Tensor *W3 = tensor_create((int[]){dim, hidden}, 2, 0);
    Tensor *W2 = tensor_create((int[]){hidden, dim}, 2, 0);
    Tensor *G = tensor_create((int[]){seq_len, dim}, 2, 0);

    tensor_fill_random(x, -1.0, 1.0, 1);
    tensor_fill_random(W1, -0.5, 0.5, 2);
    tensor_fill_random(W3, -0.5, 0.5, 3);
    tensor_fill_random(W2, -0.5, 0.5, 4);
    tensor_fill_random(G, -1.0, 1.0, 5);

    Tensor *h1 = tensor_create((int[]){seq_len, hidden}, 2, 0);
    Tensor *h2 = tensor_create((int[]){seq_len, hidden}, 2, 0);
    Tensor *a = tensor_create((int[]){seq_len, hidden}, 2, 0);
    Tensor *hidden_t = tensor_create((int[]){seq_len, hidden}, 2, 0);
    Tensor *out = tensor_create((int[]){seq_len, dim}, 2, 0);

    swiglu_forward_for_backward(x, W1, W3, W2, h1, h2, a, hidden_t, out);

    Tensor *dX_a = tensor_create((int[]){seq_len, dim}, 2, 0);
    Tensor *dW1_a = tensor_create((int[]){dim, hidden}, 2, 0);
    Tensor *dW3_a = tensor_create((int[]){dim, hidden}, 2, 0);
    Tensor *dW2_a = tensor_create((int[]){hidden, dim}, 2, 0);
    swiglu_backward(x, W1, W3, W2, h1, h2, a, hidden_t, G, dX_a, dW1_a, dW3_a, dW2_a);

    Tensor *dX_n = tensor_create((int[]){seq_len, dim}, 2, 0);
    for (size_t i = 0; i < x->size; i++) {
        double orig = x->data[i];
        x->data[i] = orig + FD_EPS;
        swiglu_forward_for_backward(x, W1, W3, W2, h1, h2, a, hidden_t, out);
        double loss_p = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_p += out->data[k] * G->data[k];

        x->data[i] = orig - FD_EPS;
        swiglu_forward_for_backward(x, W1, W3, W2, h1, h2, a, hidden_t, out);
        double loss_m = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_m += out->data[k] * G->data[k];

        x->data[i] = orig;
        dX_n->data[i] = (loss_p - loss_m) / (2.0 * FD_EPS);
    }

    Tensor *dW1_n = tensor_create((int[]){dim, hidden}, 2, 0);
    for (size_t i = 0; i < W1->size; i++) {
        double orig = W1->data[i];
        W1->data[i] = orig + FD_EPS;
        swiglu_forward_for_backward(x, W1, W3, W2, h1, h2, a, hidden_t, out);
        double loss_p = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_p += out->data[k] * G->data[k];

        W1->data[i] = orig - FD_EPS;
        swiglu_forward_for_backward(x, W1, W3, W2, h1, h2, a, hidden_t, out);
        double loss_m = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_m += out->data[k] * G->data[k];

        W1->data[i] = orig;
        dW1_n->data[i] = (loss_p - loss_m) / (2.0 * FD_EPS);
    }

    double diff_x = max_abs_diff(dX_a, dX_n);
    double diff_w1 = max_abs_diff(dW1_a, dW1_n);
    printf("Max diff dX: %.8f, dW1: %.8f\n", diff_x, diff_w1);
    int pass = (diff_x < CHECK_TOL) && (diff_w1 < CHECK_TOL);
    printf("SwiGLU backward: %s\n\n", pass ? "LOLOS" : "GAGAL");

    tensor_free(x); tensor_free(W1); tensor_free(W3); tensor_free(W2); tensor_free(G);
    tensor_free(h1); tensor_free(h2); tensor_free(a); tensor_free(hidden_t); tensor_free(out);
    tensor_free(dX_a); tensor_free(dW1_a); tensor_free(dW3_a); tensor_free(dW2_a);
    tensor_free(dX_n); tensor_free(dW1_n);
    return pass;
}

static int test_retention_multihead_backward(void) {
    printf("=== Test 4: Retention multi-head backward ===\n");
    int seq_len = 4, num_heads = 2, head_dim = 3;

    double *decays = make_multiscale_decay(num_heads, 0.7, 0.95);
    MultiHeadRetentionConfig cfg = {seq_len, num_heads, head_dim, decays};

    int shape[] = {num_heads, seq_len, head_dim};
    Tensor *Qh = tensor_create(shape, 3, 0);
    Tensor *Kh = tensor_create(shape, 3, 0);
    Tensor *Vh = tensor_create(shape, 3, 0);
    Tensor *G = tensor_create(shape, 3, 0);
    Tensor *out = tensor_create(shape, 3, 0);

    tensor_fill_random(Qh, -1.0, 1.0, 1);
    tensor_fill_random(Kh, -1.0, 1.0, 2);
    tensor_fill_random(Vh, -1.0, 1.0, 3);
    tensor_fill_random(G, -1.0, 1.0, 4);

    Tensor *dQh_a = tensor_create(shape, 3, 0);
    Tensor *dKh_a = tensor_create(shape, 3, 0);
    Tensor *dVh_a = tensor_create(shape, 3, 0);
    retention_multihead_backward(&cfg, Qh, Kh, Vh, G, dQh_a, dKh_a, dVh_a);

    Tensor *dQh_n = tensor_create(shape, 3, 0);
    for (size_t i = 0; i < Qh->size; i++) {
        double orig = Qh->data[i];
        Qh->data[i] = orig + FD_EPS;
        retention_multihead_parallel_forward(&cfg, Qh, Kh, Vh, out);
        double loss_p = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_p += out->data[k] * G->data[k];

        Qh->data[i] = orig - FD_EPS;
        retention_multihead_parallel_forward(&cfg, Qh, Kh, Vh, out);
        double loss_m = 0.0;
        for (size_t k = 0; k < out->size; k++) loss_m += out->data[k] * G->data[k];

        Qh->data[i] = orig;
        dQh_n->data[i] = (loss_p - loss_m) / (2.0 * FD_EPS);
    }

    double diff_q = max_abs_diff(dQh_a, dQh_n);
    printf("Max diff dQh: %.8f\n", diff_q);
    int pass = (diff_q < CHECK_TOL);
    printf("Retention multihead backward: %s\n\n", pass ? "LOLOS" : "GAGAL");

    free(decays);
    tensor_free(Qh); tensor_free(Kh); tensor_free(Vh); tensor_free(G); tensor_free(out);
    tensor_free(dQh_a); tensor_free(dKh_a); tensor_free(dVh_a); tensor_free(dQh_n);
    return pass;
}

int main(void) {
    int p1 = test_matmul_backward();
    int p2 = test_rmsnorm_backward();
    int p3 = test_swiglu_backward();
    int p4 = test_retention_multihead_backward();

    int all_pass = p1 && p2 && p3 && p4;
    printf("=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA BACKWARD OPS LOLOS" : "ADA YANG GAGAL");
    return all_pass ? 0 : 1;
}
