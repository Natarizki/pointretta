// test_retention_backward.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 5
#define DIM 3
#define DECAY 0.85
#define FD_EPS 1e-6
#define CHECK_TOL 1e-4

static double compute_loss(const RetentionConfig *cfg, const Tensor *Q, const Tensor *K,
                            const Tensor *V, const Tensor *G, Tensor *scratch_out) {
    retention_parallel_forward(cfg, Q, K, V, scratch_out);
    double loss = 0.0;
    for (size_t i = 0; i < scratch_out->size; i++) {
        loss += scratch_out->data[i] * G->data[i];
    }
    return loss;
}

static void numerical_gradient(const RetentionConfig *cfg, Tensor *target,
                                Tensor *Q, Tensor *K, Tensor *V, const Tensor *G,
                                Tensor *scratch_out, Tensor *grad_numeric) {
    for (size_t i = 0; i < target->size; i++) {
        double original = target->data[i];

        target->data[i] = original + FD_EPS;
        double loss_plus = compute_loss(cfg, Q, K, V, G, scratch_out);

        target->data[i] = original - FD_EPS;
        double loss_minus = compute_loss(cfg, Q, K, V, G, scratch_out);

        target->data[i] = original;

        grad_numeric->data[i] = (loss_plus - loss_minus) / (2.0 * FD_EPS);
    }
}

static double max_abs_diff(const Tensor *a, const Tensor *b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a->size; i++) {
        double diff = fabs(a->data[i] - b->data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

int main(void) {
    RetentionConfig cfg = { .seq_len = SEQ_LEN, .dim = DIM, .decay = DECAY };
    int shape[] = {SEQ_LEN, DIM};

    Tensor *Q = tensor_create(shape, 2, 0);
    Tensor *K = tensor_create(shape, 2, 0);
    Tensor *V = tensor_create(shape, 2, 0);
    Tensor *G = tensor_create(shape, 2, 0);
    Tensor *out = tensor_create(shape, 2, 0);

    tensor_fill_random(Q, -1.0, 1.0, 1);
    tensor_fill_random(K, -1.0, 1.0, 2);
    tensor_fill_random(V, -1.0, 1.0, 3);
    tensor_fill_random(G, -1.0, 1.0, 4);

    Tensor *dQ_analytic = tensor_create(shape, 2, 0);
    Tensor *dK_analytic = tensor_create(shape, 2, 0);
    Tensor *dV_analytic = tensor_create(shape, 2, 0);

    int rc = retention_parallel_backward(&cfg, Q, K, V, G, dQ_analytic, dK_analytic, dV_analytic);
    printf("retention_parallel_backward return: %d (harusnya 0)\n", rc);

    Tensor *dQ_numeric = tensor_create(shape, 2, 0);
    Tensor *dK_numeric = tensor_create(shape, 2, 0);
    Tensor *dV_numeric = tensor_create(shape, 2, 0);

    numerical_gradient(&cfg, Q, Q, K, V, G, out, dQ_numeric);
    numerical_gradient(&cfg, K, Q, K, V, G, out, dK_numeric);
    numerical_gradient(&cfg, V, Q, K, V, G, out, dV_numeric);

    tensor_print(dQ_analytic, "dQ (analitik)");
    tensor_print(dQ_numeric, "dQ (numerik)");
    printf("\n");
    tensor_print(dK_analytic, "dK (analitik)");
    tensor_print(dK_numeric, "dK (numerik)");
    printf("\n");
    tensor_print(dV_analytic, "dV (analitik)");
    tensor_print(dV_numeric, "dV (numerik)");
    printf("\n");

    double diff_q = max_abs_diff(dQ_analytic, dQ_numeric);
    double diff_k = max_abs_diff(dK_analytic, dK_numeric);
    double diff_v = max_abs_diff(dV_analytic, dV_numeric);

    printf("Max diff dQ: %.10f\n", diff_q);
    printf("Max diff dK: %.10f\n", diff_k);
    printf("Max diff dV: %.10f\n", diff_v);

    int pass = (diff_q < CHECK_TOL) && (diff_k < CHECK_TOL) && (diff_v < CHECK_TOL);

    if (pass) {
        printf("\nHASIL: LOLOS — gradient analitik cocok dengan gradient numerik (toleransi %.0e)\n", CHECK_TOL);
    } else {
        printf("\nHASIL: GAGAL — ada kesalahan derivasi gradient, perlu dicek ulang!\n");
    }

    tensor_free(Q); tensor_free(K); tensor_free(V); tensor_free(G); tensor_free(out);
    tensor_free(dQ_analytic); tensor_free(dK_analytic); tensor_free(dV_analytic);
    tensor_free(dQ_numeric); tensor_free(dK_numeric); tensor_free(dV_numeric);

    return pass ? 0 : 1;
}
