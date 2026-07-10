// retention_parallel_neon_wrapper.c
#include "tensor.h"
#include "retention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern double dot_neon64_core(const double *a, const double *b, long n_even);
extern void accum_scale_neon64_core(double *out, const double *v, double weight, long n_even);

static double dot_neon(const double *a, const double *b, long n) {
    long n_even = (n % 2 == 0) ? n : n - 1;
    double result = (n_even > 0) ? dot_neon64_core(a, b, n_even) : 0.0;
    if (n_even < n) result += a[n - 1] * b[n - 1];
    return result;
}

static void accum_scale_neon(double *out, const double *v, double weight, long n) {
    long n_even = (n % 2 == 0) ? n : n - 1;
    if (n_even > 0) accum_scale_neon64_core(out, v, weight, n_even);
    if (n_even < n) out[n - 1] += weight * v[n - 1];
}

int retention_parallel_forward_neon(const RetentionConfig *cfg,
                                     const Tensor *Q, const Tensor *K, const Tensor *V,
                                     Tensor *out) {
    if (!cfg || !Q || !K || !V || !out) return -1;

    long seq_len = cfg->seq_len;
    long dim = cfg->dim;
    double decay = cfg->decay;

    if (Q->shape[0] != seq_len || Q->shape[1] != dim ||
        K->shape[0] != seq_len || K->shape[1] != dim ||
        V->shape[0] != seq_len || V->shape[1] != dim ||
        out->shape[0] != seq_len || out->shape[1] != dim) {
        fprintf(stderr, "[retention_parallel_forward_neon] shape tidak sesuai\n");
        return -1;
    }

    double *decay_pows = (double *)malloc(sizeof(double) * seq_len);
    decay_pows[0] = 1.0;
    for (long d = 1; d < seq_len; d++) decay_pows[d] = decay_pows[d - 1] * decay;

    tensor_fill(out, 0.0);

    for (long i = 0; i < seq_len; i++) {
        const double *q_i = &Q->data[i * dim];
        double *out_i = &out->data[i * dim];

        for (long j = 0; j <= i; j++) {
            const double *k_j = &K->data[j * dim];
            const double *v_j = &V->data[j * dim];

            double qk = dot_neon(q_i, k_j, dim);
            double weight = qk * decay_pows[i - j];

            accum_scale_neon(out_i, v_j, weight, dim);
        }
    }

    free(decay_pows);
    return 0;
}
