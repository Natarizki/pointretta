// retention_parallel.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>
#include <math.h>

int retention_parallel_forward(const RetentionConfig *cfg,
                                const Tensor *Q, const Tensor *K, const Tensor *V,
                                Tensor *out) {
    if (!cfg || !Q || !K || !V || !out) return -1;

    int seq_len = cfg->seq_len;
    int dim = cfg->dim;
    double decay = cfg->decay;

    if (Q->ndim != 2 || K->ndim != 2 || V->ndim != 2 || out->ndim != 2) {
        fprintf(stderr, "[retention_parallel_forward] semua tensor harus 2D\n");
        return -1;
    }
    if (Q->shape[0] != seq_len || Q->shape[1] != dim ||
        K->shape[0] != seq_len || K->shape[1] != dim ||
        V->shape[0] != seq_len || V->shape[1] != dim ||
        out->shape[0] != seq_len || out->shape[1] != dim) {
        fprintf(stderr, "[retention_parallel_forward] shape tidak sesuai config (seq_len=%d, dim=%d)\n",
                seq_len, dim);
        return -1;
    }

    tensor_fill(out, 0.0);

    for (int i = 0; i < seq_len; i++) {
        double *q_i = &Q->data[i * dim];
        double *out_i = &out->data[i * dim];

        for (int j = 0; j <= i; j++) {
            double *k_j = &K->data[j * dim];
            double *v_j = &V->data[j * dim];

            double qk = vec_dot(q_i, k_j, dim);
            double decay_factor = pow(decay, (double)(i - j));
            double weight = qk * decay_factor;

            for (int d = 0; d < dim; d++) {
                out_i[d] += weight * v_j[d];
            }
        }
    }

    return 0;
}
