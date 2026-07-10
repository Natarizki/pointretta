// backward_retention.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

int retention_parallel_backward(const RetentionConfig *cfg,
                                 const Tensor *Q, const Tensor *K, const Tensor *V,
                                 const Tensor *grad_out,
                                 Tensor *dQ, Tensor *dK, Tensor *dV) {
    if (!cfg || !Q || !K || !V || !grad_out || !dQ || !dK || !dV) return -1;

    int seq_len = cfg->seq_len;
    int dim = cfg->dim;
    double decay = cfg->decay;

    const Tensor *all[] = {Q, K, V, grad_out, dQ, dK, dV};
    for (int i = 0; i < 7; i++) {
        if (all[i]->ndim != 2 || all[i]->shape[0] != seq_len || all[i]->shape[1] != dim) {
            fprintf(stderr, "[retention_parallel_backward] shape tensor ke-%d tidak sesuai\n", i);
            return -1;
        }
    }

    tensor_fill(dQ, 0.0);
    tensor_fill(dK, 0.0);
    tensor_fill(dV, 0.0);

    for (int i = 0; i < seq_len; i++) {
        double *g_i = &grad_out->data[i * dim];
        double *dq_i = &dQ->data[i * dim];

        for (int j = 0; j <= i; j++) {
            double *v_j = &V->data[j * dim];
            double *k_j = &K->data[j * dim];

            double gv = vec_dot(g_i, v_j, dim);
            double decay_factor = pow(decay, (double)(i - j));
            double weight = gv * decay_factor;

            for (int d = 0; d < dim; d++) {
                dq_i[d] += weight * k_j[d];
            }
        }
    }

    for (int j = 0; j < seq_len; j++) {
        double *v_j = &V->data[j * dim];
        double *dk_j = &dK->data[j * dim];

        for (int i = j; i < seq_len; i++) {
            double *g_i = &grad_out->data[i * dim];
            double *q_i = &Q->data[i * dim];

            double gv = vec_dot(g_i, v_j, dim);
            double decay_factor = pow(decay, (double)(i - j));
            double weight = gv * decay_factor;

            for (int d = 0; d < dim; d++) {
                dk_j[d] += weight * q_i[d];
            }
        }
    }

    for (int j = 0; j < seq_len; j++) {
        double *k_j = &K->data[j * dim];
        double *dv_j = &dV->data[j * dim];

        for (int i = j; i < seq_len; i++) {
            double *q_i = &Q->data[i * dim];
            double *g_i = &grad_out->data[i * dim];

            double qk = vec_dot(q_i, k_j, dim);
            double decay_factor = pow(decay, (double)(i - j));
            double weight = qk * decay_factor;

            for (int d = 0; d < dim; d++) {
                dv_j[d] += weight * g_i[d];
            }
        }
    }

    return 0;
}
