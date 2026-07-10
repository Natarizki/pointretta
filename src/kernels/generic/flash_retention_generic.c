// flash_retention_generic.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

int retention_flash_forward(const RetentionConfig *cfg,
                             const Tensor *Q, const Tensor *K, const Tensor *V,
                             Tensor *out, int chunk_size) {
    if (!cfg || !Q || !K || !V || !out) return -1;

    int seq_len = cfg->seq_len;
    int dim = cfg->dim;
    double decay = cfg->decay;

    if (chunk_size <= 0) chunk_size = seq_len;

    const Tensor *all[] = {Q, K, V, out};
    for (int i = 0; i < 4; i++) {
        if (all[i]->ndim != 2 || all[i]->shape[0] != seq_len || all[i]->shape[1] != dim) {
            fprintf(stderr, "[retention_flash_forward] shape tensor ke-%d tidak sesuai\n", i);
            return -1;
        }
    }

    tensor_fill(out, 0.0);

    double *state = (double *)calloc((size_t)dim * dim, sizeof(double));
    double *new_contrib = (double *)calloc((size_t)dim * dim, sizeof(double));
    if (!state || !new_contrib) {
        fprintf(stderr, "[retention_flash_forward] gagal alokasi state\n");
        free(state); free(new_contrib);
        return -1;
    }

    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    for (int c = 0; c < num_chunks; c++) {
        int chunk_start = c * chunk_size;
        int chunk_len = chunk_size;
        if (chunk_start + chunk_len > seq_len) chunk_len = seq_len - chunk_start;

        for (int ip = 0; ip < chunk_len; ip++) {
            int i = chunk_start + ip;
            double *q_i = &Q->data[i * dim];
            double *out_i = &out->data[i * dim];

            for (int jp = 0; jp <= ip; jp++) {
                int j = chunk_start + jp;
                double *k_j = &K->data[j * dim];
                double *v_j = &V->data[j * dim];

                double qk = vec_dot(q_i, k_j, dim);
                double decay_factor = pow(decay, (double)(ip - jp));
                double weight = qk * decay_factor;

                for (int d = 0; d < dim; d++) {
                    out_i[d] += weight * v_j[d];
                }
            }

            double decay_cross = pow(decay, (double)(ip + 1));
            for (int b = 0; b < dim; b++) {
                double sum = 0.0;
                for (int a = 0; a < dim; a++) {
                    sum += q_i[a] * state[a * dim + b];
                }
                out_i[b] += decay_cross * sum;
            }
        }

        memset(new_contrib, 0, (size_t)dim * dim * sizeof(double));
        for (int jp = 0; jp < chunk_len; jp++) {
            int j = chunk_start + jp;
            double *k_j = &K->data[j * dim];
            double *v_j = &V->data[j * dim];
            double decay_factor = pow(decay, (double)(chunk_len - 1 - jp));

            for (int a = 0; a < dim; a++) {
                for (int b = 0; b < dim; b++) {
                    new_contrib[a * dim + b] += decay_factor * k_j[a] * v_j[b];
                }
            }
        }

        double decay_state = pow(decay, (double)chunk_len);
        for (int a = 0; a < dim; a++) {
            for (int b = 0; b < dim; b++) {
                state[a * dim + b] = decay_state * state[a * dim + b] + new_contrib[a * dim + b];
            }
        }
    }

    free(state);
    free(new_contrib);
    return 0;
}
