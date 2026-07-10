// retention_multihead.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention_multihead.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double *make_multiscale_decay(int num_heads, double decay_min, double decay_max) {
    double *decays = (double *)malloc(sizeof(double) * num_heads);
    if (!decays) return NULL;

    if (num_heads == 1) {
        decays[0] = decay_max;
        return decays;
    }

    double log_min = log(1.0 - decay_min);
    double log_max = log(1.0 - decay_max);

    for (int h = 0; h < num_heads; h++) {
        double t = (double)h / (double)(num_heads - 1);
        double log_val = log_min + t * (log_max - log_min);
        decays[h] = 1.0 - exp(log_val);
    }

    return decays;
}

Tensor *retention_multihead_state_create(int num_heads, int head_dim) {
    int shape[] = {num_heads, head_dim, head_dim};
    Tensor *state = tensor_create(shape, 3, 0);
    if (state) tensor_fill(state, 0.0);
    return state;
}

int retention_multihead_parallel_forward(const MultiHeadRetentionConfig *cfg,
                                          const Tensor *Q, const Tensor *K, const Tensor *V,
                                          Tensor *out) {
    if (!cfg || !Q || !K || !V || !out) return -1;

    int seq_len = cfg->seq_len;
    int num_heads = cfg->num_heads;
    int head_dim = cfg->head_dim;

    const Tensor *all[] = {Q, K, V, out};
    for (int i = 0; i < 4; i++) {
        if (all[i]->ndim != 3 || all[i]->shape[0] != num_heads ||
            all[i]->shape[1] != seq_len || all[i]->shape[2] != head_dim) {
            fprintf(stderr, "[retention_multihead_parallel_forward] shape tensor ke-%d tidak sesuai\n", i);
            return -1;
        }
    }

    tensor_fill(out, 0.0);

    int head_stride = seq_len * head_dim;

    for (int h = 0; h < num_heads; h++) {
        double decay = cfg->decay_per_head[h];
        double *Qh = &Q->data[h * head_stride];
        double *Kh = &K->data[h * head_stride];
        double *Vh = &V->data[h * head_stride];
        double *outh = &out->data[h * head_stride];

        for (int i = 0; i < seq_len; i++) {
            double *q_i = &Qh[i * head_dim];
            double *out_i = &outh[i * head_dim];

            for (int j = 0; j <= i; j++) {
                double *k_j = &Kh[j * head_dim];
                double *v_j = &Vh[j * head_dim];

                double qk = vec_dot(q_i, k_j, head_dim);
                double decay_factor = pow(decay, (double)(i - j));
                double weight = qk * decay_factor;

                for (int d = 0; d < head_dim; d++) {
                    out_i[d] += weight * v_j[d];
                }
            }
        }
    }

    return 0;
}

int retention_multihead_recurrent_step(const MultiHeadRetentionConfig *cfg,
                                        Tensor *state,
                                        const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                        Tensor *out_t) {
    if (!cfg || !state || !q_t || !k_t || !v_t || !out_t) return -1;

    int num_heads = cfg->num_heads;
    int head_dim = cfg->head_dim;

    if (state->ndim != 3 || state->shape[0] != num_heads ||
        state->shape[1] != head_dim || state->shape[2] != head_dim) {
        fprintf(stderr, "[retention_multihead_recurrent_step] state shape tidak sesuai\n");
        return -1;
    }
    if (q_t->size != (size_t)(num_heads * head_dim) ||
        k_t->size != (size_t)(num_heads * head_dim) ||
        v_t->size != (size_t)(num_heads * head_dim) ||
        out_t->size != (size_t)(num_heads * head_dim)) {
        fprintf(stderr, "[retention_multihead_recurrent_step] q_t/k_t/v_t/out_t size tidak sesuai\n");
        return -1;
    }

    int state_head_stride = head_dim * head_dim;
    int vec_head_stride = head_dim;

    for (int h = 0; h < num_heads; h++) {
        double decay = cfg->decay_per_head[h];
        double *state_h = &state->data[h * state_head_stride];
        double *q_h = &q_t->data[h * vec_head_stride];
        double *k_h = &k_t->data[h * vec_head_stride];
        double *v_h = &v_t->data[h * vec_head_stride];
        double *out_h = &out_t->data[h * vec_head_stride];

        for (int a = 0; a < head_dim; a++) {
            for (int b = 0; b < head_dim; b++) {
                double outer_val = k_h[a] * v_h[b];
                state_h[a * head_dim + b] = decay * state_h[a * head_dim + b] + outer_val;
            }
        }

        for (int b = 0; b < head_dim; b++) {
            double sum = 0.0;
            for (int a = 0; a < head_dim; a++) {
                sum += q_h[a] * state_h[a * head_dim + b];
            }
            out_h[b] = sum;
        }
    }

    return 0;
}
