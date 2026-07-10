// retention_multihead.h
#ifndef POINTRETTA_RETENTION_MULTIHEAD_H
#define POINTRETTA_RETENTION_MULTIHEAD_H

#include "tensor.h"

typedef struct {
    int seq_len;
    int num_heads;
    int head_dim;
    double *decay_per_head;
} MultiHeadRetentionConfig;

int retention_multihead_parallel_forward(const MultiHeadRetentionConfig *cfg,
                                          const Tensor *Q, const Tensor *K, const Tensor *V,
                                          Tensor *out);

int retention_multihead_recurrent_step(const MultiHeadRetentionConfig *cfg,
                                        Tensor *state,
                                        const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                        Tensor *out_t);

Tensor *retention_multihead_state_create(int num_heads, int head_dim);

double *make_multiscale_decay(int num_heads, double decay_min, double decay_max);

#endif // POINTRETTA_RETENTION_MULTIHEAD_H
