// retention.h
#ifndef POINTRETTA_RETENTION_H
#define POINTRETTA_RETENTION_H

#include "tensor.h"

typedef struct {
    int seq_len;
    int dim;
    double decay;
} RetentionConfig;

int retention_parallel_forward(const RetentionConfig *cfg,
                                const Tensor *Q, const Tensor *K, const Tensor *V,
                                Tensor *out);

int retention_recurrent_step(const RetentionConfig *cfg,
                              Tensor *state,
                              const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                              Tensor *out_t);

Tensor *retention_state_create(int dim);

int retention_parallel_backward(const RetentionConfig *cfg,
                                 const Tensor *Q, const Tensor *K, const Tensor *V,
                                 const Tensor *grad_out,
                                 Tensor *dQ, Tensor *dK, Tensor *dV);

int retention_flash_forward(const RetentionConfig *cfg,
                             const Tensor *Q, const Tensor *K, const Tensor *V,
                             Tensor *out, int chunk_size);

#endif // POINTRETTA_RETENTION_H
