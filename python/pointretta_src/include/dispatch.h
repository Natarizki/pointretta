// dispatch.h
#ifndef POINTRETTA_DISPATCH_H
#define POINTRETTA_DISPATCH_H

#include "tensor.h"
#include "retention.h"
#include "retention_multihead.h"
#include "ffn.h"

const char *dispatch_active_backend(void);

int tensor_matmul_fast(Tensor *out, const Tensor *A, const Tensor *B);
int rmsnorm_forward_fast(const Tensor *x, const Tensor *gain, Tensor *out, double eps);
int retention_parallel_forward_fast(const RetentionConfig *cfg,
                                     const Tensor *Q, const Tensor *K, const Tensor *V,
                                     Tensor *out);
int retention_recurrent_step_fast(const RetentionConfig *cfg,
                                   Tensor *state,
                                   const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                   Tensor *out_t);
int retention_multihead_parallel_forward_fast(const MultiHeadRetentionConfig *cfg,
                                               const Tensor *Qh, const Tensor *Kh, const Tensor *Vh,
                                               Tensor *out_h);
int retention_multihead_recurrent_step_fast(const MultiHeadRetentionConfig *cfg,
                                             Tensor *state,
                                             const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                             Tensor *out_t);
#endif // POINTRETTA_DISPATCH_H
