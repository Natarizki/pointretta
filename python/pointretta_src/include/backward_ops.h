// backward_ops.h
#ifndef POINTRETTA_BACKWARD_OPS_H
#define POINTRETTA_BACKWARD_OPS_H

#include "tensor.h"
#include "retention_multihead.h"

int matmul_backward(const Tensor *dOut, const Tensor *A, const Tensor *B,
                     Tensor *dA, Tensor *dB);

int rmsnorm_backward(const Tensor *x, const Tensor *gain, const Tensor *dOut, double eps,
                      Tensor *dX, Tensor *dGain);

int swiglu_forward_for_backward(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                                 Tensor *h1, Tensor *h2, Tensor *a, Tensor *hidden, Tensor *out);

int swiglu_backward(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                     const Tensor *h1, const Tensor *h2, const Tensor *a, const Tensor *hidden,
                     const Tensor *dOut,
                     Tensor *dX, Tensor *dW1, Tensor *dW3, Tensor *dW2);

int retention_multihead_backward(const MultiHeadRetentionConfig *cfg,
                                  const Tensor *Qh, const Tensor *Kh, const Tensor *Vh,
                                  const Tensor *dOut_h,
                                  Tensor *dQh, Tensor *dKh, Tensor *dVh);

#endif // POINTRETTA_BACKWARD_OPS_H
