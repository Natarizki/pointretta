// ffn.h
#ifndef POINTRETTA_FFN_H
#define POINTRETTA_FFN_H

#include "tensor.h"

int swiglu_forward(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                    Tensor *out, Tensor *scratch_hidden1, Tensor *scratch_hidden2);

double silu(double z);

#endif // POINTRETTA_FFN_H
