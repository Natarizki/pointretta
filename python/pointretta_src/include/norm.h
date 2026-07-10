// norm.h
#ifndef POINTRETTA_NORM_H
#define POINTRETTA_NORM_H

#include "tensor.h"

int rmsnorm_forward(const Tensor *x, const Tensor *gain, Tensor *out, double eps);

#endif // POINTRETTA_NORM_H
