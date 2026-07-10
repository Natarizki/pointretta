// tensor_ops.h
#ifndef POINTRETTA_TENSOR_OPS_H
#define POINTRETTA_TENSOR_OPS_H

#include "tensor.h"

int tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
int tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
int tensor_mul_elementwise(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_scale(Tensor *out, const Tensor *a, double scalar);
int tensor_matmul(Tensor *out, const Tensor *a, const Tensor *b);
double vec_dot(const double *a, const double *b, int dim);
void vec_outer(const double *a, int dim_a, const double *b, int dim_b, double *out);

#endif // POINTRETTA_TENSOR_OPS_H
