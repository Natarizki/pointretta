// tensor.h
// Struktur data Tensor dasar untuk PointRetta.
// Semua modul (retention, ffn, autograd, optimizer) pakai struct ini.

#ifndef POINTRETTA_TENSOR_H
#define POINTRETTA_TENSOR_H

#include <stddef.h>

#define TENSOR_MAX_DIMS 4

typedef struct {
    double *data;
    int shape[TENSOR_MAX_DIMS];
    int strides[TENSOR_MAX_DIMS];
    int ndim;
    size_t size;
    int requires_grad;
    double *grad;
} Tensor;

Tensor *tensor_create(const int *shape, int ndim, int requires_grad);
void tensor_free(Tensor *t);
void tensor_fill(Tensor *t, double value);
void tensor_fill_random(Tensor *t, double min, double max, unsigned int seed);
int tensor_copy(Tensor *dst, const Tensor *src);
void tensor_zero_grad(Tensor *t);
size_t tensor_index(const Tensor *t, const int *coords);
double tensor_get(const Tensor *t, const int *coords);
void tensor_set(Tensor *t, const int *coords, double value);
void tensor_print(const Tensor *t, const char *name);

#endif // POINTRETTA_TENSOR_H
