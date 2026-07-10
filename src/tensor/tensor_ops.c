// tensor_ops.c
#include "tensor.h"
#include "tensor_ops.h"
#include <stdio.h>

int tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) return -1;
    if (out->size != a->size || out->size != b->size) {
        fprintf(stderr, "[tensor_add] size tidak cocok: out=%zu a=%zu b=%zu\n",
                out->size, a->size, b->size);
        return -1;
    }
    for (size_t i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return 0;
}

int tensor_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) return -1;
    if (out->size != a->size || out->size != b->size) {
        fprintf(stderr, "[tensor_sub] size tidak cocok: out=%zu a=%zu b=%zu\n",
                out->size, a->size, b->size);
        return -1;
    }
    for (size_t i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return 0;
}

int tensor_mul_elementwise(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) return -1;
    if (out->size != a->size || out->size != b->size) {
        fprintf(stderr, "[tensor_mul_elementwise] size tidak cocok: out=%zu a=%zu b=%zu\n",
                out->size, a->size, b->size);
        return -1;
    }
    for (size_t i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return 0;
}

void tensor_scale(Tensor *out, const Tensor *a, double scalar) {
    if (!out || !a) return;
    for (size_t i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] * scalar;
    }
}

int tensor_matmul(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) return -1;
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        fprintf(stderr, "[tensor_matmul] semua tensor harus 2D\n");
        return -1;
    }

    int M = a->shape[0];
    int K = a->shape[1];
    int K2 = b->shape[0];
    int N = b->shape[1];

    if (K != K2) {
        fprintf(stderr, "[tensor_matmul] dimensi dalam tidak cocok: a=[%d,%d] b=[%d,%d]\n",
                M, K, K2, N);
        return -1;
    }
    if (out->shape[0] != M || out->shape[1] != N) {
        fprintf(stderr, "[tensor_matmul] shape out salah: harus [%d,%d], dapat [%d,%d]\n",
                M, N, out->shape[0], out->shape[1]);
        return -1;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }
    return 0;
}

double vec_dot(const double *a, const double *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vec_outer(const double *a, int dim_a, const double *b, int dim_b, double *out) {
    for (int i = 0; i < dim_a; i++) {
        for (int j = 0; j < dim_b; j++) {
            out[i * dim_b + j] = a[i] * b[j];
        }
    }
}
