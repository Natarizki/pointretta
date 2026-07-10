// norm.c
#include "tensor.h"
#include "norm.h"
#include <stdio.h>
#include <math.h>

int rmsnorm_forward(const Tensor *x, const Tensor *gain, Tensor *out, double eps) {
    if (!x || !gain || !out) return -1;
    if (x->ndim != 2 || out->ndim != 2) {
        fprintf(stderr, "[rmsnorm_forward] x dan out harus 2D [seq_len, dim]\n");
        return -1;
    }
    int seq_len = x->shape[0];
    int dim = x->shape[1];
    if (out->shape[0] != seq_len || out->shape[1] != dim) {
        fprintf(stderr, "[rmsnorm_forward] shape out tidak cocok dengan x\n");
        return -1;
    }
    if (gain->size != (size_t)dim) {
        fprintf(stderr, "[rmsnorm_forward] gain harus sepanjang dim (%d)\n", dim);
        return -1;
    }

    for (int i = 0; i < seq_len; i++) {
        double *x_i = &x->data[i * dim];
        double *out_i = &out->data[i * dim];

        double sum_sq = 0.0;
        for (int d = 0; d < dim; d++) {
            sum_sq += x_i[d] * x_i[d];
        }
        double mean_sq = sum_sq / (double)dim;
        double rms = sqrt(mean_sq + eps);

        for (int d = 0; d < dim; d++) {
            out_i[d] = (x_i[d] / rms) * gain->data[d];
        }
    }

    return 0;
}
