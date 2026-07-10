// rmsnorm_neon64_wrapper.c
#include "tensor.h"
#include <math.h>
#include <stdio.h>

extern double sumsq_neon64_core(const double *x, long n);
extern void scale_gain_neon64_core(const double *x, const double *gain, double *out, long n, double inv_rms);

int rmsnorm_forward_neon(const Tensor *x, const Tensor *gain, Tensor *out, double eps) {
    if (!x || !gain || !out) return -1;
    if (x->ndim != 2 || out->ndim != 2) return -1;

    long seq_len = x->shape[0];
    long dim = x->shape[1];
    if (out->shape[0] != seq_len || out->shape[1] != dim) return -1;
    if (gain->size != (size_t)dim) return -1;

    long dim_even = (dim % 2 == 0) ? dim : dim - 1;

    for (long i = 0; i < seq_len; i++) {
        const double *x_i = &x->data[i * dim];
        double *out_i = &out->data[i * dim];

        double sum_sq = (dim_even > 0) ? sumsq_neon64_core(x_i, dim_even) : 0.0;
        if (dim_even < dim) {
            double last = x_i[dim - 1];
            sum_sq += last * last;
        }

        double mean_sq = sum_sq / (double)dim;
        double rms = sqrt(mean_sq + eps);
        double inv_rms = 1.0 / rms;

        if (dim_even > 0) {
            scale_gain_neon64_core(x_i, gain->data, out_i, dim_even, inv_rms);
        }
        if (dim_even < dim) {
            out_i[dim - 1] = x_i[dim - 1] * inv_rms * gain->data[dim - 1];
        }
    }

    return 0;
}
