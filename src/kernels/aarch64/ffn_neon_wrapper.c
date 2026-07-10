// ffn_neon_wrapper.c
#include "tensor.h"
#include "tensor_ops.h"
#include "ffn.h"
#include <stdio.h>
#include <math.h>

extern void vecmul_neon64_core(const double *a, const double *b, double *out, long n_even);

static void vecmul_neon(const double *a, const double *b, double *out, long n) {
    long n_even = (n % 2 == 0) ? n : n - 1;
    if (n_even > 0) vecmul_neon64_core(a, b, out, n_even);
    if (n_even < n) out[n - 1] = a[n - 1] * b[n - 1];
}

int swiglu_forward_neon(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                         Tensor *out, Tensor *scratch_hidden1, Tensor *scratch_hidden2) {
    if (!x || !W1 || !W3 || !W2 || !out || !scratch_hidden1 || !scratch_hidden2) return -1;

    int rc1 = tensor_matmul(scratch_hidden1, x, W1);
    int rc2 = tensor_matmul(scratch_hidden2, x, W3);
    if (rc1 != 0 || rc2 != 0) return -1;

    for (size_t i = 0; i < scratch_hidden1->size; i++) {
        double z = scratch_hidden1->data[i];
        double sigmoid = 1.0 / (1.0 + exp(-z));
        scratch_hidden1->data[i] = z * sigmoid;
    }

    vecmul_neon(scratch_hidden1->data, scratch_hidden2->data, scratch_hidden1->data, scratch_hidden1->size);

    int rc3 = tensor_matmul(out, scratch_hidden1, W2);
    return rc3;
}
