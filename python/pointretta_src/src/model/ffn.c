// ffn.c
#include "tensor.h"
#include "tensor_ops.h"
#include "ffn.h"
#include <stdio.h>
#include <math.h>

double silu(double z) {
    double sigmoid = 1.0 / (1.0 + exp(-z));
    return z * sigmoid;
}

int swiglu_forward(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                    Tensor *out, Tensor *scratch_hidden1, Tensor *scratch_hidden2) {
    if (!x || !W1 || !W3 || !W2 || !out || !scratch_hidden1 || !scratch_hidden2) return -1;

    if (x->ndim != 2 || W1->ndim != 2 || W3->ndim != 2 || W2->ndim != 2 || out->ndim != 2) {
        fprintf(stderr, "[swiglu_forward] semua tensor harus 2D\n");
        return -1;
    }

    int seq_len = x->shape[0];
    int dim = x->shape[1];
    int hidden = W1->shape[1];

    if (W1->shape[0] != dim || W3->shape[0] != dim || W3->shape[1] != hidden) {
        fprintf(stderr, "[swiglu_forward] shape W1/W3 tidak konsisten dengan dim/hidden\n");
        return -1;
    }
    if (W2->shape[0] != hidden || W2->shape[1] != dim) {
        fprintf(stderr, "[swiglu_forward] shape W2 harus [hidden, dim]\n");
        return -1;
    }
    if (out->shape[0] != seq_len || out->shape[1] != dim) {
        fprintf(stderr, "[swiglu_forward] shape out harus [seq_len, dim]\n");
        return -1;
    }
    if (scratch_hidden1->shape[0] != seq_len || scratch_hidden1->shape[1] != hidden ||
        scratch_hidden2->shape[0] != seq_len || scratch_hidden2->shape[1] != hidden) {
        fprintf(stderr, "[swiglu_forward] shape scratch harus [seq_len, hidden]\n");
        return -1;
    }

    int rc1 = tensor_matmul(scratch_hidden1, x, W1);
    if (rc1 != 0) return -1;

    int rc2 = tensor_matmul(scratch_hidden2, x, W3);
    if (rc2 != 0) return -1;

    for (size_t i = 0; i < scratch_hidden1->size; i++) {
        double activated = silu(scratch_hidden1->data[i]);
        scratch_hidden1->data[i] = activated * scratch_hidden2->data[i];
    }

    int rc3 = tensor_matmul(out, scratch_hidden1, W2);
    if (rc3 != 0) return -1;

    return 0;
}
