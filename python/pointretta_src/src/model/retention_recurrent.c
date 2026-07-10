// retention_recurrent.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>

Tensor *retention_state_create(int dim) {
    int shape[] = {dim, dim};
    Tensor *state = tensor_create(shape, 2, 0);
    if (state) {
        tensor_fill(state, 0.0);
    }
    return state;
}

int retention_recurrent_step(const RetentionConfig *cfg,
                              Tensor *state,
                              const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                              Tensor *out_t) {
    if (!cfg || !state || !q_t || !k_t || !v_t || !out_t) return -1;

    int dim = cfg->dim;
    double decay = cfg->decay;

    if (state->ndim != 2 || state->shape[0] != dim || state->shape[1] != dim) {
        fprintf(stderr, "[retention_recurrent_step] state harus shape [%d,%d]\n", dim, dim);
        return -1;
    }
    if (q_t->size != (size_t)dim || k_t->size != (size_t)dim ||
        v_t->size != (size_t)dim || out_t->size != (size_t)dim) {
        fprintf(stderr, "[retention_recurrent_step] q_t/k_t/v_t/out_t harus size %d\n", dim);
        return -1;
    }

    for (int a = 0; a < dim; a++) {
        for (int b = 0; b < dim; b++) {
            double outer_val = k_t->data[a] * v_t->data[b];
            state->data[a * dim + b] = decay * state->data[a * dim + b] + outer_val;
        }
    }

    for (int b = 0; b < dim; b++) {
        double sum = 0.0;
        for (int a = 0; a < dim; a++) {
            sum += q_t->data[a] * state->data[a * dim + b];
        }
        out_t->data[b] = sum;
    }

    return 0;
}
