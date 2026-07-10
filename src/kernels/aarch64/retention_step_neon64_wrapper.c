// retention_step_neon64_wrapper.c
#include "tensor.h"
#include <stdlib.h>

extern void state_update_neon64_core(double *state, const double *k, const double *v,
                                      double decay, long head_dim, long b_even);
extern void matmul_neon64_core(const double *A, const double *B, double *out,
                                long M, long K, long N_stride, long N_compute);

static void state_update_neon(double *state, const double *k, const double *v, double decay, long head_dim) {
    long b_even = (head_dim % 2 == 0) ? head_dim : head_dim - 1;
    if (b_even > 0) state_update_neon64_core(state, k, v, decay, head_dim, b_even);
    if (b_even < head_dim) {
        long b = head_dim - 1;
        for (long a = 0; a < head_dim; a++) {
            state[a * head_dim + b] = decay * state[a * head_dim + b] + k[a] * v[b];
        }
    }
}

static void vecmat_neon(const double *q, const double *state, double *out, long head_dim) {
    long n_even = (head_dim % 2 == 0) ? head_dim : head_dim - 1;
    if (n_even > 0) matmul_neon64_core(q, state, out, 1, head_dim, head_dim, n_even);
    if (n_even < head_dim) {
        long j = head_dim - 1;
        double sum = 0.0;
        for (long a = 0; a < head_dim; a++) sum += q[a] * state[a * head_dim + j];
        out[j] = sum;
    }
}

// State: Tensor [head_dim, head_dim]; q_t,k_t,v_t,out_t: Tensor [head_dim]
int retention_recurrent_step_neon(double decay, int head_dim,
                                   Tensor *state, const Tensor *q_t, const Tensor *k_t,
                                   const Tensor *v_t, Tensor *out_t) {
    if (!state || !q_t || !k_t || !v_t || !out_t) return -1;
    if (state->size != (size_t)(head_dim * head_dim)) return -1;
    if (q_t->size != (size_t)head_dim || k_t->size != (size_t)head_dim ||
        v_t->size != (size_t)head_dim || out_t->size != (size_t)head_dim) return -1;

    state_update_neon(state->data, k_t->data, v_t->data, decay, head_dim);
    vecmat_neon(q_t->data, state->data, out_t->data, head_dim);

    return 0;
}
