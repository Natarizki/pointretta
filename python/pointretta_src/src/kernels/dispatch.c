// dispatch.c
// Implementasi dispatch layer. Isi tiap #ifdef block beda-beda sesuai arch,
// tapi signature fungsi publik ("_fast") selalu sama.

#include "dispatch.h"
#include "tensor_ops.h"
#include "norm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ============================================================
#if defined(ARCH_AARCH64)
// ============================================================

extern void matmul_neon64_core(const double *A, const double *B, double *out,
                                long M, long K, long N_stride, long N_compute);
extern double sumsq_neon64_core(const double *x, long n);
extern void scale_gain_neon64_core(const double *x, const double *gain, double *out, long n, double inv_rms);
extern void state_update_neon64_core(double *state, const double *k, const double *v,
                                      double decay, long head_dim, long b_even);
extern double dot_neon64_core(const double *a, const double *b, long n_even);
extern void accum_scale_neon64_core(double *out, const double *v, double weight, long n_even);

const char *dispatch_active_backend(void) { return "aarch64_neon"; }

int tensor_matmul_fast(Tensor *out, const Tensor *A, const Tensor *B) {
    long M = A->shape[0], K = A->shape[1], N = B->shape[1];
    long n_even = (N % 2 == 0) ? N : N - 1;
    if (n_even > 0) matmul_neon64_core(A->data, B->data, out->data, M, K, N, n_even);
    if (n_even < N) {
        long j = N - 1;
        for (long i = 0; i < M; i++) {
            double sum = 0.0;
            for (long k = 0; k < K; k++) sum += A->data[i * K + k] * B->data[k * N + j];
            out->data[i * N + j] = sum;
        }
    }
    return 0;
}

int rmsnorm_forward_fast(const Tensor *x, const Tensor *gain, Tensor *out, double eps) {
    long seq_len = x->shape[0], dim = x->shape[1];
    long dim_even = (dim % 2 == 0) ? dim : dim - 1;
    for (long i = 0; i < seq_len; i++) {
        const double *x_i = &x->data[i * dim];
        double *out_i = &out->data[i * dim];
        double sum_sq = (dim_even > 0) ? sumsq_neon64_core(x_i, dim_even) : 0.0;
        if (dim_even < dim) { double last = x_i[dim - 1]; sum_sq += last * last; }
        double rms = sqrt(sum_sq / (double)dim + eps);
        double inv_rms = 1.0 / rms;
        if (dim_even > 0) scale_gain_neon64_core(x_i, gain->data, out_i, dim_even, inv_rms);
        if (dim_even < dim) out_i[dim - 1] = x_i[dim - 1] * inv_rms * gain->data[dim - 1];
    }
    return 0;
}

int retention_parallel_forward_fast(const RetentionConfig *cfg,
                                     const Tensor *Q, const Tensor *K, const Tensor *V,
                                     Tensor *out) {
    long seq_len = cfg->seq_len, dim = cfg->dim;
    double decay = cfg->decay;
    double *decay_pows = (double *)malloc(sizeof(double) * seq_len);
    decay_pows[0] = 1.0;
    for (long d = 1; d < seq_len; d++) decay_pows[d] = decay_pows[d - 1] * decay;
    tensor_fill(out, 0.0);
    long dim_even = (dim % 2 == 0) ? dim : dim - 1;
    for (long i = 0; i < seq_len; i++) {
        const double *q_i = &Q->data[i * dim];
        double *out_i = &out->data[i * dim];
        for (long j = 0; j <= i; j++) {
            const double *k_j = &K->data[j * dim];
            const double *v_j = &V->data[j * dim];
            double qk = (dim_even > 0) ? dot_neon64_core(q_i, k_j, dim_even) : 0.0;
            if (dim_even < dim) qk += q_i[dim - 1] * k_j[dim - 1];
            double weight = qk * decay_pows[i - j];
            if (dim_even > 0) accum_scale_neon64_core(out_i, v_j, weight, dim_even);
            if (dim_even < dim) out_i[dim - 1] += weight * v_j[dim - 1];
        }
    }
    free(decay_pows);
    return 0;
}

int retention_recurrent_step_fast(const RetentionConfig *cfg,
                                   Tensor *state,
                                   const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                   Tensor *out_t) {
    long dim = cfg->dim;
    double decay = cfg->decay;
    long dim_even = (dim % 2 == 0) ? dim : dim - 1;

    if (dim_even > 0) state_update_neon64_core(state->data, k_t->data, v_t->data, decay, dim, dim_even);
    if (dim_even < dim) {
        long b = dim - 1;
        for (long a = 0; a < dim; a++) {
            state->data[a * dim + b] = decay * state->data[a * dim + b] + k_t->data[a] * v_t->data[b];
        }
    }

    if (dim_even > 0) matmul_neon64_core(q_t->data, state->data, out_t->data, 1, dim, dim, dim_even);
    if (dim_even < dim) {
        long j = dim - 1;
        double sum = 0.0;
        for (long a = 0; a < dim; a++) sum += q_t->data[a] * state->data[a * dim + j];
        out_t->data[j] = sum;
    }
    return 0;
}

int retention_multihead_parallel_forward_fast(const MultiHeadRetentionConfig *cfg,
                                               const Tensor *Qh, const Tensor *Kh, const Tensor *Vh,
                                               Tensor *out_h) {
    long seq_len = cfg->seq_len, heads = cfg->num_heads, dim = cfg->head_dim;
    long head_stride = seq_len * dim;
    long dim_even = (dim % 2 == 0) ? dim : dim - 1;

    tensor_fill(out_h, 0.0);

    for (long h = 0; h < heads; h++) {
        double decay = cfg->decay_per_head[h];
        double *Qp = &Qh->data[h * head_stride];
        double *Kp = &Kh->data[h * head_stride];
        double *Vp = &Vh->data[h * head_stride];
        double *Op = &out_h->data[h * head_stride];

        double *decay_pows = (double *)malloc(sizeof(double) * seq_len);
        decay_pows[0] = 1.0;
        for (long d = 1; d < seq_len; d++) decay_pows[d] = decay_pows[d - 1] * decay;

        for (long i = 0; i < seq_len; i++) {
            double *q_i = &Qp[i * dim];
            double *o_i = &Op[i * dim];
            for (long j = 0; j <= i; j++) {
                double *k_j = &Kp[j * dim];
                double *v_j = &Vp[j * dim];
                double qk = (dim_even > 0) ? dot_neon64_core(q_i, k_j, dim_even) : 0.0;
                if (dim_even < dim) qk += q_i[dim - 1] * k_j[dim - 1];
                double weight = qk * decay_pows[i - j];
                if (dim_even > 0) accum_scale_neon64_core(o_i, v_j, weight, dim_even);
                if (dim_even < dim) o_i[dim - 1] += weight * v_j[dim - 1];
            }
        }
        free(decay_pows);
    }
    return 0;
}

int retention_multihead_recurrent_step_fast(const MultiHeadRetentionConfig *cfg,
                                             Tensor *state,
                                             const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                             Tensor *out_t) {
    long heads = cfg->num_heads, dim = cfg->head_dim;
    long state_stride = dim * dim, vec_stride = dim;
    long dim_even = (dim % 2 == 0) ? dim : dim - 1;

    for (long h = 0; h < heads; h++) {
        double decay = cfg->decay_per_head[h];
        double *state_h = &state->data[h * state_stride];
        double *q_h = &q_t->data[h * vec_stride];
        double *k_h = &k_t->data[h * vec_stride];
        double *v_h = &v_t->data[h * vec_stride];
        double *o_h = &out_t->data[h * vec_stride];

        if (dim_even > 0) state_update_neon64_core(state_h, k_h, v_h, decay, dim, dim_even);
        if (dim_even < dim) {
            long b = dim - 1;
            for (long a = 0; a < dim; a++) {
                state_h[a * dim + b] = decay * state_h[a * dim + b] + k_h[a] * v_h[b];
            }
        }

        if (dim_even > 0) matmul_neon64_core(q_h, state_h, o_h, 1, dim, dim, dim_even);
        if (dim_even < dim) {
            long j = dim - 1;
            double sum = 0.0;
            for (long a = 0; a < dim; a++) sum += q_h[a] * state_h[a * dim + j];
            o_h[j] = sum;
        }
    }
    return 0;
}

// ============================================================
#elif defined(ARCH_X86_64)
// ============================================================

extern void matmul_avx2_core(const double *A, const double *B, double *out,
                              long M, long K, long N_stride, long N_compute);
extern double sumsq_avx2_core(const double *x, long n);
extern void scale_gain_avx2_core(const double *x, const double *gain, double *out, long n, double inv_rms);
extern void state_update_avx2_core(double *state, const double *k, const double *v,
                                    long head_dim, long b_vec, double decay);
extern double dot_avx2_core(const double *a, const double *b, long n);
extern void accum_scale_avx2_core(double *out, const double *v, long n, double weight);

const char *dispatch_active_backend(void) { return "x86_64_avx2"; }

int tensor_matmul_fast(Tensor *out, const Tensor *A, const Tensor *B) {
    long M = A->shape[0], K = A->shape[1], N = B->shape[1];
    long n_vec = N - (N % 4);
    if (n_vec > 0) matmul_avx2_core(A->data, B->data, out->data, M, K, N, n_vec);
    for (long j = n_vec; j < N; j++) {
        for (long i = 0; i < M; i++) {
            double sum = 0.0;
            for (long k = 0; k < K; k++) sum += A->data[i * K + k] * B->data[k * N + j];
            out->data[i * N + j] = sum;
        }
    }
    return 0;
}

int rmsnorm_forward_fast(const Tensor *x, const Tensor *gain, Tensor *out, double eps) {
    long seq_len = x->shape[0], dim = x->shape[1];
    long dv = dim - (dim % 4);
    for (long i = 0; i < seq_len; i++) {
        const double *x_i = &x->data[i * dim];
        double *out_i = &out->data[i * dim];
        double sum_sq = (dv > 0) ? sumsq_avx2_core(x_i, dv) : 0.0;
        for (long d = dv; d < dim; d++) sum_sq += x_i[d] * x_i[d];
        double rms = sqrt(sum_sq / (double)dim + eps);
        double inv_rms = 1.0 / rms;
        if (dv > 0) scale_gain_avx2_core(x_i, gain->data, out_i, dv, inv_rms);
        for (long d = dv; d < dim; d++) out_i[d] = x_i[d] * inv_rms * gain->data[d];
    }
    return 0;
}

int retention_parallel_forward_fast(const RetentionConfig *cfg,
                                     const Tensor *Q, const Tensor *K, const Tensor *V,
                                     Tensor *out) {
    long seq_len = cfg->seq_len, dim = cfg->dim;
    double decay = cfg->decay;
    double *decay_pows = (double *)malloc(sizeof(double) * seq_len);
    decay_pows[0] = 1.0;
    for (long d = 1; d < seq_len; d++) decay_pows[d] = decay_pows[d - 1] * decay;
    tensor_fill(out, 0.0);
    long dv = dim - (dim % 4);
    for (long i = 0; i < seq_len; i++) {
        const double *q_i = &Q->data[i * dim];
        double *out_i = &out->data[i * dim];
        for (long j = 0; j <= i; j++) {
            const double *k_j = &K->data[j * dim];
            const double *v_j = &V->data[j * dim];
            double qk = (dv > 0) ? dot_avx2_core(q_i, k_j, dv) : 0.0;
            for (long d = dv; d < dim; d++) qk += q_i[d] * k_j[d];
            double weight = qk * decay_pows[i - j];
            if (dv > 0) accum_scale_avx2_core(out_i, v_j, dv, weight);
            for (long d = dv; d < dim; d++) out_i[d] += weight * v_j[d];
        }
    }
    free(decay_pows);
    return 0;
}

int retention_recurrent_step_fast(const RetentionConfig *cfg,
                                   Tensor *state,
                                   const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                   Tensor *out_t) {
    long dim = cfg->dim;
    double decay = cfg->decay;
    long dv = dim - (dim % 4);

    if (dv > 0) state_update_avx2_core(state->data, k_t->data, v_t->data, dim, dv, decay);
    for (long a = 0; a < dim; a++) {
        for (long b = dv; b < dim; b++) {
            state->data[a * dim + b] = decay * state->data[a * dim + b] + k_t->data[a] * v_t->data[b];
        }
    }

    if (dv > 0) matmul_avx2_core(q_t->data, state->data, out_t->data, 1, dim, dim, dv);
    for (long j = dv; j < dim; j++) {
        double sum = 0.0;
        for (long a = 0; a < dim; a++) sum += q_t->data[a] * state->data[a * dim + j];
        out_t->data[j] = sum;
    }
    return 0;
}

int retention_multihead_parallel_forward_fast(const MultiHeadRetentionConfig *cfg,
                                               const Tensor *Qh, const Tensor *Kh, const Tensor *Vh,
                                               Tensor *out_h) {
    long seq_len = cfg->seq_len, heads = cfg->num_heads, dim = cfg->head_dim;
    long head_stride = seq_len * dim;
    long dv = dim - (dim % 4);

    tensor_fill(out_h, 0.0);

    for (long h = 0; h < heads; h++) {
        double decay = cfg->decay_per_head[h];
        double *Qp = &Qh->data[h * head_stride];
        double *Kp = &Kh->data[h * head_stride];
        double *Vp = &Vh->data[h * head_stride];
        double *Op = &out_h->data[h * head_stride];

        double *decay_pows = (double *)malloc(sizeof(double) * seq_len);
        decay_pows[0] = 1.0;
        for (long d = 1; d < seq_len; d++) decay_pows[d] = decay_pows[d - 1] * decay;

        for (long i = 0; i < seq_len; i++) {
            double *q_i = &Qp[i * dim];
            double *o_i = &Op[i * dim];
            for (long j = 0; j <= i; j++) {
                double *k_j = &Kp[j * dim];
                double *v_j = &Vp[j * dim];
                double qk = (dv > 0) ? dot_avx2_core(q_i, k_j, dv) : 0.0;
                for (long d = dv; d < dim; d++) qk += q_i[d] * k_j[d];
                double weight = qk * decay_pows[i - j];
                if (dv > 0) accum_scale_avx2_core(o_i, v_j, dv, weight);
                for (long d = dv; d < dim; d++) o_i[d] += weight * v_j[d];
            }
        }
        free(decay_pows);
    }
    return 0;
}

int retention_multihead_recurrent_step_fast(const MultiHeadRetentionConfig *cfg,
                                             Tensor *state,
                                             const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                             Tensor *out_t) {
    long heads = cfg->num_heads, dim = cfg->head_dim;
    long state_stride = dim * dim, vec_stride = dim;
    long dv = dim - (dim % 4);

    for (long h = 0; h < heads; h++) {
        double decay = cfg->decay_per_head[h];
        double *state_h = &state->data[h * state_stride];
        double *q_h = &q_t->data[h * vec_stride];
        double *k_h = &k_t->data[h * vec_stride];
        double *v_h = &v_t->data[h * vec_stride];
        double *o_h = &out_t->data[h * vec_stride];

        if (dv > 0) state_update_avx2_core(state_h, k_h, v_h, dim, dv, decay);
        for (long a = 0; a < dim; a++) {
            for (long b = dv; b < dim; b++) {
                state_h[a * dim + b] = decay * state_h[a * dim + b] + k_h[a] * v_h[b];
            }
        }

        if (dv > 0) matmul_avx2_core(q_h, state_h, o_h, 1, dim, dim, dv);
        for (long j = dv; j < dim; j++) {
            double sum = 0.0;
            for (long a = 0; a < dim; a++) sum += q_h[a] * state_h[a * dim + j];
            o_h[j] = sum;
        }
    }
    return 0;
}

// ============================================================
#else
// ============================================================
// Fallback generic C -- dipakai kalau arch tidak dikenal (arm 32-bit, x86 32-bit, dll)

#include "retention.h"

const char *dispatch_active_backend(void) { return "generic"; }

int tensor_matmul_fast(Tensor *out, const Tensor *A, const Tensor *B) {
    return tensor_matmul(out, A, B);
}

int rmsnorm_forward_fast(const Tensor *x, const Tensor *gain, Tensor *out, double eps) {
    return rmsnorm_forward(x, gain, out, eps);
}

int retention_parallel_forward_fast(const RetentionConfig *cfg,
                                     const Tensor *Q, const Tensor *K, const Tensor *V,
                                     Tensor *out) {
    return retention_parallel_forward(cfg, Q, K, V, out);
}

int retention_recurrent_step_fast(const RetentionConfig *cfg,
                                   Tensor *state,
                                   const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                   Tensor *out_t) {
    return retention_recurrent_step(cfg, state, q_t, k_t, v_t, out_t);
}

int retention_multihead_parallel_forward_fast(const MultiHeadRetentionConfig *cfg,
                                               const Tensor *Qh, const Tensor *Kh, const Tensor *Vh,
                                               Tensor *out_h) {
    return retention_multihead_parallel_forward(cfg, Qh, Kh, Vh, out_h);
}

int retention_multihead_recurrent_step_fast(const MultiHeadRetentionConfig *cfg,
                                             Tensor *state,
                                             const Tensor *q_t, const Tensor *k_t, const Tensor *v_t,
                                             Tensor *out_t) {
    return retention_multihead_recurrent_step(cfg, state, q_t, k_t, v_t, out_t);
}

#endif
