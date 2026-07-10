// loss_neon_wrapper.c
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern double max_neon64_core(const double *x, long n_even);
extern double sum_neon64_core(const double *x, long n_even);
extern void vecscale_neon64_core(const double *x, double scalar, double *out, long n_even);

static double max_neon(const double *x, long n) {
    long n_even = (n % 2 == 0) ? n : n - 1;
    double result;
    if (n_even >= 2) {
        result = max_neon64_core(x, n_even);
        if (n_even < n && x[n - 1] > result) result = x[n - 1];
    } else {
        result = x[0];
        for (long i = 1; i < n; i++) if (x[i] > result) result = x[i];
    }
    return result;
}

static double sum_neon(const double *x, long n) {
    long n_even = (n % 2 == 0) ? n : n - 1;
    double result;
    if (n_even >= 2) {
        result = sum_neon64_core(x, n_even);
        if (n_even < n) result += x[n - 1];
    } else {
        result = 0.0;
        for (long i = 0; i < n; i++) result += x[i];
    }
    return result;
}

static void vecscale_neon(const double *x, double scalar, double *out, long n) {
    long n_even = (n % 2 == 0) ? n : n - 1;
    if (n_even >= 2) {
        vecscale_neon64_core(x, scalar, out, n_even);
        if (n_even < n) out[n - 1] = x[n - 1] * scalar;
    } else {
        for (long i = 0; i < n; i++) out[i] = x[i] * scalar;
    }
}

double cross_entropy_loss_neon(const Tensor *logits, const int *target_ids, int count,
                                Tensor *dLogits_out) {
    if (!logits || !target_ids || !dLogits_out) return -1.0;

    long seq_len = logits->shape[0];
    long vocab_size = logits->shape[1];

    tensor_fill(dLogits_out, 0.0);

    double total_loss = 0.0;
    double *probs = (double *)malloc(sizeof(double) * vocab_size);
    double *shifted = (double *)malloc(sizeof(double) * vocab_size);

    for (long i = 0; i < count && i < seq_len; i++) {
        double *logits_i = &logits->data[i * vocab_size];
        double *dlogits_i = &dLogits_out->data[i * vocab_size];

        double max_val = max_neon(logits_i, vocab_size);

        for (long v = 0; v < vocab_size; v++) {
            shifted[v] = exp(logits_i[v] - max_val);
        }

        double sum_exp = sum_neon(shifted, vocab_size);
        double inv_sum = 1.0 / sum_exp;
        vecscale_neon(shifted, inv_sum, probs, vocab_size);

        int target = target_ids[i];
        double loss_i = -log(probs[target] + 1e-12);
        total_loss += loss_i;

        for (long v = 0; v < vocab_size; v++) {
            double one_hot = (v == target) ? 1.0 : 0.0;
            dlogits_i[v] = (probs[v] - one_hot) / (double)count;
        }
    }

    free(probs);
    free(shifted);
    return total_loss / (double)count;
}
