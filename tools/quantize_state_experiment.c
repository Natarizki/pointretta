// quantize_state_experiment.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 8
#define DECAY 0.95
#define NUM_SEEDS 5

static void quantize_dequantize_int8(double *arr, int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(arr[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-12) return;

    double scale = max_abs / 127.0;

    for (int i = 0; i < n; i++) {
        int q = (int)round(arr[i] / scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        arr[i] = q * scale;
    }
}

static void run_recurrent_quantized(RetentionConfig *cfg, Tensor *Q, Tensor *K, Tensor *V,
                                     Tensor *out_recurrent, int quantize_every_step) {
    int dim = cfg->dim;
    Tensor *state = retention_state_create(dim);
    int shape_vec[] = {dim};
    Tensor *q_t = tensor_create(shape_vec, 1, 0);
    Tensor *k_t = tensor_create(shape_vec, 1, 0);
    Tensor *v_t = tensor_create(shape_vec, 1, 0);
    Tensor *out_t = tensor_create(shape_vec, 1, 0);

    for (int t = 0; t < cfg->seq_len; t++) {
        for (int d = 0; d < dim; d++) {
            q_t->data[d] = Q->data[t * dim + d];
            k_t->data[d] = K->data[t * dim + d];
            v_t->data[d] = V->data[t * dim + d];
        }

        retention_recurrent_step(cfg, state, q_t, k_t, v_t, out_t);

        if (quantize_every_step) {
            quantize_dequantize_int8(state->data, dim * dim);
        }

        for (int d = 0; d < dim; d++) {
            out_recurrent->data[t * dim + d] = out_t->data[d];
        }
    }

    tensor_free(state); tensor_free(q_t); tensor_free(k_t);
    tensor_free(v_t); tensor_free(out_t);
}

int main(void) {
    int test_lengths[] = {8, 16, 32, 64, 128, 256};
    int num_lengths = sizeof(test_lengths) / sizeof(test_lengths[0]);

    printf("=== Eksperimen Drift Kuantisasi State Retention (int8, per-step) ===\n");
    printf("DIM=%d, DECAY=%.2f, rata-rata dari %d seed random\n\n", DIM, DECAY, NUM_SEEDS);
    printf("%-10s | %-20s | %-20s\n", "SeqLen", "Avg Max Diff", "Avg Relative Diff (%)");
    printf("--------------------------------------------------------------\n");

    for (int li = 0; li < num_lengths; li++) {
        int seq_len = test_lengths[li];
        RetentionConfig cfg = { .seq_len = seq_len, .dim = DIM, .decay = DECAY };

        double total_max_diff = 0.0;
        double total_rel_diff = 0.0;

        for (int seed = 0; seed < NUM_SEEDS; seed++) {
            int shape[] = {seq_len, DIM};
            Tensor *Q = tensor_create(shape, 2, 0);
            Tensor *K = tensor_create(shape, 2, 0);
            Tensor *V = tensor_create(shape, 2, 0);
            Tensor *out_fp64 = tensor_create(shape, 2, 0);
            Tensor *out_int8 = tensor_create(shape, 2, 0);

            tensor_fill_random(Q, -1.0, 1.0, 100 + seed);
            tensor_fill_random(K, -1.0, 1.0, 200 + seed);
            tensor_fill_random(V, -1.0, 1.0, 300 + seed);

            run_recurrent_quantized(&cfg, Q, K, V, out_fp64, 0);
            run_recurrent_quantized(&cfg, Q, K, V, out_int8, 1);

            double max_diff = 0.0;
            double sum_abs_fp64 = 0.0;
            for (size_t i = 0; i < out_fp64->size; i++) {
                double diff = fabs(out_fp64->data[i] - out_int8->data[i]);
                if (diff > max_diff) max_diff = diff;
                sum_abs_fp64 += fabs(out_fp64->data[i]);
            }
            double avg_magnitude = sum_abs_fp64 / out_fp64->size;
            double rel_diff = (avg_magnitude > 1e-9) ? (max_diff / avg_magnitude) * 100.0 : 0.0;

            total_max_diff += max_diff;
            total_rel_diff += rel_diff;

            tensor_free(Q); tensor_free(K); tensor_free(V);
            tensor_free(out_fp64); tensor_free(out_int8);
        }

        printf("%-10d | %-20.6f | %-20.2f\n",
               seq_len, total_max_diff / NUM_SEEDS, total_rel_diff / NUM_SEEDS);
    }

    printf("\nCatatan: kalau 'Avg Max Diff' naik signifikan seiring SeqLen membesar,\n");
    printf("itu bukti error kuantisasi NUMPUK sepanjang sequence (hipotesis awal kita).\n");
    printf("Kalau relatif stabil/flat, artinya quantization int8 di state cukup aman\n");
    printf("bahkan untuk sequence panjang.\n");

    return 0;
}
