// quantize_decay_experiment.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 8
#define SEQ_LEN 256
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
    double decay_values[] = {0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999};
    int num_decays = sizeof(decay_values) / sizeof(decay_values[0]);

    printf("=== Eksperimen: Pengaruh Decay Rate terhadap Drift Kuantisasi State ===\n");
    printf("DIM=%d, SEQ_LEN=%d (sudah masuk plateau), rata-rata %d seed\n\n", DIM, SEQ_LEN, NUM_SEEDS);
    printf("%-12s | %-20s | %-20s\n", "Decay", "Avg Max Diff", "Avg Relative Diff (%)");
    printf("--------------------------------------------------------------\n");

    for (int di = 0; di < num_decays; di++) {
        double decay = decay_values[di];
        RetentionConfig cfg = { .seq_len = SEQ_LEN, .dim = DIM, .decay = decay };

        double total_max_diff = 0.0;
        double total_rel_diff = 0.0;

        for (int seed = 0; seed < NUM_SEEDS; seed++) {
            int shape[] = {SEQ_LEN, DIM};
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

        printf("%-12.3f | %-20.6f | %-20.2f\n",
               decay, total_max_diff / NUM_SEEDS, total_rel_diff / NUM_SEEDS);
    }

    printf("\nInterpretasi:\n");
    printf("- Kalau relative diff MENINGKAT signifikan seiring decay mendekati 1.0,\n");
    printf("  berarti head 'ingat jangka panjang' butuh precision lebih tinggi\n");
    printf("  (state tidak boleh di-quantize seagresif head jangka pendek).\n");
    printf("- Kalau relatif FLAT di semua decay, quantization int8 di state\n");
    printf("  aman dipakai seragam untuk semua head.\n");

    return 0;
}
