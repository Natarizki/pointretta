// test_retention_multihead.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention_multihead.h"
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 8
#define NUM_HEADS 4
#define HEAD_DIM 4
#define EPS 1e-9

static double max_abs_diff(const Tensor *a, const Tensor *b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a->size; i++) {
        double diff = fabs(a->data[i] - b->data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

int main(void) {
    double *decays = make_multiscale_decay(NUM_HEADS, 0.7, 0.99);
    printf("Decay per head: ");
    for (int h = 0; h < NUM_HEADS; h++) printf("%.4f ", decays[h]);
    printf("\n\n");

    MultiHeadRetentionConfig cfg = {
        .seq_len = SEQ_LEN,
        .num_heads = NUM_HEADS,
        .head_dim = HEAD_DIM,
        .decay_per_head = decays
    };

    int shape[] = {NUM_HEADS, SEQ_LEN, HEAD_DIM};
    Tensor *Q = tensor_create(shape, 3, 0);
    Tensor *K = tensor_create(shape, 3, 0);
    Tensor *V = tensor_create(shape, 3, 0);
    Tensor *out_parallel = tensor_create(shape, 3, 0);

    tensor_fill_random(Q, -1.0, 1.0, 42);
    tensor_fill_random(K, -1.0, 1.0, 123);
    tensor_fill_random(V, -1.0, 1.0, 7);

    int rc = retention_multihead_parallel_forward(&cfg, Q, K, V, out_parallel);
    printf("retention_multihead_parallel_forward return: %d (harusnya 0)\n", rc);

    Tensor *state = retention_multihead_state_create(NUM_HEADS, HEAD_DIM);
    int vec_size = NUM_HEADS * HEAD_DIM;
    int shape_vec[] = {vec_size};
    Tensor *q_t = tensor_create(shape_vec, 1, 0);
    Tensor *k_t = tensor_create(shape_vec, 1, 0);
    Tensor *v_t = tensor_create(shape_vec, 1, 0);
    Tensor *out_t = tensor_create(shape_vec, 1, 0);
    Tensor *out_recurrent = tensor_create(shape, 3, 0);

    for (int t = 0; t < SEQ_LEN; t++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int src_idx = h * (SEQ_LEN * HEAD_DIM) + t * HEAD_DIM + d;
                int dst_idx = h * HEAD_DIM + d;
                q_t->data[dst_idx] = Q->data[src_idx];
                k_t->data[dst_idx] = K->data[src_idx];
                v_t->data[dst_idx] = V->data[src_idx];
            }
        }

        int step_rc = retention_multihead_recurrent_step(&cfg, state, q_t, k_t, v_t, out_t);
        if (step_rc != 0) {
            printf("GAGAL di step %d\n", t);
            return 1;
        }

        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int dst_idx = h * (SEQ_LEN * HEAD_DIM) + t * HEAD_DIM + d;
                int src_idx = h * HEAD_DIM + d;
                out_recurrent->data[dst_idx] = out_t->data[src_idx];
            }
        }
    }

    double diff = max_abs_diff(out_parallel, out_recurrent);
    printf("\nMax perbedaan numerik: %.12f\n", diff);

    if (diff < EPS) {
        printf("HASIL: LOLOS — multi-head parallel == recurrent (versi Tensor, toleransi %.0e)\n", EPS);
    } else {
        printf("HASIL: GAGAL — cek ulang implementasi!\n");
    }

    free(decays);
    tensor_free(Q); tensor_free(K); tensor_free(V);
    tensor_free(out_parallel); tensor_free(out_recurrent);
    tensor_free(state); tensor_free(q_t); tensor_free(k_t);
    tensor_free(v_t); tensor_free(out_t);

    return (diff < EPS) ? 0 : 1;
}
