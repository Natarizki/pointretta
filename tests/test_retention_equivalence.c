// test_retention_equivalence.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 6
#define DIM 4
#define DECAY 0.9
#define EPS 1e-9

int main(void) {
    RetentionConfig cfg = { .seq_len = SEQ_LEN, .dim = DIM, .decay = DECAY };

    int shape[] = {SEQ_LEN, DIM};
    Tensor *Q = tensor_create(shape, 2, 0);
    Tensor *K = tensor_create(shape, 2, 0);
    Tensor *V = tensor_create(shape, 2, 0);
    Tensor *out_parallel = tensor_create(shape, 2, 0);

    tensor_fill_random(Q, -1.0, 1.0, 42);
    tensor_fill_random(K, -1.0, 1.0, 123);
    tensor_fill_random(V, -1.0, 1.0, 7);

    int rc = retention_parallel_forward(&cfg, Q, K, V, out_parallel);
    printf("retention_parallel_forward return: %d (harusnya 0)\n", rc);
    tensor_print(out_parallel, "Output parallel");

    Tensor *state = retention_state_create(DIM);
    int shape_vec[] = {DIM};
    Tensor *q_t = tensor_create(shape_vec, 1, 0);
    Tensor *k_t = tensor_create(shape_vec, 1, 0);
    Tensor *v_t = tensor_create(shape_vec, 1, 0);
    Tensor *out_t = tensor_create(shape_vec, 1, 0);
    Tensor *out_recurrent = tensor_create(shape, 2, 0);

    for (int t = 0; t < SEQ_LEN; t++) {
        for (int d = 0; d < DIM; d++) {
            q_t->data[d] = Q->data[t * DIM + d];
            k_t->data[d] = K->data[t * DIM + d];
            v_t->data[d] = V->data[t * DIM + d];
        }

        int step_rc = retention_recurrent_step(&cfg, state, q_t, k_t, v_t, out_t);
        if (step_rc != 0) {
            printf("GAGAL di step %d\n", t);
            return 1;
        }

        for (int d = 0; d < DIM; d++) {
            out_recurrent->data[t * DIM + d] = out_t->data[d];
        }
    }

    tensor_print(out_recurrent, "Output recurrent");

    double max_diff = 0.0;
    for (size_t i = 0; i < out_parallel->size; i++) {
        double diff = fabs(out_parallel->data[i] - out_recurrent->data[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("\nMax perbedaan numerik: %.12f\n", max_diff);
    if (max_diff < EPS) {
        printf("HASIL: LOLOS — retention_parallel_forward == retention_recurrent_step (toleransi %.0e)\n", EPS);
    } else {
        printf("HASIL: GAGAL — ada bug di implementasi Tensor, cek ulang!\n");
    }

    tensor_free(Q); tensor_free(K); tensor_free(V);
    tensor_free(out_parallel); tensor_free(out_recurrent);
    tensor_free(state); tensor_free(q_t); tensor_free(k_t);
    tensor_free(v_t); tensor_free(out_t);

    return (max_diff < EPS) ? 0 : 1;
}
