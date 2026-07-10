// test_flash_retention.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention.h"
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 12
#define DIM 4
#define DECAY 0.9
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
    RetentionConfig cfg = { .seq_len = SEQ_LEN, .dim = DIM, .decay = DECAY };
    int shape[] = {SEQ_LEN, DIM};

    Tensor *Q = tensor_create(shape, 2, 0);
    Tensor *K = tensor_create(shape, 2, 0);
    Tensor *V = tensor_create(shape, 2, 0);
    Tensor *out_naive = tensor_create(shape, 2, 0);
    Tensor *out_flash = tensor_create(shape, 2, 0);

    tensor_fill_random(Q, -1.0, 1.0, 11);
    tensor_fill_random(K, -1.0, 1.0, 22);
    tensor_fill_random(V, -1.0, 1.0, 33);

    retention_parallel_forward(&cfg, Q, K, V, out_naive);
    tensor_print(out_naive, "Output naive (baseline)");

    int chunk_sizes[] = {1, 2, 3, 4, 6, 12};
    int num_tests = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
    int all_pass = 1;

    for (int t = 0; t < num_tests; t++) {
        int cs = chunk_sizes[t];
        retention_flash_forward(&cfg, Q, K, V, out_flash, cs);

        double diff = max_abs_diff(out_naive, out_flash);
        printf("\nChunk size = %2d | Max diff vs naive: %.12f", cs, diff);

        if (diff < EPS) {
            printf(" -> LOLOS");
        } else {
            printf(" -> GAGAL");
            all_pass = 0;
        }
    }

    printf("\n\n");
    if (all_pass) {
        printf("HASIL AKHIR: SEMUA CHUNK SIZE LOLOS — FlashRetention identik dengan forward pass biasa\n");
        printf("untuk semua variasi ukuran chunk (toleransi %.0e)\n", EPS);
    } else {
        printf("HASIL AKHIR: ADA CHUNK SIZE YANG GAGAL — cek ulang derivasi chunk-wise!\n");
    }

    tensor_free(Q); tensor_free(K); tensor_free(V);
    tensor_free(out_naive); tensor_free(out_flash);

    return all_pass ? 0 : 1;
}
