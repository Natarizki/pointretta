// test_model.c
#include "tensor.h"
#include "tensor_ops.h"
#include "model.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    printf("=== Test 1: split_heads / merge_heads round-trip ===\n");
    int seq_len = 4, num_heads = 2, head_dim = 3, dim = num_heads * head_dim;

    int shape2d[] = {seq_len, dim};
    Tensor *x = tensor_create(shape2d, 2, 0);
    tensor_fill_random(x, -1.0, 1.0, 99);

    int shape3d[] = {num_heads, seq_len, head_dim};
    Tensor *xh = tensor_create(shape3d, 3, 0);
    Tensor *x_back = tensor_create(shape2d, 2, 0);

    int rc1 = split_heads(x, num_heads, head_dim, xh);
    int rc2 = merge_heads(xh, num_heads, head_dim, x_back);
    printf("split_heads return: %d, merge_heads return: %d (harusnya 0, 0)\n", rc1, rc2);

    double max_diff = 0.0;
    for (size_t i = 0; i < x->size; i++) {
        double diff = fabs(x->data[i] - x_back->data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff round-trip: %.12f (harusnya 0.0)\n", max_diff);
    int roundtrip_pass = (max_diff < 1e-12);
    printf("Round-trip: %s\n\n", roundtrip_pass ? "LOLOS" : "GAGAL");

    tensor_free(x); tensor_free(xh); tensor_free(x_back);

    printf("=== Test 2: Model create + forward pass (smoke test) ===\n");
    ModelConfig cfg = {
        .vocab_size = 300,
        .dim = 16,
        .num_heads = 4,
        .head_dim = 4,
        .ffn_hidden = 32,
        .num_layers = 3,
        .decay_min = 0.7,
        .decay_max = 0.99
    };

    Model *m = model_create(cfg, 42);
    if (!m) {
        printf("GAGAL: model_create return NULL\n");
        return 1;
    }
    printf("Model berhasil dibuat: vocab=%d dim=%d heads=%d head_dim=%d layers=%d\n",
           cfg.vocab_size, cfg.dim, cfg.num_heads, cfg.head_dim, cfg.num_layers);

    int token_ids[] = {5, 10, 15, 20, 25, 30};
    int test_seq_len = 6;

    int shape_logits[] = {test_seq_len, cfg.vocab_size};
    Tensor *logits = tensor_create(shape_logits, 2, 0);

    int rc = model_forward(m, token_ids, test_seq_len, logits);
    printf("model_forward return: %d (harusnya 0)\n", rc);

    printf("Shape logits: [%d, %d] (harusnya [%d, %d])\n",
           logits->shape[0], logits->shape[1], test_seq_len, cfg.vocab_size);

    int has_nan_or_inf = 0;
    double min_val = 1e18, max_val = -1e18;
    for (size_t i = 0; i < logits->size; i++) {
        double v = logits->data[i];
        if (isnan(v) || isinf(v)) has_nan_or_inf = 1;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    printf("Range logits: [%.4f, %.4f]\n", min_val, max_val);
    printf("Ada NaN/Inf: %s (harusnya TIDAK)\n", has_nan_or_inf ? "YA" : "TIDAK");

    int shape_pass = (logits->shape[0] == test_seq_len && logits->shape[1] == cfg.vocab_size);
    int forward_pass = (rc == 0) && shape_pass && !has_nan_or_inf;
    printf("Forward pass smoke test: %s\n\n", forward_pass ? "LOLOS" : "GAGAL");

    printf("=== Test 3: Forward pass dua kali dengan token sama -> hasil harus identik ===\n");
    Tensor *logits2 = tensor_create(shape_logits, 2, 0);
    model_forward(m, token_ids, test_seq_len, logits2);

    double max_diff_deterministic = 0.0;
    for (size_t i = 0; i < logits->size; i++) {
        double diff = fabs(logits->data[i] - logits2->data[i]);
        if (diff > max_diff_deterministic) max_diff_deterministic = diff;
    }
    printf("Max diff antara 2 forward pass (input sama): %.12f (harusnya 0.0)\n", max_diff_deterministic);
    int deterministic_pass = (max_diff_deterministic < 1e-12);
    printf("Determinism: %s\n\n", deterministic_pass ? "LOLOS" : "GAGAL");

    int all_pass = roundtrip_pass && forward_pass && deterministic_pass;
    printf("=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA TEST LOLOS" : "ADA TEST GAGAL");

    tensor_free(logits); tensor_free(logits2);
    model_free(m);

    return all_pass ? 0 : 1;
}
