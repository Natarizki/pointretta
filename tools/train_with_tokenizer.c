// train_with_tokenizer.c
#include "tensor.h"
#include "model.h"
#include "train.h"
#include "optimizer.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    const char *corpus_text =
        "pointretta adalah model retention pointretta sangat cepat "
        "pointretta belajar bahasa indonesia dengan pointretta "
        "retention lebih cepat dari attention biasa pointretta hebat "
        "model kecil bisa belajar pola sederhana dengan cepat";
    size_t corpus_len = strlen(corpus_text);

    printf("=== Langkah 1: Training BPE Tokenizer ===\n");
    printf("Corpus (%zu byte): \"%s\"\n\n", corpus_len, corpus_text);

    BPETokenizer *tok = bpe_tokenizer_create(280);
    int merges = bpe_train(tok, (const unsigned char *)corpus_text, corpus_len);
    printf("Vocab size aktual: %d (merge: %d)\n\n", tok->vocab_size, merges);

    printf("=== Langkah 2: Encode Corpus ===\n");
    int *all_ids = NULL;
    int all_count = 0;
    bpe_encode(tok, (const unsigned char *)corpus_text, corpus_len, &all_ids, &all_count);
    printf("Total token hasil encode: %d\n", all_count);

    int seq_len = all_count < 24 ? all_count : 24;
    printf("Dipakai buat training: %d token pertama\n\n", seq_len);

    printf("=== Langkah 3: Buat Model ===\n");
    ModelConfig cfg = {
        .vocab_size = tok->vocab_size,
        .dim = 32, .num_heads = 4, .head_dim = 8,
        .ffn_hidden = 64, .num_layers = 2,
        .decay_min = 0.7, .decay_max = 0.95
    };
    Model *m = model_create(cfg, 2026);
    printf("Model: vocab=%d dim=%d heads=%d layers=%d\n\n", cfg.vocab_size, cfg.dim, cfg.num_heads, cfg.num_layers);

    int *target_ids = (int *)malloc(sizeof(int) * (seq_len - 1));
    for (int i = 0; i < seq_len - 1; i++) target_ids[i] = all_ids[i + 1];
    int count = seq_len - 1;

    printf("=== Langkah 4: Training Loop ===\n");
    Tensor **params;
    int num_params;
    model_collect_params(m, &params, &num_params);

    AdafactorState **states = (AdafactorState **)malloc(sizeof(AdafactorState *) * num_params);
    for (int i = 0; i < num_params; i++) states[i] = adafactor_state_create(params[i], 0.999, 1e-30, 1e-3);

    ModelCache *cache = model_cache_create(m, seq_len);
    Tensor *logits = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);
    Tensor *dLogits = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);

    double lr = 0.05;
    int num_iters = 400;

    for (int iter = 0; iter < num_iters; iter++) {
        model_forward_train(m, all_ids, seq_len, cache, logits);
        double loss = cross_entropy_loss(logits, target_ids, count, dLogits);

        if (iter % 50 == 0 || iter == num_iters - 1) {
            printf("  Iter %3d | Loss: %.4f\n", iter, loss);
        }

        model_zero_grad(m);
        model_backward(m, all_ids, seq_len, cache, dLogits, 1e-6);

        for (int i = 0; i < num_params; i++) adafactor_step(params[i], states[i], lr);
    }

    printf("\n=== Langkah 5: Demo Prediksi (greedy argmax) ===\n");
    model_forward_train(m, all_ids, seq_len, cache, logits);

    printf("Perbandingan token asli vs prediksi model:\n");
    for (int i = 0; i < seq_len - 1; i++) {
        double *logits_i = &logits->data[i * cfg.vocab_size];
        int best_id = 0;
        double best_val = logits_i[0];
        for (int v = 1; v < cfg.vocab_size; v++) {
            if (logits_i[v] > best_val) { best_val = logits_i[v]; best_id = v; }
        }

        unsigned char *actual_text = NULL; size_t actual_len = 0;
        unsigned char *pred_text = NULL; size_t pred_len = 0;
        bpe_decode(tok, &target_ids[i], 1, &actual_text, &actual_len);
        bpe_decode(tok, &best_id, 1, &pred_text, &pred_len);

        printf("  posisi %2d: asli=\"", i);
        for (size_t k = 0; k < actual_len; k++) putchar(actual_text[k]);
        printf("\" prediksi=\"");
        for (size_t k = 0; k < pred_len; k++) putchar(pred_text[k]);
        printf("\" %s\n", (best_id == target_ids[i]) ? "[COCOK]" : "");

        free(actual_text); free(pred_text);
    }

    int correct = 0;
    for (int i = 0; i < seq_len - 1; i++) {
        double *logits_i = &logits->data[i * cfg.vocab_size];
        int best_id = 0;
        double best_val = logits_i[0];
        for (int v = 1; v < cfg.vocab_size; v++) {
            if (logits_i[v] > best_val) { best_val = logits_i[v]; best_id = v; }
        }
        if (best_id == target_ids[i]) correct++;
    }
    printf("\nAkurasi next-token prediction (data training): %d/%d (%.1f%%)\n",
           correct, seq_len - 1, 100.0 * correct / (seq_len - 1));

    for (int i = 0; i < num_params; i++) adafactor_state_free(states[i]);
    free(states); free(params);
    free(all_ids); free(target_ids);
    tensor_free(logits); tensor_free(dLogits);
    model_cache_free(cache);
    model_free(m);
    bpe_tokenizer_free(tok);

    return 0;
}
