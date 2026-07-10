// test_pr_format.c
#include "tensor.h"
#include "model.h"
#include "tokenizer.h"
#include "pr_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("=== Test: Save & Load Format .prtm/.pr ===\n");

    const char *corpus = "pointretta model retention pointretta cepat pointretta hebat";
    BPETokenizer *tok = bpe_tokenizer_create(270);
    bpe_train(tok, (const unsigned char *)corpus, strlen(corpus));
    printf("Tokenizer dibuat: vocab_size=%d, merges=%d\n", tok->vocab_size, tok->num_merges);

    ModelConfig cfg = {
        .vocab_size = tok->vocab_size, .dim = 16, .num_heads = 4, .head_dim = 4,
        .ffn_hidden = 24, .num_layers = 2, .decay_min = 0.7, .decay_max = 0.95
    };
    Model *m1 = model_create(cfg, 999);
    printf("Model dibuat: vocab=%d dim=%d layers=%d\n\n", cfg.vocab_size, cfg.dim, cfg.num_layers);

    int token_ids[] = {5, 10, 15, 20, 25};
    int seq_len = 5;
    Tensor *logits_before = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);
    model_forward(m1, token_ids, seq_len, logits_before);

    const char *path = "test_model.pr";
    int save_rc = pr_file_save(path, m1, tok, 1);
    printf("pr_file_save return: %d (harusnya 0)\n", save_rc);

    Model *m2 = NULL;
    BPETokenizer *tok2 = NULL;
    int trained_flag = -1;
    int load_rc = pr_file_load(path, &m2, &tok2, &trained_flag);
    printf("pr_file_load return: %d (harusnya 0)\n", load_rc);
    printf("Trained flag: %d (harusnya 1)\n\n", trained_flag);

    printf("=== Verifikasi Config ===\n");
    int config_match = (m2->cfg.vocab_size == cfg.vocab_size) &&
                        (m2->cfg.dim == cfg.dim) &&
                        (m2->cfg.num_heads == cfg.num_heads) &&
                        (m2->cfg.num_layers == cfg.num_layers);
    printf("Config cocok: %s\n\n", config_match ? "YA" : "TIDAK");

    printf("=== Verifikasi Tokenizer ===\n");
    int tok_match = (tok2->vocab_size == tok->vocab_size) && (tok2->num_merges == tok->num_merges);
    for (int i = 0; i < tok->num_merges && tok_match; i++) {
        if (tok2->merges[i].left_id != tok->merges[i].left_id ||
            tok2->merges[i].right_id != tok->merges[i].right_id ||
            tok2->merges[i].new_id != tok->merges[i].new_id) {
            tok_match = 0;
        }
    }
    printf("Tokenizer cocok: %s\n\n", tok_match ? "YA" : "TIDAK");

    printf("=== Verifikasi Forward Pass Identik ===\n");
    Tensor *logits_after = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);
    model_forward(m2, token_ids, seq_len, logits_after);

    double max_diff = 0.0;
    for (size_t i = 0; i < logits_before->size; i++) {
        double diff = fabs(logits_before->data[i] - logits_after->data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff logits sebelum vs sesudah save/load: %.15f\n", max_diff);
    int forward_match = (max_diff < 1e-12);
    printf("Forward pass identik: %s\n\n", forward_match ? "YA" : "TIDAK");

    printf("=== Test Deteksi File Korup ===\n");
    FILE *f = fopen(path, "r+b");
    fseek(f, 20, SEEK_SET);
    unsigned char corrupt_byte = 0xFF;
    fwrite(&corrupt_byte, 1, 1, f);
    fclose(f);

    Model *m3 = NULL; BPETokenizer *tok3 = NULL; int flag3;
    int corrupt_load_rc = pr_file_load(path, &m3, &tok3, &flag3);
    printf("Load file korup return: %d (harusnya -1)\n", corrupt_load_rc);
    int corrupt_detect_pass = (corrupt_load_rc == -1);
    printf("Deteksi korup: %s\n\n", corrupt_detect_pass ? "LOLOS" : "GAGAL");

    int all_pass = (save_rc == 0) && (load_rc == 0) && config_match && tok_match &&
                   forward_match && corrupt_detect_pass;

    printf("=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA TEST LOLOS" : "ADA TEST GAGAL");

    tensor_free(logits_before); tensor_free(logits_after);
    model_free(m1); model_free(m2);
    bpe_tokenizer_free(tok); bpe_tokenizer_free(tok2);
    remove(path);

    return all_pass ? 0 : 1;
}
