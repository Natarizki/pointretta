// cmd_train.c
// Implementation of "pointretta train file.prtm dataset1 dataset2 ..."
// Loads .prtm -> reads dataset(s) -> extends the BPE tokenizer -> encodes ->
// runs the training loop -> saves as .pr

#include "cmd_train.h"
#include "tensor.h"
#include "model.h"
#include "train.h"
#include "optimizer.h"
#include "tokenizer.h"
#include "pr_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW_SIZE 32
#define TOTAL_ITERS 400
#define LEARNING_RATE 0.05

static char *read_file_to_string(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open dataset '%s'\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = (char *)malloc(size + 1);
    size_t read_size = fread(buf, 1, size, f);
    buf[read_size] = '\0';
    fclose(f);

    if (out_len) *out_len = read_size;
    return buf;
}

static void make_pr_path(const char *prtm_path, char *out_path, size_t out_size) {
    strncpy(out_path, prtm_path, out_size - 1);
    out_path[out_size - 1] = '\0';
    size_t len = strlen(out_path);
    if (len > 5 && strcmp(out_path + len - 5, ".prtm") == 0) {
        out_path[len - 5] = '\0';
        strncat(out_path, ".pr", out_size - strlen(out_path) - 1);
    } else {
        strncat(out_path, ".pr", out_size - strlen(out_path) - 1);
    }
}

int cmd_train(const char *prtm_path, char **dataset_paths, int num_datasets) {
    printf("PointRetta train\n");
    printf("  model:    %s\n", prtm_path);
    printf("  datasets: %d file(s)\n\n", num_datasets);

    Model *m = NULL;
    BPETokenizer *tok = NULL;
    int trained_flag = 0;
    if (pr_file_load(prtm_path, &m, &tok, &trained_flag) != 0) {
        fprintf(stderr, "Error: failed to load %s\n", prtm_path);
        return -1;
    }
    printf("Model loaded: vocab_size=%d dim=%d layers=%d (trained_flag=%d)\n\n",
           m->cfg.vocab_size, m->cfg.dim, m->cfg.num_layers, trained_flag);

    size_t total_corpus_len = 0;
    char **dataset_texts = (char **)malloc(sizeof(char *) * num_datasets);
    size_t *dataset_lens = (size_t *)malloc(sizeof(size_t) * num_datasets);

    for (int i = 0; i < num_datasets; i++) {
        dataset_texts[i] = read_file_to_string(dataset_paths[i], &dataset_lens[i]);
        if (!dataset_texts[i]) {
            fprintf(stderr, "Error: failed to read dataset %s\n", dataset_paths[i]);
            model_free(m); bpe_tokenizer_free(tok);
            free(dataset_texts); free(dataset_lens);
            return -1;
        }
        total_corpus_len += dataset_lens[i] + 1;
        printf("  Dataset %d: %s (%zu bytes)\n", i + 1, dataset_paths[i], dataset_lens[i]);
    }

    char *corpus = (char *)malloc(total_corpus_len + 1);
    size_t pos = 0;
    for (int i = 0; i < num_datasets; i++) {
        memcpy(corpus + pos, dataset_texts[i], dataset_lens[i]);
        pos += dataset_lens[i];
        corpus[pos++] = '\n';
        free(dataset_texts[i]);
    }
    corpus[pos] = '\0';
    size_t corpus_len = pos;

    free(dataset_texts);
    free(dataset_lens);

    printf("\nTotal corpus: %zu bytes\n\n", corpus_len);

    printf("=== Training BPE Tokenizer ===\n");
    int merges_done = bpe_train(tok, (const unsigned char *)corpus, corpus_len);
    printf("New merges: %d (total vocab now: %d)\n\n", merges_done, tok->vocab_size);

    int *all_ids = NULL;
    int all_count = 0;
    bpe_encode(tok, (const unsigned char *)corpus, corpus_len, &all_ids, &all_count);
    printf("Total tokens after encoding: %d\n\n", all_count);
    free(corpus);

    if (all_count < 2) {
        fprintf(stderr, "Error: dataset too small to train on (need at least 2 tokens)\n");
        model_free(m); bpe_tokenizer_free(tok); free(all_ids);
        return -1;
    }

    int window_size = all_count < WINDOW_SIZE ? all_count : WINDOW_SIZE;
    int num_windows = all_count / window_size;
    if (num_windows < 1) num_windows = 1;
    printf("Window size: %d, number of windows: %d\n\n", window_size, num_windows);

    printf("=== Training Loop ===\n");
    Tensor **params;
    int num_params;
    model_collect_params(m, &params, &num_params);

    AdafactorState **states = (AdafactorState **)malloc(sizeof(AdafactorState *) * num_params);
    for (int i = 0; i < num_params; i++) states[i] = adafactor_state_create(params[i], 0.999, 1e-30, 1e-3);

    ModelCache *cache = model_cache_create(m, window_size);
    Tensor *logits = tensor_create((int[]){window_size, m->cfg.vocab_size}, 2, 0);
    Tensor *dLogits = tensor_create((int[]){window_size, m->cfg.vocab_size}, 2, 0);

    int *target_ids = (int *)malloc(sizeof(int) * (window_size - 1));

    for (int iter = 0; iter < TOTAL_ITERS; iter++) {
        int w = iter % num_windows;
        int *window_ids = &all_ids[w * window_size];
        for (int i = 0; i < window_size - 1; i++) target_ids[i] = window_ids[i + 1];
        int count = window_size - 1;

        model_forward_train(m, window_ids, window_size, cache, logits);
        double loss = cross_entropy_loss(logits, target_ids, count, dLogits);

        if (iter % 50 == 0 || iter == TOTAL_ITERS - 1) {
            printf("  Iter %3d | Window %d | Loss: %.4f\n", iter, w, loss);
        }

        model_zero_grad(m);
        model_backward(m, window_ids, window_size, cache, dLogits, 1e-6);

        for (int i = 0; i < num_params; i++) adafactor_step(params[i], states[i], LEARNING_RATE);
    }

    char pr_path[512];
    make_pr_path(prtm_path, pr_path, sizeof(pr_path));

    int save_rc = pr_file_save(pr_path, m, tok, 1);
    printf("\n=== Done ===\n");
    if (save_rc == 0) {
        printf("Model saved: %s\n", pr_path);
    } else {
        fprintf(stderr, "Error: failed to save %s\n", pr_path);
    }

    for (int i = 0; i < num_params; i++) adafactor_state_free(states[i]);
    free(states); free(params);
    free(all_ids); free(target_ids);
    tensor_free(logits); tensor_free(dLogits);
    model_cache_free(cache);
    model_free(m);
    bpe_tokenizer_free(tok);

    return save_rc;
}
