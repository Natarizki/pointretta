// pr_file.c
#include "pr_format.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define PR_MAGIC "PRTA"
#define PR_VERSION 1

static uint64_t fnv1a(const unsigned char *data, size_t len) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static int write_i32(FILE *f, int v) { return fwrite(&v, sizeof(int), 1, f) == 1 ? 0 : -1; }
static int write_u8(FILE *f, unsigned char v) { return fwrite(&v, 1, 1, f) == 1 ? 0 : -1; }
static int write_double(FILE *f, double v) { return fwrite(&v, sizeof(double), 1, f) == 1 ? 0 : -1; }
static int write_doubles(FILE *f, const double *arr, size_t n) {
    return fwrite(arr, sizeof(double), n, f) == n ? 0 : -1;
}

static int read_i32(FILE *f, int *v) { return fread(v, sizeof(int), 1, f) == 1 ? 0 : -1; }
static int read_u8(FILE *f, unsigned char *v) { return fread(v, 1, 1, f) == 1 ? 0 : -1; }
static int read_double(FILE *f, double *v) { return fread(v, sizeof(double), 1, f) == 1 ? 0 : -1; }
static int read_doubles(FILE *f, double *arr, size_t n) {
    return fread(arr, sizeof(double), n, f) == n ? 0 : -1;
}

int pr_file_save(const char *path, const Model *m, const BPETokenizer *tok, int trained_flag) {
    if (!path || !m || !tok) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[pr_file_save] gagal buka file %s\n", path); return -1; }

    int rc = 0;
    rc |= fwrite(PR_MAGIC, 4, 1, f) == 1 ? 0 : -1;
    rc |= write_i32(f, PR_VERSION);
    rc |= write_u8(f, (unsigned char)trained_flag);

    rc |= write_i32(f, m->cfg.vocab_size);
    rc |= write_i32(f, m->cfg.dim);
    rc |= write_i32(f, m->cfg.num_heads);
    rc |= write_i32(f, m->cfg.head_dim);
    rc |= write_i32(f, m->cfg.ffn_hidden);
    rc |= write_i32(f, m->cfg.num_layers);
    rc |= write_double(f, m->cfg.decay_min);
    rc |= write_double(f, m->cfg.decay_max);

    rc |= write_i32(f, tok->vocab_size);
    rc |= write_i32(f, tok->num_merges);
    for (int i = 0; i < tok->vocab_size; i++) {
        rc |= write_u8(f, (unsigned char)tok->vocab[i].len);
        rc |= fwrite(tok->vocab[i].bytes, 1, tok->vocab[i].len, f) == (size_t)tok->vocab[i].len ? 0 : -1;
    }
    for (int i = 0; i < tok->num_merges; i++) {
        rc |= write_i32(f, tok->merges[i].left_id);
        rc |= write_i32(f, tok->merges[i].right_id);
        rc |= write_i32(f, tok->merges[i].new_id);
    }

    rc |= write_doubles(f, m->embedding->data, m->embedding->size);
    rc |= write_doubles(f, m->output_head->data, m->output_head->size);
    rc |= write_doubles(f, m->final_norm_gain->data, m->final_norm_gain->size);

    for (int l = 0; l < m->cfg.num_layers; l++) {
        ModelLayer *layer = &m->layers[l];
        rc |= write_doubles(f, layer->Wq->data, layer->Wq->size);
        rc |= write_doubles(f, layer->Wk->data, layer->Wk->size);
        rc |= write_doubles(f, layer->Wv->data, layer->Wv->size);
        rc |= write_doubles(f, layer->Wo->data, layer->Wo->size);
        rc |= write_doubles(f, layer->norm1_gain->data, layer->norm1_gain->size);
        rc |= write_doubles(f, layer->norm2_gain->data, layer->norm2_gain->size);
        rc |= write_doubles(f, layer->W1->data, layer->W1->size);
        rc |= write_doubles(f, layer->W3->data, layer->W3->size);
        rc |= write_doubles(f, layer->W2->data, layer->W2->size);
        rc |= write_doubles(f, layer->decay_per_head, m->cfg.num_heads);
    }

    fclose(f);
    if (rc != 0) { fprintf(stderr, "[pr_file_save] gagal menulis data\n"); return -1; }

    f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(file_size);
    fread(buf, 1, file_size, f);
    fclose(f);

    uint64_t checksum = fnv1a(buf, file_size);
    free(buf);

    f = fopen(path, "ab");
    if (!f) return -1;
    fwrite(&checksum, sizeof(uint64_t), 1, f);
    fclose(f);

    return 0;
}

int pr_file_load(const char *path, Model **out_model, BPETokenizer **out_tok, int *out_trained_flag) {
    if (!path || !out_model || !out_tok) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[pr_file_load] file %s tidak ditemukan\n", path); return -1; }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size < 8) { fclose(f); return -1; }

    long payload_size = file_size - (long)sizeof(uint64_t);
    unsigned char *buf = (unsigned char *)malloc(payload_size);
    fread(buf, 1, payload_size, f);
    uint64_t stored_checksum;
    fread(&stored_checksum, sizeof(uint64_t), 1, f);

    uint64_t computed_checksum = fnv1a(buf, payload_size);
    free(buf);

    if (computed_checksum != stored_checksum) {
        fprintf(stderr, "[pr_file_load] checksum tidak cocok -- file korup atau rusak\n");
        fclose(f);
        return -1;
    }

    fseek(f, 0, SEEK_SET);
    char magic[5] = {0};
    fread(magic, 4, 1, f);
    if (strcmp(magic, PR_MAGIC) != 0) {
        fprintf(stderr, "[pr_file_load] magic bytes tidak cocok, bukan file PointRetta\n");
        fclose(f);
        return -1;
    }

    int version;
    read_i32(f, &version);
    if (version != PR_VERSION) {
        fprintf(stderr, "[pr_file_load] versi format %d tidak didukung (support: %d)\n", version, PR_VERSION);
        fclose(f);
        return -1;
    }

    unsigned char trained_flag;
    read_u8(f, &trained_flag);
    if (out_trained_flag) *out_trained_flag = (int)trained_flag;

    ModelConfig cfg;
    read_i32(f, &cfg.vocab_size);
    read_i32(f, &cfg.dim);
    read_i32(f, &cfg.num_heads);
    read_i32(f, &cfg.head_dim);
    read_i32(f, &cfg.ffn_hidden);
    read_i32(f, &cfg.num_layers);
    read_double(f, &cfg.decay_min);
    read_double(f, &cfg.decay_max);

    int tok_vocab_size, tok_num_merges;
    read_i32(f, &tok_vocab_size);
    read_i32(f, &tok_num_merges);

    int tok_capacity = cfg.vocab_size > tok_vocab_size ? cfg.vocab_size : tok_vocab_size;
    BPETokenizer *tok = (BPETokenizer *)malloc(sizeof(BPETokenizer));
    tok->vocab_capacity = tok_capacity;
    tok->vocab_size = tok_vocab_size;
    tok->vocab = (BPEToken *)malloc(sizeof(BPEToken) * tok_capacity);
    for (int i = 0; i < tok_vocab_size; i++) {
        unsigned char len;
        read_u8(f, &len);
        tok->vocab[i].len = len;
        fread(tok->vocab[i].bytes, 1, len, f);
    }

    int merges_capacity = tok_capacity - 256;
    if (merges_capacity < 1) merges_capacity = 1;
    tok->merges_capacity = merges_capacity;
    tok->num_merges = tok_num_merges;
    tok->merges = (BPEMergeRule *)malloc(sizeof(BPEMergeRule) * tok->merges_capacity);
    for (int i = 0; i < tok_num_merges; i++) {
        read_i32(f, &tok->merges[i].left_id);
        read_i32(f, &tok->merges[i].right_id);
        read_i32(f, &tok->merges[i].new_id);
    }

    Model *m = model_create(cfg, 1);
    if (!m) {
        fprintf(stderr, "[pr_file_load] gagal alokasi model\n");
        fclose(f);
        bpe_tokenizer_free(tok);
        return -1;
    }

    read_doubles(f, m->embedding->data, m->embedding->size);
    read_doubles(f, m->output_head->data, m->output_head->size);
    read_doubles(f, m->final_norm_gain->data, m->final_norm_gain->size);

    for (int l = 0; l < cfg.num_layers; l++) {
        ModelLayer *layer = &m->layers[l];
        read_doubles(f, layer->Wq->data, layer->Wq->size);
        read_doubles(f, layer->Wk->data, layer->Wk->size);
        read_doubles(f, layer->Wv->data, layer->Wv->size);
        read_doubles(f, layer->Wo->data, layer->Wo->size);
        read_doubles(f, layer->norm1_gain->data, layer->norm1_gain->size);
        read_doubles(f, layer->norm2_gain->data, layer->norm2_gain->size);
        read_doubles(f, layer->W1->data, layer->W1->size);
        read_doubles(f, layer->W3->data, layer->W3->size);
        read_doubles(f, layer->W2->data, layer->W2->size);
        read_doubles(f, layer->decay_per_head, cfg.num_heads);
    }

    fclose(f);

    *out_model = m;
    *out_tok = tok;
    return 0;
}
