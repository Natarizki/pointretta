// bpe_encode.c
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bpe_encode(const BPETokenizer *tok, const unsigned char *text, size_t text_len,
               int **out_ids, int *out_count) {
    if (!tok || !text || !out_ids || !out_count) return -1;
    if (text_len == 0) {
        *out_ids = NULL;
        *out_count = 0;
        return 0;
    }

    int *seq = (int *)malloc(sizeof(int) * text_len);
    if (!seq) return -1;
    size_t seq_len = text_len;
    for (size_t i = 0; i < text_len; i++) {
        seq[i] = (int)text[i];
    }

    int *new_seq = (int *)malloc(sizeof(int) * text_len);
    if (!new_seq) { free(seq); return -1; }

    for (int m = 0; m < tok->num_merges; m++) {
        int left_id = tok->merges[m].left_id;
        int right_id = tok->merges[m].right_id;
        int new_id = tok->merges[m].new_id;

        size_t new_len = 0;
        size_t i = 0;
        int changed = 0;
        while (i < seq_len) {
            if (i + 1 < seq_len && seq[i] == left_id && seq[i + 1] == right_id) {
                new_seq[new_len++] = new_id;
                i += 2;
                changed = 1;
            } else {
                new_seq[new_len++] = seq[i];
                i += 1;
            }
        }

        if (changed) {
            memcpy(seq, new_seq, sizeof(int) * new_len);
            seq_len = new_len;
        }
    }

    free(new_seq);

    int *final_ids = (int *)malloc(sizeof(int) * seq_len);
    if (!final_ids) { free(seq); return -1; }
    memcpy(final_ids, seq, sizeof(int) * seq_len);
    free(seq);

    *out_ids = final_ids;
    *out_count = (int)seq_len;
    return 0;
}

int bpe_decode(const BPETokenizer *tok, const int *ids, int count,
               unsigned char **out_text, size_t *out_len) {
    if (!tok || !ids || !out_text || !out_len) return -1;
    if (count == 0) {
        *out_text = NULL;
        *out_len = 0;
        return 0;
    }

    size_t total_len = 0;
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        if (id < 0 || id >= tok->vocab_size) {
            fprintf(stderr, "[bpe_decode] token id %d di luar jangkauan vocab\n", id);
            return -1;
        }
        total_len += (size_t)tok->vocab[id].len;
    }

    unsigned char *text = (unsigned char *)malloc(total_len);
    if (!text) return -1;

    size_t pos = 0;
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        memcpy(text + pos, tok->vocab[id].bytes, tok->vocab[id].len);
        pos += tok->vocab[id].len;
    }

    *out_text = text;
    *out_len = total_len;
    return 0;
}
