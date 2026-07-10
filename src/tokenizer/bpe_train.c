// bpe_train.c
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

BPETokenizer *bpe_tokenizer_create(int target_vocab_size) {
    BPETokenizer *tok = (BPETokenizer *)malloc(sizeof(BPETokenizer));
    if (!tok) return NULL;

    tok->vocab_capacity = target_vocab_size > BPE_BASE_VOCAB ? target_vocab_size : BPE_BASE_VOCAB;
    tok->vocab = (BPEToken *)malloc(sizeof(BPEToken) * tok->vocab_capacity);
    if (!tok->vocab) { free(tok); return NULL; }

    for (int i = 0; i < BPE_BASE_VOCAB; i++) {
        tok->vocab[i].bytes[0] = (unsigned char)i;
        tok->vocab[i].len = 1;
    }
    tok->vocab_size = BPE_BASE_VOCAB;

    tok->merges_capacity = tok->vocab_capacity - BPE_BASE_VOCAB;
    if (tok->merges_capacity < 1) tok->merges_capacity = 1;
    tok->merges = (BPEMergeRule *)malloc(sizeof(BPEMergeRule) * tok->merges_capacity);
    if (!tok->merges) { free(tok->vocab); free(tok); return NULL; }
    tok->num_merges = 0;

    return tok;
}

void bpe_tokenizer_free(BPETokenizer *tok) {
    if (!tok) return;
    if (tok->vocab) free(tok->vocab);
    if (tok->merges) free(tok->merges);
    free(tok);
}

typedef struct {
    long long key;
    int count;
} PairEntry;

static long long make_pair_key(int left, int right) {
    return ((long long)left << 20) | (long long)right;
}

static int find_slot(PairEntry *table, int table_size, long long key) {
    unsigned long long h = (unsigned long long)key;
    h = (h ^ (h >> 16)) * 0x9E3779B1u;
    int idx = (int)(h % (unsigned long long)table_size);
    while (table[idx].key != -1 && table[idx].key != key) {
        idx = (idx + 1) % table_size;
    }
    return idx;
}

int bpe_train(BPETokenizer *tok, const unsigned char *corpus, size_t corpus_len) {
    if (!tok || !corpus || corpus_len == 0) return 0;

    int *seq = (int *)malloc(sizeof(int) * corpus_len);
    if (!seq) return 0;
    size_t seq_len = corpus_len;
    for (size_t i = 0; i < corpus_len; i++) {
        seq[i] = (int)corpus[i];
    }

    int target_merges = tok->merges_capacity;
    int merges_done = 0;

    int table_size = 1024;
    while ((size_t)table_size < seq_len * 4 && table_size < (1 << 22)) {
        table_size *= 2;
    }
    PairEntry *table = (PairEntry *)malloc(sizeof(PairEntry) * table_size);
    if (!table) { free(seq); return 0; }

    int *new_seq = (int *)malloc(sizeof(int) * corpus_len);
    if (!new_seq) { free(seq); free(table); return 0; }

    for (int step = 0; step < target_merges; step++) {
        if (seq_len < 2) break;

        for (int i = 0; i < table_size; i++) table[i].key = -1;

        for (size_t i = 0; i + 1 < seq_len; i++) {
            long long key = make_pair_key(seq[i], seq[i + 1]);
            int idx = find_slot(table, table_size, key);
            if (table[idx].key == -1) {
                table[idx].key = key;
                table[idx].count = 1;
            } else {
                table[idx].count++;
            }
        }

        long long best_key = -1;
        int best_count = 1;
        for (int i = 0; i < table_size; i++) {
            if (table[i].key != -1 && table[i].count > best_count) {
                best_count = table[i].count;
                best_key = table[i].key;
            }
        }

        if (best_key == -1) break;

        int left_id = (int)(best_key >> 20);
        int right_id = (int)(best_key & 0xFFFFF);

        int combined_len = tok->vocab[left_id].len + tok->vocab[right_id].len;
        if (combined_len > BPE_MAX_TOKEN_LEN) break;
        if (tok->vocab_size >= tok->vocab_capacity) break;

        int new_id = tok->vocab_size;
        BPEToken *new_tok = &tok->vocab[new_id];
        memcpy(new_tok->bytes, tok->vocab[left_id].bytes, tok->vocab[left_id].len);
        memcpy(new_tok->bytes + tok->vocab[left_id].len, tok->vocab[right_id].bytes, tok->vocab[right_id].len);
        new_tok->len = combined_len;
        tok->vocab_size++;

        tok->merges[tok->num_merges].left_id = left_id;
        tok->merges[tok->num_merges].right_id = right_id;
        tok->merges[tok->num_merges].new_id = new_id;
        tok->num_merges++;
        merges_done++;

        size_t new_len = 0;
        size_t i = 0;
        while (i < seq_len) {
            if (i + 1 < seq_len && seq[i] == left_id && seq[i + 1] == right_id) {
                new_seq[new_len++] = new_id;
                i += 2;
            } else {
                new_seq[new_len++] = seq[i];
                i += 1;
            }
        }
        memcpy(seq, new_seq, sizeof(int) * new_len);
        seq_len = new_len;
    }

    free(seq);
    free(table);
    free(new_seq);

    return merges_done;
}

void bpe_print_vocab_sample(const BPETokenizer *tok, int max_entries) {
    if (!tok) return;
    printf("Vocab size: %d, jumlah merge: %d\n", tok->vocab_size, tok->num_merges);
    printf("Sample token hasil merge (bukan byte dasar):\n");

    int shown = 0;
    for (int i = BPE_BASE_VOCAB; i < tok->vocab_size && shown < max_entries; i++) {
        printf("  id=%d len=%d bytes=[", i, tok->vocab[i].len);
        for (int b = 0; b < tok->vocab[i].len; b++) {
            unsigned char c = tok->vocab[i].bytes[b];
            if (c >= 32 && c < 127) printf("%c", c);
            else printf("\\x%02x", c);
        }
        printf("]\n");
        shown++;
    }
}
