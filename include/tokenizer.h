// tokenizer.h
#ifndef POINTRETTA_TOKENIZER_H
#define POINTRETTA_TOKENIZER_H

#include <stddef.h>

#define BPE_MAX_TOKEN_LEN 32
#define BPE_BASE_VOCAB 256

typedef struct {
    unsigned char bytes[BPE_MAX_TOKEN_LEN];
    int len;
} BPEToken;

typedef struct {
    int left_id;
    int right_id;
    int new_id;
} BPEMergeRule;

typedef struct {
    BPEToken *vocab;
    int vocab_size;
    int vocab_capacity;

    BPEMergeRule *merges;
    int num_merges;
    int merges_capacity;
} BPETokenizer;

BPETokenizer *bpe_tokenizer_create(int target_vocab_size);
void bpe_tokenizer_free(BPETokenizer *tok);

int bpe_train(BPETokenizer *tok, const unsigned char *corpus, size_t corpus_len);

int bpe_encode(const BPETokenizer *tok, const unsigned char *text, size_t text_len,
               int **out_ids, int *out_count);

int bpe_decode(const BPETokenizer *tok, const int *ids, int count,
               unsigned char **out_text, size_t *out_len);

void bpe_print_vocab_sample(const BPETokenizer *tok, int max_entries);

#endif // POINTRETTA_TOKENIZER_H
