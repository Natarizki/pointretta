// test_tokenizer.c
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const char *corpus_text =
        "pointretta adalah model retention pointretta sangat cepat "
        "pointretta belajar bahasa indonesia dengan pointretta "
        "retention lebih cepat dari attention biasa pointretta hebat";

    size_t corpus_len = strlen(corpus_text);

    printf("=== Training BPE Tokenizer ===\n");
    printf("Corpus: \"%s\"\n", corpus_text);
    printf("Panjang corpus: %zu byte\n\n", corpus_len);

    BPETokenizer *tok = bpe_tokenizer_create(300);
    if (!tok) {
        printf("GAGAL: bpe_tokenizer_create return NULL\n");
        return 1;
    }

    int merges_done = bpe_train(tok, (const unsigned char *)corpus_text, corpus_len);
    printf("Jumlah merge yang dilakukan: %d\n\n", merges_done);

    bpe_print_vocab_sample(tok, 15);

    printf("\n=== Test Encode ===\n");
    const char *test_text = "pointretta sangat cepat";
    int *ids = NULL;
    int id_count = 0;
    int rc = bpe_encode(tok, (const unsigned char *)test_text, strlen(test_text), &ids, &id_count);
    printf("bpe_encode return: %d (harusnya 0)\n", rc);
    printf("Teks asli: \"%s\" (%zu byte)\n", test_text, strlen(test_text));
    printf("Jumlah token hasil encode: %d\n", id_count);
    printf("Token ids: [");
    for (int i = 0; i < id_count; i++) {
        printf("%d%s", ids[i], (i < id_count - 1) ? ", " : "");
    }
    printf("]\n");

    printf("\n=== Test Decode (round-trip) ===\n");
    unsigned char *decoded_text = NULL;
    size_t decoded_len = 0;
    int decode_rc = bpe_decode(tok, ids, id_count, &decoded_text, &decoded_len);
    printf("bpe_decode return: %d (harusnya 0)\n", decode_rc);

    printf("Teks hasil decode (%zu byte): \"", decoded_len);
    for (size_t i = 0; i < decoded_len; i++) {
        putchar(decoded_text[i]);
    }
    printf("\"\n");

    int match = (decoded_len == strlen(test_text)) &&
                (memcmp(decoded_text, test_text, decoded_len) == 0);

    printf("\nHASIL: %s — teks asli %s teks hasil decode\n",
           match ? "LOLOS" : "GAGAL",
           match ? "SAMA PERSIS dengan" : "BERBEDA dengan");

    printf("\n=== Efisiensi Kompresi ===\n");
    printf("Tanpa BPE (byte per byte): %zu token\n", strlen(test_text));
    printf("Dengan BPE: %d token\n", id_count);
    printf("Rasio kompresi: %.2fx\n", (double)strlen(test_text) / id_count);

    free(ids);
    free(decoded_text);
    bpe_tokenizer_free(tok);

    return match ? 0 : 1;
}
