#include "tensor.h"
#include "retention.h"
#include "ffn.h"
#include "train.h"
#include <stdio.h>
#include <math.h>

extern int retention_parallel_forward_neon(const RetentionConfig *cfg,
                                            const Tensor *Q, const Tensor *K, const Tensor *V,
                                            Tensor *out);
extern int swiglu_forward_neon(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                                Tensor *out, Tensor *scratch_hidden1, Tensor *scratch_hidden2);
extern double cross_entropy_loss_neon(const Tensor *logits, const int *target_ids, int count,
                                       Tensor *dLogits_out);

int main(void) {
    int all_pass = 1;

    printf("=== Test 1: Retention Parallel ASM vs Generic ===\n");
    RetentionConfig cfg = { .seq_len = 20, .dim = 8, .decay = 0.9 };
    Tensor *Q = tensor_create((int[]){20, 8}, 2, 0);
    Tensor *K = tensor_create((int[]){20, 8}, 2, 0);
    Tensor *V = tensor_create((int[]){20, 8}, 2, 0);
    Tensor *out_c = tensor_create((int[]){20, 8}, 2, 0);
    Tensor *out_asm = tensor_create((int[]){20, 8}, 2, 0);
    tensor_fill_random(Q, -1, 1, 1);
    tensor_fill_random(K, -1, 1, 2);
    tensor_fill_random(V, -1, 1, 3);
    retention_parallel_forward(&cfg, Q, K, V, out_c);
    retention_parallel_forward_neon(&cfg, Q, K, V, out_asm);
    double diff1 = 0;
    for (size_t i = 0; i < out_c->size; i++) {
        double d = fabs(out_c->data[i] - out_asm->data[i]);
        if (d > diff1) diff1 = d;
    }
    printf("Max diff: %.12f | %s\n\n", diff1, diff1 < 1e-6 ? "LOLOS" : "GAGAL");
    all_pass &= (diff1 < 1e-6);

    printf("=== Test 2: SwiGLU ASM vs Generic ===\n");
    Tensor *x = tensor_create((int[]){5, 16}, 2, 0);
    Tensor *W1 = tensor_create((int[]){16, 32}, 2, 0);
    Tensor *W3 = tensor_create((int[]){16, 32}, 2, 0);
    Tensor *W2 = tensor_create((int[]){32, 16}, 2, 0);
    Tensor *fout_c = tensor_create((int[]){5, 16}, 2, 0);
    Tensor *fout_asm = tensor_create((int[]){5, 16}, 2, 0);
    Tensor *s1 = tensor_create((int[]){5, 32}, 2, 0);
    Tensor *s2 = tensor_create((int[]){5, 32}, 2, 0);
    Tensor *s1b = tensor_create((int[]){5, 32}, 2, 0);
    Tensor *s2b = tensor_create((int[]){5, 32}, 2, 0);
    tensor_fill_random(x, -1, 1, 4);
    tensor_fill_random(W1, -0.5, 0.5, 5);
    tensor_fill_random(W3, -0.5, 0.5, 6);
    tensor_fill_random(W2, -0.5, 0.5, 7);
    swiglu_forward(x, W1, W3, W2, fout_c, s1, s2);
    swiglu_forward_neon(x, W1, W3, W2, fout_asm, s1b, s2b);
    double diff2 = 0;
    for (size_t i = 0; i < fout_c->size; i++) {
        double d = fabs(fout_c->data[i] - fout_asm->data[i]);
        if (d > diff2) diff2 = d;
    }
    printf("Max diff: %.12f | %s\n\n", diff2, diff2 < 1e-9 ? "LOLOS" : "GAGAL");
    all_pass &= (diff2 < 1e-9);

    printf("=== Test 3: Cross-Entropy Loss ASM vs Generic ===\n");
    Tensor *logits = tensor_create((int[]){4, 50}, 2, 0);
    tensor_fill_random(logits, -5, 5, 8);
    int targets[] = {3, 10, 25, 40};
    Tensor *dlogits_c = tensor_create((int[]){4, 50}, 2, 0);
    Tensor *dlogits_asm = tensor_create((int[]){4, 50}, 2, 0);
    double loss_c = cross_entropy_loss(logits, targets, 4, dlogits_c);
    double loss_asm = cross_entropy_loss_neon(logits, targets, 4, dlogits_asm);
    double diff3 = fabs(loss_c - loss_asm);
    for (size_t i = 0; i < dlogits_c->size; i++) {
        double d = fabs(dlogits_c->data[i] - dlogits_asm->data[i]);
        if (d > diff3) diff3 = d;
    }
    printf("Loss C: %.10f | Loss ASM: %.10f | Max diff: %.12f | %s\n\n",
           loss_c, loss_asm, diff3, diff3 < 1e-9 ? "LOLOS" : "GAGAL");
    all_pass &= (diff3 < 1e-9);

    printf("=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA TEST LOLOS" : "ADA TEST GAGAL");
    return all_pass ? 0 : 1;
}
