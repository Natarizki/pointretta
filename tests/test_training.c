// test_training.c
#include "tensor.h"
#include "model.h"
#include "train.h"
#include "optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FD_EPS 1e-5
#define CHECK_TOL 1e-2

static double compute_total_loss(const Model *m, const int *token_ids, const int *target_ids,
                                  int seq_len, int count, Tensor *logits_scratch) {
    ModelCache *cache = model_cache_create(m, seq_len);
    model_forward_train(m, token_ids, seq_len, cache, logits_scratch);
    Tensor *dLogits_dummy = tensor_create((int[]){seq_len, m->cfg.vocab_size}, 2, 0);
    double loss = cross_entropy_loss(logits_scratch, target_ids, count, dLogits_dummy);
    tensor_free(dLogits_dummy);
    model_cache_free(cache);
    return loss;
}

static int test_end_to_end_gradient_check(void) {
    printf("=== Test 1: Numerical Gradient Check END-TO-END ===\n");

    ModelConfig cfg = {
        .vocab_size = 20, .dim = 8, .num_heads = 2, .head_dim = 4,
        .ffn_hidden = 12, .num_layers = 2, .decay_min = 0.7, .decay_max = 0.95
    };
    Model *m = model_create(cfg, 7);

    int seq_len = 5;
    int token_ids[] = {1, 3, 5, 7, 9};
    int target_ids[] = {3, 5, 7, 9, 1};
    int count = seq_len;

    ModelCache *cache = model_cache_create(m, seq_len);
    Tensor *logits = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);
    model_forward_train(m, token_ids, seq_len, cache, logits);

    Tensor *dLogits = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);
    cross_entropy_loss(logits, target_ids, count, dLogits);

    model_zero_grad(m);
    model_backward(m, token_ids, seq_len, cache, dLogits, 1e-6);

    int all_pass = 1;

    {
        int tok = token_ids[0];
        int check_dim = 2;
        for (int d = 0; d < check_dim; d++) {
            int idx = tok * cfg.dim + d;
            double analytic = m->embedding->grad[idx];

            double orig = m->embedding->data[idx];
            m->embedding->data[idx] = orig + FD_EPS;
            double loss_p = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            m->embedding->data[idx] = orig - FD_EPS;
            double loss_m = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            m->embedding->data[idx] = orig;

            double numeric = (loss_p - loss_m) / (2.0 * FD_EPS);
            double diff = fabs(analytic - numeric);
            printf("  embedding[%d,%d]: analytic=%.6f numeric=%.6f diff=%.6f\n", tok, d, analytic, numeric, diff);
            if (diff > CHECK_TOL) all_pass = 0;
        }
    }

    {
        Tensor *Wq = m->layers[0].Wq;
        for (int i = 0; i < 2; i++) {
            double analytic = Wq->grad[i];
            double orig = Wq->data[i];
            Wq->data[i] = orig + FD_EPS;
            double loss_p = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            Wq->data[i] = orig - FD_EPS;
            double loss_m = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            Wq->data[i] = orig;
            double numeric = (loss_p - loss_m) / (2.0 * FD_EPS);
            double diff = fabs(analytic - numeric);
            printf("  layer0.Wq[%d]: analytic=%.6f numeric=%.6f diff=%.6f\n", i, analytic, numeric, diff);
            if (diff > CHECK_TOL) all_pass = 0;
        }
    }

    {
        Tensor *W1 = m->layers[1].W1;
        for (int i = 0; i < 2; i++) {
            double analytic = W1->grad[i];
            double orig = W1->data[i];
            W1->data[i] = orig + FD_EPS;
            double loss_p = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            W1->data[i] = orig - FD_EPS;
            double loss_m = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            W1->data[i] = orig;
            double numeric = (loss_p - loss_m) / (2.0 * FD_EPS);
            double diff = fabs(analytic - numeric);
            printf("  layer1.W1[%d]: analytic=%.6f numeric=%.6f diff=%.6f\n", i, analytic, numeric, diff);
            if (diff > CHECK_TOL) all_pass = 0;
        }
    }

    {
        for (int i = 0; i < 2; i++) {
            double analytic = m->output_head->grad[i];
            double orig = m->output_head->data[i];
            m->output_head->data[i] = orig + FD_EPS;
            double loss_p = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            m->output_head->data[i] = orig - FD_EPS;
            double loss_m = compute_total_loss(m, token_ids, target_ids, seq_len, count, logits);
            m->output_head->data[i] = orig;
            double numeric = (loss_p - loss_m) / (2.0 * FD_EPS);
            double diff = fabs(analytic - numeric);
            printf("  output_head[%d]: analytic=%.6f numeric=%.6f diff=%.6f\n", i, analytic, numeric, diff);
            if (diff > CHECK_TOL) all_pass = 0;
        }
    }

    printf("\nEnd-to-end gradient check: %s\n\n", all_pass ? "LOLOS" : "GAGAL");

    tensor_free(logits); tensor_free(dLogits);
    model_cache_free(cache);
    model_free(m);
    return all_pass;
}

static int test_tiny_overfit(void) {
    printf("=== Test 2: Training Loop Kecil (Overfit Demo) ===\n");

    ModelConfig cfg = {
        .vocab_size = 15, .dim = 16, .num_heads = 4, .head_dim = 4,
        .ffn_hidden = 24, .num_layers = 2, .decay_min = 0.7, .decay_max = 0.95
    };
    Model *m = model_create(cfg, 123);

    int token_ids[] = {2, 5, 8, 11, 3, 6};
    int target_ids[] = {5, 8, 11, 3, 6, 2};
    int seq_len = 6;
    int count = seq_len;

    Tensor **params;
    int num_params;
    model_collect_params(m, &params, &num_params);

    AdafactorState **states = (AdafactorState **)malloc(sizeof(AdafactorState *) * num_params);
    for (int i = 0; i < num_params; i++) {
        states[i] = adafactor_state_create(params[i], 0.999, 1e-30, 1e-3);
    }

    ModelCache *cache = model_cache_create(m, seq_len);
    Tensor *logits = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);
    Tensor *dLogits = tensor_create((int[]){seq_len, cfg.vocab_size}, 2, 0);

    double lr = 0.05;
    int num_iters = 300;
    double loss_start = 0.0, loss_end = 0.0;

    for (int iter = 0; iter < num_iters; iter++) {
        model_forward_train(m, token_ids, seq_len, cache, logits);
        double loss = cross_entropy_loss(logits, target_ids, count, dLogits);

        if (iter == 0) loss_start = loss;
        if (iter == num_iters - 1) loss_end = loss;
        if (iter % 50 == 0) printf("  Iter %3d | Loss: %.4f\n", iter, loss);

        model_zero_grad(m);
        model_backward(m, token_ids, seq_len, cache, dLogits, 1e-6);

        for (int i = 0; i < num_params; i++) {
            adafactor_step(params[i], states[i], lr);
        }
    }

    printf("Loss awal: %.4f -> Loss akhir: %.4f\n", loss_start, loss_end);
    int pass = (loss_end < loss_start * 0.3);
    printf("Overfit test: %s\n\n", pass ? "LOLOS (model belajar)" : "GAGAL (loss tidak turun cukup)");

    for (int i = 0; i < num_params; i++) adafactor_state_free(states[i]);
    free(states);
    free(params);
    tensor_free(logits); tensor_free(dLogits);
    model_cache_free(cache);
    model_free(m);

    return pass;
}

int main(void) {
    int p1 = test_end_to_end_gradient_check();
    int p2 = test_tiny_overfit();

    int all_pass = p1 && p2;
    printf("=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA TEST LOLOS -- POINTRETTA BISA BELAJAR!" : "ADA TEST GAGAL");
    return all_pass ? 0 : 1;
}
