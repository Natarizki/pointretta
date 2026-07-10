// train.h
#ifndef POINTRETTA_TRAIN_H
#define POINTRETTA_TRAIN_H

#include "tensor.h"
#include "model.h"

typedef struct {
    Tensor *x_in;
    Tensor *normed1;
    Tensor *Q, *K, *V;
    Tensor *Qh, *Kh, *Vh;
    Tensor *retention_out_h;
    Tensor *retention_merged;
    Tensor *attn_out;
    Tensor *x_mid;
    Tensor *normed2;
    Tensor *ffn_h1, *ffn_h2, *ffn_a, *ffn_hidden;
    Tensor *ffn_out;
} LayerCache;

typedef struct {
    Tensor *x0;
    LayerCache *layers;
    Tensor *x_final;
    Tensor *final_normed;
    int seq_len;
    int num_layers;
} ModelCache;

ModelCache *model_cache_create(const Model *m, int seq_len);
void model_cache_free(ModelCache *cache);

int model_forward_train(const Model *m, const int *token_ids, int seq_len,
                         ModelCache *cache, Tensor *logits_out);

int model_backward(const Model *m, const int *token_ids, int seq_len,
                    const ModelCache *cache, const Tensor *dLogits, double eps);

void model_zero_grad(Model *m);

int model_collect_params(Model *m, Tensor ***out_params, int *out_count);

double cross_entropy_loss(const Tensor *logits, const int *target_ids, int count,
                           Tensor *dLogits_out);

#endif // POINTRETTA_TRAIN_H
