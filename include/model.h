// model.h
#ifndef POINTRETTA_MODEL_H
#define POINTRETTA_MODEL_H

#include "tensor.h"

typedef struct {
    int vocab_size;
    int dim;
    int num_heads;
    int head_dim;
    int ffn_hidden;
    int num_layers;
    double decay_min;
    double decay_max;
} ModelConfig;

typedef struct {
    Tensor *Wq, *Wk, *Wv, *Wo;
    Tensor *norm1_gain;
    Tensor *norm2_gain;
    Tensor *W1, *W3, *W2;
    double *decay_per_head;
} ModelLayer;

typedef struct {
    ModelConfig cfg;
    Tensor *embedding;
    ModelLayer *layers;
    Tensor *final_norm_gain;
    Tensor *output_head;
} Model;

Model *model_create(ModelConfig cfg, unsigned int seed);
void model_free(Model *m);

int model_forward(const Model *m, const int *token_ids, int seq_len, Tensor *logits_out);

int split_heads(const Tensor *x, int num_heads, int head_dim, Tensor *out);
int merge_heads(const Tensor *x, int num_heads, int head_dim, Tensor *out);

#endif // POINTRETTA_MODEL_H
