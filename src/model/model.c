// model.c
#include "tensor.h"
#include "tensor_ops.h"
#include "retention_multihead.h"
#include "norm.h"
#include "ffn.h"
#include "model.h"
#include "dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int split_heads(const Tensor *x, int num_heads, int head_dim, Tensor *out) {
    if (!x || !out) return -1;
    if (x->ndim != 2) return -1;
    int seq_len = x->shape[0];
    int dim = x->shape[1];
    if (dim != num_heads * head_dim) return -1;
    if (out->ndim != 3 || out->shape[0] != num_heads || out->shape[1] != seq_len || out->shape[2] != head_dim) {
        return -1;
    }

    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                double val = x->data[i * dim + h * head_dim + d];
                out->data[h * (seq_len * head_dim) + i * head_dim + d] = val;
            }
        }
    }
    return 0;
}

int merge_heads(const Tensor *x, int num_heads, int head_dim, Tensor *out) {
    if (!x || !out) return -1;
    if (x->ndim != 3 || x->shape[0] != num_heads || x->shape[2] != head_dim) return -1;
    int seq_len = x->shape[1];
    int dim = num_heads * head_dim;
    if (out->ndim != 2 || out->shape[0] != seq_len || out->shape[1] != dim) return -1;

    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                double val = x->data[h * (seq_len * head_dim) + i * head_dim + d];
                out->data[i * dim + h * head_dim + d] = val;
            }
        }
    }
    return 0;
}

static Tensor *alloc_param_2d(int rows, int cols, double init_range, unsigned int seed) {
    int shape[] = {rows, cols};
    Tensor *t = tensor_create(shape, 2, 1);
    if (t) tensor_fill_random(t, -init_range, init_range, seed);
    return t;
}

static Tensor *alloc_param_1d(int n, double init_value) {
    int shape[] = {n};
    Tensor *t = tensor_create(shape, 1, 1);
    if (t) tensor_fill(t, init_value);
    return t;
}

Model *model_create(ModelConfig cfg, unsigned int seed) {
    if (cfg.dim != cfg.num_heads * cfg.head_dim) {
        fprintf(stderr, "[model_create] dim (%d) harus sama dengan num_heads*head_dim (%d*%d=%d)\n",
                cfg.dim, cfg.num_heads, cfg.head_dim, cfg.num_heads * cfg.head_dim);
        return NULL;
    }

    Model *m = (Model *)malloc(sizeof(Model));
    if (!m) return NULL;
    m->cfg = cfg;

    double init_range = 1.0 / sqrt((double)cfg.dim);

    m->embedding = alloc_param_2d(cfg.vocab_size, cfg.dim, init_range, seed + 1);
    m->output_head = alloc_param_2d(cfg.dim, cfg.vocab_size, init_range, seed + 2);
    m->final_norm_gain = alloc_param_1d(cfg.dim, 1.0);

    m->layers = (ModelLayer *)malloc(sizeof(ModelLayer) * cfg.num_layers);
    if (!m->layers) { free(m); return NULL; }

    for (int l = 0; l < cfg.num_layers; l++) {
        ModelLayer *layer = &m->layers[l];
        unsigned int layer_seed = seed + 100 + l * 10;

        layer->Wq = alloc_param_2d(cfg.dim, cfg.dim, init_range, layer_seed + 1);
        layer->Wk = alloc_param_2d(cfg.dim, cfg.dim, init_range, layer_seed + 2);
        layer->Wv = alloc_param_2d(cfg.dim, cfg.dim, init_range, layer_seed + 3);
        layer->Wo = alloc_param_2d(cfg.dim, cfg.dim, init_range, layer_seed + 4);

        layer->norm1_gain = alloc_param_1d(cfg.dim, 1.0);
        layer->norm2_gain = alloc_param_1d(cfg.dim, 1.0);

        layer->W1 = alloc_param_2d(cfg.dim, cfg.ffn_hidden, init_range, layer_seed + 5);
        layer->W3 = alloc_param_2d(cfg.dim, cfg.ffn_hidden, init_range, layer_seed + 6);
        layer->W2 = alloc_param_2d(cfg.ffn_hidden, cfg.dim, init_range, layer_seed + 7);

        layer->decay_per_head = make_multiscale_decay(cfg.num_heads, cfg.decay_min, cfg.decay_max);
    }

    return m;
}

void model_free(Model *m) {
    if (!m) return;

    tensor_free(m->embedding);
    tensor_free(m->output_head);
    tensor_free(m->final_norm_gain);

    for (int l = 0; l < m->cfg.num_layers; l++) {
        ModelLayer *layer = &m->layers[l];
        tensor_free(layer->Wq); tensor_free(layer->Wk);
        tensor_free(layer->Wv); tensor_free(layer->Wo);
        tensor_free(layer->norm1_gain); tensor_free(layer->norm2_gain);
        tensor_free(layer->W1); tensor_free(layer->W3); tensor_free(layer->W2);
        if (layer->decay_per_head) free(layer->decay_per_head);
    }
    free(m->layers);
    free(m);
}

int model_forward(const Model *m, const int *token_ids, int seq_len, Tensor *logits_out) {
    if (!m || !token_ids || !logits_out) return -1;

    int dim = m->cfg.dim;
    int num_heads = m->cfg.num_heads;
    int head_dim = m->cfg.head_dim;
    int hidden = m->cfg.ffn_hidden;
    int vocab_size = m->cfg.vocab_size;

    if (logits_out->ndim != 2 || logits_out->shape[0] != seq_len || logits_out->shape[1] != vocab_size) {
        fprintf(stderr, "[model_forward] shape logits_out harus [%d, %d]\n", seq_len, vocab_size);
        return -1;
    }

    int shape_seq_dim[] = {seq_len, dim};
    Tensor *x = tensor_create(shape_seq_dim, 2, 0);
    for (int i = 0; i < seq_len; i++) {
        int tok = token_ids[i];
        if (tok < 0 || tok >= vocab_size) {
            fprintf(stderr, "[model_forward] token id %d di luar jangkauan vocab\n", tok);
            tensor_free(x);
            return -1;
        }
        memcpy(&x->data[i * dim], &m->embedding->data[tok * dim], sizeof(double) * dim);
    }

    Tensor *normed = tensor_create(shape_seq_dim, 2, 0);
    Tensor *Q = tensor_create(shape_seq_dim, 2, 0);
    Tensor *K = tensor_create(shape_seq_dim, 2, 0);
    Tensor *V = tensor_create(shape_seq_dim, 2, 0);

    int shape_heads[] = {num_heads, seq_len, head_dim};
    Tensor *Qh = tensor_create(shape_heads, 3, 0);
    Tensor *Kh = tensor_create(shape_heads, 3, 0);
    Tensor *Vh = tensor_create(shape_heads, 3, 0);
    Tensor *retention_out_h = tensor_create(shape_heads, 3, 0);

    Tensor *retention_merged = tensor_create(shape_seq_dim, 2, 0);
    Tensor *attn_out = tensor_create(shape_seq_dim, 2, 0);

    int shape_hidden[] = {seq_len, hidden};
    Tensor *ffn_scratch1 = tensor_create(shape_hidden, 2, 0);
    Tensor *ffn_scratch2 = tensor_create(shape_hidden, 2, 0);
    Tensor *ffn_out = tensor_create(shape_seq_dim, 2, 0);

    double eps = 1e-6;
    int rc = 0;

    for (int l = 0; l < m->cfg.num_layers && rc == 0; l++) {
        ModelLayer *layer = &m->layers[l];

        rc |= rmsnorm_forward_fast(x, layer->norm1_gain, normed, eps);
        rc |= tensor_matmul_fast(Q, normed, layer->Wq);
        rc |= tensor_matmul_fast(K, normed, layer->Wk);
        rc |= tensor_matmul_fast(V, normed, layer->Wv);

        rc |= split_heads(Q, num_heads, head_dim, Qh);
        rc |= split_heads(K, num_heads, head_dim, Kh);
        rc |= split_heads(V, num_heads, head_dim, Vh);

        MultiHeadRetentionConfig ret_cfg = {
            .seq_len = seq_len, .num_heads = num_heads, .head_dim = head_dim,
            .decay_per_head = layer->decay_per_head
        };
        rc |= retention_multihead_parallel_forward_fast(&ret_cfg, Qh, Kh, Vh, retention_out_h);

        rc |= merge_heads(retention_out_h, num_heads, head_dim, retention_merged);
        rc |= tensor_matmul_fast(attn_out, retention_merged, layer->Wo);

        for (size_t i = 0; i < x->size; i++) x->data[i] += attn_out->data[i];

        rc |= rmsnorm_forward_fast(x, layer->norm2_gain, normed, eps);
        rc |= swiglu_forward(normed, layer->W1, layer->W3, layer->W2, ffn_out, ffn_scratch1, ffn_scratch2);

        for (size_t i = 0; i < x->size; i++) x->data[i] += ffn_out->data[i];
    }

    if (rc == 0) {
        rc |= rmsnorm_forward_fast(x, m->final_norm_gain, normed, eps);
        rc |= tensor_matmul_fast(logits_out, normed, m->output_head);
    }

    tensor_free(x); tensor_free(normed);
    tensor_free(Q); tensor_free(K); tensor_free(V);
    tensor_free(Qh); tensor_free(Kh); tensor_free(Vh); tensor_free(retention_out_h);
    tensor_free(retention_merged); tensor_free(attn_out);
    tensor_free(ffn_scratch1); tensor_free(ffn_scratch2); tensor_free(ffn_out);

    return rc;
}
