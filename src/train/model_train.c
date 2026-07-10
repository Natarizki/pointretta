// model_train.c
// Forward pass "cached" + backward pass penuh lewat seluruh model.

#include "tensor.h"
#include "tensor_ops.h"
#include "retention_multihead.h"
#include "norm.h"
#include "ffn.h"
#include "backward_ops.h"
#include "model.h"
#include "train.h"
#include "dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Tensor *t2(int a, int b) { int s[] = {a, b}; return tensor_create(s, 2, 0); }
static Tensor *t3(int a, int b, int c) { int s[] = {a, b, c}; return tensor_create(s, 3, 0); }

ModelCache *model_cache_create(const Model *m, int seq_len) {
    ModelCache *cache = (ModelCache *)malloc(sizeof(ModelCache));
    if (!cache) return NULL;

    int dim = m->cfg.dim;
    int heads = m->cfg.num_heads;
    int head_dim = m->cfg.head_dim;
    int hidden = m->cfg.ffn_hidden;
    int num_layers = m->cfg.num_layers;

    cache->seq_len = seq_len;
    cache->num_layers = num_layers;
    cache->x0 = t2(seq_len, dim);
    cache->x_final = t2(seq_len, dim);
    cache->final_normed = t2(seq_len, dim);

    cache->layers = (LayerCache *)malloc(sizeof(LayerCache) * num_layers);
    for (int l = 0; l < num_layers; l++) {
        LayerCache *lc = &cache->layers[l];
        lc->x_in = t2(seq_len, dim);
        lc->normed1 = t2(seq_len, dim);
        lc->Q = t2(seq_len, dim); lc->K = t2(seq_len, dim); lc->V = t2(seq_len, dim);
        lc->Qh = t3(heads, seq_len, head_dim);
        lc->Kh = t3(heads, seq_len, head_dim);
        lc->Vh = t3(heads, seq_len, head_dim);
        lc->retention_out_h = t3(heads, seq_len, head_dim);
        lc->retention_merged = t2(seq_len, dim);
        lc->attn_out = t2(seq_len, dim);
        lc->x_mid = t2(seq_len, dim);
        lc->normed2 = t2(seq_len, dim);
        lc->ffn_h1 = t2(seq_len, hidden);
        lc->ffn_h2 = t2(seq_len, hidden);
        lc->ffn_a = t2(seq_len, hidden);
        lc->ffn_hidden = t2(seq_len, hidden);
        lc->ffn_out = t2(seq_len, dim);
    }

    return cache;
}

void model_cache_free(ModelCache *cache) {
    if (!cache) return;
    tensor_free(cache->x0);
    tensor_free(cache->x_final);
    tensor_free(cache->final_normed);

    for (int l = 0; l < cache->num_layers; l++) {
        LayerCache *lc = &cache->layers[l];
        tensor_free(lc->x_in); tensor_free(lc->normed1);
        tensor_free(lc->Q); tensor_free(lc->K); tensor_free(lc->V);
        tensor_free(lc->Qh); tensor_free(lc->Kh); tensor_free(lc->Vh);
        tensor_free(lc->retention_out_h); tensor_free(lc->retention_merged);
        tensor_free(lc->attn_out); tensor_free(lc->x_mid); tensor_free(lc->normed2);
        tensor_free(lc->ffn_h1); tensor_free(lc->ffn_h2);
        tensor_free(lc->ffn_a); tensor_free(lc->ffn_hidden); tensor_free(lc->ffn_out);
    }
    free(cache->layers);
    free(cache);
}

int model_forward_train(const Model *m, const int *token_ids, int seq_len,
                         ModelCache *cache, Tensor *logits_out) {
    if (!m || !token_ids || !cache || !logits_out) return -1;

    int dim = m->cfg.dim;
    int heads = m->cfg.num_heads;
    int head_dim = m->cfg.head_dim;
    int vocab_size = m->cfg.vocab_size;
    double eps = 1e-6;
    int rc = 0;

    for (int i = 0; i < seq_len; i++) {
        int tok = token_ids[i];
        if (tok < 0 || tok >= vocab_size) return -1;
        memcpy(&cache->x0->data[i * dim], &m->embedding->data[tok * dim], sizeof(double) * dim);
    }

    Tensor *x = cache->x0;

    for (int l = 0; l < m->cfg.num_layers; l++) {
        ModelLayer *layer = &m->layers[l];
        LayerCache *lc = &cache->layers[l];

        memcpy(lc->x_in->data, x->data, sizeof(double) * x->size);

        rc |= rmsnorm_forward_fast(lc->x_in, layer->norm1_gain, lc->normed1, eps);
        rc |= tensor_matmul_fast(lc->Q, lc->normed1, layer->Wq);
        rc |= tensor_matmul_fast(lc->K, lc->normed1, layer->Wk);
        rc |= tensor_matmul_fast(lc->V, lc->normed1, layer->Wv);

        rc |= split_heads(lc->Q, heads, head_dim, lc->Qh);
        rc |= split_heads(lc->K, heads, head_dim, lc->Kh);
        rc |= split_heads(lc->V, heads, head_dim, lc->Vh);

        MultiHeadRetentionConfig ret_cfg = {seq_len, heads, head_dim, layer->decay_per_head};
        rc |= retention_multihead_parallel_forward_fast(&ret_cfg, lc->Qh, lc->Kh, lc->Vh, lc->retention_out_h);

        rc |= merge_heads(lc->retention_out_h, heads, head_dim, lc->retention_merged);
        rc |= tensor_matmul_fast(lc->attn_out, lc->retention_merged, layer->Wo);

        for (size_t i = 0; i < lc->x_in->size; i++) {
            lc->x_mid->data[i] = lc->x_in->data[i] + lc->attn_out->data[i];
        }

        rc |= rmsnorm_forward_fast(lc->x_mid, layer->norm2_gain, lc->normed2, eps);
        rc |= swiglu_forward_for_backward(lc->normed2, layer->W1, layer->W3, layer->W2,
                                           lc->ffn_h1, lc->ffn_h2, lc->ffn_a, lc->ffn_hidden, lc->ffn_out);

        Tensor *x_out = (l == m->cfg.num_layers - 1) ? cache->x_final : cache->layers[l + 1].x_in;
        for (size_t i = 0; i < lc->x_mid->size; i++) {
            x_out->data[i] = lc->x_mid->data[i] + lc->ffn_out->data[i];
        }
        x = x_out;
    }

    if (m->cfg.num_layers == 0) {
        memcpy(cache->x_final->data, cache->x0->data, sizeof(double) * cache->x0->size);
    }

    rc |= rmsnorm_forward_fast(cache->x_final, m->final_norm_gain, cache->final_normed, eps);
    rc |= tensor_matmul_fast(logits_out, cache->final_normed, m->output_head);

    return rc;
}

void model_zero_grad(Model *m) {
    tensor_zero_grad(m->embedding);
    tensor_zero_grad(m->output_head);
    tensor_zero_grad(m->final_norm_gain);
    for (int l = 0; l < m->cfg.num_layers; l++) {
        ModelLayer *layer = &m->layers[l];
        tensor_zero_grad(layer->Wq); tensor_zero_grad(layer->Wk);
        tensor_zero_grad(layer->Wv); tensor_zero_grad(layer->Wo);
        tensor_zero_grad(layer->norm1_gain); tensor_zero_grad(layer->norm2_gain);
        tensor_zero_grad(layer->W1); tensor_zero_grad(layer->W3); tensor_zero_grad(layer->W2);
    }
}

int model_collect_params(Model *m, Tensor ***out_params, int *out_count) {
    int count = 3 + m->cfg.num_layers * 9;
    Tensor **params = (Tensor **)malloc(sizeof(Tensor *) * count);
    if (!params) return -1;

    int idx = 0;
    params[idx++] = m->embedding;
    params[idx++] = m->output_head;
    params[idx++] = m->final_norm_gain;
    for (int l = 0; l < m->cfg.num_layers; l++) {
        ModelLayer *layer = &m->layers[l];
        params[idx++] = layer->Wq; params[idx++] = layer->Wk;
        params[idx++] = layer->Wv; params[idx++] = layer->Wo;
        params[idx++] = layer->norm1_gain; params[idx++] = layer->norm2_gain;
        params[idx++] = layer->W1; params[idx++] = layer->W3; params[idx++] = layer->W2;
    }

    *out_params = params;
    *out_count = count;
    return 0;
}

static void accumulate_grad(Tensor *param, const Tensor *temp_grad) {
    for (size_t i = 0; i < param->size; i++) {
        param->grad[i] += temp_grad->data[i];
    }
}

int model_backward(const Model *m, const int *token_ids, int seq_len,
                    const ModelCache *cache, const Tensor *dLogits, double eps) {
    if (!m || !token_ids || !cache || !dLogits) return -1;

    int dim = m->cfg.dim;
    int heads = m->cfg.num_heads;
    int head_dim = m->cfg.head_dim;
    int hidden = m->cfg.ffn_hidden;
    int vocab_size = m->cfg.vocab_size;

    Tensor *dFinalNormed = t2(seq_len, dim);
    Tensor *dOutputHead = t2(dim, vocab_size);
    matmul_backward(dLogits, cache->final_normed, m->output_head, dFinalNormed, dOutputHead);
    accumulate_grad(m->output_head, dOutputHead);

    Tensor *dX = t2(seq_len, dim);
    Tensor *dFinalGain1d; { int s[] = {dim}; dFinalGain1d = tensor_create(s, 1, 0); }
    rmsnorm_backward(cache->x_final, m->final_norm_gain, dFinalNormed, eps, dX, dFinalGain1d);
    accumulate_grad(m->final_norm_gain, dFinalGain1d);

    tensor_free(dFinalNormed); tensor_free(dOutputHead); tensor_free(dFinalGain1d);

    Tensor *dNormed2 = t2(seq_len, dim);
    Tensor *dW1 = t2(dim, hidden), *dW3 = t2(dim, hidden), *dW2 = t2(hidden, dim);
    Tensor *dNorm2Gain; { int s[] = {dim}; dNorm2Gain = tensor_create(s, 1, 0); }
    Tensor *dXMidFromNorm = t2(seq_len, dim);
    Tensor *dAttnOut = t2(seq_len, dim);
    Tensor *dRetentionMerged = t2(seq_len, dim);
    Tensor *dWo = t2(dim, dim);
    Tensor *dRetentionOutH = t3(heads, seq_len, head_dim);
    Tensor *dQh = t3(heads, seq_len, head_dim), *dKh = t3(heads, seq_len, head_dim), *dVh = t3(heads, seq_len, head_dim);
    Tensor *dQ = t2(seq_len, dim), *dK = t2(seq_len, dim), *dV = t2(seq_len, dim);
    Tensor *dWq = t2(dim, dim), *dWk = t2(dim, dim), *dWv = t2(dim, dim);
    Tensor *dNormed1_q = t2(seq_len, dim), *dNormed1_k = t2(seq_len, dim), *dNormed1_v = t2(seq_len, dim);
    Tensor *dNormed1_total = t2(seq_len, dim);
    Tensor *dNorm1Gain; { int s[] = {dim}; dNorm1Gain = tensor_create(s, 1, 0); }
    Tensor *dXInFromNorm1 = t2(seq_len, dim);

    for (int l = m->cfg.num_layers - 1; l >= 0; l--) {
        ModelLayer *layer = &m->layers[l];
        LayerCache *lc = &cache->layers[l];

        swiglu_backward(lc->normed2, layer->W1, layer->W3, layer->W2,
                         lc->ffn_h1, lc->ffn_h2, lc->ffn_a, lc->ffn_hidden,
                         dX, dNormed2, dW1, dW3, dW2);
        accumulate_grad(layer->W1, dW1);
        accumulate_grad(layer->W3, dW3);
        accumulate_grad(layer->W2, dW2);

        rmsnorm_backward(lc->x_mid, layer->norm2_gain, dNormed2, eps, dXMidFromNorm, dNorm2Gain);
        accumulate_grad(layer->norm2_gain, dNorm2Gain);

        Tensor *dXMid = dX;
        for (size_t i = 0; i < dXMid->size; i++) dXMid->data[i] += dXMidFromNorm->data[i];

        memcpy(dAttnOut->data, dXMid->data, sizeof(double) * dXMid->size);

        matmul_backward(dAttnOut, lc->retention_merged, layer->Wo, dRetentionMerged, dWo);
        accumulate_grad(layer->Wo, dWo);

        split_heads(dRetentionMerged, heads, head_dim, dRetentionOutH);

        MultiHeadRetentionConfig ret_cfg = {seq_len, heads, head_dim, layer->decay_per_head};
        retention_multihead_backward(&ret_cfg, lc->Qh, lc->Kh, lc->Vh, dRetentionOutH, dQh, dKh, dVh);

        merge_heads(dQh, heads, head_dim, dQ);
        merge_heads(dKh, heads, head_dim, dK);
        merge_heads(dVh, heads, head_dim, dV);

        matmul_backward(dQ, lc->normed1, layer->Wq, dNormed1_q, dWq);
        matmul_backward(dK, lc->normed1, layer->Wk, dNormed1_k, dWk);
        matmul_backward(dV, lc->normed1, layer->Wv, dNormed1_v, dWv);
        accumulate_grad(layer->Wq, dWq);
        accumulate_grad(layer->Wk, dWk);
        accumulate_grad(layer->Wv, dWv);

        for (size_t i = 0; i < dNormed1_total->size; i++) {
            dNormed1_total->data[i] = dNormed1_q->data[i] + dNormed1_k->data[i] + dNormed1_v->data[i];
        }

        rmsnorm_backward(lc->x_in, layer->norm1_gain, dNormed1_total, eps, dXInFromNorm1, dNorm1Gain);
        accumulate_grad(layer->norm1_gain, dNorm1Gain);

        for (size_t i = 0; i < dX->size; i++) {
            dX->data[i] = dXMid->data[i] + dXInFromNorm1->data[i];
        }
    }

    for (int i = 0; i < seq_len; i++) {
        int tok = token_ids[i];
        for (int d = 0; d < dim; d++) {
            m->embedding->grad[tok * dim + d] += dX->data[i * dim + d];
        }
    }

    tensor_free(dX);
    tensor_free(dNormed2); tensor_free(dW1); tensor_free(dW3); tensor_free(dW2); tensor_free(dNorm2Gain);
    tensor_free(dXMidFromNorm); tensor_free(dAttnOut); tensor_free(dRetentionMerged); tensor_free(dWo);
    tensor_free(dRetentionOutH); tensor_free(dQh); tensor_free(dKh); tensor_free(dVh);
    tensor_free(dQ); tensor_free(dK); tensor_free(dV);
    tensor_free(dWq); tensor_free(dWk); tensor_free(dWv);
    tensor_free(dNormed1_q); tensor_free(dNormed1_k); tensor_free(dNormed1_v);
    tensor_free(dNormed1_total); tensor_free(dNorm1Gain); tensor_free(dXInFromNorm1);

    return 0;
}
