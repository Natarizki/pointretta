// loss.c
#include "tensor.h"
#include "train.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double cross_entropy_loss(const Tensor *logits, const int *target_ids, int count,
                           Tensor *dLogits_out) {
    if (!logits || !target_ids || !dLogits_out) return -1.0;

    int seq_len = logits->shape[0];
    int vocab_size = logits->shape[1];

    tensor_fill(dLogits_out, 0.0);

    double total_loss = 0.0;
    double *probs = (double *)malloc(sizeof(double) * vocab_size);

    for (int i = 0; i < count && i < seq_len; i++) {
        double *logits_i = &logits->data[i * vocab_size];
        double *dlogits_i = &dLogits_out->data[i * vocab_size];

        double max_val = logits_i[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits_i[v] > max_val) max_val = logits_i[v];
        }
        double sum_exp = 0.0;
        for (int v = 0; v < vocab_size; v++) {
            probs[v] = exp(logits_i[v] - max_val);
            sum_exp += probs[v];
        }
        for (int v = 0; v < vocab_size; v++) {
            probs[v] /= sum_exp;
        }

        int target = target_ids[i];
        double loss_i = -log(probs[target] + 1e-12);
        total_loss += loss_i;

        for (int v = 0; v < vocab_size; v++) {
            double one_hot = (v == target) ? 1.0 : 0.0;
            dlogits_i[v] = (probs[v] - one_hot) / (double)count;
        }
    }

    free(probs);
    return total_loss / (double)count;
}
