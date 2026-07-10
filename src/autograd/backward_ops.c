// backward_ops.c
#include "tensor.h"
#include "tensor_ops.h"
#include "backward_ops.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

int matmul_backward(const Tensor *dOut, const Tensor *A, const Tensor *B,
                     Tensor *dA, Tensor *dB) {
    if (!dOut || !A || !B || !dA || !dB) return -1;
    if (A->ndim != 2 || B->ndim != 2 || dOut->ndim != 2) return -1;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    if (B->shape[0] != K || dOut->shape[0] != M || dOut->shape[1] != N) return -1;
    if (dA->shape[0] != M || dA->shape[1] != K) return -1;
    if (dB->shape[0] != K || dB->shape[1] != N) return -1;

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            double sum = 0.0;
            for (int n = 0; n < N; n++) {
                sum += dOut->data[m * N + n] * B->data[k * N + n];
            }
            dA->data[m * K + k] = sum;
        }
    }

    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int m = 0; m < M; m++) {
                sum += A->data[m * K + k] * dOut->data[m * N + n];
            }
            dB->data[k * N + n] = sum;
        }
    }

    return 0;
}

int rmsnorm_backward(const Tensor *x, const Tensor *gain, const Tensor *dOut, double eps,
                      Tensor *dX, Tensor *dGain) {
    if (!x || !gain || !dOut || !dX || !dGain) return -1;
    if (x->ndim != 2 || dOut->ndim != 2 || dX->ndim != 2) return -1;

    int seq_len = x->shape[0];
    int dim = x->shape[1];

    tensor_fill(dGain, 0.0);

    for (int i = 0; i < seq_len; i++) {
        double *x_i = &x->data[i * dim];
        double *dOut_i = &dOut->data[i * dim];
        double *dX_i = &dX->data[i * dim];

        double sum_sq = 0.0;
        for (int d = 0; d < dim; d++) sum_sq += x_i[d] * x_i[d];
        double mean_sq = sum_sq / (double)dim;
        double rms = sqrt(mean_sq + eps);
        double s = 1.0 / rms;

        double sum_term = 0.0;
        for (int d = 0; d < dim; d++) {
            sum_term += dOut_i[d] * gain->data[d] * x_i[d];
        }

        double rms3 = rms * rms * rms;
        for (int d = 0; d < dim; d++) {
            dX_i[d] = dOut_i[d] * gain->data[d] * s - (x_i[d] / ((double)dim * rms3)) * sum_term;
            dGain->data[d] += dOut_i[d] * x_i[d] * s;
        }
    }

    return 0;
}

static double silu_val(double z) {
    double sig = 1.0 / (1.0 + exp(-z));
    return z * sig;
}
static double silu_grad(double z) {
    double sig = 1.0 / (1.0 + exp(-z));
    return sig * (1.0 + z * (1.0 - sig));
}

int swiglu_forward_for_backward(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                                 Tensor *h1, Tensor *h2, Tensor *a, Tensor *hidden, Tensor *out) {
    if (!x || !W1 || !W3 || !W2 || !h1 || !h2 || !a || !hidden || !out) return -1;

    int rc = 0;
    rc |= tensor_matmul(h1, x, W1);
    rc |= tensor_matmul(h2, x, W3);
    if (rc != 0) return -1;

    for (size_t i = 0; i < h1->size; i++) {
        a->data[i] = silu_val(h1->data[i]);
        hidden->data[i] = a->data[i] * h2->data[i];
    }

    rc |= tensor_matmul(out, hidden, W2);
    return rc;
}

int swiglu_backward(const Tensor *x, const Tensor *W1, const Tensor *W3, const Tensor *W2,
                     const Tensor *h1, const Tensor *h2, const Tensor *a, const Tensor *hidden,
                     const Tensor *dOut,
                     Tensor *dX, Tensor *dW1, Tensor *dW3, Tensor *dW2) {
    if (!x || !W1 || !W3 || !W2 || !h1 || !h2 || !a || !hidden || !dOut) return -1;
    if (!dX || !dW1 || !dW3 || !dW2) return -1;

    int seq_len = x->shape[0];
    int hidden_dim = W1->shape[1];

    int shape_hidden[] = {seq_len, hidden_dim};
    Tensor *dHidden = tensor_create(shape_hidden, 2, 0);
    Tensor *dA = tensor_create(shape_hidden, 2, 0);
    Tensor *dH2 = tensor_create(shape_hidden, 2, 0);
    Tensor *dH1 = tensor_create(shape_hidden, 2, 0);

    int dim = W2->shape[1];
    for (int m = 0; m < seq_len; m++) {
        for (int k = 0; k < hidden_dim; k++) {
            double sum = 0.0;
            for (int n = 0; n < dim; n++) {
                sum += dOut->data[m * dim + n] * W2->data[k * dim + n];
            }
            dHidden->data[m * hidden_dim + k] = sum;
        }
    }

    for (int k = 0; k < hidden_dim; k++) {
        for (int n = 0; n < dim; n++) {
            double sum = 0.0;
            for (int m = 0; m < seq_len; m++) {
                sum += hidden->data[m * hidden_dim + k] * dOut->data[m * dim + n];
            }
            dW2->data[k * dim + n] = sum;
        }
    }

    for (size_t i = 0; i < dHidden->size; i++) {
        dA->data[i] = dHidden->data[i] * h2->data[i];
        dH2->data[i] = dHidden->data[i] * a->data[i];
        dH1->data[i] = dA->data[i] * silu_grad(h1->data[i]);
    }

    int x_dim = x->shape[1];
    tensor_fill(dX, 0.0);

    for (int m = 0; m < seq_len; m++) {
        for (int d = 0; d < x_dim; d++) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int h = 0; h < hidden_dim; h++) {
                sum1 += dH1->data[m * hidden_dim + h] * W1->data[d * hidden_dim + h];
                sum2 += dH2->data[m * hidden_dim + h] * W3->data[d * hidden_dim + h];
            }
            dX->data[m * x_dim + d] = sum1 + sum2;
        }
    }

    for (int d = 0; d < x_dim; d++) {
        for (int h = 0; h < hidden_dim; h++) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int m = 0; m < seq_len; m++) {
                sum1 += x->data[m * x_dim + d] * dH1->data[m * hidden_dim + h];
                sum2 += x->data[m * x_dim + d] * dH2->data[m * hidden_dim + h];
            }
            dW1->data[d * hidden_dim + h] = sum1;
            dW3->data[d * hidden_dim + h] = sum2;
        }
    }

    tensor_free(dHidden); tensor_free(dA); tensor_free(dH2); tensor_free(dH1);
    return 0;
}

int retention_multihead_backward(const MultiHeadRetentionConfig *cfg,
                                  const Tensor *Qh, const Tensor *Kh, const Tensor *Vh,
                                  const Tensor *dOut_h,
                                  Tensor *dQh, Tensor *dKh, Tensor *dVh) {
    if (!cfg || !Qh || !Kh || !Vh || !dOut_h || !dQh || !dKh || !dVh) return -1;

    int seq_len = cfg->seq_len;
    int num_heads = cfg->num_heads;
    int head_dim = cfg->head_dim;
    int head_stride = seq_len * head_dim;

    tensor_fill(dQh, 0.0);
    tensor_fill(dKh, 0.0);
    tensor_fill(dVh, 0.0);

    for (int h = 0; h < num_heads; h++) {
        double decay = cfg->decay_per_head[h];
        double *Qh_ptr = &Qh->data[h * head_stride];
        double *Kh_ptr = &Kh->data[h * head_stride];
        double *Vh_ptr = &Vh->data[h * head_stride];
        double *g_ptr = &dOut_h->data[h * head_stride];
        double *dQ_ptr = &dQh->data[h * head_stride];
        double *dK_ptr = &dKh->data[h * head_stride];
        double *dV_ptr = &dVh->data[h * head_stride];

        for (int i = 0; i < seq_len; i++) {
            double *g_i = &g_ptr[i * head_dim];
            double *dq_i = &dQ_ptr[i * head_dim];
            for (int j = 0; j <= i; j++) {
                double *v_j = &Vh_ptr[j * head_dim];
                double *k_j = &Kh_ptr[j * head_dim];
                double gv = vec_dot(g_i, v_j, head_dim);
                double decay_factor = pow(decay, (double)(i - j));
                double weight = gv * decay_factor;
                for (int d = 0; d < head_dim; d++) dq_i[d] += weight * k_j[d];
            }
        }

        for (int j = 0; j < seq_len; j++) {
            double *v_j = &Vh_ptr[j * head_dim];
            double *dk_j = &dK_ptr[j * head_dim];
            for (int i = j; i < seq_len; i++) {
                double *g_i = &g_ptr[i * head_dim];
                double *q_i = &Qh_ptr[i * head_dim];
                double gv = vec_dot(g_i, v_j, head_dim);
                double decay_factor = pow(decay, (double)(i - j));
                double weight = gv * decay_factor;
                for (int d = 0; d < head_dim; d++) dk_j[d] += weight * q_i[d];
            }
        }

        for (int j = 0; j < seq_len; j++) {
            double *k_j = &Kh_ptr[j * head_dim];
            double *dv_j = &dV_ptr[j * head_dim];
            for (int i = j; i < seq_len; i++) {
                double *q_i = &Qh_ptr[i * head_dim];
                double *g_i = &g_ptr[i * head_dim];
                double qk = vec_dot(q_i, k_j, head_dim);
                double decay_factor = pow(decay, (double)(i - j));
                double weight = qk * decay_factor;
                for (int d = 0; d < head_dim; d++) dv_j[d] += weight * g_i[d];
            }
        }
    }

    return 0;
}
