#include "tensor.h"
#include "retention.h"
#include <stdio.h>
#include <math.h>

extern int retention_recurrent_step_neon(double decay, int head_dim,
                                          Tensor *state, const Tensor *q_t, const Tensor *k_t,
                                          const Tensor *v_t, Tensor *out_t);

int main(void) {
    int head_dim = 8;
    double decay = 0.9;
    RetentionConfig cfg = { .seq_len = 20, .dim = head_dim, .decay = decay };

    Tensor *state_c = retention_state_create(head_dim);
    Tensor *state_asm = retention_state_create(head_dim);

    Tensor *q = tensor_create((int[]){head_dim}, 1, 0);
    Tensor *k = tensor_create((int[]){head_dim}, 1, 0);
    Tensor *v = tensor_create((int[]){head_dim}, 1, 0);
    Tensor *out_c = tensor_create((int[]){head_dim}, 1, 0);
    Tensor *out_asm = tensor_create((int[]){head_dim}, 1, 0);

    double max_diff = 0.0;
    for (int t = 0; t < 20; t++) {
        tensor_fill_random(q, -1.0, 1.0, 100 + t);
        tensor_fill_random(k, -1.0, 1.0, 200 + t);
        tensor_fill_random(v, -1.0, 1.0, 300 + t);

        retention_recurrent_step(&cfg, state_c, q, k, v, out_c);
        retention_recurrent_step_neon(decay, head_dim, state_asm, q, k, v, out_asm);

        for (int d = 0; d < head_dim; d++) {
            double diff = fabs(out_c->data[d] - out_asm->data[d]);
            if (diff > max_diff) max_diff = diff;
        }
    }

    printf("Max diff retention_recurrent_step vs versi ASM (20 langkah): %.15f\n", max_diff);
    printf("%s\n", (max_diff < 1e-9) ? "LOLOS" : "GAGAL");

    return (max_diff < 1e-9) ? 0 : 1;
}
