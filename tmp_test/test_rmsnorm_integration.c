#include "tensor.h"
#include "norm.h"
#include <stdio.h>
#include <math.h>

extern int rmsnorm_forward_neon(const Tensor *x, const Tensor *gain, Tensor *out, double eps);

int main(void) {
    int shape[] = {5, 64};
    Tensor *x = tensor_create(shape, 2, 0);
    Tensor *gain = tensor_create((int[]){64}, 1, 0);
    Tensor *out_c = tensor_create(shape, 2, 0);
    Tensor *out_asm = tensor_create(shape, 2, 0);

    tensor_fill_random(x, -1.0, 1.0, 1);
    tensor_fill_random(gain, 0.5, 1.5, 2);

    double eps = 1e-6;
    rmsnorm_forward(x, gain, out_c, eps);
    rmsnorm_forward_neon(x, gain, out_asm, eps);

    double max_diff = 0.0;
    for (size_t i = 0; i < out_c->size; i++) {
        double diff = fabs(out_c->data[i] - out_asm->data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff rmsnorm_forward vs rmsnorm_forward_neon: %.15f\n", max_diff);
    printf("%s\n", (max_diff < 1e-9) ? "LOLOS" : "GAGAL");

    return (max_diff < 1e-9) ? 0 : 1;
}
