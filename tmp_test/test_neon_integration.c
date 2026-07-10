#include "tensor.h"
#include "tensor_ops.h"
#include <stdio.h>
#include <math.h>

extern int tensor_matmul_neon(Tensor *out, const Tensor *A, const Tensor *B);

int main(void) {
    int shapeA[] = {5, 6};
    int shapeB[] = {6, 7};
    int shapeOut[] = {5, 7};
    Tensor *A = tensor_create(shapeA, 2, 0);
    Tensor *B = tensor_create(shapeB, 2, 0);
    Tensor *out_c = tensor_create(shapeOut, 2, 0);
    Tensor *out_asm = tensor_create(shapeOut, 2, 0);

    tensor_fill_random(A, -1.0, 1.0, 1);
    tensor_fill_random(B, -1.0, 1.0, 2);

    tensor_matmul(out_c, A, B);
    tensor_matmul_neon(out_asm, A, B);

    double max_diff = 0.0;
    for (size_t i = 0; i < out_c->size; i++) {
        double diff = fabs(out_c->data[i] - out_asm->data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff tensor_matmul vs tensor_matmul_neon: %.15f\n", max_diff);
    printf("%s\n", (max_diff < 1e-9) ? "LOLOS" : "GAGAL");

    return (max_diff < 1e-9) ? 0 : 1;
}
