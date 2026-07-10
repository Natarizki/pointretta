// test_tensor_ops.c
#include "tensor.h"
#include "tensor_ops.h"
#include <stdio.h>

int main(void) {
    printf("=== Test 1: tensor_add ===\n");
    int shape[] = {2, 2};
    Tensor *a = tensor_create(shape, 2, 0);
    Tensor *b = tensor_create(shape, 2, 0);
    Tensor *out = tensor_create(shape, 2, 0);

    a->data[0] = 1; a->data[1] = 2; a->data[2] = 3; a->data[3] = 4;
    b->data[0] = 10; b->data[1] = 20; b->data[2] = 30; b->data[3] = 40;

    tensor_add(out, a, b);
    tensor_print(out, "a + b (harusnya 11,22,33,44)");

    printf("\n=== Test 2: tensor_sub ===\n");
    tensor_sub(out, b, a);
    tensor_print(out, "b - a (harusnya 9,18,27,36)");

    printf("\n=== Test 3: tensor_mul_elementwise ===\n");
    tensor_mul_elementwise(out, a, b);
    tensor_print(out, "a * b elementwise (harusnya 10,40,90,160)");

    printf("\n=== Test 4: tensor_scale ===\n");
    tensor_scale(out, a, 2.5);
    tensor_print(out, "a * 2.5 (harusnya 2.5,5,7.5,10)");

    printf("\n=== Test 5: tensor_matmul ===\n");
    int shapeA[] = {2, 3};
    int shapeB[] = {3, 2};
    int shapeOut[] = {2, 2};
    Tensor *A = tensor_create(shapeA, 2, 0);
    Tensor *B = tensor_create(shapeB, 2, 0);
    Tensor *C = tensor_create(shapeOut, 2, 0);

    double a_vals[] = {1,2,3,4,5,6};
    for (int i = 0; i < 6; i++) A->data[i] = a_vals[i];

    double b_vals[] = {7,8,9,10,11,12};
    for (int i = 0; i < 6; i++) B->data[i] = b_vals[i];

    int result = tensor_matmul(C, A, B);
    printf("matmul result code: %d (harusnya 0)\n", result);
    tensor_print(C, "A @ B (harusnya 58,64,139,154)");

    printf("\n=== Test 6: vec_dot ===\n");
    double v1[] = {1, 2, 3};
    double v2[] = {4, 5, 6};
    double dot = vec_dot(v1, v2, 3);
    printf("dot([1,2,3],[4,5,6]) = %.2f (harusnya 32.00)\n", dot);

    printf("\n=== Test 7: vec_outer ===\n");
    double v3[] = {1, 2};
    double v4[] = {3, 4, 5};
    double outer_result[6];
    vec_outer(v3, 2, v4, 3, outer_result);
    printf("outer([1,2],[3,4,5]) = [");
    for (int i = 0; i < 6; i++) printf("%.2f ", outer_result[i]);
    printf("] (harusnya 3,4,5,6,8,10)\n");

    printf("\n=== Test 8: Matmul dimensi salah (harus gagal) ===\n");
    int shapeBad[] = {5, 5};
    Tensor *bad = tensor_create(shapeBad, 2, 0);
    int fail = tensor_matmul(bad, A, B);
    printf("matmul dengan shape salah: %d (harusnya -1)\n", fail);

    tensor_free(a); tensor_free(b); tensor_free(out);
    tensor_free(A); tensor_free(B); tensor_free(C); tensor_free(bad);

    printf("\n=== SEMUA TEST SELESAI ===\n");
    return 0;
}
