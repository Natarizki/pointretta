// test_tensor.c
#include "tensor.h"
#include <stdio.h>

int main(void) {
    printf("=== Test 1: Create tensor 2x3 ===\n");
    int shape1[] = {2, 3};
    Tensor *a = tensor_create(shape1, 2, 0);
    if (!a) {
        printf("GAGAL: tensor_create return NULL\n");
        return 1;
    }
    tensor_fill(a, 5.0);
    tensor_print(a, "a");

    printf("\n=== Test 2: Set & Get elemen ===\n");
    int coord[] = {1, 2};
    tensor_set(a, coord, 99.0);
    double val = tensor_get(a, coord);
    printf("a[1][2] = %.2f (harusnya 99.00)\n", val);
    tensor_print(a, "a setelah set");

    printf("\n=== Test 3: Random fill ===\n");
    Tensor *b = tensor_create(shape1, 2, 0);
    tensor_fill_random(b, -1.0, 1.0, 42);
    tensor_print(b, "b (random)");

    printf("\n=== Test 4: Copy tensor ===\n");
    Tensor *c = tensor_create(shape1, 2, 0);
    int copy_result = tensor_copy(c, b);
    printf("copy result: %d (harusnya 0)\n", copy_result);
    tensor_print(c, "c (copy dari b)");

    printf("\n=== Test 5: Tensor dengan gradient ===\n");
    Tensor *d = tensor_create(shape1, 2, 1);
    tensor_fill(d, 1.0);
    d->grad[0] = 0.5;
    d->grad[1] = 0.25;
    printf("d->grad[0] = %.2f, d->grad[1] = %.2f\n", d->grad[0], d->grad[1]);
    tensor_zero_grad(d);
    printf("Setelah zero_grad: d->grad[0] = %.2f, d->grad[1] = %.2f (harusnya 0.00 semua)\n",
           d->grad[0], d->grad[1]);

    printf("\n=== Test 6: Copy dengan size beda (harus gagal) ===\n");
    int shape2[] = {3, 3};
    Tensor *e = tensor_create(shape2, 2, 0);
    int fail_result = tensor_copy(e, a);
    printf("copy result: %d (harusnya -1, karena size beda)\n", fail_result);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(e);

    printf("\n=== SEMUA TEST SELESAI ===\n");
    return 0;
}
