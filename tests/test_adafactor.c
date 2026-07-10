// test_adafactor.c
#include "tensor.h"
#include "optimizer.h"
#include <stdio.h>
#include <math.h>

#define NUM_ITERS 500
#define LR 0.1

int main(void) {
    printf("=== Test 1: Adafactor pada parameter 2D (factored) ===\n");
    int shape2d[] = {4, 3};
    Tensor *param2d = tensor_create(shape2d, 2, 1);
    Tensor *target2d = tensor_create(shape2d, 2, 0);

    tensor_fill_random(param2d, -1.0, 1.0, 1);
    tensor_fill_random(target2d, 5.0, 10.0, 2);

    AdafactorState *state2d = adafactor_state_create(param2d, 0.999, 1e-30, 1e-3);
    if (!state2d) {
        printf("GAGAL: adafactor_state_create return NULL\n");
        return 1;
    }

    double loss_start = 0.0, loss_end = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        double loss = 0.0;
        for (size_t i = 0; i < param2d->size; i++) {
            double diff = param2d->data[i] - target2d->data[i];
            loss += diff * diff;
            param2d->grad[i] = 2.0 * diff;
        }
        if (iter == 0) loss_start = loss;
        if (iter == NUM_ITERS - 1) loss_end = loss;

        adafactor_step(param2d, state2d, LR);

        if (iter % 100 == 0) {
            printf("  Iter %3d | Loss: %.6f\n", iter, loss);
        }
    }

    printf("Loss awal: %.6f -> Loss akhir: %.6f\n", loss_start, loss_end);
    int pass2d = (loss_end < loss_start * 0.01);
    printf("Test 2D: %s\n\n", pass2d ? "LOLOS (loss turun signifikan)" : "GAGAL (loss tidak konvergen)");

    printf("=== Test 2: Adafactor pada parameter 1D (full second moment) ===\n");
    int shape1d[] = {10};
    Tensor *param1d = tensor_create(shape1d, 1, 1);
    Tensor *target1d = tensor_create(shape1d, 1, 0);

    tensor_fill_random(param1d, -1.0, 1.0, 10);
    tensor_fill_random(target1d, 3.0, 6.0, 20);

    AdafactorState *state1d = adafactor_state_create(param1d, 0.999, 1e-30, 1e-3);

    double loss1d_start = 0.0, loss1d_end = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        double loss = 0.0;
        for (size_t i = 0; i < param1d->size; i++) {
            double diff = param1d->data[i] - target1d->data[i];
            loss += diff * diff;
            param1d->grad[i] = 2.0 * diff;
        }
        if (iter == 0) loss1d_start = loss;
        if (iter == NUM_ITERS - 1) loss1d_end = loss;

        adafactor_step(param1d, state1d, LR);
    }

    printf("Loss awal: %.6f -> Loss akhir: %.6f\n", loss1d_start, loss1d_end);
    int pass1d = (loss1d_end < loss1d_start * 0.01);
    printf("Test 1D: %s\n\n", pass1d ? "LOLOS (loss turun signifikan)" : "GAGAL (loss tidak konvergen)");

    printf("=== Perbandingan Memory: Adafactor vs Adam-style ===\n");
    size_t param_elements = param2d->size;
    size_t adam_memory = param_elements * sizeof(double);
    size_t adafactor_memory = (state2d->rows + state2d->cols) * sizeof(double);
    printf("Parameter size: %zu elemen [%d x %d]\n", param_elements, state2d->rows, state2d->cols);
    printf("Adam-style (full second moment): %zu byte\n", adam_memory);
    printf("Adafactor (row+col factor):      %zu byte\n", adafactor_memory);
    printf("Penghematan: %.1fx lebih kecil\n", (double)adam_memory / adafactor_memory);

    int all_pass = pass2d && pass1d;
    printf("\n=== HASIL AKHIR: %s ===\n", all_pass ? "SEMUA TEST LOLOS" : "ADA TEST GAGAL");

    adafactor_state_free(state2d);
    adafactor_state_free(state1d);
    tensor_free(param2d); tensor_free(target2d);
    tensor_free(param1d); tensor_free(target1d);

    return all_pass ? 0 : 1;
}
