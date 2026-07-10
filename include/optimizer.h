// optimizer.h
#ifndef POINTRETTA_OPTIMIZER_H
#define POINTRETTA_OPTIMIZER_H

#include "tensor.h"

typedef struct {
    int is_factored;
    int rows, cols;

    double *row_factor;
    double *col_factor;
    double *full_second_moment;

    double beta2;
    double eps1;
    double eps2;
    int step;
} AdafactorState;

AdafactorState *adafactor_state_create(const Tensor *param, double beta2, double eps1, double eps2);
void adafactor_state_free(AdafactorState *state);
int adafactor_step(Tensor *param, AdafactorState *state, double lr);

#endif // POINTRETTA_OPTIMIZER_H
