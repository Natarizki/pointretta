// adafactor.c
#include "tensor.h"
#include "optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

AdafactorState *adafactor_state_create(const Tensor *param, double beta2, double eps1, double eps2) {
    if (!param) return NULL;

    AdafactorState *state = (AdafactorState *)malloc(sizeof(AdafactorState));
    if (!state) return NULL;

    state->beta2 = beta2;
    state->eps1 = eps1;
    state->eps2 = eps2;
    state->step = 0;
    state->row_factor = NULL;
    state->col_factor = NULL;
    state->full_second_moment = NULL;

    if (param->ndim == 2) {
        state->is_factored = 1;
        state->rows = param->shape[0];
        state->cols = param->shape[1];

        state->row_factor = (double *)calloc(state->rows, sizeof(double));
        state->col_factor = (double *)calloc(state->cols, sizeof(double));

        if (!state->row_factor || !state->col_factor) {
            free(state->row_factor);
            free(state->col_factor);
            free(state);
            return NULL;
        }
    } else {
        state->is_factored = 0;
        state->rows = 0;
        state->cols = 0;
        state->full_second_moment = (double *)calloc(param->size, sizeof(double));
        if (!state->full_second_moment) {
            free(state);
            return NULL;
        }
    }

    return state;
}

void adafactor_state_free(AdafactorState *state) {
    if (!state) return;
    if (state->row_factor) free(state->row_factor);
    if (state->col_factor) free(state->col_factor);
    if (state->full_second_moment) free(state->full_second_moment);
    free(state);
}

int adafactor_step(Tensor *param, AdafactorState *state, double lr) {
    if (!param || !state || !param->grad) return -1;

    state->step++;
    double beta2 = state->beta2;
    double eps1 = state->eps1;
    double eps2 = state->eps2;

    if (state->is_factored) {
        int rows = state->rows;
        int cols = state->cols;

        if (param->shape[0] != rows || param->shape[1] != cols) {
            fprintf(stderr, "[adafactor_step] shape param berubah dari saat state dibuat\n");
            return -1;
        }

        double *row_means = (double *)calloc(rows, sizeof(double));
        double *col_means = (double *)calloc(cols, sizeof(double));
        if (!row_means || !col_means) {
            free(row_means); free(col_means);
            return -1;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double g = param->grad[i * cols + j];
                double g2 = g * g + eps1;
                row_means[i] += g2;
                col_means[j] += g2;
            }
        }
        for (int i = 0; i < rows; i++) row_means[i] /= (double)cols;
        for (int j = 0; j < cols; j++) col_means[j] /= (double)rows;

        for (int i = 0; i < rows; i++) {
            state->row_factor[i] = beta2 * state->row_factor[i] + (1.0 - beta2) * row_means[i];
        }
        for (int j = 0; j < cols; j++) {
            state->col_factor[j] = beta2 * state->col_factor[j] + (1.0 - beta2) * col_means[j];
        }

        double row_sum = 0.0;
        for (int i = 0; i < rows; i++) row_sum += state->row_factor[i];
        if (row_sum < 1e-12) row_sum = 1e-12;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v_ij = (state->row_factor[i] * state->col_factor[j]) / row_sum;
                double g = param->grad[i * cols + j];
                double update = g / sqrt(v_ij + eps2);
                param->data[i * cols + j] -= lr * update;
            }
        }

        free(row_means);
        free(col_means);
    } else {
        for (size_t i = 0; i < param->size; i++) {
            double g = param->grad[i];
            double g2 = g * g;
            state->full_second_moment[i] = beta2 * state->full_second_moment[i] + (1.0 - beta2) * g2;

            double update = g / sqrt(state->full_second_moment[i] + eps2);
            param->data[i] -= lr * update;
        }
    }

    return 0;
}
