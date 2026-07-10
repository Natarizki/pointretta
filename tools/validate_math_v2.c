// validate_math_v2.c
// Membuktikan: Retention parallel form == recurrent form
// VERSI MULTI-HEAD + MULTI-SCALE DECAY (tiap head punya decay rate sendiri)
//
// Compile: gcc -O2 -o validate_math_v2 validate_math_v2.c -lm
// Run:     ./validate_math_v2

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define SEQ_LEN 6
#define DIM 4
#define NUM_HEADS 3
#define EPS 1e-6

double DECAY_PER_HEAD[NUM_HEADS] = {0.7, 0.9, 0.99};

void fill_random(double *arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        arr[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void print_head_output(const char *name, double *out, int head) {
    printf("%s (head %d, decay=%.2f):\n", name, head, DECAY_PER_HEAD[head]);
    for (int i = 0; i < SEQ_LEN; i++) {
        printf("  [");
        for (int d = 0; d < DIM; d++) {
            printf("%8.5f ", out[(head * SEQ_LEN + i) * DIM + d]);
        }
        printf("]\n");
    }
}

void retention_parallel_multihead(double *Q, double *K, double *V, double *out) {
    for (int h = 0; h < NUM_HEADS; h++) {
        double decay = DECAY_PER_HEAD[h];
        double *Qh = Q + h * SEQ_LEN * DIM;
        double *Kh = K + h * SEQ_LEN * DIM;
        double *Vh = V + h * SEQ_LEN * DIM;
        double *outh = out + h * SEQ_LEN * DIM;

        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < DIM; d++) outh[i * DIM + d] = 0.0;

            for (int j = 0; j <= i; j++) {
                double qk = 0.0;
                for (int d = 0; d < DIM; d++) {
                    qk += Qh[i * DIM + d] * Kh[j * DIM + d];
                }
                double decay_factor = pow(decay, (double)(i - j));
                double weight = qk * decay_factor;

                for (int d = 0; d < DIM; d++) {
                    outh[i * DIM + d] += weight * Vh[j * DIM + d];
                }
            }
        }
    }
}

void retention_recurrent_multihead(double *Q, double *K, double *V, double *out) {
    for (int h = 0; h < NUM_HEADS; h++) {
        double decay = DECAY_PER_HEAD[h];
        double *Qh = Q + h * SEQ_LEN * DIM;
        double *Kh = K + h * SEQ_LEN * DIM;
        double *Vh = V + h * SEQ_LEN * DIM;
        double *outh = out + h * SEQ_LEN * DIM;

        double state[DIM][DIM];
        memset(state, 0, sizeof(state));

        for (int t = 0; t < SEQ_LEN; t++) {
            for (int a = 0; a < DIM; a++) {
                for (int b = 0; b < DIM; b++) {
                    state[a][b] = decay * state[a][b] + Kh[t * DIM + a] * Vh[t * DIM + b];
                }
            }

            for (int b = 0; b < DIM; b++) {
                double sum = 0.0;
                for (int a = 0; a < DIM; a++) {
                    sum += Qh[t * DIM + a] * state[a][b];
                }
                outh[t * DIM + b] = sum;
            }
        }
    }
}

int main(void) {
    int total_size = NUM_HEADS * SEQ_LEN * DIM;

    double *Q = malloc(total_size * sizeof(double));
    double *K = malloc(total_size * sizeof(double));
    double *V = malloc(total_size * sizeof(double));
    double *out_parallel = malloc(total_size * sizeof(double));
    double *out_recurrent = malloc(total_size * sizeof(double));

    fill_random(Q, total_size, 42);
    fill_random(K, total_size, 123);
    fill_random(V, total_size, 7);

    retention_parallel_multihead(Q, K, V, out_parallel);
    retention_recurrent_multihead(Q, K, V, out_recurrent);

    for (int h = 0; h < NUM_HEADS; h++) {
        print_head_output("Output parallel", out_parallel, h);
        print_head_output("Output recurrent", out_recurrent, h);
        printf("\n");
    }

    double max_diff = 0.0;
    for (int i = 0; i < total_size; i++) {
        double diff = fabs(out_parallel[i] - out_recurrent[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Max perbedaan numerik (semua head): %.10f\n", max_diff);

    if (max_diff < EPS) {
        printf("HASIL: LOLOS — multi-head + multi-scale decay tetap identik (toleransi %.0e)\n", EPS);
        free(Q); free(K); free(V); free(out_parallel); free(out_recurrent);
        return 0;
    } else {
        printf("HASIL: GAGAL — ada gap numerik, cek ulang derivasi!\n");
        free(Q); free(K); free(V); free(out_parallel); free(out_recurrent);
        return 1;
    }
}
