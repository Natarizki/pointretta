// validate_math_v3.c
// Membuktikan: Retention + xPos (rotation-based position encoding)
// parallel form == recurrent form

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

void compute_freqs(double *freqs) {
    for (int k = 0; k < DIM / 2; k++) {
        freqs[k] = 1.0 / pow(10000.0, (2.0 * k) / DIM);
    }
}

void apply_xpos_rotation(double *v, int pos, double *freqs, double *v_out) {
    for (int k = 0; k < DIM / 2; k++) {
        double angle = pos * freqs[k];
        double c = cos(angle);
        double s = sin(angle);
        double x0 = v[2 * k];
        double x1 = v[2 * k + 1];
        v_out[2 * k]     = x0 * c - x1 * s;
        v_out[2 * k + 1] = x0 * s + x1 * c;
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

void retention_parallel_xpos(double *Q, double *K, double *V, double *out, double *freqs) {
    double q_rot[DIM], k_rot[DIM];

    for (int h = 0; h < NUM_HEADS; h++) {
        double decay = DECAY_PER_HEAD[h];
        double *Qh = Q + h * SEQ_LEN * DIM;
        double *Kh = K + h * SEQ_LEN * DIM;
        double *Vh = V + h * SEQ_LEN * DIM;
        double *outh = out + h * SEQ_LEN * DIM;

        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < DIM; d++) outh[i * DIM + d] = 0.0;

            apply_xpos_rotation(&Qh[i * DIM], i, freqs, q_rot);

            for (int j = 0; j <= i; j++) {
                apply_xpos_rotation(&Kh[j * DIM], j, freqs, k_rot);

                double qk = 0.0;
                for (int d = 0; d < DIM; d++) {
                    qk += q_rot[d] * k_rot[d];
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

void retention_recurrent_xpos(double *Q, double *K, double *V, double *out, double *freqs) {
    double q_rot[DIM], k_rot[DIM];

    for (int h = 0; h < NUM_HEADS; h++) {
        double decay = DECAY_PER_HEAD[h];
        double *Qh = Q + h * SEQ_LEN * DIM;
        double *Kh = K + h * SEQ_LEN * DIM;
        double *Vh = V + h * SEQ_LEN * DIM;
        double *outh = out + h * SEQ_LEN * DIM;

        double state[DIM][DIM];
        memset(state, 0, sizeof(state));

        for (int t = 0; t < SEQ_LEN; t++) {
            apply_xpos_rotation(&Kh[t * DIM], t, freqs, k_rot);
            apply_xpos_rotation(&Qh[t * DIM], t, freqs, q_rot);

            for (int a = 0; a < DIM; a++) {
                for (int b = 0; b < DIM; b++) {
                    state[a][b] = decay * state[a][b] + k_rot[a] * Vh[t * DIM + b];
                }
            }

            for (int b = 0; b < DIM; b++) {
                double sum = 0.0;
                for (int a = 0; a < DIM; a++) {
                    sum += q_rot[a] * state[a][b];
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
    double freqs[DIM / 2];

    fill_random(Q, total_size, 42);
    fill_random(K, total_size, 123);
    fill_random(V, total_size, 7);
    compute_freqs(freqs);

    retention_parallel_xpos(Q, K, V, out_parallel, freqs);
    retention_recurrent_xpos(Q, K, V, out_recurrent, freqs);

    for (int h = 0; h < NUM_HEADS; h++) {
        print_head_output("Output parallel+xPos", out_parallel, h);
        print_head_output("Output recurrent+xPos", out_recurrent, h);
        printf("\n");
    }

    double max_diff = 0.0;
    for (int i = 0; i < total_size; i++) {
        double diff = fabs(out_parallel[i] - out_recurrent[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Max perbedaan numerik (semua head, dengan xPos): %.10f\n", max_diff);

    if (max_diff < EPS) {
        printf("HASIL: LOLOS — retention + xPos tetap identik parallel vs recurrent (toleransi %.0e)\n", EPS);
        free(Q); free(K); free(V); free(out_parallel); free(out_recurrent);
        return 0;
    } else {
        printf("HASIL: GAGAL — xPos merusak ekuivalensi, perlu revisi derivasi!\n");
        free(Q); free(K); free(V); free(out_parallel); free(out_recurrent);
        return 1;
    }
}
