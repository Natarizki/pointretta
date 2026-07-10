// validate_math.c
// Membuktikan: Retention parallel form == recurrent form (secara numerik)
//
// Parallel form:   Output = (Q K^T ⊙ D) V
// Recurrent form:  state_t = decay * state_{t-1} + K_t^T V_t
//                  output_t = Q_t * state_t
//
// Compile: gcc -O2 -o validate_math validate_math.c -lm
// Run:     ./validate_math

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define SEQ_LEN 6      // panjang sequence (kecil biar gampang dicek)
#define DIM 4          // dimensi Q/K/V per head (kecil dulu)
#define DECAY 0.9      // decay rate tunggal (single-scale dulu, belum multi-scale)
#define EPS 1e-6       // toleransi perbedaan numerik yang dianggap "sama"

// ------- Util: random kecil untuk isi Q, K, V -------
void fill_random(double *arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        arr[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // range [-1, 1]
    }
}

void print_matrix(const char *name, double *m, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%8.5f ", m[i * cols + j]);
        }
        printf("]\n");
    }
}

// ------- PARALLEL FORM -------
// Output[i] = sum_j ( (Q[i] . K[j]) * decay^(i-j) * [j <= i] ) * V[j]
void retention_parallel(double *Q, double *K, double *V, double *out) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < DIM; d++) {
            out[i * DIM + d] = 0.0;
        }
        for (int j = 0; j <= i; j++) {  // causal: j <= i doang
            // hitung Q[i] . K[j] (dot product)
            double qk = 0.0;
            for (int d = 0; d < DIM; d++) {
                qk += Q[i * DIM + d] * K[j * DIM + d];
            }
            // decay factor berdasarkan jarak posisi
            double decay_factor = pow(DECAY, (double)(i - j));
            double weight = qk * decay_factor;

            // akumulasi ke output: weight * V[j]
            for (int d = 0; d < DIM; d++) {
                out[i * DIM + d] += weight * V[j * DIM + d];
            }
        }
    }
}

// ------- RECURRENT FORM -------
// state adalah matrix ukuran [DIM x DIM], di-update tiap token:
//   state = decay * state + K[t]^T outer V[t]
// output[t] = Q[t] . state   (matrix-vector product)
void retention_recurrent(double *Q, double *K, double *V, double *out) {
    double state[DIM][DIM];
    memset(state, 0, sizeof(state));

    for (int t = 0; t < SEQ_LEN; t++) {
        // update state: state = decay * state + outer(K[t], V[t])
        for (int a = 0; a < DIM; a++) {
            for (int b = 0; b < DIM; b++) {
                state[a][b] = DECAY * state[a][b] + K[t * DIM + a] * V[t * DIM + b];
            }
        }

        // output[t] = Q[t] * state  (vector-matrix product)
        for (int b = 0; b < DIM; b++) {
            double sum = 0.0;
            for (int a = 0; a < DIM; a++) {
                sum += Q[t * DIM + a] * state[a][b];
            }
            out[t * DIM + b] = sum;
        }
    }
}

int main(void) {
    double Q[SEQ_LEN * DIM];
    double K[SEQ_LEN * DIM];
    double V[SEQ_LEN * DIM];
    double out_parallel[SEQ_LEN * DIM];
    double out_recurrent[SEQ_LEN * DIM];

    fill_random(Q, SEQ_LEN * DIM, 42);
    fill_random(K, SEQ_LEN * DIM, 123);
    fill_random(V, SEQ_LEN * DIM, 7);

    retention_parallel(Q, K, V, out_parallel);
    retention_recurrent(Q, K, V, out_recurrent);

    print_matrix("Output (parallel form)", out_parallel, SEQ_LEN, DIM);
    print_matrix("Output (recurrent form)", out_recurrent, SEQ_LEN, DIM);

    // Bandingkan elemen per elemen
    double max_diff = 0.0;
    int total = SEQ_LEN * DIM;
    for (int i = 0; i < total; i++) {
        double diff = fabs(out_parallel[i] - out_recurrent[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("\nMax perbedaan numerik: %.10f\n", max_diff);

    if (max_diff < EPS) {
        printf("HASIL: LOLOS — parallel form dan recurrent form identik (dalam toleransi %.0e)\n", EPS);
        return 0;
    } else {
        printf("HASIL: GAGAL — ada gap numerik signifikan, cek ulang derivasi matematika!\n");
        return 1;
    }
}
